from ..utils.general_utils import AttrDict, listdict2dictlist
from ..utils.rl_utils import ReplayCache

import os
import torch
import numpy as np
import PIL.Image as Image

import pybullet as p

from surrol.tasks.needle_pick import NeedlePick
from surrol.tasks.needle_pick_sphere import NeedlePickSphere
from surrol.tasks.needle_pick_cylinder import NeedlePickCylinder
from surrol.tasks.needle_pick_wound_for_clf import NeedlePickWoundCLF
from surrol.tasks.needle_pick_lung_clf_cbf import NeedlePickLungCLF
from surrol.tasks.needle_pick_trajectory_clf import NeedlePickTrajectoryCLF
from surrol.tasks.needle_pick_trajectory_cbf import NeedlePickTrajectoryCBF
from surrol.tasks.gauze_retrieve import GauzeRetrieve
from surrol.tasks.gauze_retrieve_sphere import GauzeRetrieveSphere
from surrol.tasks.gauze_retrieve_cylinder import GauzeRetrieveCylinder
from surrol.tasks.needle_reach_sphere import NeedleReach as NeedleReachSphere
from surrol.tasks.needle_reach_plate_obstacle import NeedleReach as NeedleReachPlate
from surrol.tasks.peg_transfer_sphere_obstacle import PegTransfer as PegTransferSphere
from surrol.tasks.peg_transfer_plate_obstacle import PegTransfer as PegTransferPlate

from NeuralODE.node import NeuralODE
from CBF.cbf import CBF
from CLF.clf import CLF


class Sampler:
    """Collects rollouts from the environment using the given agent."""

    def __init__(self, env, agent, max_episode_len, config):
        self._env = env
        self._agent = agent
        self._max_episode_len = max_episode_len
        self.cfg = config

        self._obs = None
        self._episode_step = 0
        self._episode_cache = ReplayCache(max_episode_len)

        # ===============================================================
        #                 Integrate Neural ODE, CBF and CLF
        # ===============================================================

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Seed:', self.cfg.seed)
        self.supported_envs = (
            NeedlePick,
            GauzeRetrieve,
            GauzeRetrieveCylinder, 
            GauzeRetrieveSphere, 
            NeedlePickCylinder, 
            NeedlePickSphere,
            NeedlePickWoundCLF,
            NeedlePickLungCLF,
            NeedlePickTrajectoryCLF,
            NeedlePickTrajectoryCBF,
            NeedleReachSphere,
            NeedleReachPlate,
            PegTransferSphere,
            PegTransferPlate
        )

        # Initialize Neural ODE
        # the neural ode has dims [x_dim, 64, x_dim + x_dim * u_dim]
        # position only: x_dim=3, u_dim=3, output=3+3*3=12
        self.node = NeuralODE([3, 64, 12]).to(self.device)

        if self.cfg.use_dcbf:

            # Initialize Neural ODE
            # the neural ode has dims [x_dim, 64, x_dim + x_dim * u_dim]
            # position only
            self.node = NeuralODE([3, 64, 12]).to(self.device)

            self.node.load_latest_weight(self.cfg.task, type='pos_')
            self.node.eval()

            # Initialize CBF
            self.cbf = CBF(self.node.net, self.device)
        if  self.cfg.use_dclf:
            # position and orientation and obj position
            self.pos_ori_node = NeuralODE([6, 64, 30]).to(self.device)

        # Initialize CLF
        # self.clf = PositionCLF(self.node.net, self.device)
        # self.clf = CLF(self.orn_node.net, self.device)
        self.pos_ori_node.load_latest_weight(self.cfg.task, type='')
        traj_type = getattr(self.cfg, 'traj_type', 'line')  # Default to 'line' if not specified
        self.clf = CLF(self.pos_ori_node.net, self.device, traj_type=traj_type)
        print(f'CLF initialized with traj_type: {traj_type}')
        
        # Sync traj_type to environment for obstacle placement
        if isinstance(self._env.env, NeedlePickTrajectoryCBF):
            self._env.env.TRAJ_TYPE = traj_type
            print(f'Environment TRAJ_TYPE set to: {traj_type}')


    def init(self):
        """Starts a new rollout. Render indicates whether output should contain image."""
        self._episode_reset()

    def sample_action(self, obs, is_train):
        return self._agent.get_action(obs, noise=is_train)

    def sample_episode(self, is_train, ep=-1, render=False, random_act=False, render_three_views=False):
        """Samples one episode from the environment."""
        self.init()
        episode, done = [], False
        
        # Store number of violations
        num_violations = 0
        deviation = []
        cbf_triggered_steps = 0
        cbf_deviation = []
        clf_deviation = []
        
        # Determine path type
        path_type = "CLF" if self.cfg.use_dclf else "CBF" if self.cfg.use_dcbf else "NONE"

        # Create full path with seed
        base_path = f"saved_eval_pic/{path_type}/{self.cfg.task}/s{self.cfg.seed}/{ep:02}"
        os.makedirs(base_path, exist_ok=True)
        
        # Store states and actions for real demo
        states = []
        actions = []
        
        # Log object position in different episodes
        print("Episode", ep, "Object position:", p.getBasePositionAndOrientation(self._env.env.obj_ids['rigid'][0]))

        # NOTE: Must change while loop's condition back to run train.py normally
        # while not done and self._episode_step < self._max_episode_len:
        # Each step is 0.1 s, 100 steps is 10 s.
        while self._episode_step < self.cfg.max_episode_steps:
            action = self._env.action_space.sample(
            ) if random_act else self.sample_action(self._obs, is_train)
            if action is None:
                break
            if render:
                if render_three_views:
                    front_rgb_array, right_rgb_array, top_rgb_array = self._env.render_three_views('rgb_array')
                    render_obs = np.concatenate([front_rgb_array, right_rgb_array, top_rgb_array], axis=1)
                else:
                    render_obs = self._env.render('rgb_array')

                img = Image.fromarray(render_obs)
                img.save(f'{base_path}/image_{self._episode_step}.png')
                
            # ===============================================================
            #                       Check Collision
            # ===============================================================

            if not is_train and isinstance(self._env.env, self.supported_envs):
                env = self._env.env
                if hasattr(env, 'check_collision'):
                    # only some environments have collision constraints
                    violate_constraint = env.check_collision()
                else:
                    violate_constraint = False
                
                if violate_constraint:
                    num_violations += 1
                    print(f'Episode {ep:02}: warning: violate the constraint at episode step {self._episode_step}')                    
            # ===============================================================
            #                  Control Lyapunov Function
            # ===============================================================
            # NOTE: Only use CLF during inference
            if not is_train and self.cfg.use_dclf and isinstance(self._env.env, self.supported_envs):
                with torch.no_grad():
                    u_pos = 0.01 * self._env.env.SCALING * action[0:3]
                    u_ori = action[[3]] * np.deg2rad(30)

                    u = torch.tensor(np.concatenate((u_pos, u_ori))).unsqueeze(0).float().to(self.device)

                    if isinstance(self._env.env, NeedlePickWoundCLF):
                        env = self._env.env
                        modified_action, p_ref = self.clf.traj_tracking(u, env)
                    elif isinstance(self._env.env, NeedlePickLungCLF):
                        env = self._env.env
                        modified_action, p_ref = self.clf.traj_tracking_linear(u, env)
                    elif isinstance(self._env.env, NeedlePickTrajectoryCLF):
                        env = self._env.env
                        modified_action, p_ref = self.clf.traj_tracking_linear(u, env)
                    elif isinstance(self._env.env, NeedlePickTrajectoryCBF):
                        env = self._env.env
                        modified_action, p_ref = self.clf.traj_tracking_linear(u, env)
                    elif isinstance(self._env.env, GauzeRetrieve):
                        env = self._env.env
                        modified_action, p_ref = self.clf.traj_tracking(u, env)
                    else:
                        raise ValueError("Unsupported environment for CLF, such as no CLF defined for this env.")
                    # Scale back the action before input into gym environment
                    action[0:3] = modified_action[:, 0:3].cpu().numpy() / (0.01 * self._env.env.SCALING)
                    action[3] = modified_action[:, 3].cpu().numpy() / np.deg2rad(30)
            # ===============================================================
            #                  Control Barrier Function
            # ===============================================================
            # NOTE: Only use CBF during inference
            cbf_triggered = False  # Track if CBF modified the action
            if not is_train and self.cfg.use_dcbf and isinstance(self._env.env, self.supported_envs):
                with torch.no_grad():
                    u = 0.01 * self._env.env.SCALING * action[0:3]
                    u_original = u.copy()
                    u = torch.tensor(u).unsqueeze(0).float().to(self.device)

                    if isinstance(self._env.env, NeedlePickSphere) or isinstance(self._env.env, NeedleReachSphere) or isinstance(self._env.env, GauzeRetrieveSphere) or isinstance(self._env.env, PegTransferSphere) \
                        or isinstance(self._env.env, NeedlePickLungCLF) or isinstance(self._env.env, NeedlePickTrajectoryCLF) or isinstance(self._env.env, NeedlePickTrajectoryCBF):
                        env = self._env.env
                        modified_action = self.cbf.sphere(u, env)
                    elif isinstance(self._env.env, NeedlePickCylinder) or isinstance(self._env.env, GauzeRetrieveCylinder) or isinstance(self._env.env, NeedleReachPlate) or isinstance(self._env.env, PegTransferPlate):
                        env = self._env.env
                        modified_action = self.cbf.cylinder(u, env)
                    else:
                        raise ValueError("Unsupported environment for CBF, such as no constraints defined for this env.")
                    
                    # Check if CBF actually modified the action
                    modified_action_np = modified_action.cpu().numpy()
                    if not np.allclose(u_original, modified_action_np, atol=1e-3):
                        cbf_triggered = True
                        cbf_triggered_steps += 1
                    # Scale back the action before input into gym environment
                    action[0:3] = modified_action_np / (0.01 * self._env.env.SCALING)


            # Append final action for real demo (only when CBF/CLF is used)
            if not is_train and (self.cfg.use_dcbf or self.cfg.use_dclf):
                states.append(self._env.env._get_robot_state(0)[:6])
                actions.append(action)
            
            obs, reward, done, info = self._env.step(action)
            episode.append(AttrDict(
                reward=reward,
                success=info['is_success'],
                info=info
            ))
            self._episode_cache.store_transition(obs, action, done)
            if render:
                episode[-1].update(AttrDict(image=render_obs))

            # record deviation
            if not is_train and self.cfg.use_dclf and isinstance(self._env.env, self.supported_envs):
                if p_ref is not None:
                    current_pos = env._get_robot_state(0)[:3]
                    traj_key = env.__class__.__name__
                    
                    # 如果 CBF 触发了修改，计算到小球表面的距离
                    if cbf_triggered and hasattr(env, 'get_sphere_prop'):
                        sphere_center, sphere_radius = env.get_sphere_prop()
                        dist_to_center = np.linalg.norm(current_pos - sphere_center)
                        current_dev = dist_to_center - sphere_radius
                        # 如果在球内部，距离为负，取绝对值或设为0
                        current_dev = max(0, current_dev)
                        cbf_deviation.append(current_dev)
                    elif self.cfg.use_new_deviation and traj_key in self.clf._traj_states:
                        traj_state = self.clf._traj_states[traj_key]
                        traj_type = traj_state.get('traj_type', 'line')
                        start_pos = traj_state['start']
                        goal_pos = traj_state['goal']
                        
                        if traj_type == 'line':
                            # 直线轨迹: Point-to-line distance
                            line_vec = goal_pos - start_pos
                            line_len = np.linalg.norm(line_vec)
                            if line_len > 1e-6:
                                line_unit = line_vec / line_len
                                point_vec = current_pos - start_pos
                                proj_len = np.dot(point_vec, line_unit)
                                proj_len = np.clip(proj_len, 0, line_len)
                                closest_point = start_pos + proj_len * line_unit
                                current_dev = np.linalg.norm(current_pos - closest_point)
                            else:
                                current_dev = np.linalg.norm(current_pos - start_pos)
                                
                        elif traj_type == 'circle':
                            # 圆形轨迹: Distance to circle = |distance_to_center - radius|
                            circle_params = traj_state['circle_params']
                            center_xy = circle_params['center_xy']
                            radius = circle_params['radius']
                            z_center = circle_params['z_center']
                            
                            # Distance in XY plane from center
                            dist_to_center_xy = np.linalg.norm(current_pos[:2] - center_xy)
                            # Radial deviation (distance from the circle in XY plane)
                            radial_dev = np.abs(dist_to_center_xy - radius)
                            # Z deviation (distance from the z plane of the circle)
                            z_dev = np.abs(current_pos[2] - z_center)
                            # Total deviation
                            current_dev = np.sqrt(radial_dev**2 + z_dev**2)
                            
                        elif traj_type == 'triangle':
                            # 三角形轨迹: Minimum distance to any of the three edges
                            vertices = traj_state['triangle_vertices']
                            # Triangle has 3 edges: start->v1, v1->v2, v2->goal
                            edges = [
                                (vertices[0], vertices[1]),  # start -> vertex1
                                (vertices[1], vertices[2]),  # vertex1 -> vertex2
                                (vertices[2], vertices[3]),  # vertex2 -> goal
                            ]
                            
                            min_dist = float('inf')
                            for p1, p2 in edges:
                                # Calculate point-to-segment distance
                                edge_vec = p2 - p1
                                edge_len = np.linalg.norm(edge_vec)
                                if edge_len > 1e-6:
                                    edge_unit = edge_vec / edge_len
                                    point_vec = current_pos - p1
                                    proj_len = np.dot(point_vec, edge_unit)
                                    proj_len = np.clip(proj_len, 0, edge_len)
                                    closest_point = p1 + proj_len * edge_unit
                                    dist = np.linalg.norm(current_pos - closest_point)
                                else:
                                    dist = np.linalg.norm(current_pos - p1)
                                min_dist = min(min_dist, dist)
                            current_dev = min_dist
                        else:
                            # Unknown trajectory type, fallback
                            current_dev = np.linalg.norm(current_pos - p_ref[0:3])
                        # Record CLF deviation (when CBF is not triggered)
                        clf_deviation.append(current_dev)
                    else:
                        # Fallback to original calculation
                        current_dev = (np.linalg.norm(current_pos - p_ref[0:3]) +
                                       self.clf.yaw_difference(env._get_robot_state(0)[5], p_ref[3]))
                        # Record CLF deviation (when CBF is not triggered)
                        clf_deviation.append(current_dev)
                    
                    deviation.append(current_dev)

            # update stored observation
            self._obs = obs
            self._episode_step += 1
        print(f'mean deviation:', np.array(deviation).mean() if len(deviation) > 0 else 0.0)
        print(f'CLF mean deviation (trajectory tracking): {np.array(clf_deviation).mean():.4f}' if len(clf_deviation) > 0 else 'CLF mean deviation: N/A')
        print(f'CBF triggered steps: {cbf_triggered_steps} / {self._episode_step}')
        print(f'CBF mean deviation (distance to sphere surface): {np.array(cbf_deviation).mean():.4f}' if len(cbf_deviation) > 0 else 'CBF mean deviation: N/A')
        if hasattr(self, 'clf') and self.clf is not None and env.__class__.__name__ in self.clf._traj_states:
            traj = self.clf._traj_states[env.__class__.__name__]['traj']
            # Calculate total trajectory distance (sum of distances between consecutive points)
            traj_distance = 0.0
            for i in range(1, len(traj)):
                traj_distance += np.linalg.norm(traj[i][:3] - traj[i-1][:3])
            print(f"trajectory distance: {traj_distance:.4f}")

        if not is_train and episode[-1]['success'] == 1.0:
            # Just a file to indicate which episode is success.
            success_file = f"{base_path}/success.txt"
            open(success_file, 'w').close()
        
        if not is_train and isinstance(self._env.env, self.supported_envs):
            # Save state and action sequence for real world demonstration
            states.append(env._get_robot_state(0)[:6])
            states = np.array(states)
            actions = np.array(actions)
            action_filename = f"{base_path}/actions.npy"
            states_filename = f"{base_path}/states.npy"
            np.save(action_filename, actions)
            np.save(states_filename, states)
            print("Images, states and actions are saved at", base_path)
        
        # make sure episode is marked as done at final time step
        episode[-1].done = True
        rollouts = self._episode_cache.pop()
        assert self._episode_step == self._max_episode_len
        return listdict2dictlist(episode), rollouts, self._episode_step, num_violations

    def _episode_reset(self, global_step=None):
        """Resets sampler at the end of an episode."""
        self._episode_step, self._episode_reward = 0, 0.
        self._obs = self._reset_env()
        self._episode_cache.store_obs(self._obs)

    def _reset_env(self):
        return self._env.reset()
