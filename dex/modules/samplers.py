from ..utils.general_utils import AttrDict, listdict2dictlist
from ..utils.rl_utils import ReplayCache

import os
import torch
import numpy as np
import PIL.Image as Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
from surrol.tasks.needle_reach_sphere import NeedleReachSphere
from surrol.tasks.needle_reach_plate_obstacle import NeedleReachPlate
from surrol.tasks.peg_transfer_sphere_obstacle import PegTransferSphere
from surrol.tasks.peg_transfer_plate_obstacle import PegTransferPlate

from NeuralODE.node import NeuralODE
from CBF.cbf import CBF
from CLF.clf import CLF, PositionCLF
import time
from torchdiffeq import odeint

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

        # [NEW] List to store the minimum safe margin of each episode
        self.global_min_safe_margins = []
        self.global_inference_times = []
        self.global_max_node_pred_errors = []  # Store max node_pred_error for each episode
        self.global_max_state_pred_errors = []  # Store max state_pred_error for each episode

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
        self.pos_envs = (
            GauzeRetrieve,
            GauzeRetrieveCylinder, 
            GauzeRetrieveSphere, 
            NeedleReachSphere,
            NeedleReachPlate,
            NeedleReachSphere,
        )
        # Initialize Neural ODE
        # the neural ode has dims [x_dim, 64, x_dim + x_dim * u_dim]
        # position only: x_dim=3, u_dim=3, output=3+3*3=12
        self.node = NeuralODE([3, 64, 12]).to(self.device)
        # self.node = NeuralODE([3, 64, 64, 12]).to(self.device)


        # self.node = NeuralODE([3, 64, 12]).to(self.device)

        self.node.load_latest_weight(self.cfg.task, type='pos_')
        self.node.eval()

        # Initialize CBF
        if self.cfg.use_dcbf and not self.cfg.use_dclf:
            self.cbf = CBF(self.node.net, self.device, gamma=1)
        else:
            self.cbf = CBF(self.node.net, self.device, gamma=10)
            
        if  self.cfg.use_dclf:
            # position and orientation and obj position
            if isinstance(self._env.env, self.pos_envs):
                self.pos_ori_node = NeuralODE([3, 64, 12]).to(self.device)
                self.pos_ori_node.load_latest_weight(self.cfg.task, type='pos_')
                self.clf = PositionCLF(self.pos_ori_node.net, self.device)
            else:
                self.pos_ori_node = NeuralODE([6, 64, 30]).to(self.device)
                self.pos_ori_node.load_latest_weight(self.cfg.task, type='')
                traj_type = getattr(self.cfg, 'traj_type', 'line')  # Default to 'line' if not specified
                self.clf = CLF(self.pos_ori_node.net, self.device, traj_type=traj_type)
                print(f'CLF initialized with traj_type: {traj_type}')

            # # Initialize CLF
            # # self.clf = PositionCLF(self.node.net, self.device)
            # # self.clf = CLF(self.orn_node.net, self.device)
            # self.pos_ori_node.load_latest_weight(self.cfg.task, type='')
            # traj_type = getattr(self.cfg, 'traj_type', 'line')  # Default to 'line' if not specified
            # self.clf = CLF(self.pos_ori_node.net, self.device, traj_type=traj_type)
            # print(f'CLF initialized with traj_type: {traj_type}')
        
        # Sync traj_type to environment for obstacle placement
        if isinstance(self._env.env, NeedlePickTrajectoryCBF):
            self._env.env.TRAJ_TYPE = traj_type
            print(f'Environment TRAJ_TYPE set to: {traj_type}')
        elif isinstance(self._env.env, NeedlePickWoundCLF):
            self._env.env.TRAJ_TYPE = traj_type
            print(f'NeedlePickWoundCLF TRAJ_TYPE set to: {traj_type}')



    def init(self):
        """Starts a new rollout. Render indicates whether output should contain image."""
        self._episode_reset()
        # Reset CLF trajectory for environments where waypoints depend on object positions
        if self.cfg.use_dclf and hasattr(self, 'clf'):
            # Use actual environment class name as key
            env_name = self._env.env.__class__.__name__
            self.clf.reset_trajectory(env_name)

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
        cbf_triggered_step_indices = []
        cbf_deviation = []
        clf_deviation = []
        min_episode_margin = float('inf')
        inference_time = 0.0
        safe_margin = float('-inf')
        max_episode_node_pred_error = 0.0  # Track max node_pred_error for this episode
        max_episode_state_pred_error = 0.0  # Track max state_pred_error for this episode
        
        # Determine path type
        # path_type = "CLF" if self.cfg.use_dclf else "CBF" if self.cfg.use_dcbf else "NONE"
        if self.cfg.use_dclf and self.cfg.use_dcbf:
            path_type = "CLF_CBF"
        elif self.cfg.use_dclf:
            path_type = "CLF"
        elif self.cfg.use_dcbf:
            path_type = "CBF"
        else:
            path_type = "NONE"

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
            if render:
                if render_three_views:
                    front_rgb_array, right_rgb_array, top_rgb_array = self._env.render_three_views('rgb_array')
                    render_obs = np.concatenate([front_rgb_array, right_rgb_array, top_rgb_array], axis=1)
                else:
                    render_obs = self._env.render('rgb_array')

                img = Image.fromarray(render_obs)
                img.save(f'{base_path}/image_{self._episode_step}.png')
            time_start = time.time()
            action = self._env.action_space.sample(
            ) if random_act else self.sample_action(self._obs, is_train)
            if action is None:
                break
                
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
                        modified_action, p_ref, gripper_state = self.clf.traj_tracking_linear(u, env)
                        # Set gripper state from trajectory (periodic open/close)
                        if gripper_state is not None:
                            action[4] = gripper_state
                    elif isinstance(self._env.env, NeedlePickTrajectoryCLF):
                        env = self._env.env
                        modified_action, p_ref, gripper_state = self.clf.traj_tracking_linear(u, env)
                        # Set gripper state from trajectory if available
                        if gripper_state is not None:
                            action[4] = gripper_state
                    elif isinstance(self._env.env, NeedlePickTrajectoryCBF):
                        env = self._env.env
                        modified_action, p_ref, gripper_state = self.clf.traj_tracking_linear(u, env)
                        # Set gripper state from trajectory if available
                        if gripper_state is not None:
                            action[4] = gripper_state
                    elif isinstance(self._env.env, (NeedlePick, NeedlePickSphere, NeedlePickCylinder, 
                                    PegTransferPlate, PegTransferSphere)):
                        env = self._env.env
                        modified_action, p_ref, gripper_state = self.clf.traj_tracking_waypoints(u, env)
                        # Set gripper state from trajectory (automatically close gripper at grasp point)
                        if gripper_state is not None:
                            action[4] = gripper_state
                    elif isinstance(self._env.env, self.pos_envs):
                        env = self._env.env
                        modified_action, p_ref, gripper_state = self.clf.traj_tracking_waypoints(u, env)
                        if gripper_state is not None:
                            action[4] = gripper_state
                    else:
                        raise ValueError("Unsupported environment for CLF, such as no CLF defined for this env.")
                    # Scale back the action before input into gym environment
                    action[0:3] = modified_action[:, 0:3].cpu().numpy() / (0.01 * self._env.env.SCALING)
                    action[3] = modified_action[:, 3].cpu().numpy() / np.deg2rad(30)
            # ===============================================================
            #                  Control Barrier Function
            # ===============================================================
            # NOTE: Only use CBF during inference
            # Environments with sphere obstacles
            sphere_envs = (NeedlePickSphere, NeedleReachSphere, GauzeRetrieveSphere, PegTransferSphere,
                            NeedlePickLungCLF, NeedlePickTrajectoryCLF, NeedlePickTrajectoryCBF, NeedleReachSphere)
            # Environments with cylinder/plate obstacles
            cylinder_envs = (NeedlePickCylinder, GauzeRetrieveCylinder, NeedleReachPlate, PegTransferPlate)
            cbf_triggered = False  # Track if CBF modified the action
            if not is_train and self.cfg.use_dcbf and isinstance(self._env.env, self.supported_envs):
                with torch.no_grad():
                    u = 0.01 * self._env.env.SCALING * action[0:3]
                    u_original = u.copy()
                    u = torch.tensor(u).unsqueeze(0).float().to(self.device)

                    
                    env = self._env.env
                    if isinstance(env, sphere_envs):
                        modified_action = self.cbf.sphere(u, env)
                    elif isinstance(env, cylinder_envs):
                        modified_action = self.cbf.cylinder(u, env)
                    else:
                        raise ValueError("Unsupported environment for CBF, such as no constraints defined for this env.")
                    # speed_scale = 0.1  # <--- 修改这里：0.5 表示 50% 的速度，0.2 表示 20%
                    # modified_action = modified_action * speed_scale
                    # Check if CBF actually modified the action
                    modified_action_np = modified_action.cpu().numpy()
                    if not np.allclose(u_original, modified_action_np, atol=1e-4):
                        cbf_triggered = True
                        cbf_triggered_steps += 1
                        cbf_triggered_step_indices.append(self._episode_step)
                    # Scale back the action before input into gym environment
                    action[0:3] = modified_action_np / (0.01 * self._env.env.SCALING)
            if not is_train and isinstance(self._env.env, sphere_envs):
                safe_margin = self.cbf.sphere(None, env, return_b=True)
            elif not is_train and isinstance(self._env.env, cylinder_envs):
                safe_margin = self.cbf.cylinder(None, env, return_b=True)

            inference_time = time.time() - time_start
            # Append final action for real demo (only when CBF/CLF is used)
            if not is_train and (self.cfg.use_dcbf or self.cfg.use_dclf):
                states.append(self._env.env._get_robot_state(0)[:6])
                actions.append(action)
            
            node_pred_error = 0.0
            if isinstance(self._env.env, sphere_envs + cylinder_envs):
                base_env = self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env
                prev_pos = base_env._get_robot_state(0)[:3].copy()
                u_for_node = 0.01 * self._env.env.SCALING * action[0:3]

            obs, reward, done, info = self._env.step(action)

            if isinstance(self._env.env, sphere_envs + cylinder_envs):
                base_env = self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env
                next_pos = base_env._get_robot_state(0)[:3].copy()
                with torch.no_grad():
                    x = torch.from_numpy(prev_pos).float().unsqueeze(0).to(self.device)
                    u = torch.from_numpy(u_for_node).float().unsqueeze(0).to(self.device)
                    self.node.u = u
                    dt = 0.1
                    true_next_pos = torch.from_numpy(next_pos).float().unsqueeze(0).to(self.device)
                    true_dxdt = (true_next_pos - x) / dt

                    t_step_vec = torch.tensor([0.0, dt], device=self.device)
                    pred = odeint(self.node, x, t_step_vec, method='dopri8')
                    pred_next_pos = pred[-1, :, :]
                    state_pred_error = torch.mean(torch.abs(true_next_pos - pred_next_pos), dim=-1).item()

                    pred_dxdt = self.node(torch.tensor(0.0, device=self.device), x)
                    node_pred_error = torch.mean(torch.abs(true_dxdt - pred_dxdt), dim=-1).item()
                    max_episode_node_pred_error = max(max_episode_node_pred_error, node_pred_error)
                    max_episode_state_pred_error = max(max_episode_state_pred_error, state_pred_error)

            episode.append(AttrDict(
                reward=reward,
                success=info['is_success'],
                info=info,
                safe_margin=safe_margin,
                inference_time=inference_time,
                state_pred_error=state_pred_error,
                node_pred_error=node_pred_error
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
                        current_dev = (np.linalg.norm(current_pos - p_ref[0:3]))
                                    #    self.clf.yaw_difference(env._get_robot_state(0)[5], p_ref[3]))
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
        all_margins = [e.safe_margin.item() if hasattr(e.safe_margin, 'item') else e.safe_margin 
                       for e in episode if e.safe_margin is not None]
        if all_margins:
            min_episode_margin = min(all_margins)
            self.global_min_safe_margins.append(min_episode_margin)
            
            print(f'Episode {ep} Min Safe Margin: {min_episode_margin:.6f}')
            print(f'Mean of Min Safe Margins (All Episodes): {np.mean(self.global_min_safe_margins):.6f}')
        else:
            print('Episode Min Safe Margin: N/A (CBF not active)')


        all_inference_times = [e.inference_time.item() if hasattr(e.inference_time, 'item') else e.inference_time 
                       for e in episode if e.inference_time is not None]
        print(f'Mean inference time: {np.array(all_inference_times).mean():.4f}')
        self.global_inference_times.append(np.array(all_inference_times).mean())
        print(f'Mean inference time (All Episodes): {np.array(self.global_inference_times).mean():.4f}')

        # Node prediction error statistics
        all_node_pred_errors = [e.node_pred_error for e in episode if e.node_pred_error is not None and e.node_pred_error > 0]
        if all_node_pred_errors:
            self.global_max_node_pred_errors.append(max_episode_node_pred_error)
            print(f'Episode {ep} Max Node Pred Error: {max_episode_node_pred_error:.6f}')
            print(f'Mean of Max Node Pred Errors (All Episodes): {np.mean(self.global_max_node_pred_errors):.6f}')
        else:
            print('Episode Max Node Pred Error: N/A (not computed)')

        all_state_pred_errors = [e.state_pred_error for e in episode if e.state_pred_error is not None and e.state_pred_error > 0]
        if all_state_pred_errors:
            self.global_max_state_pred_errors.append(max_episode_state_pred_error)
            print(f'Episode {ep} Max State Pred Error: {max_episode_state_pred_error:.6f}')
            print(f'Mean of Max State Pred Errors (All Episodes): {np.mean(self.global_max_state_pred_errors):.6f}')
        else:
            print('Episode Max State Pred Error: N/A (not computed)')

        if isinstance(self._obs, dict):
            last_dist = np.linalg.norm(self._obs['achieved_goal'] - self._obs['desired_goal'])
            print(f'last-step goal distance: {last_dist:.6f}')
        print(f'episode step: {self._episode_step},success: {episode[-1]["success"]}')
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
        # Plot safe_margin per step for this episode
        # if all_margins:
        #     # 1. 准备数据和元信息
        #     steps = np.arange(len(all_margins))
        #     margins = np.array(all_margins)
            
        #     # 2. 开始绘图 (使用更美观的配置)
        #     # 设置风格 (尝试使用 seaborn 风格，如果不可用则回退)
        #     try:
        #         plt.style.use('seaborn-v0_8-whitegrid')
        #     except:
        #         plt.grid(True, linestyle='--', alpha=0.5)

        #     fig, ax = plt.subplots(figsize=(12, 6))
            
        #     # 绘制主曲线
        #     ax.plot(steps, margins, linewidth=2.0, color='#1f77b4', label='Safe Margin $b(x)$', zorder=10)
            
        #     # 绘制安全边界 (b=0)
        #     ax.axhline(y=0, color='#d62728', linestyle='--', linewidth=1.5, label='Safety Boundary', zorder=11)
            
        #     # 3. 区域填充 (美观关键)
        #     # 安全区域 (>0) 填充淡蓝色
        #     ax.fill_between(steps, margins, 0, where=(margins >= 0), 
        #                     interpolate=True, color='#1f77b4', alpha=0.15)
            
        #     # 危险区域 (<0) 填充淡红色，强调违规
        #     ax.fill_between(steps, margins, 0, where=(margins < 0), 
        #                     interpolate=True, color='#d62728', alpha=0.2)

        #     # Phase dividers (vertical lines)
        #     if len(cbf_triggered_step_indices) > 0:
        #         cbf_start = cbf_triggered_step_indices[0]
        #         cbf_end = cbf_triggered_step_indices[-1]

        #         # Detect Goal Reaching: find the step after CBF end where margin stabilizes
        #         # (absolute change between consecutive steps stays below threshold)
        #         goal_reach_step = len(margins) - 1  # default: last step
        #         if cbf_end + 1 < len(margins):
        #             margin_diffs = np.abs(np.diff(margins[cbf_end:]))
        #             stable_window = 5  # require this many consecutive stable steps
        #             stable_thresh = 0.05 * (np.max(margins) - np.min(margins) + 1e-8)
        #             count = 0
        #             for k, d in enumerate(margin_diffs):
        #                 if d < stable_thresh:
        #                     count += 1
        #                     if count >= stable_window:
        #                         goal_reach_step = cbf_end + k - stable_window + 2
        #                         break
        #                 else:
        #                     count = 0

        #         # ax.axvline(x=cbf_start, color='green', linestyle='-.', linewidth=1.5, alpha=0.8, label=f'CBF Start')
        #         # ax.axvline(x=cbf_end, color='purple', linestyle='-.', linewidth=1.5, alpha=0.8, label=f'CBF End')
        #         # if goal_reach_step > cbf_end and goal_reach_step < len(margins) - 1:
        #         #     ax.axvline(x=goal_reach_step, color='gray', linestyle='-.', linewidth=1.5, alpha=0.8, label=f'Goal Reaching')

        #         # # Phase labels at top
        #         # y_top = ax.get_ylim()[1]
        #         # # ax.text(cbf_start / 2, y_top, 'Approach', ha='center', va='bottom', fontsize=9, color='green', fontweight='bold')
        #         # ax.text((cbf_start + cbf_end) / 2, y_top, 'CBF Active', ha='center', va='bottom', fontsize=12, color='purple', fontweight='bold')
        #         # gr_start = goal_reach_step if goal_reach_step > cbf_end else cbf_end
        #         # if goal_reach_step > cbf_end and goal_reach_step < len(margins) - 1:
        #         #     # ax.text((cbf_end + goal_reach_step) / 2, y_top, 'Transition', ha='center', va='bottom', fontsize=9, color='orange', fontweight='bold')
        #         #     ax.text((goal_reach_step + len(steps)) / 2, y_top, 'Goal Reaching', ha='center', va='bottom', fontsize=12, color='gray', fontweight='bold')
        #         # else:
        #         #     ax.text((cbf_end + len(steps)) / 2, y_top, 'Goal Reaching', ha='center', va='bottom', fontsize=12, color='gray', fontweight='bold')
                
        #         ax.axvspan(cbf_start, cbf_end, color='purple', alpha=0.1, label='CBF Active Phase')
        #         gr_start = goal_reach_step if goal_reach_step > cbf_end else cbf_end
        #         if gr_start < len(steps):
        #              ax.axvspan(gr_start, len(steps)-1, color='gray', alpha=0.15, label='Goal Reaching Phase')

        #         # # 3. Transition 阶段 (如果有)
        #         # if gr_start > cbf_end:
        #         #      ax.axvspan(cbf_end, gr_start, color='orange', alpha=0.1, label='Transition Phase')

        #         ax.axvline(x=cbf_start, color='purple', linestyle=':', linewidth=1.0, alpha=0.6)
        #         ax.axvline(x=cbf_end, color='purple', linestyle=':', linewidth=1.0, alpha=0.6)
        #         if gr_start > cbf_end:
        #             ax.axvline(x=gr_start, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
        #     # 5. 标签与修饰
        #     ax.set_xlabel('Simulation Step', fontsize=18, fontweight='bold')
        #     ax.set_ylabel('Control Barrier Function Value b(x)', fontsize=18, fontweight='bold')
        #     ax.tick_params(axis='both', labelsize=14)
            
        #     # 设置主标题和副标题
        #     ax.set_title(f'Safety Margin Analysis', fontsize=20, fontweight='bold')
            
        #     # 图例
        #     ax.legend(loc='lower right', bbox_to_anchor=(1, 0.15), 
        #               frameon=True, framealpha=0.9, shadow=True, fontsize=14)
        #     y_lo = min(np.min(margins), 0)  # always include y=0 so the boundary line is visible
        #     y_hi = np.max(margins)
        #     y_range = y_hi - y_lo
        #     if y_range == 0: y_range = 1.0
        #     ax.set_ylim(y_lo - 0.1 * y_range, y_hi + 0.1 * y_range)
        #     ax.set_xlim(0, len(steps))

        #     # 保存图片
        #     plt.tight_layout()
        #     # 调整 layout 给 suptitle 留空间
        #     plt.subplots_adjust(top=0.88) 
            
        #     margin_plot_path = f"{base_path}/safe_margin_ep{ep:02}.png"
        #     fig.savefig(margin_plot_path, dpi=200, bbox_inches='tight') # 提高 DPI
        #     plt.close(fig)
        #     print(f'Safe margin plot saved to {margin_plot_path}')

            

        #     # ===============================================================
        #     #              Plot 3D Trajectory for this episode
        #     # ===============================================================
        #     if len(states) > 1:
        #         traj_pos = states[:, :3]  # (N, 3) — x, y, z positions

        #         fig = plt.figure(figsize=(10, 8))
        #         ax = fig.add_subplot(111, projection='3d')

        #         # --- 1. Actual robot trajectory (colored by step) ---
        #         # Use sqrt mapping so early (moving) steps get more color variation
        #         N = len(traj_pos)
        #         t_norm = np.sqrt(np.linspace(0, 1, N))  # sqrt stretches early steps
        #         colors = plt.cm.viridis(t_norm)
        #         for i in range(N - 1):
        #             ax.plot(traj_pos[i:i+2, 0], traj_pos[i:i+2, 1], traj_pos[i:i+2, 2],
        #                     color=colors[i], linewidth=2.0)
        #         sc = ax.scatter(traj_pos[:, 0], traj_pos[:, 1], traj_pos[:, 2],
        #                         c=np.arange(N), cmap='viridis', s=12, zorder=5,
        #                         norm=matplotlib.colors.PowerNorm(gamma=0.5, vmin=0, vmax=N-1))
        #         cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.00)
        #         cbar.set_label('Step', fontsize=14)
        #         cbar.ax.tick_params(labelsize=14)
        #         cbar.ax.yaxis.set_ticks_position('left')

        #         # Mark start and end
        #         ax.scatter(*traj_pos[0], color='limegreen', s=120, marker='o',
        #                    edgecolors='black', linewidths=1.2, zorder=10, label='Start')
        #         ax.scatter(*traj_pos[-1], color='red', s=120, marker='*',
        #                    edgecolors='black', linewidths=1.2, zorder=10, label='End')

        #         # --- 2. Reference trajectory (if CLF is active) ---
        #         env = self._env.env
        #         env_cls_name = env.__class__.__name__
        #         if hasattr(self, 'clf') and self.clf is not None and env_cls_name in self.clf._traj_states:
        #             ref_traj = np.array(self.clf._traj_states[env_cls_name]['traj'])
        #             ax.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2],
        #                     color='orange', linewidth=2.0, linestyle='--', alpha=0.8, label='Ref Trajectory')
        #             ax.scatter(ref_traj[0, 0], ref_traj[0, 1], ref_traj[0, 2],
        #                        color='orange', s=60, marker='D', edgecolors='black', zorder=9)
        #             ax.scatter(ref_traj[-1, 0], ref_traj[-1, 1], ref_traj[-1, 2],
        #                        color='orange', s=60, marker='D', edgecolors='black', zorder=9)

        #         # --- 3. Draw obstacle as a single point ---
        #         if hasattr(env, 'get_sphere_prop'):
        #             try:
        #                 sph_center, sph_radius = env.get_sphere_prop()
        #                 # Center point
        #                 ax.scatter(*sph_center, color='red', s=150, marker='o',
        #                            edgecolors='darkred', linewidths=1.5, zorder=10, label='Obstacle Center')
        #                 # Top point (center + radius along z-axis)
        #                 sph_top = sph_center.copy()
        #                 sph_top[2] += sph_radius
        #                 ax.scatter(*sph_top, color='red', s=80, marker='^',
        #                            edgecolors='darkred', linewidths=1.2, zorder=10, label=f'Obstacle Top')
        #                 # Draw a vertical line connecting center and top
        #                 ax.plot([sph_center[0], sph_top[0]],
        #                         [sph_center[1], sph_top[1]],
        #                         [sph_center[2], sph_top[2]],
        #                         color='red', linewidth=1.5, linestyle=':', alpha=0.7)
        #             except Exception:
        #                 pass

        #         if hasattr(env, 'get_cylinder_prop'):
        #             try:
        #                 cyl_center, cyl_axis, cyl_length, cyl_radius = env.get_cylinder_prop()
        #                 cyl_center = np.array(cyl_center, dtype=float)
        #                 cyl_axis = np.array(cyl_axis, dtype=float)
        #                 cyl_axis_norm = cyl_axis / (np.linalg.norm(cyl_axis) + 1e-8)
        #                 # Build two perpendicular vectors
        #                 if abs(cyl_axis_norm[0]) < 0.9:
        #                     perp1 = np.cross(cyl_axis_norm, np.array([1, 0, 0]))
        #                 else:
        #                     perp1 = np.cross(cyl_axis_norm, np.array([0, 1, 0]))
        #                 perp1 /= (np.linalg.norm(perp1) + 1e-8)
        #                 perp2 = np.cross(cyl_axis_norm, perp1)
        #                 perp2 /= (np.linalg.norm(perp2) + 1e-8)
        #                 # Cylinder surface mesh
        #                 theta_cyl = np.linspace(0, 2 * np.pi, 30)
        #                 h_cyl = np.linspace(-cyl_length / 2, cyl_length / 2, 2)
        #                 theta_grid, h_grid = np.meshgrid(theta_cyl, h_cyl)
        #                 cx = cyl_center[0] + cyl_radius * (np.cos(theta_grid) * perp1[0] + np.sin(theta_grid) * perp2[0]) + h_grid * cyl_axis_norm[0]
        #                 cy = cyl_center[1] + cyl_radius * (np.cos(theta_grid) * perp1[1] + np.sin(theta_grid) * perp2[1]) + h_grid * cyl_axis_norm[1]
        #                 cz = cyl_center[2] + cyl_radius * (np.cos(theta_grid) * perp1[2] + np.sin(theta_grid) * perp2[2]) + h_grid * cyl_axis_norm[2]
        #                 ax.plot_surface(cx, cy, cz, color='orange', alpha=0.2)
        #                 ax.plot_wireframe(cx, cy, cz, color='darkorange', alpha=0.3, linewidth=0.4)
        #                 # Mark center point
        #                 ax.scatter(*cyl_center, color='orange', s=80, marker='o',
        #                            edgecolors='darkorange', linewidths=1.2, zorder=10, label='Obstacle')
        #             except Exception:
        #                 pass

        #         # --- 4. Labels and styling ---
        #         ax.set_xlabel('X', fontsize=18, fontweight='bold', labelpad=10)
        #         ax.set_ylabel('Y', fontsize=18, fontweight='bold', labelpad=10)
        #         ax.set_zlabel('Z', fontsize=18, fontweight='bold', labelpad=10)
        #         ax.tick_params(axis='both', labelsize=14, pad=5)
        #         ax.view_init(elev=20, azim=10)
        #         ax.dist = 11  # zoom out slightly so the 3D plot doesn't overlap legend

        #         ax.set_title(r'$\bf{3D\ Trajectory\ Visualization}$', fontsize=20, pad=15)
        #         ax.legend(loc='upper left', fontsize=14, framealpha=0.8,
        #                   bbox_to_anchor=(-0.02, 1.02))

        #         fig.tight_layout()
        #         fig.subplots_adjust(left=0.05, bottom=0.05, top=0.92)
        #         traj_plot_path = f"{base_path}/trajectory_3d_ep{ep:02}.png"
        #         fig.savefig(traj_plot_path, dpi=200, bbox_inches='tight')
        #         plt.close(fig)
        #         print(f'3D trajectory plot saved to {traj_plot_path}')
        
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
