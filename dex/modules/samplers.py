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
from surrol.tasks.gauze_retrieve import GauzeRetrieve
from surrol.tasks.gauze_retrieve_sphere import GauzeRetrieveSphere
from surrol.tasks.gauze_retrieve_cylinder import GauzeRetrieveCylinder

from NeuralODE.node import NeuralODE
from CBF.cbf import CBF
from CLF.clf import PositionCLF, CLF


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
            NeedlePickSphere
        )

        # Initialize Neural ODE
        # the neural ode has dims [x_dim, 64, x_dim + x_dim * u_dim]
        # position and orientation
        self.node = NeuralODE([6, 64, 30]).to(self.device)
        # position only
        # self.node = NeuralODE([3, 64, 12]).to(self.device)

        self.node.load_latest_weight(self.cfg.task)
        self.node.eval()

        # Initialize CBF
        self.cbf = CBF(self.node.net, self.device)

        # Initialize CLF
        # self.clf = PositionCLF(self.node.net, self.device)
        self.clf = CLF(self.node.net, self.device)


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

        # Store deviation from the CLF trajectory
        deviation = []
        
        # Determine path type
        path_type = "CLF" if self.cfg.use_dclf else "CBF" if self.cfg.use_dcbf else "NONE"

        # Create full path with seed
        base_path = f"saved_eval_pic/{path_type}/{self.cfg.task}/s{self.cfg.seed}/{ep:02}"
        os.makedirs(base_path, exist_ok=True)
        
        # Store actions for real demo
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
            #                  Control Barrier Function
            # ===============================================================
            # NOTE: Only use CBF during inference
            if not is_train and self.cfg.use_dcbf and isinstance(self._env.env, self.supported_envs):
                with torch.no_grad():
                    u = 0.01 * self._env.env.SCALING * action[0:3]
                    u = torch.tensor(u).unsqueeze(0).float().to(self.device)

                    if isinstance(self._env.env, NeedlePickSphere):
                        env = self._env.env
                        modified_action = self.cbf.needle_pick_sphere(u, env)
                        
                    elif isinstance(self._env.env, NeedlePickCylinder):
                        env = self._env.env
                        modified_action = self.cbf.needle_pick_cylinder(u, env)
                        
                    elif isinstance(self._env.env, GauzeRetrieveSphere):
                        env = self._env.env
                        modified_action = self.cbf.gauze_retrieve_sphere(u, env)
                        
                    elif isinstance(self._env.env, GauzeRetrieveCylinder):
                        env = self._env.env
                        modified_action = self.cbf.gauze_retrieve_cylinder(u, env)
                    else:
                        raise ValueError("Unsupported environment for CBF, such as no constraints defined for this env.")
                    
                    # Scale back the action before input into gym environment
                    action[0:3] = modified_action.cpu().numpy() / (0.01 * self._env.env.SCALING)

            # ===============================================================
            #                  Control Lyapunov Function
            # ===============================================================
            # NOTE: Only use CLF during inference
            if not is_train and self.cfg.use_dclf and isinstance(self._env.env, self.supported_envs):
                with torch.no_grad():
                    u_pos = 0.01 * self._env.env.SCALING * action[0:3]
                    u_ori = action[[3]] * np.deg2rad(30)

                    u = torch.tensor(np.concatenate((u_pos, u_ori))).unsqueeze(0).float().to(self.device)

                    if isinstance(self._env.env, NeedlePick):
                        env = self._env.env
                        modified_action, p_ref = self.clf.traj_tracking(u, env)
                    elif isinstance(self._env.env, GauzeRetrieve):
                        env = self._env.env
                        modified_action, p_ref = self.clf.traj_tracking(u, env)
                    else:
                        raise ValueError("Unsupported environment for CLF, such as no CLF defined for this env.")
                    # Scale back the action before input into gym environment
                    action[0:3] = modified_action[:, 0:3].cpu().numpy() / (0.01 * self._env.env.SCALING)
                    action[3] = modified_action[:, 3].cpu().numpy() / np.deg2rad(30)

            # Append final action for real demo
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

                    current_dev = (np.linalg.norm(env._get_robot_state(0)[:3]-p_ref[0:3])+
                                   self.clf.yaw_difference(env._get_robot_state(0)[5], p_ref[3]))
                    deviation.append(current_dev)

            # update stored observation
            self._obs = obs
            self._episode_step += 1

        print(f'mean deviation:', np.array(deviation).mean() if len(deviation) > 0 else 0.0)

        if not is_train and episode[-1]['success'] == 1.0:
            # Just a file to indicate which episode is success.
            success_file = f"{base_path}/success.txt"
            open(success_file, 'w').close()
        
        if not is_train and self.cfg.use_dcbf and isinstance(self._env.env, self.supported_envs):
            # Save action sequence for real world demonstration
            actions = np.array(actions)
            action_filename = f"{base_path}/actions.npy"
            np.save(action_filename, actions)
            print("Images and actions are saved at", base_path)
        
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
