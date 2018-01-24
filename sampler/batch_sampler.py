import numpy as np
import os
import cloudpickle

class BatchSampler(object):

    def __init__(self,
                 env,
                 policy,
                 sampler_hyperparams,
                 state_preprocessor=None,
                 reward_preprocessor=None,
                 logger=None, **kwargs):

        self.env = env
        self.policy = policy
        self.batch_size = sampler_hyperparams['batch_size']
        self.episode_horizon = sampler_hyperparams['max_episode_timesteps']
        self.init_batch_size = sampler_hyperparams['init_batch_size']
        if logger:
            self.logger = logger
        self.state_preprocessor = state_preprocessor
        self.reward_preprocessor = reward_preprocessor

        self.unscaled_batch = [] # this is the batch before normalization
        self.scaled_batch = [] # this is the batch after normalization

        if self.init_batch_size:
            print("Initializing preprocessor ...")
            for _ in range(self.init_batch_size):
                self.run_episode()
            print("Finished preprocessor initializing")
                
    def run_episode(self, deterministic=False):
        """ Run single episode with option to animate
        Args:
        
        Returns: two dictionaries with keys "Observations", "Actions", "Rewards"
        Observations: shape = (episode len, obs_dim)
        Actions: shape = (episode len, act_dim)
        Rewards: shape = (episode len,)

        The first dictionary is original trajectory
        The second is normalized trajectory (same as the first if no state or reward preprocessor specified)
        """


        obs = self.env.reset()
        Observations, Actions, Rewards = [], [], [] # original trajectory
        n_Observations, n_Rewards = [], [] # normalized trajectory
        done = False
        timestep = 0
        while not done and timestep < self.episode_horizon:
            Observations.append(obs)
            if self.state_preprocessor:
                n_obs = self.state_preprocessor.get_scaled_x(obs)
            else:
                n_obs = obs
            n_Observations.append(n_obs)
            action = self.policy.get_action(obs.astype(np.float32).reshape((1,-1)), deterministic=deterministic)
            Actions.append(action.flatten())
            obs, reward, done, _ = self.env.step(np.squeeze(action, axis=0))
            Rewards.append(reward)
            if self.reward_preprocessor:
                n_reward = self.reward_preprocessor.get_scaled_x(reward)
            else:
                n_reward = reward
            n_Rewards.append(n_reward)
            timestep += 1

            
        # append the last state
        Observations.append(obs)
        if self.state_preprocessor:
            n_obs = self.state_preprocessor.get_scaled_x(obs)
        else:
            n_obs = obs
        n_Observations.append(n_obs)

        unscaled_traj = {"Observations": np.array(Observations), "Actions": np.array(Actions), "Rewards": np.array(Rewards)}
        scaled_traj = {"Observations": np.array(n_Observations), "Actions": np.array(Actions), "Rewards": np.array(n_Rewards)}

        # update preprocessers
        if self.state_preprocessor:
            self.state_preprocessor.update(unscaled_traj['Observations'])
            # save preprocessor params for restoration
            self.state_preprocessor.save_params(os.path.join(self.logger.info_dir, "state_preprocessor_params.pkl"))
        if self.reward_preprocessor:
            self.reward_preprocessor.update(unscaled_traj['Rewards'])
            self.reward_preprocessor.save_params(os.path.join(self.logger.info_dir, "reward_preprocessor_params.pkl"))
            
        return unscaled_traj, scaled_traj
            
    def get_batch(self, deterministic=False):
        '''
        returns two list of dictionarys [{"Observations":obs, "Actions":act, "Rewards": reward}, ...] where
        obs has dimension steps_per_episode x obs_dim
        act has dimention steps_per_episode-1 x act_dim
        reward has dimension steps_per_episode-1 x 1

        the first list is original (unscaled) trajectories, second is scaled
        '''
        unscaled_batch = []
        scaled_batch = []
        for i in range(self.batch_size):
            unscaled_traj, scaled_traj = self.run_episode(deterministic)
            unscaled_batch.append(unscaled_traj)
            scaled_batch.append(scaled_traj)

                
        return unscaled_batch, scaled_batch


    def save_batch(self, batch, path):
        '''
        This function saves batch Observations, Actions as a pkl file
        '''
        res = cloudpickle.dumps(batch)
        with open(path, "wb") as f:
            f.write(res)
   
        
if __name__ == "__main__":
    from baxter_learn.go_to_goal.env.baxter_r_arm_vel_control_env import BaxterRightArmVelControlEnv
    from baxter_learn.go_to_goal.rlfps_execution.config import CONFIG, configuration
    from baxter_learn.go_to_goal.ppo.policies.mlp_policy import MlpPolicy
    from baxter_learn.go_to_goal.ppo.preprocessing.running_average import RunningAverage
    
    # Update CONFIG each iteration through hyperparameter tuning
    hyperparam_str = "lr0.001"
    configuration.create_hyperparam_exp(hyperparam_str)
    CONFIG['env_params'].update(monitor_dir=configuration.log_hyperparam_exp_dir)
    CONFIG['logging_params']['log_params']['summary_spec'].update(directory=configuration.log_hyperparam_exp_dir)
    CONFIG['logging_params']['log_params']['saver_spec'].update(directory=configuration.model_hyperparam_exp_dir)

    # create environment
    env = BaxterRightArmVelControlEnv(CONFIG['env_params'])
    
    # update config with env specs
    policy_config = CONFIG['train_params']['hyperparams']['policy_hyperparams']
    policy_config['upper_bound'] = env.action_space['upper_bound']
    policy_config['lower_bound'] = env.action_space['lower_bound']
    policy_config['obs_dim'] = env.state_space['shape'][0]
    
    sampler_config = CONFIG['train_params']['hyperparams']['sampler_hyperparams']

    policy = MlpPolicy(policy_config)
    
    preprocessor_config = CONFIG['preprocessing_params']
    
    state_preprocessor = RunningAverage(env.state_space['shape'][0])
    
    bs = BatchSampler(env, policy, sampler_config, state_preprocessor)

    print(bs.get_batch())