from baxter_learn.go_to_goal.env.baxter_r_arm_vel_control_env import BaxterRightArmVelControlEnv
from baxter_learn.go_to_goal.rlfps_execution.config import CONFIG, configuration
from baxter_learn.go_to_goal.ppo.policies.mlp_policy import MlpPolicy
from baxter_learn.go_to_goal.ppo.value_functions.mlp_value import MlpValue
from baxter_learn.go_to_goal.ppo.preprocessing.running_average import RunningAverage
from baxter_learn.go_to_goal.ppo.agent.PPOAgent import PPOAgent
from baxter_learn.go_to_goal.ppo.sampler.batch_sampler import BatchSampler

from rl_pipeline.logging.tensorboard_logger import TensorboardLogger
from rl_pipeline.utils.utils import make_hyperparam_str

import numpy as np
import pandas as pd

import os

class PPORunner(object):

    def __init__(self, config):

        self.config = config
        
        if config['mode'] != "hyperparam_tuning":
            hyperparam_str = "exp"
            configuration.create_hyperparam_exp(hyperparam_str)
            self.create_agent()
            self.agent.logger.save_config(config)
                        
        
    def get_next_hyperparam_dict(self):
        '''
        returns new hyperparam dict as {"lr": 0.001, "nb_layers": 3, ...}
        '''
        hyperparam_dict = self.config['train_params']['hyperparams_tuning']
        if hyperparam_dict['method'] == "random_search":
            new_hyperparam_dict = {}
            for key, value in hyperparam_dict['params_dict'].iteritems():
                if value['type'] == "float":
                    new_hyperparam_dict[key] = np.random.uniform(value['range'][0],value['range'][1])
                if value['type'] == "int":
                    new_hyperparam_dict[key] = np.random.randint(value['range'][0],value['range'][1])
            return new_hyperparam_dict

        else:
            raise ValueError("hyperparam tuning method not supported")
            
        
    def update_config(self, hyperparam_dict):
        '''
        update corresponding item in self.config to match hyperparam_dict
        '''
        
        # update policy parameters
        for key, value in self.config['train_params']['hyperparams']['policy_hyperparams'].iteritems():
            if key in hyperparam_dict.keys():
                self.config['train_params']['hyperparams']['policy_hyperparams'][key] = hyperparam_dict[key]
        # update baseline parameters
        for key, value in self.config['train_params']['hyperparams']['baseline_hyperparams'].iteritems():
            if key in hyperparam_dict.keys():
                self.config['train_params']['hyperparams']['baseline_hyperparams'][key] = hyperparam_dict[key]
        # update baseline parameters
        for key, value in self.config['train_params']['hyperparams']['agent_hyperparams'].iteritems():
            if key in hyperparam_dict.keys():
                self.config['train_params']['hyperparams']['agent_hyperparams'][key] = hyperparam_dict[key]
  
                
    def store_hyperparam_performance(self):
        pass
        
    def create_agent(self):
        #################
        # create logger #
        #################
        logger = TensorboardLogger(csv_data_dir=configuration.csv_data_hyperparam_exp_dir,
                                   log_dir=configuration.log_hyperparam_exp_dir,
                                   model_dir=configuration.model_hyperparam_exp_dir,
                                   config_dir=configuration.config_hyperparam_exp_dir,
                                   info_dir=configuration.info_hyperparam_exp_dir,
                                   transitions_dir=configuration.transitions_hyperparam_exp_dir,
                                   logging_params=self.config['logging_params']
                               )

    
        ######################
        # create environment #
        ######################
        env = BaxterRightArmVelControlEnv(self.config['env_params'])
        
        #################
        # create policy #
        #################
        policy_config = self.config['train_params']['hyperparams']['policy_hyperparams']
        # update policy config
        policy_config['upper_bound'] = env.action_space['upper_bound']
        policy_config['lower_bound'] = env.action_space['lower_bound']
        policy_config['obs_dim'] = env.state_space['shape'][0]
        policy = MlpPolicy(policy_config, logger=logger)
        
        ###################
        # create baseline #
        ###################
        baseline_config = self.config['train_params']['hyperparams']['baseline_hyperparams']
        # update baseline config
        baseline_config['obs_dim'] = env.state_space['shape'][0]
        baseline = MlpValue(baseline_config, logger)
        
        ########################
        # create preprocessors #
        ########################
        preprocessor_config = self.config['preprocessing_params']
        state_preprocessor = RunningAverage(env.state_space['shape'][0])
        
        ########################
        # create batch sampler #
        ########################
        sampler_config = self.config['train_params']['hyperparams']['sampler_hyperparams']
        batch_sampler = BatchSampler(env, policy, sampler_config, state_preprocessor, logger=logger)
        
        ################
        # create agent #
        ################
        agent_config = self.config['train_params']['hyperparams']['agent_hyperparams']
        self.agent = PPOAgent(agent_config, batch_sampler, policy, baseline, logger)


    def train(self):
        if self.config['mode'] == "train":
            self.agent.train()
        if self.config['mode'] == "hyperparam_tuning":
            hyperparam_log = {self.config['train_params']['hyperparams_tuning']['performance_metric']:[]}
            for i in range(self.config['train_params']['hyperparams_tuning']['nb_steps']):
                hyperparam_dict = self.get_next_hyperparam_dict()
                print(hyperparam_dict)
                hyperparam_str = make_hyperparam_str(hyperparam_dict)
                configuration.create_hyperparam_exp(hyperparam_str)
                self.update_config(hyperparam_dict)
                self.create_agent()
                self.agent.logger.save_config(self.config)

                self.agent.train()
                # get a batch for evaluation
                print("Obtaining evaluation batch")
                unscaled_batch, _ = self.agent.sampler.get_batch(deterministic=True)
                # get average batch return
                avg_return = []
                for traj in unscaled_batch:
                    avg_return.append(np.sum(traj['Rewards']))
                print("performance metric: %f" %np.mean(np.array(avg_return)))
                # update hyperparam_log
                for key, value in hyperparam_dict.iteritems():
                    if key not in hyperparam_log.keys():
                        hyperparam_log[key] = [value]
                    else:
                        hyperparam_log[key].append(value)
                    
                hyperparam_log[self.config['train_params']['hyperparams_tuning']['performance_metric']] = np.mean(np.array(avg_return))
                self.agent.logger.save_content(content=pd.DataFrame(data=hyperparam_log),
                                               path=os.path.join(self.agent.logger.info_dir, "hyperparam_log.csv"),
                                               file_type="csv"
                                           )        
                # clear current models for the new one
                self.agent.exit()


if __name__ == "__main__":
    ppo_runner = PPORunner(CONFIG)
    ppo_runner.train()





















