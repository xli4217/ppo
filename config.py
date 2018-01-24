import os
from rl_pipeline.configuration.configuration import Configuration
import numpy as np


file_path = os.path.realpath(__file__)
file_dir = os.getcwd()
experiment_dir = os.path.join(file_dir, "experiments")
current_exp_dir = os.path.join(experiment_dir, "test")

# create the experiment directories
configuration = Configuration(experiment_dir, current_exp_dir)


# Define and calculate steps, episodes, iterations
joint_cmd_publish_rate = 25 # hz
steps_per_episode = 100 # cmd publishes at 25hz, this is 6 secs of execution per episode
episodes_per_iteration = 30


CONFIG = {
    'configuration': configuration,
    "mode": "hyperparam_tuning",                   # this can be "train", "hyperparam_tuning", "deploy", "demonstration", etc
    "env_params": {
    },
    "preprocessing_params":{
        "state_preprocessor": "running_average",
        "reward_preprocessor": None
    },
    "train_params":{
        "hyperparams":{
            "policy_hyperparams": {
                "upper_bound": [], # fill in run script
                "lower_bound": [], # fill in run script
                "obs_dim": None, # fill in run script
                "hidden_layers": ["relu", "relu"],
                "policy_lr": 0.01,
                "lr_decay_rate": 1e-6,
                "beta": 1.0,
                "eta": 50,
                "kl_targ": 0.006,
                "lr_multiplier": 1.0,
                "epochs": 20,
                "entropy_loss_coeff": 0.02,
                "scope": "policy",
                "state_preprocessing": "running_average" # state preprocessing should be part of policy
            },
            "baseline_hyperparams": {
                "obs_dim": None, # fill in run script
                "epochs": 20,
                "value_lr": 0.01,
                "lr_decay_rate":1e-6,
            },
            "agent_hyperparams": {
                "gamma": 0.99, # discount factor
                "max_itr": 2,
                'lam': 0.99
            },
            "sampler_hyperparams":{
                'batch_size': 2, # in units of episodes
                "max_episode_timesteps": steps_per_episode,  
                "init_batch_size": 1, # run a few episode of untrained policy to initialize scaler
            },
            "task_specific_params": {}         # this can for instance be the PID gains of a robot controller or camera properties, or image information
        },
        "hyperparams_tuning": { # what's in this depends on the hyperparam opt method/package used
            "method": "random_search",
            "performance_metric": "average_return",
            "nb_steps": 2,
            "params_dict":{
                "policy_lr": {"type": "float", "range": [0.001, 0.01]},
                "value_lr": {"type": "float", "range": [0.001,0.01]}
            }
        },          
    },
    "evaluation_params":{},
    "deploy_params":{},
    "logging_params": {
        "csv_data_params": {},
        "log_params": {},
        "model_params": {
            "interval": 1 # save model every 10 iterations
        },
        "transitions_params":{
            "interval": 100 # record trajecory batch every 100 iterations 
        }
    },
}

