from rl_pipeline.agent.agent_base import AgentBase
import scipy.signal
import numpy as np

import tensorflow as tf
import os

class PPOAgent(AgentBase):

    def __init__(self,
                 agent_params,
                 sampler,
                 policy,
                 value=None,
                 logger=None,
                 **kwargs):

        super(PPOAgent, self).__init__(agent_params,
                                       sampler,
                                       policy,
                                       value,
                                       logger)

    def discount(self, x, gamma):
        """ Calculate discounted forward sum of a sequence at each point """
        return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

    def add_disc_sum_rew(self, batch, gamma):
        """ Adds discounted sum of rewards to all time steps of all trajectories
        Args:
        batch: as returned by sampler.get_batch()
        gamma: discount

        Returns:
        updated batch (mutates the batch dictionary to add 'Disc_sum_rewards')
        """
        for traj in batch:
            rewards = traj['Rewards']
            disc_sum_rew = self.discount(rewards, gamma)
            traj['Disc_sum_rew'] = disc_sum_rew

        return batch
            
    def add_value(self, batch):
        """ Adds estimated value to all time steps of all trajectories
        Args:
        batch: as returned by sampler.get_batch()
 
        
        Returns:
        updated batch (mutates batch dictionary to add 'Values')
        """
        for traj in batch:
            Obs = traj['Observations']
            values = self.value.predict(Obs)
            traj['Values'] = values

        return batch

    def add_gae(self, batch, gamma, lam):
        """ Add generalized advantage estimator.
        https://arxiv.org/pdf/1506.02438.pdf
        Args:
        batch: as returned by sampler.get_batch(), must include 'Values' key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
        lam=0 : use TD residuals
        lam=1 : A =  Sum Discounted Rewards - V_hat(s)
        
        Returns:
        updated batch (mutates batch dictionary to add 'Advantages')
        """
        for traj in batch:
            values = traj['Values'][:-1] # values has one more data point (for last obs) than rewards
            rewards = traj['Rewards']
            # temporal differences
            tds = rewards - values + np.append(values[1:] * gamma, 0)
            advantages = self.discount(tds, gamma * lam)
            traj['Advantages'] = advantages
            
        return batch
            
    def build_train_set(self, batch):
        """
        Args:
        batch: batch after processing by add_disc_sum_rew(),
        add_value(), and add_gae()
        
        Returns: dictionary with keys "Observations", "Actions", "Advantages", "Disc_sum_rew"
        """
        # Observations has one more row than other quantities, make everyone same length
        Observations = np.concatenate([traj['Observations'][:-1,:] for traj in batch]) 
        Actions = np.concatenate([traj['Actions'] for traj in batch])
        Disc_sum_rew = np.concatenate([traj['Disc_sum_rew'] for traj in batch])
        Advantages = np.concatenate([traj['Advantages'] for traj in batch])
        
        # normalize advantages
        Advantages = (Advantages - Advantages.mean()) / (Advantages.std() + 1e-6)

        return {"Observations": Observations,
                "Actions": Actions,
                "Disc_sum_rew": Disc_sum_rew,
                "Advantages": Advantages} 

            
    def log_batch_stats(self, Observations, Actions, Advantages, Disc_sum_rew, iteration):
        """ Log various batch statistics """
        
        self.logger.csv_record_tabular("mean_adv",np.mean(Advantages))
        self.logger.csv_record_tabular("min_adv",np.min(Advantages))
        self.logger.csv_record_tabular("max_adv",np.max(Advantages))
        self.logger.csv_record_tabular("std_adv",np.var(Advantages))
        self.logger.csv_record_tabular("mean_discounted_return", np.mean(Disc_sum_rew))
        self.logger.csv_record_tabular("min_discounted_return", np.min(Disc_sum_rew))
        self.logger.csv_record_tabular("max_discounted_return", np.max(Disc_sum_rew))
        self.logger.csv_record_tabular("std_discounted_return", np.var(Disc_sum_rew))
        self.logger.csv_record_tabular("iteration", iteration)

            
    def update(self, batch, itr):
        '''
        performs one update

        batch has keys "Observations", "Actions", "Rewards"
        '''
        batch_with_value = self.add_value(batch)
        batch_with_value_disc_sum_rew = self.add_disc_sum_rew(batch_with_value, self.agent_params['gamma'])
        batch_with_value_disc_sum_rew_gae = self.add_gae(batch_with_value_disc_sum_rew,self.agent_params['gamma'], self.agent_params['lam'])

        
        # concatenate all episodes into single NumPy array
        training_set_dict = self.build_train_set(batch_with_value_disc_sum_rew_gae)

        self.log_batch_stats(training_set_dict['Observations'],
                             training_set_dict['Actions'],
                             training_set_dict['Advantages'],
                             training_set_dict['Disc_sum_rew'],
                             itr
                         )

        # update policy
        self.policy.update(training_set_dict['Observations'],
                           training_set_dict['Actions'],
                           training_set_dict['Advantages'],
                           itr,
                       )
        # update value function
        self.value.fit(training_set_dict['Observations'], training_set_dict['Disc_sum_rew'], itr)
        
    def train(self):
        '''
        train the policy for the specified max_itr
       
        '''
        for itr in range(self.agent_params['max_itr']):
            unscaled_batch, scaled_batch = self.sampler.get_batch(deterministic=False)
            self.update(scaled_batch, itr)
            # saves performace metric in csv
            self.logger.csv_dump_tabular()
            # saves policy and baseline models
            self.logger.save_model(self.policy, model_name="policy", itr=itr)
            self.logger.save_model(self.value, model_name="baseline", itr=itr)
            # saves batch
            self.logger.save_content(unscaled_batch,
                                     path=os.path.join(self.logger.transitions_dir,"itr_"+str(it)+".pkl"),
                                     itr=itr,
                                     interval=self.logger.logging_params['transitions_params']['interval'],
                                     file_type="pkl"
            )
            
    def exit(self):
        '''
        clear session and exit gracefully
        '''
        #tf.get_default_session().__exit__()
        #tf.reset_default_graph()
        pass