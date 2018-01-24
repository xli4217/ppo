import numpy as np
import tensorflow as tf
import os

class MlpPolicy():
    def __init__(self, config, logger=None, deploy=False):
        self.config = config
        self.beta = self.config['beta']
        self.eta = self.config['eta']
        self.kl_targ = self.config['kl_targ']
        self.epochs = self.config['epochs']
        self.lr = self.config['policy_lr']
        self.lr_multiplier = self.config['lr_multiplier']
        self.entcoeff = self.config['entropy_loss_coeff']
        
        self.obs_dim = config['obs_dim']
        self.act_dim = len(config['upper_bound'])
        self.act_ub = config['upper_bound']
        self.act_lb = config['lower_bound']
    
        
        self._build_graph()
        if not deploy:
            self._init_session()
        else:
            self.sess = tf.Session(graph=self.g)


        if logger:
            self.logger = logger
            if self.logger.name == "tensorboard_logger":
                self.logger.tf_writer_add_graph(self.g)

        self._build_tensorboard_summary()
                
    def save(self, model_dir, model_name="model"):
        self.saver.save(self.sess, os.path.join(model_dir, model_name))

    def restore(self, model_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))
        
    def _build_graph(self):
        self.g = tf.Graph()

        with tf.variable_scope(self.config['scope']): # this is to enable loading of multiple policies in the same session, needed for policy switching
            with self.g.as_default():
                self._placeholders()
                self._policy_nn()
                self._logprob()
                self._kl_entropy()
                self._sample()
                self._deterministic_action()
                self._loss_train_op()
                self.init = tf.global_variables_initializer()
                self.saver = tf.train.Saver(max_to_keep=5)

    def _build_tensorboard_summary(self):
        # setup logging
        self.logger.tf_writer_add_summary("scalar", name="policy_loss", summary=self.loss)
        self.logger.tf_writer_add_summary("scalar", name="policy_entropy", summary=self.entropy)
        self.merged_summary = self.logger.tf_writer_summary_merge()

                
    def _placeholders(self):
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'observations')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'actions')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')

        self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
        self.eta_ph = tf.placeholder(tf.float32, (), 'eta')
        self.entcoeff_ph = tf.placeholder(tf.float32, (), 'entcoeff')
        
        self.lr_ph = tf.placeholder(tf.float32, (), 'learning_rate')

        self.old_log_vars_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_log_vars')
        self.old_means_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')

    def _policy_nn(self):
        # hid1_size = self.obs_dim * 10
        # hid3_size = self.act_dim * 10
        # hid2_size = int(np.sqrt(hid1_size*hid3_size))

        hid1_size = 20
        hid2_size = 20
        hid3_size = 20
        
        # define the policy network
        out = tf.layers.dense(self.obs_ph,
                              hid1_size,
                              tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(stddev=0.05),
                              name='h1'
                          )

        out = tf.layers.dense(out,
                              hid2_size,
                              tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(stddev=0.05),
                              name='h2'
        )

        # out = tf.layers.dense(out,
        #                       hid3_size,
        #                       tf.tanh,
        #                       kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1/hid2_size)),
        #                       name='h3'
        # )

        self.means = tf.layers.dense(out,
                                     self.act_dim,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.2),
                                     name='means'
        )

        logvar_speed = (10*hid3_size) // 48 # ???????????
        log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32, tf.constant_initializer(0.0))

        self.log_vars = tf.reduce_sum(log_vars, axis=0) + 0.5

        print('Policy Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}, logvar_speed: {}'
              .format(hid1_size, hid2_size, hid3_size, self.lr, logvar_speed))


    def _logprob(self):
        """ Calculate log probabilities of a batch of observations & actions
        Calculates log probabilities using previous step's model parameters and
        new parameters being trained.
        """

        logp = -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                     tf.exp(self.log_vars), axis=1)
        self.logp = logp

        logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) /
                                         tf.exp(self.old_log_vars_ph), axis=1)
        self.logp_old = logp_old


    def _kl_entropy(self):
        """
        Add to Graph:
        1. KL divergence between old and new distributions
        2. Entropy of present policy given states and actions
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
        """

        log_det_cov_old = tf.reduce_sum(self.old_log_vars_ph)
        log_det_cov_new = tf.reduce_sum(self.log_vars)
        tr_old_new = tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars))

        self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                       tf.reduce_sum(tf.square(self.means - self.old_means_ph) /
                                                     tf.exp(self.log_vars), axis=1) -
                                       self.act_dim)
        self.entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) + tf.reduce_sum(self.log_vars))

        
    def _sample(self):
        """ Sample from distribution, given observation """
        self.sampled_act = (self.means +
                            tf.exp(self.log_vars / 2.0) *
                            tf.random_normal(shape=(self.act_dim,)))

    def _deterministic_action(self):
        '''
        get the mean of the distribution, use for evaluation and deployment
        '''

        self.deterministic_act = self.means
        
    def _loss_train_op(self):
        """
        Three loss terms:
        1) standard policy gradient
        2) D_KL(pi_old || pi_new)
        3) Hinge loss on [D_KL - kl_targ]^2
        See: https://arxiv.org/pdf/1707.02286.pdf
        """
        # --- original loss ---
        # loss1 = -tf.reduce_mean(self.advantages_ph *
        #                    tf.exp(self.logp - self.logp_old))
        # loss2 = tf.reduce_mean(self.beta_ph * self.kl) # penalizes large KL
        # loss3 = self.eta_ph * tf.square(tf.maximum(0.0, self.kl - 2.0 * self.kl_targ)) # reinforce the penalty of loss2
        loss4 = -self.entcoeff_ph * self.entropy # prevent policy entropy to decrease too quickly
        #self.loss = loss1 + loss2 + loss3

        # ---- cliped loss ------
        loss1 = -tf.reduce_mean(tf.minimum(self.advantages_ph * tf.exp(self.logp - self.logp_old),
                                           tf.clip_by_value(tf.exp(self.logp - self.logp_old), 1-0.2, 1+0.2) * self.advantages_ph
                                       ))
        self.loss = loss1 + loss4
        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(self.loss)

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.__enter__()
        self.sess.run(self.init)
        #print(self.sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
        
    def get_action(self, obs, deterministic=True):
        """get action from distribution"""
        
        feed_dict = {self.obs_ph: obs}

        if deterministic:
            action = np.clip(self.sess.run(self.deterministic_act, feed_dict=feed_dict), self.act_lb, self.act_ub)
        else:
            action = np.clip(self.sess.run(self.sampled_act, feed_dict=feed_dict), self.act_lb, self.act_ub)
        return action

    def update(self, observes, actions, advantages, itr):
        """ Update policy based on observations, actions and advantages
        Args:
        observes: observations, shape = (N, obs_dim)
        actions: actions, shape = (N, act_dim)
        advantages: advantages, shape = (N,)
        logger: Logger object, see utils.py
        """
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: self.beta,
                     self.eta_ph: self.eta,
                     self.entcoeff_ph: self.entcoeff,
                     self.lr_ph: self.lr}
        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars], feed_dict)
        
        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.old_means_ph] = old_means_np
        loss, kl, entropy = 0, 0, 0
        for e in range(self.epochs):
            # TODO: need to improve data pipeline - re-feeding data every epoch
            self.lr *= 1./(1. + self.config['lr_decay_rate']*(itr*self.epochs + e))
            feed_dict[self.lr_ph] = self.lr
            self.sess.run(self.train_op, feed_dict)
            loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)
            if kl > self.kl_targ * 4 or entropy < 0.01:  # early stopping if D_KL diverges badly
                break

            # save logs to tensorboard
            self.logger.tf_writer_update_summary(self.merged_summary, feed_dict, step=int(itr*self.epochs+e))
        
    
        if self.logger:
            self.logger.csv_record_tabular("policy_loss", loss)
            self.logger.csv_record_tabular("policy_entropy", entropy)
            self.logger.csv_record_tabular("KL", kl)
            self.logger.csv_record_tabular('Beta', self.beta)
            self.logger.csv_record_tabular('_lr_multiplier', self.lr_multiplier)
            self.logger.csv_record_tabular('policy_lr', self.lr)
        
    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()