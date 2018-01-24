import numpy as np
import tensorflow as tf
import os
from sklearn.utils import shuffle

class MlpValue():
    def __init__(self, config, logger):
        self.config = config

        self.obs_dim = config['obs_dim']
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.epochs = self.config['epochs']
        self.lr = None # learning rate set in _build_graph()
        self._build_graph()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)
        # with self.g.as_default():
        #     print(self.sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
    
        #logger.add_tabular_output(os.path.join(config['data_dir'], "value_data.csv"))
        self.logger = logger
        
    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')
            self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')
            self.lr_ph = tf.placeholder(tf.float32, (), 'value_lr')
            
            # hid1_size = self.obs_dim * 10
            # hid3_size = 5
            # hid2_size = int(np.sqrt(hid1_size * hid3_size))

            hid1_size = 20
            hid2_size = 20
            hid3_size = 20
            
            self.lr = self.config['value_lr']

            print('Value Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}'
                  .format(hid1_size, hid2_size, hid3_size, self.lr))

            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=0.05), name="h1")
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=0.05), name="h2")
            # out = tf.layers.dense(out, hid3_size, tf.tanh,
            #                       kernel_initializer=tf.random_normal_initializer(
            #                           stddev=np.sqrt(1 / hid2_size)), name="h3")
            out = tf.layers.dense(out, 1,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=0.05), name='output')

            self.out = tf.squeeze(out)
            self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))
            optimizer = tf.train.AdamOptimizer(self.lr_ph)
            self.train_op = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=5)
        self.sess = tf.Session(graph=self.g)

    def save(self, model_dir, model_name="model"):
        self.saver.save(self.sess, os.path.join(model_dir, model_name))

    def restore(self, model_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))
   
        
    def fit(self, x, y, itr):
        """ Fit model to current data batch + previous data batch
        Args:
        x: features
        y: target
        """
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        y_hat = self.predict(x)  # check explained variance prior to update
        old_exp_var = 1 - np.var(y - y_hat)/np.var(y)
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
            self.replay_buffer_x = x
            self.replay_buffer_y = y
        for e in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train)
            self.lr *= 1./(1. + self.config['lr_decay_rate']*(itr*self.epochs + e))
            feed_dict = {self.lr_ph: self.lr}
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict[self.obs_ph] = x_train[start:end, :]
                feed_dict[self.val_ph] = y_train[start:end]
                             
                _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        y_hat = self.predict(x)
        loss = np.mean(np.square(y_hat - y))         # explained variance after update
        exp_var = 1 - np.var(y - y_hat) / np.var(y)  # diagnose over-fitting of val func

        self.logger.csv_record_tabular("val_func_loss", loss)
        self.logger.csv_record_tabular("explained_var_new", exp_var)
        self.logger.csv_record_tabular("explained_var_old", old_exp_var)
        self.logger.csv_record_tabular("val_lr", self.lr)
        
    def predict(self, x):
        """ Predict method """
        feed_dict = {self.obs_ph: x}
        y_hat = self.sess.run(self.out, feed_dict=feed_dict)
            
        return np.squeeze(y_hat)

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()