import numpy as np
from baxter_learn.go_to_goal.ppo.preprocessing.preprocessor import Preprocessor
import cloudpickle

class RunningAverage(Preprocessor):
    """ Generate scale and offset based on running mean and stddev along axis=0

        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, dim):
        """
        Args:
            obs_dim: dimension of axis=1
        """

        super(RunningAverage, self).__init__(dim)
        
        self.vars = np.zeros(dim)
        self.means = np.zeros(dim)
        self.m = 0
        self.n = 0
        self.first_pass = True

        
    def update(self, X):
        """ Update running mean and variance (this is an exact method)
        Args:
            X: NumPy array, shape = (N, dim)

        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        if self.first_pass:
            self.means = np.mean(X, axis=0)
            self.vars = np.var(X, axis=0)
            self.m = X.shape[0]
            self.first_pass = False
        else:
            n = X.shape[0]
            new_data_var = np.var(X, axis=0)
            new_data_mean = np.mean(X, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.m += n

    def get_params(self):
        return( {"means": self.means, "vars": self.vars, "m":self.m} )

    def save_params(self, save_path):
        dump = cloudpickle.dumps(self.get_params)
        with open(save_path, "wb") as f:
            f.write(dump)

    def restore_preprocessor(self, restore_path):
        with open(restore_path, 'rb') as f:
            params = cloudpickle.load(f)

        self.m = params['m']
        self.vars = params['vars']
        self.means = params['means']
            
    def get_scaled_x(self, x):
        """ returns 2-tuple: (scale, offset) """
        scale = 1/(np.sqrt(self.vars) + 0.1)/3
        offset = self.means
        scaled_x = (x - offset) * scale

        return scaled_x

