class Preprocessor(object):

    def __init__(self, dim):
        self.dim = dim
        
    def update(self, X):
        '''
        this is needed for some preprocessors such as running_average
        Args:
        X: NumPy array, shape = (N, dim)

        '''        
        self.preprocessor.update(X)

    def get_params(self):
        '''
        returns the parameters necessary to restore the preprocessor (as a dictionary)
        '''
        raise NotImplementError("")

    def save_params(self, save_path):
        '''
        save the parameters as a dictionary
        '''
        raise NotImplementError("")
        
        
    def restore_preprocessor(self, restore_path):
        '''
        restore the preprocessor 
        '''
        raise NotImplementError("")
        
    def get_scaled_x(x):
        raise NotImplementError("")
        