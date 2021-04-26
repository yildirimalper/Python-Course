import linear_regress
import unittest

class linear_regrees_Test(unittest.TestCase):

    def array_test(self):
    	if type(X) is np.ndarray:
    		pass
    	else:
    		raise TypeError("You must use NumPy arrays as inputs.")

    	if type(y) is np.ndarray:
    		pass
    	else:
    		raise TypeError("You must use NumPy arrays as inputs.")

    def number_of_obs_test(self):
        if X.shape[0] != y.shape[0]:
            raise CustomError("The number of observations must be equal in both arrays.")

    
if __name__ == '__main__':
  unittest.main()
