import utils
import preprocess
import numpy as np
from sklearn.linear_model import Lasso
# Training
X_train = utils.load_training_x()
X_train = preprocess.linear_fill_and_indicators(X_train)
Y_train = utils.load_training_y()
regressor = Lasso(alpha=0.004)
regressor.fit(X_train, Y_train)


X_test = utils.load_test_x()
X_test = preprocess.linear_fill_and_indicators(X_test)

Y = regressor.predict(X_test)
Y[Y > 5] = 5
Y[Y < 1] = 1
Y = Y.T
np.savetxt("submissions//new_submission.csv", Y, delimiter=',')







