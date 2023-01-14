import utils
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

# Training data
X = utils.load_training_x()
Y = np.reshape(utils.load_training_y(), -1)
# replace Y to a classification problem, Yi = 1 if Y>= 3, Yi = -1 if Y < 3
Y[Y < 3] = -1
Y[Y >= 3] = 1
# train - test split
X_train = X[:8000, :]
Y_train = Y[:8000]
X_test = X[8000:, :]
Y_test = Y[8000:]

classfier = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=2), n_estimators=1000)
classfier.fit(X_train, Y_train)
Y_pred_train = classfier.predict(X_train)
Y_pred_test = classfier.predict(X_test)

test_error_rate = np.mean(Y_pred_test != Y_test)
train_error_rate = np.mean(Y_pred_train != Y_train)

print(f'test error rate = {test_error_rate}')
print(f'train error rate = {train_error_rate}')

