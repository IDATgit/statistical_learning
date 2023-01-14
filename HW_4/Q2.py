import utils
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

#
# num_iters = 1000
# base_model = tree.DecisionTreeClassifier(max_depth=2)
# Y_train_scaled = Y_train
# weights = np.ones(len(Y_train))
# test_boost = np.zeros(len(Y_test))
# train_boost = np.zeros(len(Y_train))
# train_error_rate_arr = [0]*num_iters
# test_error_rate_arr = [0]*num_iters
# train_boost_classifier = np.zeros(len(train_boost))
# test_boost_classifier = np.zeros(len(test_boost))
# epsilon = 0.01
# boosting_method = 'epsilon'  # epsilon / alpha (linesearch)
# for i in range(num_iters):
#     # fit new model
#     base_model.fit(X_train, Y_train_scaled, sample_weight=weights)
#     y_hat_train = base_model.predict(X_train)
#     # error and alpha calculation
#     error = np.sum(weights * (y_hat_train != Y_train)) / np.sum(weights)
#     alpha = 0.5*np.log((1-error)/error)
#     print(f'alpha = {alpha}')
#     if boosting_method == 'alpha':
#         train_boost += alpha * y_hat_train
#     elif boosting_method == 'epsilon':
#         train_boost += epsilon * y_hat_train
#     train_boost_classifier[train_boost > 0 ] = 1
#     train_boost_classifier[train_boost <= 0] = -1
#
#     # Test set
#     y_hat_test = base_model.predict(X_test)
#     if boosting_method == 'alpha':
#         test_boost += alpha * y_hat_test
#     elif boosting_method == 'epsilon':
#         test_boost += epsilon * y_hat_test
#     test_boost_classifier[test_boost > 0] = 1
#     test_boost_classifier[test_boost <= 0] = -1
#
#     # update weights
#     weights = weights * np.exp(-alpha*y_hat_train*Y_train)
#
#     train_error_rate = np.mean(train_boost_classifier != Y_train)
#     test_error_rate = np.mean(test_boost_classifier != Y_test)
#     train_error_rate_arr[i] = train_error_rate
#     test_error_rate_arr[i] = test_error_rate
#     print("Iteration ", i)
#     print("Train error rate = ", train_error_rate)
#     print('Test error rate = ', test_error_rate)
#
#
# plt.figure()
# plt.title('ADABoost error rates')
# plt.plot(test_error_rate_arr)
# plt.plot(train_error_rate_arr)
# plt.grid()
# plt.xlabel('Iteration')
# plt.ylabel('Error Rate')
# plt.legend(['test set', 'train set'])
# plt.show()



# normal boosting with MSE
num_iters = 10000
base_model = tree.DecisionTreeRegressor(min_samples_leaf=800, criterion='friedman_mse')
epsilon = 0.01
Y_train_residues = Y_train
train_boost = np.zeros(len(Y_train))
train_boost_classifier = np.zeros(len(Y_train))
test_boost = np.zeros(len(Y_test))
test_boost_classifier = np.zeros(len(Y_test))
train_error_rate_arr = [0]*num_iters
test_error_rate_arr = [0]*num_iters
min_samples_leaf = [2000] * 3000 + [1000] * 2000 + [500] * 1000 + [400] * 1000 + [200] * 1000 + [100] * 1000 + [50] * 1000
for i in range(num_iters):
    base_model = tree.DecisionTreeRegressor(min_samples_leaf=min_samples_leaf[i], criterion='friedman_mse')
    base_model.fit(X_train, Y_train_residues)
    # Train update
    train_pred = base_model.predict(X_train)
    train_boost += epsilon * train_pred
    train_boost_classifier[train_boost > 0] = 1
    train_boost_classifier[train_boost <= 0] = -1
    Y_train_residues = Y_train - train_boost
    # Test update
    test_pred = base_model.predict(X_test)
    test_boost += epsilon * test_pred
    test_boost_classifier[test_boost > 0] = 1
    test_boost_classifier[test_boost <= 0] = -1

    train_error_rate = np.mean(train_boost_classifier != Y_train)
    test_error_rate = np.mean(test_boost_classifier != Y_test)
    train_error_rate_arr[i] = train_error_rate
    test_error_rate_arr[i] = test_error_rate
    print("Iteration ", i)
    print("Train error rate = ", train_error_rate)
    print('Test error rate = ', test_error_rate)



plt.figure()
plt.title('Regular Boosting Error rate')
plt.plot(test_error_rate_arr)
plt.plot(train_error_rate_arr)
plt.xlabel('Iteration')
plt.ylabel('Error Rate')
plt.grid()
plt.legend(['test set', 'train set'])
plt.show()

