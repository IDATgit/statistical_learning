import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def read_data(path='train_data'):
    with open(path, 'r') as fp:
        lines = fp.read().splitlines()
    data = list(map(lambda x:x.strip().split(' '), lines))
    data = list(map(lambda x:[float(i) for i in x], data))
    data = np.array(data)
    x = data[:, 1:]
    y = data[:, 0]
    yn = y[(y == 2) | (y == 3)]
    xn = x[(y == 2) | (y == 3)]
    print(f"Read file: {path}, data set length is {len(yn)}")
    return xn, yn


def ols_linear_regression(X_train, Y_train, X_test, Y_test):
    ols_model = LinearRegression().fit(X_train, Y_train)
    # train error rate
    Y_train_predict = ols_model.predict(X_train)
    # Round
    Y_train_predict[Y_train_predict <= 2.5] = 2
    Y_train_predict[Y_train_predict > 2.5] = 3
    ols_train_error_rate = np.sum(Y_train_predict != Y_train) / len(Y_train)
    print(f"OLS TRAIN error rate= {ols_train_error_rate}")
    # Test error rate
    Y_test_predict = ols_model.predict(X_test)
    # Round
    Y_test_predict[Y_test_predict <= 2.5] = 2
    Y_test_predict[Y_test_predict > 2.5] = 3
    ols_test_error_rate = np.sum(Y_test_predict != Y_test) / len(Y_test)
    print(f"OLS TEST error rate = {ols_test_error_rate}")
    return ols_train_error_rate, ols_test_error_rate


def knn_classifier(X_train, Y_train, X_test, Y_test, n):
    knn_model = KNeighborsClassifier(n_neighbors=n)
    knn_model.fit(X_train, Y_train)
    # train error rate
    Y_train_predict = knn_model.predict(X_train)
    knn_train_error_rate = np.sum(Y_train_predict != Y_train) / len(Y_train)
    print(f"KNN TRAIN error rate = {knn_train_error_rate}")
    # Test error rate
    Y_test_predict = knn_model.predict(X_test)
    knn_test_error_rate = np.sum(Y_test_predict != Y_test) / len(Y_test)
    print(f"KNN TEST error rate = {knn_test_error_rate}")
    return knn_train_error_rate, knn_test_error_rate

# Load  data
X_train, Y_train = read_data('train_data')
X_test, Y_test = read_data('test_data')
ols_train_error_rate, ols_test_error_rate = ols_linear_regression(X_train, Y_train, X_test, Y_test)

n_list = [1, 3, 5, 7, 15]
knn_train_error_rate_arr = [0] * len(n_list)
knn_test_error_rate_arr = [0] * len(n_list)
for (n_idx, n) in enumerate(n_list):
    knn_train_error_rate, knn_test_error_rate = knn_classifier(X_train, Y_train, X_test, Y_test, n)
    knn_train_error_rate_arr[n_idx] = knn_train_error_rate
    knn_test_error_rate_arr[n_idx] = knn_test_error_rate


plt.figure()
plt.plot(n_list, [ols_train_error_rate]*len(n_list))
plt.plot(n_list, [ols_test_error_rate]*len(n_list))
plt.plot(n_list, knn_train_error_rate_arr)
plt.plot(n_list, knn_test_error_rate_arr)
plt.legend(['linear regression train error rate',
            'linear regression test error rate',
            'KNN train error rate',
            'KNN test error rate'])
plt.title("Linear Regression VS KNN")
plt.xlabel("k (nearest neighbors)")
plt.ylabel("error rate")
plt.grid()
plt.show()





# Predict



