import urllib.request  # the lib that handles the url stuff
import numpy as np
from sklearn.linear_model import QuantileRegressor
import matplotlib.pyplot as plt
url_train_x = 'http://www.tau.ac.il/~saharon/StatsLearn2022/train_ratings_all.dat'
url_train_y = 'http://www.tau.ac.il/~saharon/StatsLearn2022/train_y_rating.dat'
url_test_x = 'http://www.tau.ac.il/~saharon/StatsLearn2022/test_ratings_all.dat'


def load_data(url):
    data = []
    for line in urllib.request.urlopen(url):
        x = line.decode('utf-8')
        x = x.replace('\n\r', '')
        x = x.split('\t')
        x = list(map(lambda x: int(x), x))
        data.append(x)
    data = np.array(data)
    return data



X = load_data(url_train_x)
Y = load_data(url_train_y)
# train and test separation
train_X = X[:8000]
train_Y = Y[:8000].reshape(-1)
test_X = X[8000:]
test_Y = Y[8000:].reshape(-1)
# quantile regression
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
RMSES = [0] * len(quantiles)
avg_preds = [0] * len(quantiles)
preds_5 = np.zeros([len(quantiles), 5])

for (idx, quantile) in enumerate(quantiles):
    qr = QuantileRegressor(quantile=quantile, solver='highs', alpha=0).fit(train_X, train_Y)
    test_Y_pred = qr.predict(test_X)
    RMSES[idx] = np.sqrt(np.mean((test_Y_pred - test_Y)**2))
    avg_preds[idx] = np.mean(test_Y_pred)
    preds_5[idx] = test_Y_pred[:5]
    plt.figure()
    plt.plot(test_Y_pred)
    plt.title("test Y pred")

plt.figure()
plt.plot(quantiles, RMSES, 'o-')
plt.title("RMSE vs quantile regressor")
plt.xlabel('quantile')
plt.ylabel('RMSE')
plt.grid()

plt.figure()
plt.plot(quantiles, avg_preds, 'o-')
plt.title('Average pred vs quantile regressor')
plt.xlabel('quantile')
plt.ylabel('Average prediction')
plt.grid()

plt.figure()
for i in range(len(quantiles)):
    plt.plot([quantiles[i]]*5, preds_5[i], '*b')
plt.title('5 first predictions of quantile regressors')
plt.xlabel('quantile')
plt.ylabel('5 first predictions values')
plt.grid()


plt.show()




























a=1