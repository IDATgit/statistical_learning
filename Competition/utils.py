import urllib.request  # the lib that handles the url stuff
import numpy as np
url_train_x = 'http://www.tau.ac.il/~saharon/StatsLearn2022/train_ratings_all.dat'
url_train_y = 'http://www.tau.ac.il/~saharon/StatsLearn2022/train_y_rating.dat'
url_test_x = 'http://www.tau.ac.il/~saharon/StatsLearn2022/test_ratings_all.dat'


def load_test_x():
    data = _load_data_single_url(url_test_x)
    return data

def _load_data_single_url(url):
    data = []
    for line in urllib.request.urlopen(url):
        x = line.decode('utf-8')
        x = x.replace('\n\r', '')
        x = x.split('\t')
        x = list(map(lambda x: int(x), x))
        data.append(x)
    data = np.array(data)
    return data

def load_training_data():
    X = _load_data_single_url(url_train_x)
    Y = _load_data_single_url(url_train_y)
    return X, Y

def load_training_x():
    return _load_data_single_url(url_train_x)

def load_training_y():
    return _load_data_single_url(url_train_y)