import utils
import numpy as np
from sklearn.linear_model import LinearRegression




def linear_fill_and_indicators(X):
    # Add missing indicators
    indicators = np.zeros([X.shape[0], 85])
    missing_mask = X[:, 14:] == 0
    indicators[missing_mask] = 1
    X = np.concatenate((X, indicators), axis=1)
    # Linear fill missing points
    for i in range(14, 99):
        valid_rows_mask = X[:, i] != 0
        missing_rows_mask = X[:, i] == 0
        print(f"Missing rate = {np.sum(missing_rows_mask) / np.size(X[:, i])}")
        x = X[valid_rows_mask, 0:i]
        y = X[valid_rows_mask, i]
        # fit on valid rows
        reg = LinearRegression().fit(x, y)
        # in sample error
        print("Score = ", reg.score(x, y))
        # predict and clip
        x_pred = X[missing_rows_mask, 0:i]
        y_pred = reg.predict(x_pred)
        y_pred[y_pred < 1] = 1
        y_pred[y_pred > 5] = 5
        X[missing_rows_mask, i] = y_pred
    return X


if __name__ == "__main__":
    X = utils.load_training_x()
    X = linear_fill_and_indicators(X)
    fname = 'preprocess_data//linear_filled_with_indicators.npy'
    with open(fname, 'wb') as fp:
        print("saving into: ", fname)
        np.save(fp, X)




