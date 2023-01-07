import utils
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNetCV
import matplotlib.pyplot as plt
X, Y = utils.load_training_data()
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# replace X with prerocessed data
with open('preprocess_data//linear_filled_with_indicators.npy', 'rb') as fp:
    X = np.load(fp)


#regressor = LinearRegression()
#regressor = Ridge(alpha=500)




def lasso_alpha_selection():
    #alpha_list = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    alpha_list = [1e-7, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    alpha_list = [1e-7, 1e-4, 1e-3, 2e-3,3e-3 ,4e-3, 5e-3, 6e-3, 7e-3, 1e-2, 5e-2]
    mean_list = np.zeros(len(alpha_list))
    std_list = np.zeros(len(alpha_list))
    for idx, alpha in enumerate(alpha_list):
        regressor = Lasso(alpha=alpha)
        scores = cross_val_score(regressor, X, Y, cv=5, scoring='neg_root_mean_squared_error')
        mean_score = scores.mean()
        std_score = scores.std()
        mean_list[idx] = -mean_score
        std_list[idx] = std_score


    arg_win = np.argmin(mean_list + std_list)
    winning_alpha = alpha_list[arg_win]
    print("Winning Alpha = ", winning_alpha)
    print("Winning Alpha RMSE = ", mean_list[arg_win])
    print("Winning Alpha RMSE + STD = ", mean_list[arg_win] + std_list[arg_win])
    plt.semilogx(alpha_list, mean_list, 'o-')
    plt.semilogx(alpha_list, mean_list+std_list, 'o-')
    plt.semilogx(alpha_list, mean_list - std_list, 'o-')
    plt.grid()
    plt.xlabel("Alpha value")
    plt.ylabel("RMSE")
    plt.title("Alpha selection graph")
    plt.show()



def ridge_alpha_selection():
    alpha_list = [0.1, 1, 10, 100, 500, 800, 900, 1000, 1200, 1500, 2000, 5000, 10000]

    mean_list = np.zeros(len(alpha_list))
    std_list = np.zeros(len(alpha_list))
    for idx, alpha in enumerate(alpha_list):
        regressor = Ridge(alpha=alpha)
        scores = cross_val_score(regressor, X, Y, cv=5, scoring='neg_root_mean_squared_error')
        mean_score = scores.mean()
        std_score = scores.std()
        mean_list[idx] = -mean_score
        std_list[idx] = std_score

    arg_win = np.argmin(mean_list + std_list)
    winning_alpha = alpha_list[arg_win]

    print("Winning Alpha = ", winning_alpha)
    print("Winning Alpha RMSE = ", mean_list[arg_win])
    print("Winning Alpha RMSE + STD = ", mean_list[arg_win] + std_list[arg_win])
    plt.semilogx(alpha_list, mean_list, 'o-')
    plt.semilogx(alpha_list, mean_list+std_list, 'o-')
    plt.semilogx(alpha_list, mean_list - std_list, 'o-')
    plt.grid()
    plt.xlabel("Alpha value")
    plt.ylabel("RMSE")
    plt.title("Alpha selection graph")
    plt.show()



def elastic_search():
    model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], eps=1e-5, n_alphas=1000, fit_intercept=True, precompute='auto', max_iter=2000, tol=0.0001, cv=5,
                         copy_X=True, verbose=False, n_jobs=-1, positive=False, random_state=None)
    model.fit(X, Y.reshape(-1))
    scores = cross_val_score(model, X, Y, cv=5, scoring='neg_root_mean_squared_error')
    std = np.std(scores)
    rmse = np.mean(-np.array(scores))
    l1_ratio = model.l1_ratio_
    alpha = model.alpha_
    print(f'Selected l1 ratio: {l1_ratio}')
    print(f'Selected alpha: {alpha}')
    print("STD = ", std)
    print("RMSE = ", rmse)



def random_forest():
    model = RandomForestRegressor(n_estimators=1000, criterion='squared_error', max_depth=None, min_samples_split=2,
                                  min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                  max_leaf_nodes=None, min_impurity_decrease=0.0,
                                  bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
                                  warm_start=False)

    scores = cross_val_score(model, X, Y.reshape(-1), cv=5, scoring='neg_root_mean_squared_error')
    std = np.std(scores)
    rmse = np.mean(-np.array(scores))
    print("STD = ", std)
    print("RMSE = ", rmse)

def boosting():
    model = GradientBoostingRegressor(loss='squared_error', learning_rate=0.01, n_estimators=10000, subsample=1.0,
                                      criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
                                      init=None, random_state=None, max_features=None,
                                      alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False,
                                      validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)

    scores = cross_val_score(model, X, Y.reshape(-1), cv=5, scoring='neg_root_mean_squared_error')
    std = np.std(scores)
    rmse = np.mean(-np.array(scores))
    print("STD = ", std)
    print("RMSE = ", rmse)



def nn():
    from keras.wrappers.scikit_learn import KerasRegressor
    from sklearn.model_selection import cross_val_score

    def build_model():
        model = Sequential()
        model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=Adam())
        return model

    model = KerasRegressor(build_fn=build_model, epochs=10, batch_size=32, verbose=0)
    scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='neg_mean_squared_error')


boosting()
#elastic_search()
# ridge_alpha_selection()
# lasso_alpha_selection()


