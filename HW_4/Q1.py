import utils
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

# Training data
X = utils.load_training_x()
Y = np.reshape(utils.load_training_y(), -1)

def get_model_performance(model, model_name, plot_tree=False):
    model_scores = cross_val_score(model, X, Y, cv=5, scoring="neg_root_mean_squared_error")
    model_cross_val_score = np.mean(-1 * model_scores)
    model.fit(X, Y)
    model_in_train_rmse = np.sqrt(np.mean((Y - model.predict(X)) ** 2))
    print(f"------------------------------------- {model_name}"
          f" ------------------------------------- ")
    print(f"Cross Validation RMSE score = {model_cross_val_score}")
    print(f"Training data RMSE score = {model_in_train_rmse}")
    if plot_tree:
        plt.figure()
        tree.plot_tree(small_tree_model, filled=True)
        plt.title(model_name)

# Small Tree
small_tree_model = tree.DecisionTreeRegressor(max_depth=2)
get_model_performance(small_tree_model, 'small tree', plot_tree=True)
print(f"number of leaves = {small_tree_model.get_n_leaves()}")

# Large tree (no limit)
large_tree_model = tree.DecisionTreeRegressor()
get_model_performance(large_tree_model, 'large tree')
print(f"number of leaves = {large_tree_model.get_n_leaves()}")


# Large tree after pruning with 1-SE rule
# TODO: fill

# RF / Bagging big trees
rf_large_trees = RandomForestRegressor(n_estimators=100, criterion="squared_error", max_depth=None, n_jobs=-1)
get_model_performance(rf_large_trees, 'random forest, large trees')

# RF / Bagging small trees
rf_small_trees = RandomForestRegressor(n_estimators=100, criterion="squared_error", max_depth=2, n_jobs=-1)
get_model_performance(rf_small_trees, 'random forest, small trees')

plt.show()