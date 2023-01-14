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
    std = np.std(-1*model_scores)
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
    return model_cross_val_score, std

# Small Tree
small_tree_model = tree.DecisionTreeRegressor(max_depth=2)
small_tree_score, small_tree_std = get_model_performance(small_tree_model, 'small tree', plot_tree=True)
print(f"number of leaves = {small_tree_model.get_n_leaves()}")

# Large tree (no limit)
large_tree_model = tree.DecisionTreeRegressor()
large_tree_score, large_tree_std = get_model_performance(large_tree_model, 'large tree')
print(f"number of leaves = {large_tree_model.get_n_leaves()}")


# Large tree after pruning with 1-SE rule
large_tree_model = tree.DecisionTreeRegressor(ccp_alpha=0)
path = large_tree_model.cost_complexity_pruning_path(X, Y)
# Find the optimal value for alpha by selecting the value that corresponds to the best model performance
alphas = path['ccp_alphas']
alphas = alphas[::100]
scores = [0] * len(alphas)
stds = [0] * len(alphas)
for i, alpha in enumerate(alphas):
    print("loop: ", i, 'from ', len(alphas))
    model = tree.DecisionTreeRegressor(ccp_alpha=alpha)
    model_scores = cross_val_score(model, X, Y, cv=5, scoring="neg_root_mean_squared_error")
    mean = np.mean(-1*model_scores)
    scores[i] = mean
    stds[i] = np.std(-1*model_scores)
# take the best alpha mean and std
best_alpha = alphas[np.argmin(scores)]
best_std = stds[np.argmin(scores)]
# calculate the 1-sse expected mse
one_sse_score_target = np.min(scores) + best_std
# find max alpha which has lower score than the target score
idx = np.where((one_sse_score_target - scores) >= 0)[0][0]
one_sse_score = scores[idx]
one_sse_alpha = alphas[idx]
print("best alpha = ", best_alpha)
print('best std = ', best_std)
print('best mse = ', np.min(scores))
print('1sse score target = ', one_sse_score_target)
print("1sse alpha = ", one_sse_alpha)
print('1sse mse = ', one_sse_score)
sse_pruning_score, sse_pruning_std = get_model_performance(tree.DecisionTreeRegressor(ccp_alpha=one_sse_alpha), 'pruned-1SSE')


# RF / Bagging big trees
rf_large_trees = RandomForestRegressor(n_estimators=100, criterion="squared_error", max_depth=None, n_jobs=-1)
rf_large_trees_score, rf_large_trees_std = get_model_performance(rf_large_trees, 'random forest, large trees')

# RF / Bagging small trees
rf_small_trees = RandomForestRegressor(n_estimators=100, criterion="squared_error", max_depth=2, n_jobs=-1)
rf_small_trees_score, rf_small_trees_std = get_model_performance(rf_small_trees, 'random forest, small trees')


# Comparison
# Data to plot
x_labels = ['small tree', 'large tree', 'pruning', 'RF large', 'RF small']
y_values = [small_tree_score, large_tree_score, sse_pruning_score, rf_large_trees_score, rf_small_trees_score]
stds = [small_tree_std, large_tree_std, sse_pruning_std, rf_large_trees_std, rf_small_trees_std]

# Create a figure and an axis
fig, ax = plt.subplots()

# Plot the bar chart
plt.errorbar( x_labels, y_values, yerr=stds, fmt='o', color='Black', elinewidth=3,capthick=3,errorevery=1, alpha=1, ms=4, capsize = 5)

# Set the title
ax.set_title('Tree models comparison', fontsize=18)
plt.ylabel('Cross Validation Test set MSE ')
# Set the tick label size
ax.tick_params(axis='both', which='major', labelsize=12)
plt.grid(axis='y')
# Show the plot

plt.show()