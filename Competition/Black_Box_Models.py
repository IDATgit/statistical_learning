# Here we have models that are already optimized
# Each small_tree_model have train and predict function


class Lasso:
    def __init__(self):
        self.regressor = Lasso(alpha=0.004)
    def train(self, X_train, Y_train):
        self.regressor.train(X_train, Y_train)
    def predict(self, X_test):
        Y_test = self.regressor.predict(X_test)
        return Y_test



