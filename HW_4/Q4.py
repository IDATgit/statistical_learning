import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from math import ceil
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler


# read data
data_url = "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data"
df = pd.read_csv(data_url)
data = np.array(df)
data[data == 'Absent'] = 0
data[data == 'Present'] = 1
print(data)
# present absent
# train - test split 2/3 - 1/3
n = data.shape[0]
cut_idx = ceil(2 * n / 3)
x_train = data[:cut_idx, 1:-1].astype(float)
y_train = data[:cut_idx, -1].astype(float)
x_test = data[cut_idx:, 1:-1].astype(float)
y_test = data[cut_idx:, -1].astype(float)

# scaling
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

################## linear regression part #####################

# Linear regression NN #
model = Sequential()
model.add(Dense(1, input_shape=(x_train.shape[1],), activation='linear', kernel_initializer='normal'))

# compile model
model.compile(loss='mean_squared_error', optimizer='RMSprop')

# fit model on training data and predict.
model.fit(x_train, y_train, epochs=200, batch_size=32, verbose=False)
nn_pred = model.predict(x_test)
nn_pred[nn_pred > 0.5] = 1
nn_pred[nn_pred <= 0.5] = 0
nn_linear_reg_conf_matrix = confusion_matrix(y_test, nn_pred)
nn_linear_reg_conf_table = pd.DataFrame(nn_linear_reg_conf_matrix,
                                        index=['True 0', 'True 1'],
                                        columns=['Predicted 0', 'Predicted 1'])

# real linear regression #
lr = LinearRegression()
lr.fit(x_train, y_train)
linear_reg_pred = lr.predict(x_test)
linear_reg_pred[linear_reg_pred > 0.5] = 1
linear_reg_pred[linear_reg_pred <= 0.5] = 0
linear_reg_conf_matrix = confusion_matrix(y_test, linear_reg_pred)
linear_reg_conf_table = pd.DataFrame(linear_reg_conf_matrix,
                                     index=['True 0', 'True 1'],
                                     columns=['Predicted 0', 'Predicted 1'])


print("####### real linear regresssion ####### ")
print(linear_reg_conf_table)
print("####### NN linear regresssion ####### ")
print(nn_linear_reg_conf_table)

###################### logistic regression  #############################
#
# Logistic regression NN #
model = Sequential()
model.add(Dense(1, input_shape=(x_train.shape[1],), activation='sigmoid', kernel_initializer='normal'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics='accuracy')





# fit model on training data and predict.
model.fit(x_train, y_train, epochs=200, batch_size=32, verbose=False)
nn_pred = model.predict(x_test)
nn_pred[nn_pred > 0.5] = 1
nn_pred[nn_pred <= 0.5] = 0
nn_logistic_conf_matrix = confusion_matrix(y_test, nn_pred)
nn_logistic_conf_table = pd.DataFrame(nn_logistic_conf_matrix,
                                        index=['True 0', 'True 1'],
                                        columns=['Predicted 0', 'Predicted 1'])

# Real Logistic regression #
lg = LogisticRegression()
lg.fit(x_train, y_train)
logistic_reg_pred = lg.predict(x_test)
logistic_reg_pred[logistic_reg_pred > 0.5] = 1
logistic_reg_pred[logistic_reg_pred <= 0.5] = 0
logistic_reg_conf_matrix = confusion_matrix(y_test, logistic_reg_pred)
logistic_reg_conf_table = pd.DataFrame(logistic_reg_conf_matrix,
                                        index=['True 0', 'True 1'],
                                        columns=['Predicted 0', 'Predicted 1'])



print("####### real logistic regresssion ####### ")
print(logistic_reg_conf_table)
print("####### NN logistic regresssion ####### ")
print(nn_logistic_conf_table)



#### more complex structure for logistic regression

# complex Logistic regression NN #
model = Sequential()
model.add(Dense(3, input_shape=(x_train.shape[1],), activation='sigmoid', kernel_initializer='normal'))
model.add(Dense(1, input_shape=(x_train.shape[1],), activation='sigmoid', kernel_initializer='normal'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics='accuracy')





# fit model on training data and predict.
model.fit(x_train, y_train, epochs=200, batch_size=32, verbose=False)
nn_pred = model.predict(x_test)
nn_pred[nn_pred > 0.5] = 1
nn_pred[nn_pred <= 0.5] = 0
nn_logistic_conf_matrix = confusion_matrix(y_test, nn_pred)
nn_logistic_conf_table = pd.DataFrame(nn_logistic_conf_matrix,
                                        index=['True 0', 'True 1'],
                                        columns=['Predicted 0', 'Predicted 1'])

print("####### NN 3 hidden layer regresssion ####### ")
print(nn_logistic_conf_table)



# Linear regression NN with 3 hidden nodes (actually still linear regression...)#
model = Sequential()
model.add(Dense(3, input_shape=(x_train.shape[1],), activation='linear', kernel_initializer='normal'))
model.add(Dense(1, input_shape=(x_train.shape[1],), activation='linear', kernel_initializer='normal'))

# compile model
model.compile(loss='mean_squared_error', optimizer='RMSprop')

# fit model on training data and predict.
model.fit(x_train, y_train, epochs=200, batch_size=32, verbose=False)
nn_pred = model.predict(x_test)
nn_pred[nn_pred > 0.5] = 1
nn_pred[nn_pred <= 0.5] = 0
nn_linear_reg_conf_matrix = confusion_matrix(y_test, nn_pred)
nn_linear_reg_conf_table = pd.DataFrame(nn_linear_reg_conf_matrix,
                                        index=['True 0', 'True 1'],
                                        columns=['Predicted 0', 'Predicted 1'])


print("####### NN 3 hidden layer linear regression ####### ")
print(nn_linear_reg_conf_table)