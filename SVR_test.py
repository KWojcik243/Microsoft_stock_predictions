import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from tqdm import tqdm


def generate_bathes(array, batch_size):
    X, y = [], []

    for i in range(len(array) - batch_size):
        X.append([x[0] for x in array[i:i+batch_size]])
        y.append(array[i+batch_size][0])

    return X, y


# Create dataframe
df = pd.read_csv('MCFT.csv')
df = df.iloc[::-1].reset_index()
df = df.drop(['index'], axis=1)
df = df.rename(columns={'Close/Last': 'Close'})
df = df.drop('Open', 1)
df = df.drop('High', 1)
df = df.drop('Low', 1)
df = df.drop('Volume', 1)
df = df.drop('Date', 1)

scaler = MinMaxScaler()

for i in range(len(df)):
    df.Close[i] = df.Close[i].replace('$', '')

df.loc[:, ['Close']] = df.loc[:, ['Close']].astype(float)

df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
x = df.loc[:, ['Close']]


xtrain, xtest = train_test_split(df, train_size=0.8, test_size=0.2, shuffle=False)

n_steps = 130

x_train, y_train = generate_bathes(xtrain.values, n_steps)
temp = np.concatenate((xtrain[-n_steps:].values, xtest.values), axis=None)
temp = [[x] for x in temp]
x_test, y_test = generate_bathes(temp, n_steps)


# Model
model = SVR(kernel='linear', C=0.9, verbose=True)
svr_model = model.fit(x_train, y_train)


# Predict on test
pred = []
temp_data = x_train[-n_steps:]
temp_data = temp_data[0]
for i in tqdm(range(len(y_test))):
    batch = temp_data[-n_steps:]
    single_pred = svr_model.predict([batch])[0]
    temp_data.append(single_pred)
    pred.append(single_pred)



# Plot preds
plt.figure(figsize=(20, 10))
plt.title("SVM predictions for test set", fontsize=26)
plt.plot(xtrain, label='Train')
plt.plot(xtest, label='Test')
plt.plot([i for i in range(len(xtrain), len(xtrain) + len(pred))], pred, label='SVM')
plt.legend(loc='best', prop={'size': 36})
plt.show()


# Forecast
pred2 = []
temp_data = x_test[-n_steps:]
temp_data = temp_data[0]
forecast_length = 200
for i in tqdm(range(forecast_length)):
    batch = temp_data[-n_steps:]
    single_pred = svr_model.predict([batch])[0]
    temp_data.append(single_pred)
    pred2.append(single_pred)

# Plot forecast
plt.figure(figsize=(20, 10))
plt.title("SVM forecast", fontsize=26)
plt.plot(xtrain, label='Real', color='blue')
plt.plot(xtest, color='blue')
plt.plot([i for i in range(len(xtrain) + len(xtest), len(xtrain) + len(xtest) + forecast_length)], pred2, label='SVM')
plt.legend(loc='best', prop={'size': 36})
plt.show()
