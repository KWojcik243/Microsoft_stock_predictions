import copy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

# Create dataframe

df = pd.read_csv('MCFT.csv')
df = df.sort_index(ascending=False, axis=0)
df = df.rename(columns={'Close/Last': 'Close', 'Volume': 'Employees'})
df = df.drop('Open', 1)
df = df.drop('High', 1)
df = df.drop('Low', 1)
df = df.drop('Date', 1)

# Create fictional data
microsoft_employees = [114000, 124000, 131000, 144106, 163000]
records_per_year = 151
microsoft_employees_days = np.arange(len(microsoft_employees)) * records_per_year

trend = np.polyfit(x=microsoft_employees_days, y=microsoft_employees, deg=1)
f = np.poly1d(trend)
index = np.arange(len(microsoft_employees) * records_per_year)

trend_numb = []
for i in range(0, 3000):
    trend_numb.append(f(i))

scaler_trend_numb = MinMaxScaler()
scaler = MinMaxScaler()

trend_numb = np.array(trend_numb)
trend_numb = trend_numb.reshape(-1, 1)
trend_numb = scaler_trend_numb.fit_transform(trend_numb)

for i in range(len(df)):
    df.Close[i] = df.Close[i].replace('$', '')

df.loc[:, ['Close', 'Employees']] = df.loc[:, ['Close', 'Employees']].astype(float)

for i in range(len(df)):
    df.Employees[i] = trend_numb[len(df) - i]

decompose = seasonal_decompose(df.Close, model='additive', period=1)
decompose.plot()

# Prepare input
original_close = copy.deepcopy(df['Close']).values
df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
x = df.loc[:, ['Close', 'Employees']]

xtrain, xtest = train_test_split(df, train_size=0.8, test_size=0.2, shuffle=False)
xtrain, xvalid = train_test_split(xtrain, train_size=0.75, test_size=0.25, shuffle=False)

n_steps = 5
n_features = 2

train_gen = TimeseriesGenerator(xtrain.to_numpy(), xtrain['Close'].values,
                                length=n_steps, batch_size=1, sampling_rate=1)
test_gen = TimeseriesGenerator(xtest.to_numpy(), xtest['Close'].values,
                               length=n_steps, batch_size=1, sampling_rate=1)
valid_gen = TimeseriesGenerator(xvalid.to_numpy(), xvalid['Close'].values,
                                length=n_steps, batch_size=1, sampling_rate=1)


def lstm(n_steps, n_features):
    model = Sequential()
    model.add(LSTM(32, activation='linear', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(16, activation='linear', input_shape=(n_steps, n_features)))
    model.add(Dense(1, activation='linear'))
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    return model


early_stopping = EarlyStopping(monitor='loss',patience=5)
model = lstm(n_steps, n_features)
history = model.fit(train_gen, epochs=10, validation_data=valid_gen, callbacks=[early_stopping])
scores = model.evaluate(test_gen)

# Predict for test

temp_data = copy.deepcopy(xvalid)
temp_data = temp_data.to_numpy()
preds = []

for i in tqdm(range(xtest.shape[0])):
    batch = temp_data[-n_steps:]
    batch = batch.reshape(1, n_steps, n_features)
    single_prediction = model.predict(batch)
    next_batch_sample = [[single_prediction[0][0], trend_numb[len(xtrain) + i + len(xvalid)][0]]]
    temp_data = np.append(temp_data, next_batch_sample, axis=0)
    preds.append(single_prediction[0])

preds = scaler.inverse_transform(np.array(preds))
true = scaler.inverse_transform(copy.deepcopy(xtest).to_numpy())

y_pred = preds[:, 0]
y_true = true[:, 0]

plt.figure(figsize=(20, 10))
plt.title("LSTM predictions for test set", fontsize=26)
plt.plot([x for x in range(len(xtrain) + len(xvalid), len(xtrain) + len(xvalid) + len(xtest))], y_pred, label='LSTM')
plt.plot(original_close, label='Real')
plt.legend(loc='best', prop={'size': 36})
plt.show()


# Forecast
forecast_size = 200
temp_data = copy.deepcopy(xtest)
temp_data = temp_data.to_numpy()
preds2 = []

for i in tqdm(range(forecast_size)):
    batch = temp_data[-n_steps:]
    batch = batch.reshape(1, n_steps, n_features)
    single_prediction = model.predict(batch)
    next_batch_sample = [[single_prediction[0][0], trend_numb[len(xtrain) + i + len(xvalid) + len(xtest)][0]]]
    temp_data = np.append(temp_data, next_batch_sample, axis=0)
    preds2.append(single_prediction[0])

preds2 = scaler.inverse_transform(np.array(preds2))
y_pred = preds2[:, 0]

plt.figure(figsize=(20, 10))
plt.title("LSTM forecast", fontsize=26)
plt.plot([x for x in range(len(df), len(df) + forecast_size)], y_pred, label='Forecast')
plt.plot(original_close, label='Real')
plt.legend(loc='best', prop={'size':36})
plt.show()
