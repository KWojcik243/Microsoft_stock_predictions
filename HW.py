import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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

scaler_trend_numb = MinMaxScaler()
scaler = MinMaxScaler()

for i in range(len(df)):
    df.Close[i] = df.Close[i].replace('$', '')
df.loc[:, ['Close']] = df.loc[:, ['Close']].astype(float)

df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
x = df.loc[:, ['Close']]

xtrain, xtest = train_test_split(df, train_size=0.8, test_size=0.2, shuffle=False)

model = ExponentialSmoothing(xtrain, trend='add', seasonal='add', initialization_method='known', initial_level=0.5,
                             initial_trend=0.5, initial_seasonal=0.5, seasonal_periods=357, damped_trend=False)
hw_model = model.fit(optimized=True, remove_bias=False)

# Predict for test
pred = hw_model.forecast(len(xtest))

plt.figure(figsize=(20, 10))
plt.title("HW predictions for test set", fontsize=26)
plt.plot(xtrain, label='Train')
plt.plot(xtest, label='Test')
plt.plot(pred, label='Holt-Winters')
plt.legend(loc='best', prop={'size':36})
plt.show()


# Forecast
model2 = ExponentialSmoothing(x, trend='add', seasonal='add', initialization_method='known', initial_level=0.5,
                             initial_trend=0.5, initial_seasonal=0.5, seasonal_periods=357, damped_trend=False)
hw_model2 = model2.fit(optimized=True, remove_bias=False)
pred2 = hw_model2.forecast(200)

plt.figure(figsize=(20, 10))
plt.title("HW forecast", fontsize=26)
plt.plot(x, label='Real')
plt.plot(pred2, label='Holt-Winters')
plt.legend(loc='best', prop={'size': 36})
plt.show()
