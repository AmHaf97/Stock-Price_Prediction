import numpy as np #Används för numerical computation. Ger support för multi-dim. arrayer, matriser samt matematisk funktioner för dessa arrayer
import matplotlib.pyplot as plt #Bibliotek för att skapa plots(visuella representationer av data)
import pandas as pd #Bibliotek för data manipulering & analys. Hjälper med data strukturer och funktioner
import pandas_datareader as web #Delpaket från panda bibliotek som används för att hämta data från webben
import datetime as dt #Python bibliotek, används för att manipulera tid och datum

from sklearn.preprocessing import MinMaxScaler #Verktyg och algoritmer för dataprocessing. T.ex. scaling av numerisk data 
from tensorflow.keras.models import Sequential #Bibliotek för Maskin -och djupinlärning, keras används för att bygga och lära neural-networks, sequential används för att skapa liniära lager
from tensorflow.keras.layers import Dense, Dropout, LSTM #Dense, Droupout & LongShortTermMemory är dessa linjära lager som används i neural-networks

# Load Data

companyA = 'IONQ'
companyB = 'AI'
companyC = 'PLTR'

start = dt.datetime(2023,6,1)
end = dt.datetime(2023,7,1)

data = web.DataReader(companyA, "yahoo", start, end)
data = web.DataReader(companyB, "yahoo", start, end)
data = web.DataReader(companyC, "yahoo", start, end, "yahoo", start, end)

# Prepare data

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 30

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, x_train.shape[0], x_train.shape[1], 1)

# build the model

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(unit=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fix(x_train, y_train, epochs=25, batch_size=32)

# test model accuracy on existing data

# load test data

test_start = dt.datetime(2023,6,1)
test_end = dt.datetime(2023,7,1)

test_data = web.Datareader(companyA, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].value
model_inputs = model_inputs.reshape(-1, 1)  
model_inputs = scaler.transform(model_inputs)

# make predictions

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test_append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(predicted_prices)

plt.plot(actual_prices, color="black", label=f"Actual {companyA, companyB, companyC} Price")
plt.plot(prediction_prices, color="green", label=f"Predicted {companyA, companyB, companyC} Price")
plt.title(f"Share {companyA, companyB, companyC} Price")
plt.xlabel('Time')
plt.ylabel(f'Share {companyA, companyB, companyC} Price')
plt.legend()
plt.show()
























