#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import librairies and data #
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

df = pd.read_csv('Frozen_Dessert_Production.csv',index_col='DATE',parse_dates=True)
df.head()

df.columns = ['Production']

# Plotting time series #
df.plot(figsize=(12,8))


# ## Data preprocessing 

# In[ ]:


# Split the data to the train and test (for last 24 months) #
len(df)
test_size = 24
test_ind = len(df) - test_size

train = df.iloc[:test_ind]
test = df.iloc[test_ind:]

len(test)

# Scale the data #
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(train)

scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# In[ ]:


# Time series genarator #
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

length = 18
n_features=1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=1)


# ## Model creation

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

# Define model #
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(length, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()


# In[ ]:


validation_generator = TimeseriesGenerator(scaled_test,scaled_test, length=length, batch_size=1)

# Early stopping call-back #
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=2)

# Fit model #
model.fit_generator(generator,epochs=20,
                    validation_data=validation_generator,
                   callbacks=[early_stop])


# In[ ]:


loss = pd.DataFrame(model.history.history)
loss.plot()


# ## Model evaluation

# In[ ]:


test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    
true_predictions = scaler.inverse_transform(test_predictions)
test['Predictions'] = true_predictions
test
test.plot()

from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(test['Production'],test['Predictions']))

