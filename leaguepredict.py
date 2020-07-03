import csv
import os
import pandas as pd
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf



data = pd.read_csv("high_diamond_ranked_10min.csv")
y = data["blueWins"]

data = data.drop("blueWins", axis=1)
data = data.iloc[:7476, :]

print(data[:10])
print(data.columns)

data_norm = ((data-data.min())/(data.max()-data.min()))
data_norm.head()

print(str((data['gameId'][0])))



y = np.array(y)
y = y[:7476]

x = data_norm.drop(['gameId'], axis=1)

X_train = x.iloc[:5000, :]
y_train = y[:5000]

X_test = x.iloc[5000:, :]
y_test = y[5000:]

print(X_train.shape[1])
checkpoint_path = "epochs/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=5)


def get_compiled_model():
  model = tf.keras.Sequential([
                               tf.keras.layers.Input(shape=(X_train.shape[1],)),
                               tf.keras.layers.Dense(80, activation='relu'),
                               tf.keras.layers.Dense(50, activation='relu'),
                               tf.keras.layers.Dense(40, activation='relu'),
                               tf.keras.layers.Dense(1, activation='sigmoid')
  ])
  model.compile(optimizer='adam',
                loss="mean_squared_error",
                metrics=['mse', 'mae'])
  return model

model = get_compiled_model()

#model.save_weights(checkpoint_path.format(epoch=0))

history = model.fit(X_train, y_train,
                    batch_size=10,
                    epochs=20,
                    callbacks=[cp_callback],
                    validation_data=(X_test, y_test),
                    verbose=1)

def plot_result(history):
  plt.plot(history.history['mse'])
  plt.plot(history.history['val_mse'])
  plt.title('model_mse')
  plt.ylabel('mse')
  plt.xlabel('epochs')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

  plt.plot(history.history['mae'])
  plt.plot(history.history['val_mae'])
  plt.title('model_mae')
  plt.ylabel('mae')
  plt.xlabel('epochs')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

plot_result(history)
print(history.history.keys())

testing = np.array([[28,2,1,9,6,11,0,0,0,0,17210,6.6,17039,195,36,643,-8,19.5,1721.0,15,6,0,6,9,8,0,0,0,0,16567,6.8,17047,197,55,-643,8,19.7,1656.7]])
testing = testing
print("the accuracy of the prediction = ",model.predict(testing))

#!mkdir -p saved_model
model.save('saved_model/model.v2')
model.save_weights('saved_model/weights.v2')
