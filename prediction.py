import numpy as np
import tensorflow as tf

def get_compiled_model():
  model = tf.keras.Sequential([
                               tf.keras.layers.Input(shape=(38,)),
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
model.load_weights('saved_model/weights.v2')

try:
    dataPoint = np.array([[28,2,1,9,6,11,0,0,0,0,17210,6.6,17039,195,36,643,-8,19.5,1721.0,15,6,0,6,9,8,0,0,0,0,16567,6.8,17047,197,55,-643,8,19.7,1656.7]])
    print("the accuracy of the prediction = ",model.predict(dataPoint))
except:
    raise Exception("Please use valid game data as input.")
