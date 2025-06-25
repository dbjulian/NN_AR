import tensorflow as tf
from tensorflow import keras
import numpy as np


features = np.loadtxt('features_SrCs_mol.txt')
targets = np.loadtxt('targets_SrCs_mol.txt')
targets = np.sqrt(targets)
batch_size = features.shape[0]
model = keras.Sequential()
actv_func = 'softplus'
model.add(keras.Input(shape=(2,)))
model.add(keras.layers.Dense(units=4, activation = actv_func, kernel_initializer=keras.initializers.GlorotNormal(seed=None)))
model.add(keras.layers.Dense(units=4, activation = actv_func, kernel_initializer=keras.initializers.GlorotNormal(seed=None)))
model.add(keras.layers.Dense(units=4, activation = actv_func, kernel_initializer=keras.initializers.GlorotNormal(seed=None)))
model.add(keras.layers.Dense(units=4, activation = actv_func, kernel_initializer=keras.initializers.GlorotNormal(seed=None)))
model.add(keras.layers.Dense(units=4, activation = actv_func, kernel_initializer=keras.initializers.GlorotNormal(seed=None)))
model.add(keras.layers.Dense(units=4, activation = actv_func, kernel_initializer=keras.initializers.GlorotNormal(seed=None)))
model.add(keras.layers.Dense(units=4, activation = actv_func, kernel_initializer=keras.initializers.GlorotNormal(seed=None))) 
model.add(keras.layers.Dense(units=1, activation = 'sigmoid', kernel_initializer=keras.initializers.GlorotNormal(seed=None)))
features_train = tf.convert_to_tensor(features)
targets_train = tf.convert_to_tensor(targets)
model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.007), loss = keras.losses.MeanSquaredError(), metrics=[keras.metrics.MeanSquaredError()])

train_the_model = model.fit(features_train, targets_train, batch_size = batch_size, epochs =10000)
model.save_weights('model_SrCs_mol_4')
