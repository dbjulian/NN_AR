import tensorflow as tf
from tensorflow import keras
import numpy as np


for j in ['25K', '250K', '2500K', '12500K']:

    features = np.loadtxt('features_test_' + str(j) + '_mol.txt')
    batch_size = features.shape[0]
    true_features = tf.convert_to_tensor(features)
    true_features = tf.reshape(true_features, shape= [batch_size, 2])
    model = keras.Sequential()
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
    model.load_weights('model_SrCs_mol_4')

    output = model.predict(true_features, batch_size=batch_size)
    np.savetxt('SrCs_mol_' + str(j) + '_test_preds_10.txt', output)


