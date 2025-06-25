import tensorflow as tf
from tensorflow import keras
import numpy as np


for j in ['15000K']:

    features = np.loadtxt('features_test_' + str(j) + '.txt')
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
    model.load_weights('./models/model_SrCs_upto10K')

    output = model.predict(true_features, batch_size=batch_size)
    np.savetxt('SrCs_upto10K_' + str(j) + '_val_preds_1.txt', output)


