"""
This script performs CNN-based regridding of CMIP6 mean temperature outputs.
It represents the first step of the statistical downscaling workflow,
generating the CMIP6-MedPlus dataset.

Authors: Valeria Todaro and Daniele Secci

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
import h5py
import xarray as xr

var='tas' # mean temperature

######################## CNN training ########################################

# MATLAB v7.3 (HDF5-based) dataset containing ERA5 paired inputs and targets
file_name = f'Downscaling_dataset_ERA5_{var}.mat'

with h5py.File(file_name, 'r') as file:
    # TARGET (ground truth)
    #   Dimensions (140, 256, 19723)
    #   140  → number of grid points in latitude
    #   256  → number of grid points in longitude
    #   19723 → number of time steps (1jan1970 - 31dec2023)
    var_down_TRUE = file[f'{var}_down_TRUE'] 
    # INPUT (upscaled field)
    #   Dimensions (35, 64, 19723)
    #   35  → number of grid points in latitude
    #   64 → number of grid points in longitude
    #   19723 → number of time steps (1jan1970 - 31dec2023)
    var_up = file[f'{var}_up'] 

# Define timelag
timelag_train = np.arange(0, 10957) #1jan1970-31dec1999
timelag_val = np.arange(10957,14611) #1jan2000-31dec2009
timelag_test = np.arange(14611,19723) #1jan2010-31dec2023

inputData = np.reshape(var_up, (var_up.shape[0], var_up.shape[1], 1, var_up.shape[2]))
outputData = np.reshape(var_down_TRUE, (var_down_TRUE.shape[0], var_down_TRUE.shape[1], 1, var_up.shape[2]))

inputSize = inputData.shape[0:3]
outputSize = outputData.shape[0:3]
ntimes=outputData.shape[3]

X = inputData
Y = outputData

Yr = np.reshape(Y, (outputSize[0]*outputSize[1], ntimes))

X=np.transpose(X,(3,0,1,2))
Yr=np.transpose(Yr,(1,0))

X_train=X[timelag_train,:,:,:]
Yr_train=Yr[timelag_train,:]

X_val=X[timelag_val,:,:,:]
Yr_val=Yr[timelag_val,:]

X_test=X[timelag_test,:,:,:]
Yr_test=Yr[timelag_test,:]

@tf.keras.utils.register_keras_serializable()
class ConstrainLayer(layers.Layer):
    def __init__(self, inputSize, outputSize, **kwargs):
        self.inputSize = inputSize
        self.outputSize = outputSize
        super(ConstrainLayer, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config()
        config["inputSize"] = self.inputSize
        config["outputSize"] = self.outputSize
        return config
    
    def call(self, output, input_layer):
        up_factor=int(outputSize[0]/inputSize[0])
        output_resh=tf.reshape(output, (tf.shape(input_layer)[0], inputSize[0]*up_factor, inputSize[1]*up_factor))     
        y = output_resh
        y = tf.expand_dims(y, axis=-1)
        y = tf.cast(y, tf.float32)
        downscaled_y = tf.nn.avg_pool2d(y, ksize=(up_factor, up_factor), strides=(up_factor, up_factor), padding='SAME')
        factor=(input_layer - downscaled_y)
        upsampled_factor = tf.keras.layers.UpSampling2D(size=(up_factor, up_factor), interpolation='nearest')(factor)
        const_output = y + upsampled_factor
        constrained_output=tf.reshape(const_output, (tf.shape(input_layer)[0],outputSize[0]*outputSize[1]))
        return constrained_output

def create_cnn(inputSize, outputSize):
    input_layer = layers.Input(shape=inputSize)
    conv1 = layers.Conv2D(32, (3, 3), activation='elu', padding='same')(input_layer)
    conv2 = layers.Conv2D(64, (3, 3), activation='elu', padding='same')(conv1)
    flatten = layers.Flatten()(conv2)
    dense1 = layers.Dense(3000, activation='elu')(flatten)
    dense1 = layers.Dropout(0.3)(dense1)
    output = layers.Dense(outputSize[0] * outputSize[1])(dense1)
    output = ConstrainLayer(inputSize, outputSize)(output, input_layer)
    CNN = models.CNN(inputs=input_layer, outputs=output)
    return CNN

CNN = create_cnn(inputSize, outputSize)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

CNN.compile(optimizer=optimizer, loss='mse')

early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=6,          # Stop after 6 epochs with no improvement
    restore_best_weights=True)  # Restore the weights from the epoch with the best validation loss

history = CNN.fit(X_train, Yr_train, epochs=200, batch_size=100, 
                   validation_data=(X_val, Yr_val),
                   callbacks=[early_stopping])

plt.plot(history.history['val_loss'], label = 'loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

# performance in the test phase 
YPredicted_test = CNN.predict(X_test)
YPredicted_test=YPredicted_test
predictionError = Yr_test - YPredicted_test
squares = tf.square(predictionError)
rmse_test = tf.sqrt(tf.reduce_mean(squares))
mse_test = tf.reduce_mean(squares)
print(rmse_test)

######################## CNN application ######################################
# List of CMIP6 GCMs to be processed
mod_names = ['EC-Earth3-Veg', 'NorESM2-MM', 'MRI-ESM2-0', 'MPI-ESM1-2-HR', 'GFDL-ESM4']

for mod_name in mod_names:
    print(f"Processing model: {mod_name}")
    # working directory to the folder containing model data organized by variable (e.g., Models_MED/pr)
    os.chdir(f'Models_MED/{var}')
    # Open the NetCDF file for the selected model and variable ((all variables pre-interpolated to a common 1° grid))
    ds_GCM = xr.open_dataset(f'{mod_name}_{var}_coarse.nc')
    
    var_hist=ds_GCM[f'{var}_hist']
    var_hist_down= CNN.predict(var_hist)
    var_hist_down = np.reshape(var_hist_down, (var_hist_down.shape[0], outputSize[0], outputSize[1]))
    
    X_ssp126=ds_GCM[f'{var}_ssp126']
    var_ssp126_down= CNN.predict(X_ssp126)
    var_ssp126_down = np.reshape(var_ssp126_down, (var_ssp126_down.shape[0], outputSize[0], outputSize[1]))
    
    X_ssp370=ds_GCM[f'{var}_ssp370']
    var_ssp370_down= CNN.predict(X_ssp370)
    var_ssp370_down = np.reshape(var_ssp370_down, (var_ssp370_down.shape[0], outputSize[0], outputSize[1]))
    
    os.chdir('../..')
    # Open the ERA5 NetCDF file used as a spatial grid reference 
    ds_down = xr.open_dataset('ERA5_025_grid.nc')
    ds_GCM_regridded = ds_down.copy()
    ds_GCM_regridded = ds_GCM_regridded.assign_coords(time_hist=ds_GCM['time_hist'])
    ds_GCM_regridded = ds_GCM_regridded.assign_coords(time_ssp=ds_GCM['time_ssp'])
    ds_GCM_regridded = ds_GCM_regridded.assign({f"{var}_hist": (("time_hist", "lat", "lon"), var_hist_down)})
    ds_GCM_regridded = ds_GCM_regridded.assign({f"{var}_ssp126": (("time_ssp", "lat", "lon"), var_ssp126_down)})
    ds_GCM_regridded = ds_GCM_regridded.assign({f"{var}_ssp370": (("time_ssp", "lat", "lon"), var_ssp370_down)})
    ds_GCM_regridded = ds_GCM_regridded.assign(time_hist=ds_GCM['time_hist'])
    ds_GCM_regridded = ds_GCM_regridded.assign(time_ssp=ds_GCM['time_ssp'])

    encoding = {var_name: {"zlib": True, "complevel": 9} for var_name in ds_GCM_regridded.data_vars}
    encoding = {"time_hist": {"zlib": True, "complevel": 9, "units": "days since 1850-01-01"},}
    encoding = {"time_ssp": {"zlib": True, "complevel": 9, "units": "days since 1850-01-01"},}
    
    os.chdir(f'Models_MED_regridded/{var}')
    output_file = f'{mod_name}_{var}_regridded.nc'
    ds_GCM_regridded.to_netcdf(output_file, encoding=encoding, format='NETCDF4', mode='w')
    os.chdir('../..')