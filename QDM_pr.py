"""
This script performs QDM bias correction of CMIP6 precipitation outputs.
It represents the second step of the statistical downscaling workflow,
generating the CMIP6-MedPlus dataset available at https://zenodo.org/records/17898529

Authors: Valeria Todaro and Daniele Secci
"""

import numpy as np
import xarray as xr
import pandas as pd
import dask.array as da
import cftime
import os

var='pr'

mod_names = ['EC-Earth3-Veg', 'NorESM2-MM', 'MRI-ESM2-0', 'MPI-ESM1-2-HR', 'GFDL-ESM4']
# Select a single model for the current processing step
mod_name='EC-Earth3-Veg'

# Minimum precipitation threshold to filter non-physical or negligible values
thr_pr = 0.1

# Upper bound for the scaling factor applied during bias correction
MAX_SCALING_FACTOR = 1.6

# Length of the moving time window (in years) used for quantile delta mapping correction
time_window_years = 30 

#Observational ERA5 dataset
file_path_obs = f'ERA5_MED_{var}.nc'
ds_obs = xr.open_dataset(file_path_obs)

# CNN-based regridded GCM dataset for the selected model
os.chdir(f'Models_MED_regridded/{var}')
file_path_mod = f"{mod_name}_{var}_regridded.nc"
ds_mod = xr.open_dataset(file_path_mod)
os.chdir('../..')

time_slice_his = slice('1985-01-01', '2014-12-31')
time_slice_modp = slice('2015-01-01', '2100-12-31')

ds_mod = ds_mod.sel(time_hist=time_slice_his)

def convert_no_leap_to_gregorian(ds, var, time_dim):   
    time_var = ds[time_dim]
    if isinstance(time_var.values[0], cftime.DatetimeNoLeap):
        time_gregorian = pd.to_datetime([t.isoformat() for t in time_var.values])
    else:
        time_gregorian = pd.to_datetime(time_var.values)
    time_full = pd.date_range(start=time_gregorian[0], end=time_gregorian[-1], freq='D')
    ds = ds.assign_coords({time_dim: time_gregorian})  
    ds_new = ds.reindex({time_dim: time_full}, method=None, fill_value=np.nan)
    
    feb29_indices = time_full[(time_full.month == 2) & (time_full.day == 29)]
    for date in feb29_indices:
        prev_day = date - pd.Timedelta(days=1)
        next_day = date + pd.Timedelta(days=1)
        mean_value = (ds[var].sel({time_dim: prev_day}) + ds[var].sel({time_dim: next_day}))/2
        ds_new[var].loc[{time_dim: date}]=mean_value
    return ds_new

def replace_below_threshold(data, thr_pr):
    random_values = xr.DataArray(
        np.random.uniform(0+1e-5, thr_pr-1e-10, size=data.shape),
        dims=data.dims,
        coords=data.coords)
    data = xr.where(data < thr_pr, random_values, data)  
    return data

def quantile_delta_mapping_one_to_one(obs, modh, modp):
    obs = obs.chunk({obs.dims[0]: -1})  # Chunk along the first axis (time or another dimension)
    modh = modh.chunk({modh.dims[0]: -1})
    modp = modp.chunk({modp.dims[0]: -1})
    
    sorted_obs = xr.DataArray(da.map_blocks(np.sort, obs.data, axis=0), dims=obs.dims, coords=obs.coords)
    sorted_modh = xr.DataArray(da.map_blocks(np.sort, modh.data, axis=0), dims=modh.dims, coords=modh.coords)
    sorted_modp = xr.DataArray(da.map_blocks(np.sort, modp.data, axis=0), dims=modp.dims, coords=modp.coords)
 
    delta = (sorted_modp.values / sorted_modh.values).clip(max=MAX_SCALING_FACTOR)
    delta_xr = xr.DataArray(delta, dims=['time','lat','lon'])
    delta_rolling = delta_xr.rolling(time=60, center=True, min_periods=1).mean()
    corrected = sorted_obs * delta_rolling

    sorted_indices = da.map_blocks(
        lambda x: np.argsort(np.argsort(x, axis=0), axis=0),
        modp.data,
        dtype=int,
        chunks=modp.data.chunks,)
    corrected_unsorted = xr.DataArray(
        da.map_blocks(
            lambda data, idx: np.take_along_axis(data, idx, axis=0),
            corrected.data,
            sorted_indices,
            dtype=corrected.data.dtype,
            chunks=corrected.data.chunks,),
        dims=modp.dims,
        coords=modp.coords,)
    
    corrected_unsorted = corrected_unsorted.where(corrected_unsorted >= thr_pr, 0)
    return corrected_unsorted

#check and correct no leap years
tim=ds_mod['time_hist']
n_leap = len(tim[(tim.dt.month == 2) & (tim.dt.day == 29)])
if n_leap==0:
    ds_mod = convert_no_leap_to_gregorian(ds_mod, f'{var}_hist', 'time_hist')
    ds_mod = convert_no_leap_to_gregorian(ds_mod, f'{var}_ssp126', 'time_ssp')
    ds_mod = convert_no_leap_to_gregorian(ds_mod, f'{var}_ssp370', 'time_ssp')

obs_30y = ds_obs[f'{var}_ERA5'].sel(time=time_slice_his)
modh_30y = ds_mod[f'{var}_hist'].sel(time_hist=time_slice_his)

modp_ssp126 = ds_mod[f'{var}_ssp126'].sel(time_ssp=time_slice_modp)
modp_ssp370 = ds_mod[f'{var}_ssp370'].sel(time_ssp=time_slice_modp)

obs_30y=replace_below_threshold(obs_30y, thr_pr)
modh_30y=replace_below_threshold(modh_30y, thr_pr)
modp_ssp126=replace_below_threshold(modp_ssp126, thr_pr)
modp_ssp370=replace_below_threshold(modp_ssp370, thr_pr)

start_year = pd.to_datetime(time_slice_modp.start).year
end_year = pd.to_datetime(time_slice_modp.stop).year
half_window_years=time_window_years//2

ds_GCM_downscaled=ds_mod.copy(deep=True)

#correct historical period  
print('historical')
for month in range(1, 13):
    if month == 1:
        month_indices = [12, 1, 2]  # December, January, February
    elif month == 12:
        month_indices = [11, 12, 1]  # November, December, January
    else:
        month_indices = [month - 1, month, month + 1]
    obs_30y_90d=obs_30y.sel(time=obs_30y.time.dt.month.isin(month_indices))
    modh_30y_90d = modh_30y.sel(time_hist=modh_30y.time_hist.dt.month.isin(month_indices))
    modp_ssp126_30y_90d = modh_30y_90d
    
    corrected = quantile_delta_mapping_one_to_one(obs_30y_90d, modh_30y_90d, modp_ssp126_30y_90d)

    time_win=modh_30y.time_hist[(modh_30y.time_hist.dt.month == month)]
    corrected_y_m=corrected.sel(time_hist=time_win)
        
    ds_GCM_downscaled[f'{var}_hist'].loc[{'time_hist': time_win}] = corrected_y_m

#correct projection
for year in range(start_year+half_window_years, end_year-half_window_years + 2):     
    print(year)    
    modp_ssp126_30y = modp_ssp126.sel(time_ssp=slice(f"{year - half_window_years}-01-01", f"{year + half_window_years - 1}-12-31"))
    modp_ssp370_30y = modp_ssp370.sel(time_ssp=slice(f"{year - half_window_years}-01-01", f"{year + half_window_years - 1}-12-31"))
    for month in range(1, 13):
        if month == 1:
            month_indices = [12, 1, 2]  # December, January, February
        elif month == 12:
            month_indices = [11, 12, 1]  # November, December, January
        else:
            month_indices = [month - 1, month, month + 1]
        obs_30y_90d=obs_30y.sel(time=obs_30y.time.dt.month.isin(month_indices))
        modh_30y_90d = modh_30y.sel(time_hist=modh_30y.time_hist.dt.month.isin(month_indices))
        modp_ssp126_30y_90d = modp_ssp126_30y.sel(time_ssp=modp_ssp126_30y.time_ssp.dt.month.isin(month_indices))
        modp_ssp370_30y_90d = modp_ssp370_30y.sel(time_ssp=modp_ssp370_30y.time_ssp.dt.month.isin(month_indices))
        
        if obs_30y_90d.shape[0] > modp_ssp126_30y_90d.shape[0]:
            obs_30y_90d = obs_30y_90d.isel(time=slice(None, -1))
            modh_30y_90d =modh_30y_90d.isel(time_hist=slice(None, -1))
        elif obs_30y_90d.shape[0] < modp_ssp126_30y_90d.shape[0]:
             next_day_obs = min(obs_30y_90d.time[-1] + np.timedelta64(1, 'D'), obs_30y.time[-1])
             next_day_mod = min(modh_30y_90d.time_hist[-1] + np.timedelta64(1, 'D'), modh_30y.time_hist[-1])
             obs_30y_90d = xr.concat([obs_30y_90d, obs_30y.sel(time=next_day_obs)], dim="time")
             modh_30y_90d = xr.concat([modh_30y_90d, modh_30y.sel(time_hist=next_day_mod)], dim="time_hist") 
        
        corrected_126 = quantile_delta_mapping_one_to_one(obs_30y_90d, modh_30y_90d, modp_ssp126_30y_90d)
        corrected_370 = quantile_delta_mapping_one_to_one(obs_30y_90d, modh_30y_90d, modp_ssp370_30y_90d)
        
        if year in (start_year+half_window_years,end_year-half_window_years+1):
            time_win=modp_ssp126_30y.time_ssp[(modp_ssp126_30y.time_ssp.dt.month == month)]
        else:    
            time_win=modp_ssp126_30y.time_ssp[(modp_ssp126_30y.time_ssp.dt.month == month) & (modp_ssp126_30y.time_ssp.dt.year == year)]
        
        corrected_126_y_m=corrected_126.sel(time_ssp=time_win)
        corrected_370_y_m=corrected_370.sel(time_ssp=time_win)
            
        ds_GCM_downscaled[f'{var}_ssp126'].loc[{'time_ssp': time_win}] = corrected_126_y_m
        ds_GCM_downscaled[f'{var}_ssp370'].loc[{'time_ssp': time_win}] = corrected_370_y_m

ds_GCM_downscaled['time_hist'] = ds_GCM_downscaled['time_hist'].dt.floor('D')
ds_GCM_downscaled['time_ssp'] = ds_GCM_downscaled['time_ssp'].dt.floor('D')

ds_GCM_downscaled['time_hist'].encoding['units'] = 'days since 1850-01-01'
ds_GCM_downscaled['time_hist'].encoding['calendar'] = 'proleptic_gregorian'
ds_GCM_downscaled['time_ssp'].encoding['units'] = 'days since 1850-01-01'
ds_GCM_downscaled['time_ssp'].encoding['calendar'] = 'proleptic_gregorian'

encoding = {var_name: {"zlib": True, "complevel": 9} for var_name in ds_GCM_downscaled.data_vars}

os.chdir(f'Models_MED_downscaled/{var}')
output_file = f'{mod_name}_{var}.nc'
ds_GCM_downscaled.to_netcdf(output_file, encoding=encoding, format='NETCDF4', mode='w')
os.chdir('../..')
