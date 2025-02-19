import xarray as xr
import numpy as np
from matplotlib import pyplot as plt

import os

def get_glade_picture():
    def get_dataset(filepath_pattern, use_grib, parallel):
        """ Given a file pattern specification and a file format type, return an xarray dataset 
            containing data from all matching files.   
            
            `filepath_pattern` must specify a full directory path.  Wildcards may be included to match
            many subdirectories or groups of files.  Wildcard syntax follows bash shell conventions for
            greater flexibility.
            
            If `parallel = True`, use an existing Dask cluster to open data files.
        """
        # If reading GRIB data, disable saving a GRIB index file to the data directory.
        if use_grib:
            filename_extension = '.grb'
            backend_kwargs = {'indexpath': ''}
        else:
            filename_extension = '.nc'    
            backend_kwargs = None
            
        full_pattern = filepath_pattern + filename_extension
        
        # Allow bash-style syntax for file patterns
        file_listing = os.popen(f"/bin/bash -c 'ls {full_pattern}'").read()
        
        # Split the output into lines and ignore empty lines
        file_list = file_listing.split('\n')
        file_list = [filename for filename in file_list if len(filename) > 0]

        # Verify there is at least one matching file
        if len(file_list) == 0:
            raise ValueError(f'No files match the pattern {full_pattern}')
            
        ds = xr.open_mfdataset(file_list, parallel=parallel, backend_kwargs=backend_kwargs) 
        return ds

    def get_point_array(dataset, latitude, longitude, varname=None):
        """ Extract and return a DataArray object associated with some lat/lon location.
        A DataArray object is an array of time series values, along with associated metadata.
        
        'dataset' is an xarray dataset
        
        'latitude' is a scalar in the range of the dataset's latitude values.
            
        'longitude' is a scalar in the range of the dataset's longitude values.

        If 'varname' is provided, retrieve values from this data variable.  Otherwise, 
            return values from the first data variable in the dataset.
        """
        # Assert latitude value is in the dataset range
        assert(latitude >= np.min(dataset.coords['latitude'].values))
        assert(latitude <= np.max(dataset.coords['latitude'].values))
        
        # Assert longitude value is in the dataset range
        assert(longitude >= np.min(dataset.coords['longitude'].values))
        assert(longitude <= np.max(dataset.coords['longitude'].values))
        
        # If a data variable name is not provided, use the first data variable.
        if not varname:
            data_vars = list(dataset.data_vars.keys())
            varname = data_vars[0]
            
        point_array = dataset[varname].sel(latitude=latitude, longitude=longitude, method='nearest')
        return point_array

    def wind_speed(u, v, units=None):
        """Compute the wind speed from u and v-component numpy arrays.
        If units is 'mph', convert from "meters per second" to "miles per hour".
        """
        speed = np.hypot(u, v)
        if units == 'mph':
            speed = 2.369 * speed
        return speed

    def plot_winds(u_values, v_values, time_values):
        """ Compute wind speed values and plot them on a line plot.
        """
        winds = wind_speed(u_values, v_values, units='mph')
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(time_values, winds, color='r')
        ax.set_title('Hourly Average Wind Speeds for Cheyenne, Wyoming')
        ax.set_ylabel('Miles Per Hour')
        return fig

    MAX_WORKERS = 4

    def get_local_cluster():
        """ Create cluster using the Jupyter server's resources
        """
        from distributed import LocalCluster
        cluster = LocalCluster()    

        cluster.scale(MAX_WORKERS)
        return cluster

    # Obtain dask cluster in one of three ways

    cluster = get_local_cluster()

    # Connect to cluster
    from distributed import Client
    client = Client(cluster)

    # Pause notebook execution until some workers have been allocated.
    min_workers = 2
    client.wait_for_workers(min_workers)

    # This subdirectory contains surface analysis data on a 0.25 degree global grid
    data_dir = '/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.sfc/'

    # This bash-style pattern will match data for 2021 and 2022.
    year_month_pattern = '202{1,2}*/'

    data_spec = data_dir + year_month_pattern

    # These filename patterns refer to u- and v-components of winds at 10 meters above the land surface.
    filename_pattern_u = 'e5.oper.an.sfc.228_131_u10n.ll025sc.*'
    filename_pattern_v = 'e5.oper.an.sfc.228_132_v10n.ll025sc.*'  

    ds_u = get_dataset(data_spec + filename_pattern_u, False, parallel=True)
    ds_v = get_dataset(data_spec + filename_pattern_v, False, parallel=True)

    var_u = 'U10N'
    var_v = 'V10N'

    # Select data for a specific geographic location (Cheyenne, Wyoming).
    # Note that dataset longitude values are in the range [0, 360]; click the disk icon to the right of 
    #   "longitude" above to verify.
    # We convert from longitude values provided by Google in the range [-180, 180] using subtraction.

    cheyenne = {'lat': 41.14, 'lon': 360 - 104.82}

    city = cheyenne

    # Select the nearest grid cell to our lat/lon location.
    u = get_point_array(ds_u, city['lat'], city['lon'], var_u)
    v = get_point_array(ds_v, city['lat'], city['lon'], var_v)

    # Actually load the data into memory.
    u_values = u.values
    v_values = v.values

    figure = plot_winds(u_values, v_values, ds_u.time)

    cur_dir = os.getcwd()
    plotfile = cur_dir + '/app/static/glade_data_access.png'
    figure.savefig(plotfile, dpi=100)

    cluster.close()

    return '/static/glade_data_access.png'