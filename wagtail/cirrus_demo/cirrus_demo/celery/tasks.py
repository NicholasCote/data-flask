from celery import shared_task
import pandas as pd
import numpy as np
import requests
from io import StringIO
import logging

@shared_task(bind=True, max_retries=3)
def analyze_taxi_data(self):
    try:
        url = "https://github.com/dotnet/machinelearning/raw/refs/heads/main/test/data/taxi-fare-train.csv"
        
        response = requests.get(url)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text), nrows=1000)
        
        results = {
            'average_fare': round(float(df['fare_amount'].mean()), 2),
            'max_distance': round(float(df['trip_distance'].max()), 2),
            'total_passengers': int(df['passenger_count'].sum())
        }
        
        return results
        
    except requests.RequestException as exc:
        logging.error(f"HTTP error occurred: {exc}")
        self.retry(exc=exc, countdown=2 ** self.request.retries)
        
    except Exception as exc:
        logging.error(f"Error analyzing taxi data: {exc}")
        raise

@shared_task(bind=True, max_retries=3)
def max_taxi_fare(self):
    try:
        url = "https://github.com/dotnet/machinelearning/raw/refs/heads/main/test/data/taxi-fare-train.csv"
        
        response = requests.get(url)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text), nrows=1000)
        
        results = {
            'maximum_fare': round(float(df['fare_amount'].max()), 2)
        }
        
        return results
        
    except requests.RequestException as exc:
        logging.error(f"HTTP error occurred: {exc}")
        self.retry(exc=exc, countdown=2 ** self.request.retries)
        
    except Exception as exc:
        logging.error(f"Error analyzing taxi data: {exc}")
        raise

@shared_task(bind=True, max_retries=3)
def total_taxi_fare(self):
    try:
        url = "https://github.com/dotnet/machinelearning/raw/refs/heads/main/test/data/taxi-fare-train.csv"
        
        response = requests.get(url)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text), nrows=1000)
        
        results = {
            'total_fare': round(float(df['fare_amount'].sum()), 2)
        }
        
        return results
        
    except requests.RequestException as exc:
        logging.error(f"HTTP error occurred: {exc}")
        self.retry(exc=exc, countdown=2 ** self.request.retries)
        
    except Exception as exc:
        logging.error(f"Error analyzing taxi data: {exc}")
        raise

@shared_task(bind=True, max_retries=3)
def taxi_weather_analysis(self):
    """Enhanced version of taxi_weather_analysis with comprehensive debugging"""
    # Setup enhanced logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('taxi_weather_debug')
    logger.setLevel(logging.DEBUG)
    
    # Create a debug log handler to capture detailed information
    import sys
    from io import StringIO
    debug_log = StringIO()
    debug_handler = logging.StreamHandler(debug_log)
    debug_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    debug_handler.setFormatter(formatter)
    logger.addHandler(debug_handler)
    
    # Track execution stages
    execution_stages = {
        'start_time': None,
        'taxi_data_loaded': False,
        'era5_files_found': [],
        'weather_data_extracted': 0,
        'correlations_calculated': False,
        'errors': []
    }
    
    try:
        import numpy as np
        import xarray as xr
        import os
        import requests
        import pandas as pd
        import math
        import json
        from io import StringIO as TextIO
        from datetime import datetime, timedelta
        
        execution_stages['start_time'] = datetime.now().isoformat()
        logger.info(f"Starting enhanced taxi weather analysis at {execution_stages['start_time']}")
        
        # Helper function to convert NumPy types to Python native types
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                # Handle NaN values
                if np.isnan(obj):
                    return None  # Convert NaN to None, which becomes null in JSON
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(i) for i in obj)
            else:
                return obj
        
        # Function to extract and format weather data from ERA5 dataset with enhanced debugging
        def extract_era5_weather_data(ds, date_str, lat_slice, lon_slice):
            """
            Extract and format weather data from ERA5 dataset with additional debugging.
            """
            logger.debug(f"Extracting ERA5 weather data for date: {date_str}")
            logger.debug(f"Dataset dimensions: {list(ds.dims.items())}")
            logger.debug(f"Dataset coordinates: {list(ds.coords)}")
            logger.debug(f"Dataset variables: {list(ds.data_vars)}")
            
            # Get the region-specific data
            try:
                # Identify actual coordinate names in the dataset
                lat_names = [coord for coord in ds.coords if coord.lower() in ('latitude', 'lat')]
                lon_names = [coord for coord in ds.coords if coord.lower() in ('longitude', 'lon')]
                
                logger.debug(f"Found lat coordinate names: {lat_names}")
                logger.debug(f"Found lon coordinate names: {lon_names}")
                
                if lat_names and lon_names:
                    lat_name, lon_name = lat_names[0], lon_names[0]
                    
                    # Create selection dict for the region
                    selection = {
                        lat_name: lat_slice,
                        lon_name: lon_slice
                    }
                    
                    logger.debug(f"Selecting region with: {selection}")
                    
                    # First select the region with slices (without method parameter)
                    regional_ds = ds.sel(**selection)
                    
                    # Verify the regional selection worked
                    logger.debug(f"Regional dataset dimensions: {list(regional_ds.dims.items())}")
                    
                    # Then select the time with method="nearest"
                    logger.debug(f"Time values in dataset: {ds.time.values[:5]}...")
                    try:
                        # Check if time dimension exists
                        if 'time' in ds.coords:
                            logger.debug(f"Using time selection with method='nearest'")
                            regional_weather = regional_ds.sel(time=date_str, method="nearest")
                            logger.debug(f"Selected time: {regional_weather.time.values}")
                        else:
                            logger.warning("No 'time' coordinate found in dataset!")
                            # Try to find alternative time dimension names
                            time_dims = [dim for dim in ds.coords if 'time' in dim.lower()]
                            logger.debug(f"Alternative time dimensions found: {time_dims}")
                            
                            if time_dims:
                                time_dim = time_dims[0]
                                logger.debug(f"Using alternative time dimension: {time_dim}")
                                regional_weather = regional_ds.sel({time_dim: date_str}, method="nearest")
                            else:
                                logger.error("No time dimension found, using entire regional dataset without time selection")
                                regional_weather = regional_ds
                    except Exception as time_err:
                        logger.error(f"Error selecting time: {time_err}")
                        logger.debug(f"Time coordinate data type: {type(ds.time.values[0])}")
                        regional_weather = regional_ds  # Use region without time selection
                else:
                    # Fallback to just time selection if coordinates aren't as expected
                    logger.warning("Coordinate names not found as expected, falling back to time selection only")
                    regional_weather = ds.sel(time=date_str, method="nearest")
            except Exception as e:
                logger.error(f"Error extracting regional data: {e}", exc_info=True)
                # Fallback to just time selection
                try:
                    regional_weather = ds.sel(time=date_str, method="nearest")
                    logger.debug("Fell back to time selection only")
                except Exception as time_err:
                    logger.error(f"Error in time-only selection: {time_err}", exc_info=True)
                    # Last resort - return dataset with no selection
                    regional_weather = ds
                    logger.debug("Using entire dataset without any selection")
            
            # Create a more readable weather dictionary with meaningful names
            weather_data = {}
            
            # Add UTC date for debugging
            try:
                if hasattr(regional_weather, 'time') and hasattr(regional_weather.time, 'values'):
                    time_value = regional_weather.time.values
                    logger.debug(f"Selected time value: {time_value}")
                    # Convert to numeric UTC format YYYYMMDDHH
                    if isinstance(time_value, np.ndarray) and len(time_value) > 0:
                        time_value = time_value[0]
                    
                    if hasattr(time_value, 'strftime'):
                        utc_date_str = time_value.strftime('%Y%m%d%H')
                        weather_data['utc_date'] = float(utc_date_str)
                    elif isinstance(time_value, (str, np.datetime64)):
                        # Convert numpy.datetime64 to datetime
                        dt_obj = pd.to_datetime(time_value)
                        utc_date_str = dt_obj.strftime('%Y%m%d%H')
                        weather_data['utc_date'] = float(utc_date_str)
                    else:
                        logger.warning(f"Unexpected time value type: {type(time_value)}")
                else:
                    # Create from the date_str
                    try:
                        dt_obj = pd.to_datetime(date_str)
                        utc_date_str = dt_obj.strftime('%Y%m%d%H')
                        weather_data['utc_date'] = float(utc_date_str)
                    except Exception as dt_err:
                        logger.warning(f"Error creating UTC date from date_str: {dt_err}")
            except Exception as e:
                logger.error(f"Error adding UTC date: {e}")
            
            # Map of common ERA5 variable codes to more readable names
            variable_mapping = {
                't2m': 'temperature_2m_C',  # 2m temperature
                'tp': 'total_precipitation_mm',  # Total precipitation
                'sp': 'surface_pressure_hPa',  # Surface pressure
                'msl': 'mean_sea_level_pressure_hPa',  # Mean sea level pressure
                'u10': 'wind_u_component_10m_ms',  # 10m U wind component
                'v10': 'wind_v_component_10m_ms',  # 10m V wind component
                'tcc': 'total_cloud_cover_percent',  # Total cloud cover
                'tcw': 'total_column_water_kg_m2',  # Total column water
                'r': 'relative_humidity_percent',  # Relative humidity
                'skt': 'skin_temperature_K',  # Skin temperature
                'd2m': 'dewpoint_2m_C',  # 2m dewpoint temperature
            }
            
            # Process each variable in the dataset
            var_processed = 0
            var_success = 0
            
            logger.debug(f"Processing variables: {list(regional_weather.data_vars)}")
            for var_name in regional_weather.data_vars:
                var_processed += 1
                try:
                    # Get the data
                    var_data = regional_weather[var_name]
                    logger.debug(f"Processing variable: {var_name}, shape: {var_data.shape if hasattr(var_data, 'shape') else 'unknown'}")
                    
                    # Calculate mean value for the region
                    if hasattr(var_data, 'mean'):
                        # Check variable dimensions and values
                        logger.debug(f"Variable {var_name} dimensions: {list(var_data.dims)}")
                        logger.debug(f"Variable {var_name} has NaN values: {np.isnan(var_data.values).any() if hasattr(var_data.values, 'any') else 'unknown'}")
                        
                        # Calculate mean - handle different dimension combinations
                        if len(var_data.dims) == 0:
                            # Scalar value
                            mean_val = float(var_data.values)
                            logger.debug(f"Scalar value for {var_name}: {mean_val}")
                        else:
                            # Reduce all dimensions to get mean
                            mean_val = float(var_data.mean().values)
                            logger.debug(f"Mean value for {var_name}: {mean_val}")
                        
                        # Skip NaN values
                        if not np.isnan(mean_val):
                            # Convert units for specific variables
                            if var_name == 't2m' or var_name == 'd2m':
                                # Convert from Kelvin to Celsius
                                mean_val = mean_val - 273.15
                                logger.debug(f"Converted {var_name} from K to C: {mean_val}")
                            elif var_name == 'sp' or var_name == 'msl':
                                # Convert from Pa to hPa
                                mean_val = mean_val / 100.0
                                logger.debug(f"Converted {var_name} from Pa to hPa: {mean_val}")
                            elif var_name == 'tcc':
                                # Convert from 0-1 to percentage
                                mean_val = mean_val * 100.0
                                logger.debug(f"Converted {var_name} to percentage: {mean_val}")
                            elif var_name == 'tp':
                                # Convert to mm (sometimes needed)
                                mean_val = mean_val * 1000.0
                                logger.debug(f"Converted {var_name} to mm: {mean_val}")
                            
                            # Use readable name if available, otherwise use original
                            readable_name = variable_mapping.get(var_name, var_name)
                            weather_data[readable_name] = round(mean_val, 2)
                            var_success += 1
                            logger.debug(f"Added {readable_name}: {weather_data[readable_name]}")
                        else:
                            logger.warning(f"Mean value for {var_name} is NaN, skipping")
                    else:
                        logger.warning(f"Variable {var_name} does not have a mean method, skipping")
                except Exception as e:
                    logger.error(f"Error processing variable {var_name}: {e}", exc_info=True)
            
            logger.debug(f"Variables processed: {var_processed}, successfully extracted: {var_success}")
            logger.debug(f"Final weather data: {json.dumps(weather_data, default=str)}")
            
            # Calculate derived metrics if possible
            try:
                if 'wind_u_component_10m_ms' in weather_data and 'wind_v_component_10m_ms' in weather_data:
                    u = weather_data['wind_u_component_10m_ms']
                    v = weather_data['wind_v_component_10m_ms']
                    # Calculate wind speed
                    weather_data['wind_speed_ms'] = round(np.sqrt(u**2 + v**2), 2)
                    # Calculate wind direction (meteorological convention)
                    weather_data['wind_direction_degrees'] = round(np.degrees(np.arctan2(-u, -v)), 2) % 360
                    logger.debug(f"Added derived wind metrics")
            except Exception as e:
                logger.error(f"Error calculating wind metrics: {e}")
            
            # Add categorical description of weather if possible
            try:
                if 'temperature_2m_C' in weather_data:
                    temp = weather_data['temperature_2m_C']
                    precip = weather_data.get('total_precipitation_mm', 0)
                    cloud = weather_data.get('total_cloud_cover_percent', 50)
                    
                    # Simple weather condition classification
                    if precip > 5:
                        condition = "Heavy Rain"
                    elif precip > 1:
                        condition = "Moderate Rain"
                    elif precip > 0.2:
                        condition = "Light Rain"
                    elif cloud > 80:
                        condition = "Overcast"
                    elif cloud > 50:
                        condition = "Mostly Cloudy"
                    elif cloud > 20:
                        condition = "Partly Cloudy"
                    else:
                        condition = "Clear"
                    
                    # Add snow classification if temperature is near or below freezing
                    if temp < 2 and precip > 0.1:
                        condition = condition.replace("Rain", "Snow/Sleet")
                    
                    weather_data['weather_condition'] = condition
                    logger.debug(f"Added weather condition: {condition}")
            except Exception as e:
                logger.error(f"Error determining weather condition: {e}")
            
            logger.info(f"Extracted weather data with {len(weather_data)} variables")
            
            return weather_data
        
        # Step 1: Download and process taxi data
        logger.info("Starting taxi data download and processing")
        url = "https://github.com/dotnet/machinelearning/raw/refs/heads/main/test/data/taxi-fare-train.csv"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            logger.debug(f"Successfully downloaded taxi data, size: {len(response.text)} bytes")
            
            df = pd.read_csv(TextIO(response.text))
            logger.debug(f"Parsed CSV data, shape: {df.shape}")
            
            # Since the dataset doesn't have pickup_datetime, we'll create synthetic dates
            start_date = datetime(2019, 1, 1)  # Arbitrary start date
            
            # Create synthetic dates distributed over a year for demonstration
            num_rows = len(df)
            date_range = [start_date + timedelta(hours=i % (24*365)) for i in range(num_rows)]
            df['pickup_datetime'] = date_range
            
            # Extract date components
            df['year'] = df['pickup_datetime'].dt.year
            df['month'] = df['pickup_datetime'].dt.month
            df['day'] = df['pickup_datetime'].dt.day
            df['hour'] = df['pickup_datetime'].dt.hour
            
            logger.info(f"Taxi data processed, {num_rows} rows spanning {df['year'].min()}-{df['month'].min()} to {df['year'].max()}-{df['month'].max()}")
            execution_stages['taxi_data_loaded'] = True
            
        except Exception as e:
            error_msg = f"Error downloading or processing taxi data: {e}"
            logger.error(error_msg, exc_info=True)
            execution_stages['errors'].append(error_msg)
            raise
        
        # Step 2: Process ERA5 data
        era5_base_path = "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.sfc"
        
        # Verify ERA5 path exists
        if not os.path.exists(era5_base_path):
            error_msg = f"ERA5 base path does not exist: {era5_base_path}"
            logger.error(error_msg)
            execution_stages['errors'].append(error_msg)
            # Instead of raising, we'll continue with a warning and create a placeholder
            logger.warning("Creating placeholder for ERA5 data path")
            os.makedirs(era5_base_path, exist_ok=True)
        
        # Function to find the closest ERA5 file with enhanced debugging
        def find_closest_era5_file(year, month):
            logger.debug(f"Looking for ERA5 file for {year}-{month:02d}")
            # Format year and month as directories
            year_month_dir = f"{era5_base_path}/{year:04d}{month:02d}/"
            logger.debug(f"Target directory: {year_month_dir}")
            
            # Check if the path exists
            if not os.path.exists(era5_base_path):
                error_msg = f"ERA5 base path does not exist: {era5_base_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # List contents of base path for debugging
            try:
                base_contents = os.listdir(era5_base_path)
                logger.debug(f"ERA5 base path contents: {base_contents[:10]}... ({len(base_contents)} items)")
            except Exception as e:
                logger.error(f"Error listing ERA5 base path: {e}")
                base_contents = []
            
            # If the exact directory doesn't exist, find the closest one
            if not os.path.exists(year_month_dir):
                logger.debug(f"Exact year-month directory not found: {year_month_dir}")
                # Find available years
                try:
                    available_dirs = [d for d in os.listdir(era5_base_path) if os.path.isdir(os.path.join(era5_base_path, d))]
                    logger.debug(f"Available directories: {available_dirs[:10]}... ({len(available_dirs)} items)")
                    
                    available_years = [int(d) for d in available_dirs if d.isdigit() and len(d) == 6]
                    logger.debug(f"Extracted year-month integers: {available_years[:10]}... ({len(available_years)} items)")
                    
                    if not available_years:
                        error_msg = f"No valid data directories found in {era5_base_path}"
                        logger.error(error_msg)
                        raise FileNotFoundError(error_msg)
                    
                    # Find closest year-month
                    target_ym = year*100 + month
                    closest_ym = min(available_years, key=lambda ym: abs(ym - target_ym))
                    logger.debug(f"Target year-month: {target_ym}, closest available: {closest_ym}")
                    
                    year_month_dir = f"{era5_base_path}/{closest_ym}/"
                    logger.debug(f"Using closest year-month directory: {year_month_dir}")
                except Exception as e:
                    error_msg = f"Error finding closest year-month directory: {e}"
                    logger.error(error_msg, exc_info=True)
                    raise FileNotFoundError(error_msg)
            
            # List files in the year-month directory
            try:
                dir_files = os.listdir(year_month_dir)
                logger.debug(f"Files in {year_month_dir}: {dir_files[:10]}... ({len(dir_files)} files)")
            except Exception as e:
                error_msg = f"Error listing files in {year_month_dir}: {e}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # Find temperature or precipitation file
            for suffix in ["_t2m", "_tp", ""]:
                logger.debug(f"Searching for files with suffix: {suffix}")
                matching_files = [f for f in dir_files if suffix in f and f.endswith(".nc")]
                logger.debug(f"Found {len(matching_files)} matching files")
                
                if matching_files:
                    file_path = os.path.join(year_month_dir, matching_files[0])
                    logger.info(f"Selected ERA5 file: {file_path}")
                    return file_path
            
            error_msg = f"No suitable ERA5 file found in {year_month_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Determine unique year-months in taxi data
        unique_dates = df[['year', 'month']].drop_duplicates().values
        logger.info(f"Found {len(unique_dates)} unique year-month combinations in taxi data")
        
        # Initialize results
        weather_impacts = []
        
        # NYC bounding box for weather data
        nyc_lat_slice = slice(40.5, 41.0)
        nyc_lon_slice = slice(-74.3, -73.7)
        
        # Process each year-month
        for year, month in unique_dates:
            logger.info(f"Processing data for {year}-{month:02d}")
            # Find appropriate ERA5 file
            try:
                era5_file = find_closest_era5_file(year, month)
                logger.info(f"Using ERA5 file: {era5_file}")
                execution_stages['era5_files_found'].append(f"{year}-{month:02d}: {era5_file}")
                
                # Check file size
                try:
                    file_size = os.path.getsize(era5_file) / (1024 * 1024)  # Size in MB
                    logger.debug(f"ERA5 file size: {file_size:.2f} MB")
                except Exception as e:
                    logger.warning(f"Could not determine file size: {e}")
                
                # Open the dataset with enhanced error reporting
                try:
                    logger.debug(f"Opening dataset: {era5_file}")
                    ds = xr.open_dataset(era5_file)
                    
                    # Log dataset details
                    logger.debug(f"Dataset dimensions: {list(ds.dims.items())}")
                    logger.debug(f"Dataset coordinates: {list(ds.coords)}")
                    logger.debug(f"Dataset variables: {list(ds.data_vars)}")
                    logger.info(f"Successfully opened dataset with {len(ds.data_vars)} variables")
                except Exception as e:
                    error_msg = f"Error opening dataset {era5_file}: {e}"
                    logger.error(error_msg, exc_info=True)
                    execution_stages['errors'].append(error_msg)
                    continue
                
                # Get taxi data for this month
                taxi_month = df[(df['year'] == year) & (df['month'] == month)]
                logger.debug(f"Filtered taxi data for {year}-{month:02d}, {len(taxi_month)} records")
                
                # Aggregate data
                daily_stats = taxi_month.groupby(['day', 'hour']).agg({
                    'fare_amount': ['mean', 'count'],
                    'trip_distance': ['mean'],
                    'trip_time_in_secs': ['mean']
                }).reset_index()
                
                logger.debug(f"Aggregated taxi data, {len(daily_stats)} day-hour combinations")
                logger.debug(f"Aggregated columns: {daily_stats.columns.tolist()}")
                
                # Process each day
                days_processed = 0
                hours_processed = 0
                hours_with_weather = 0
                
                for day, day_group in daily_stats.groupby('day'):
                    days_processed += 1
                    logger.debug(f"Processing day {year}-{month:02d}-{day:02d}")
                    try:
                        # Format date for xarray selection
                        date_str = f"{year}-{month:02d}-{day:02d}"
                        
                        # Process each hour
                        for hour, hour_data in day_group.groupby('hour'):
                            hours_processed += 1
                            try:
                                # Format datetime for ERA5 selection
                                hour_date_str = f"{date_str}T{hour:02d}:00:00"
                                logger.debug(f"Processing hour: {hour_date_str}")
                                
                                # Get weather data for this hour
                                weather_data = extract_era5_weather_data(ds, hour_date_str, nyc_lat_slice, nyc_lon_slice)
                                
                                # If we got any weather variables beyond just the date, count it
                                if len(weather_data) > 1:  # More than just utc_date
                                    hours_with_weather += 1
                                    logger.debug(f"Successfully extracted {len(weather_data)} weather variables")
                                else:
                                    logger.warning(f"No weather variables extracted beyond UTC date")
                                
                                execution_stages['weather_data_extracted'] += 1
                                
                                # Extract taxi metrics
                                try:
                                    # Convert tuple column names to strings for compatibility
                                    columns_dict = {}
                                    for col in hour_data.columns:
                                        if isinstance(col, tuple) and len(col) == 2:
                                            if col[0] == 'fare_amount' and col[1] == 'count':
                                                columns_dict['ride_count'] = hour_data[col].values[0]
                                            elif col[0] == 'fare_amount' and col[1] == 'mean':
                                                columns_dict['avg_fare'] = hour_data[col].values[0]
                                            elif col[0] == 'trip_distance' and col[1] == 'mean':
                                                columns_dict['avg_distance'] = hour_data[col].values[0]
                                            elif col[0] == 'trip_time_in_secs' and col[1] == 'mean':
                                                columns_dict['avg_trip_time'] = hour_data[col].values[0]
                                    
                                    taxi_metrics = {
                                        'ride_count': columns_dict.get('ride_count', 0),
                                        'avg_fare': columns_dict.get('avg_fare', 0),
                                        'avg_distance': columns_dict.get('avg_distance', 0),
                                        'avg_trip_time': columns_dict.get('avg_trip_time', 0)
                                    }
                                    logger.debug(f"Extracted taxi metrics: {json.dumps(taxi_metrics, default=str)}")
                                except Exception as e:
                                    logger.warning(f"Error extracting taxi metrics: {e}")
                                    # Fallback with empty metrics
                                    taxi_metrics = {
                                        'ride_count': 0,
                                        'avg_fare': 0,
                                        'avg_distance': 0,
                                        'avg_trip_time': 0
                                    }
                                
                                # Create readable date string
                                date_time_str = f"{year}-{month:02d}-{day:02d} {hour:02d}:00"
                                
                                # Combine with weather data
                                result = {
                                    'year': int(year),
                                    'month': int(month),
                                    'day': int(day),
                                    'hour': int(hour),
                                    'taxi': taxi_metrics,
                                    'weather': weather_data,
                                    'date_str': date_time_str
                                }
                                
                                weather_impacts.append(result)
                                
                            except Exception as e:
                                error_msg = f"Error processing hour {hour}: {e}"
                                logger.warning(error_msg)
                                execution_stages['errors'].append(error_msg)
                                continue
                        
                    except Exception as e:
                        error_msg = f"Error processing day {day}: {e}"
                        logger.warning(error_msg)
                        execution_stages['errors'].append(error_msg)
                        continue
                
                logger.info(f"Processed {days_processed} days, {hours_processed} hours, extracted weather data for {hours_with_weather} hours")
                
                # Close the dataset to free memory
                ds.close()
                
            except Exception as e:
                error_msg = f"Error processing ERA5 file for {year}-{month}: {e}"
                logger.warning(error_msg, exc_info=True)
                execution_stages['errors'].append(error_msg)
                continue
        
        # Calculate correlations
        correlations = {}
        
        if weather_impacts:
            # Convert results to dataframe for analysis
            try:
                impact_records = []
                for r in weather_impacts:
                    record = {
                        'year': r['year'],
                        'month': r['month'],
                        'day': r['day'],
                        'hour': r['hour'],
                        'ride_count': r['taxi']['ride_count'],
                        'avg_fare': r['taxi']['avg_fare'],
                        'avg_distance': r['taxi']['avg_distance'],
                        'avg_trip_time': r['taxi']['avg_trip_time']
                    }
                    # Add weather variables
                    for k, v in r['weather'].items():
                        record[f"weather_{k}"] = v
                    
                    impact_records.append(record)
                
                impact_df = pd.DataFrame(impact_records)
                logger.debug(f"Created impact dataframe with shape: {impact_df.shape}")
                logger.debug(f"Impact dataframe columns: {impact_df.columns.tolist()}")
                
                # Calculate correlations
                taxi_columns = ['ride_count', 'avg_fare', 'avg_distance', 'avg_trip_time']
                weather_columns = [col for col in impact_df.columns if col.startswith('weather_')]
                
                logger.debug(f"Found {len(taxi_columns)} taxi metrics and {len(weather_columns)} weather variables")

                for taxi_col in taxi_columns:
                    for weather_col in weather_columns:
                        try:
                            # Drop NA values before calculating correlation
                            valid_data = impact_df[[taxi_col, weather_col]].dropna()
                            if len(valid_data) > 0:
                                corr = valid_data.corr().iloc[0, 1]
                                # Check if the correlation is valid (not NaN)
                                if not np.isnan(corr):
                                    correlations[f"{taxi_col}_vs_{weather_col}"] = float(corr)
                        except Exception as e:
                            logging.warning(f"Error calculating correlation for {taxi_col} vs {weather_col}: {e}")
                            continue
                
                # Get top correlations
                top_correlations = []
                if correlations:
                    # Sort correlations by absolute value
                    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                    # Take top 10
                    top_correlations = sorted_correlations[:10]
                
                # Calculate weather statistics
                weather_stats = {}
                all_weather_vars = set()
                for impact in weather_impacts:
                    all_weather_vars.update(impact.get('weather', {}).keys())
                
                # Calculate min, max, mean for each weather variable
                for var in all_weather_vars:
                    values = [impact['weather'].get(var) for impact in weather_impacts 
                             if var in impact.get('weather', {}) and impact['weather'].get(var) is not None]
                    if values:
                        weather_stats[var] = {
                            'min': min(values),
                            'max': max(values),
                            'mean': sum(values) / len(values),
                            'count': len(values)
                        }
                
                # Calculate taxi metrics statistics
                taxi_stats = {}
                for metric in taxi_columns:
                    values = [impact['taxi'].get(metric) for impact in weather_impacts 
                             if metric in impact.get('taxi', {}) and impact['taxi'].get(metric) is not None]
                    if values:
                        taxi_stats[metric] = {
                            'min': min(values),
                            'max': max(values),
                            'mean': sum(values) / len(values)
                        }
                
                # Count occurrences of different weather conditions
                weather_condition_counts = {}
                for impact in weather_impacts:
                    condition = impact.get('weather', {}).get('weather_condition')
                    if condition:
                        weather_condition_counts[condition] = weather_condition_counts.get(condition, 0) + 1
                
            except Exception as e:
                logging.error(f"Error in correlation analysis: {e}")
        
        # Apply the conversion function to ensure all NumPy types are converted
        weather_impacts = convert_numpy_types(weather_impacts)
        correlations = convert_numpy_types(correlations)
        
        # Return more informative results
        result = {
            'weather_impact_samples': weather_impacts[:15],  # First 15 samples
            'correlations': correlations,
            'top_correlations': dict(top_correlations) if 'top_correlations' in locals() else {},
            'total_days_analyzed': len(set([(impact['year'], impact['month'], impact['day']) 
                                          for impact in weather_impacts])),
            'total_hours_analyzed': len(weather_impacts),
            'weather_variables_found': list(all_weather_vars) if 'all_weather_vars' in locals() else [],
            'weather_statistics': weather_stats if 'weather_stats' in locals() else {},
            'taxi_statistics': taxi_stats if 'taxi_stats' in locals() else {},
            'analysis_summary': {
                'strongest_weather_effect': dict([top_correlations[0]]) if 'top_correlations' in locals() and top_correlations else None,
                'weather_condition_counts': weather_condition_counts if 'weather_condition_counts' in locals() else {}
            }
        }
        
        result = convert_numpy_types(result)

        return result
    
    except Exception as exc:
        logging.error(f"Error in taxi-weather analysis: {exc}")
        raise self.retry(exc=exc, countdown=60*5)  # Retry after 5 minutes