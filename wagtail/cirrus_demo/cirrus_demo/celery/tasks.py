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
    try:
        import numpy as np
        import xarray as xr
        import os
        import requests
        import pandas as pd
        import logging
        from io import StringIO
        from datetime import datetime, timedelta
        
        # Set up verbose logging
        logging.basicConfig(level=logging.INFO)
        
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
        
        # Function to safely extract a value from a pandas Series or scalar
        def safe_extract(value):
            if isinstance(value, pd.Series):
                if len(value) > 0:
                    return value.iloc[0]  # Get the first value safely
                return None
            return value
        
        # Function to find relevant ERA5 NetCDF files
        def find_era5_files(year, month, base_path="/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.sfc"):
            """Find relevant ERA5 NetCDF files for weather analysis."""
            # Ensure year and month are integers
            year = int(safe_extract(year))
            month = int(safe_extract(month))
            
            logging.info(f"Looking for ERA5 files for {year}-{month:02d}")
            
            # Format the directory path for the year and month
            year_month_dir = f"{base_path}/{year:04d}{month:02d}"
            
            # If directory doesn't exist, find the closest available directory
            if not os.path.exists(year_month_dir):
                logging.warning(f"Directory {year_month_dir} not found, looking for alternatives")
                
                try:
                    # List all available directories
                    all_dirs = os.listdir(base_path)
                    logging.info(f"Found {len(all_dirs)} potential directories in {base_path}")
                    
                    # Filter for valid year-month directories (6-digit numeric names)
                    available_dirs = [d for d in all_dirs 
                                     if os.path.isdir(os.path.join(base_path, d)) and 
                                     d.isdigit() and len(d) == 6]
                    
                    logging.info(f"Found {len(available_dirs)} valid year-month directories: {available_dirs[:5]}...")
                    
                    if not available_dirs:
                        logging.error(f"No valid year-month directories found in {base_path}")
                        return {}
                    
                    # Convert to integers for numeric comparison
                    available_ym = [int(d) for d in available_dirs]
                    
                    # Find the closest year-month
                    target_ym = year * 100 + month
                    closest_ym = min(available_ym, key=lambda ym: abs(ym - target_ym))
                    
                    # Update directory path
                    year_month_dir = f"{base_path}/{closest_ym}"
                    logging.info(f"Using alternative directory: {year_month_dir}")
                    
                except Exception as e:
                    logging.error(f"Error finding alternative directory: {str(e)}")
                    return {}
            
            # Define key variables we're interested in with their file patterns
            target_variables = {
                't2m': '_2t.ll',      # 2m temperature (167)
                'd2m': '_2d.ll',      # 2m dewpoint (168)
                'sp': '_sp.ll',       # surface pressure (134)
                'msl': '_msl.ll',     # mean sea level pressure (151)
                'u10': '_10u.ll',     # 10m U wind component (165)
                'v10': '_10v.ll',     # 10m V wind component (166)
                'tcc': '_tcc.ll',     # total cloud cover (164)
                'tcw': '_tcw.ll',     # total column water (136)
                'skt': '_skt.ll'      # skin temperature (235)
            }
            
            # Find the corresponding files
            found_files = {}
            
            try:
                # List all files in the directory
                files = os.listdir(year_month_dir)
                logging.info(f"Found {len(files)} files in {year_month_dir}")
                
                # Filter for NetCDF files
                nc_files = [f for f in files if f.endswith('.nc')]
                logging.info(f"Found {len(nc_files)} NetCDF files")
                
                # Find files for each variable
                for var, pattern in target_variables.items():
                    matching_files = [f for f in nc_files if pattern in f]
                    
                    if matching_files:
                        # Sort by file size (larger files might have more complete data)
                        matching_files.sort(key=lambda f: os.path.getsize(os.path.join(year_month_dir, f)), reverse=True)
                        found_files[var] = os.path.join(year_month_dir, matching_files[0])
                        logging.info(f"Found file for {var}: {matching_files[0]}")
                    else:
                        logging.warning(f"No file found for variable {var} with pattern {pattern}")
            
            except Exception as e:
                logging.error(f"Error scanning directory {year_month_dir}: {str(e)}")
                return {}
            
            logging.info(f"Found {len(found_files)} variable files out of {len(target_variables)} variables")
            return found_files
        
        # Update the extract_era5_weather_data function with these changes:

        def extract_era5_weather_data(datasets, date_str, nyc_lat, nyc_lon):
            """
            Extract and format weather data from ERA5 datasets for a specific date and NYC location.
            
            Args:
                datasets (dict): Dictionary of variable names to xarray datasets
                date_str (str): Date string in format YYYY-MM-DDThh:mm:ss
                nyc_lat (float): NYC latitude (around 40.7)
                nyc_lon (float): NYC longitude (adjusted to 0-360 format, around 255.7)
                
            Returns:
                dict: Weather data for the specified date and location
            """
            weather_data = {}
            
            if not datasets:
                logging.warning("No datasets provided to extract_era5_weather_data")
                return weather_data
            
            # Variable mapping between internal names and the actual variables in the ERA5 files
            var_mapping = {
                't2m': ['VAR_2T', '2T', 't2m', 'T2M'],
                'd2m': ['VAR_2D', '2D', 'd2m', 'D2M'],
                'sp': ['SP', 'sp'],
                'msl': ['MSL', 'msl'],
                'u10': ['VAR_10U', '10U', 'u10', 'U10', 'u10n', 'U10N'],
                'v10': ['VAR_10V', '10V', 'v10', 'V10', 'v10n', 'V10N'],
                'tcc': ['TCC', 'tcc'],
                'tcw': ['TCW', 'tcw'],
                'skt': ['SKT', 'skt']
            }
            
            # Process each variable dataset
            for var_name, ds in datasets.items():
                try:
                    logging.info(f"Processing variable {var_name}")
                    
                    # Identify coordinate names (they can vary between datasets)
                    lat_names = [coord for coord in ds.coords if coord.lower() in ('latitude', 'lat')]
                    lon_names = [coord for coord in ds.coords if coord.lower() in ('longitude', 'lon')]
                    
                    if not lat_names or not lon_names:
                        logging.warning(f"Could not identify lat/lon coordinates for {var_name}")
                        continue
                        
                    lat_name, lon_name = lat_names[0], lon_names[0]
                    
                    # Log coordinate ranges
                    lat_min = float(ds[lat_name].min())
                    lat_max = float(ds[lat_name].max())
                    lon_min = float(ds[lon_name].min())
                    lon_max = float(ds[lon_name].max())
                    logging.info(f"Coordinate ranges - lat: [{lat_min}, {lat_max}], lon: [{lon_min}, {lon_max}]")
                    
                    # Verify our point is in range
                    if not (lat_min <= nyc_lat <= lat_max):
                        logging.warning(f"NYC latitude {nyc_lat} is outside dataset range [{lat_min}, {lat_max}]")
                        continue
                        
                    if not (lon_min <= nyc_lon <= lon_max):
                        logging.warning(f"NYC longitude {nyc_lon} is outside dataset range [{lon_min}, {lon_max}]")
                        continue
                    
                    # Get the data variable using our mapping
                    data_var = None
                    possible_var_names = var_mapping.get(var_name, [var_name])
                    
                    # Get a list of all data variables in the dataset
                    all_vars = list(ds.data_vars.keys())
                    logging.info(f"Available variables: {all_vars}")
                    
                    # Try all possible variable names
                    for possible_name in possible_var_names:
                        if possible_name in ds.data_vars:
                            data_var_name = possible_name
                            logging.info(f"Found variable {possible_name} for {var_name}")
                            break
                    
                    # If no match found, try case-insensitive match
                    if not data_var:
                        for actual_var in all_vars:
                            if any(possible_name.lower() == actual_var.lower() for possible_name in possible_var_names):
                                data_var_name = actual_var
                                logging.info(f"Found case-insensitive match: {actual_var} for {var_name}")
                                break
                    
                    # If still no match, use the first variable that's not utc_date
                    if not data_var and all_vars:
                        var_candidates = [v for v in all_vars if v.lower() != 'utc_date']
                        if var_candidates:
                            data_var_name = var_candidates[0]
                            logging.info(f"Using first available variable {data_var_name} for {var_name}")
                        else:
                            logging.warning(f"No suitable data variable found for {var_name}")
                            continue
                    
                    # Select the time point
                    try:
                        logging.info(f"Selecting time: {date_str}")
                        time_selected = ds.sel(time=date_str, method="nearest")
                        logging.info(f"Time selection successful, shape: {time_selected.dims}")
                    except Exception as e:
                        logging.error(f"Time selection failed: {str(e)}")
                        continue
                    
                    # Select the nearest point to NYC
                    try:
                        # Use "nearest" method to find the closest grid point
                        point_data = time_selected[data_var_name].sel(
                            {lat_name: nyc_lat, lon_name: nyc_lon}, 
                            method="nearest"
                        )
                        
                        logging.info(f"Point selection successful, result: {point_data.values}")
                        
                        # Extract the value (should be a single number)
                        mean_val = float(point_data.values)
                        
                        # Skip NaN values
                        if np.isnan(mean_val):
                            logging.warning(f"Value for {var_name} is NaN, skipping")
                            continue
                        
                        # Process values based on variable type
                        if var_name == 't2m' or var_name == 'd2m' or var_name == 'skt':
                            # Convert from Kelvin to Celsius
                            mean_val = mean_val - 273.15
                            logging.info(f"Converted temperature from K to C: {mean_val}")
                        elif var_name == 'sp' or var_name == 'msl':
                            # Convert from Pa to hPa
                            mean_val = mean_val / 100.0
                            logging.info(f"Converted pressure from Pa to hPa: {mean_val}")
                        elif var_name == 'tcc':
                            # Convert from 0-1 to percentage
                            if mean_val <= 1.0:  # Only scale if in 0-1 range
                                mean_val = mean_val * 100.0
                                logging.info(f"Converted cloud cover from 0-1 to percent: {mean_val}")
                        
                        # Use readable names
                        variable_mapping = {
                            't2m': 'temperature_2m_C',
                            'tp': 'total_precipitation_mm',
                            'sp': 'surface_pressure_hPa',
                            'msl': 'mean_sea_level_pressure_hPa',
                            'u10': 'wind_u_component_10m_ms',
                            'v10': 'wind_v_component_10m_ms',
                            'tcc': 'total_cloud_cover_percent',
                            'tcw': 'total_column_water_kg_m2',
                            'd2m': 'dewpoint_2m_C',
                            'skt': 'skin_temperature_C'
                        }
                        
                        readable_name = variable_mapping.get(var_name, var_name)
                        weather_data[readable_name] = round(mean_val, 2)
                        logging.info(f"Added {readable_name}: {round(mean_val, 2)}")
                    except Exception as e:
                        logging.error(f"Error extracting point data for {var_name}: {str(e)}")
                        import traceback
                        logging.error(traceback.format_exc())
                
                except Exception as e:
                    logging.error(f"Error processing variable {var_name}: {str(e)}")
                    import traceback
                    logging.error(traceback.format_exc())
            
            # Calculate derived metrics if possible
            try:
                if 'wind_u_component_10m_ms' in weather_data and 'wind_v_component_10m_ms' in weather_data:
                    u = weather_data['wind_u_component_10m_ms']
                    v = weather_data['wind_v_component_10m_ms']
                    # Calculate wind speed
                    weather_data['wind_speed_ms'] = round(np.sqrt(u**2 + v**2), 2)
                    # Calculate wind direction (meteorological convention)
                    weather_data['wind_direction_degrees'] = round(np.degrees(np.arctan2(-u, -v)), 2) % 360
                    logging.info("Added derived wind metrics")
            except Exception as e:
                logging.error(f"Error calculating wind metrics: {str(e)}")
            
            # Add categorical description of weather if possible
            try:
                if 'temperature_2m_C' in weather_data:
                    temp = weather_data['temperature_2m_C']
                    cloud = weather_data.get('total_cloud_cover_percent', 50)
                    
                    # Simple weather condition classification
                    if cloud > 80:
                        condition = "Overcast"
                    elif cloud > 50:
                        condition = "Mostly Cloudy"
                    elif cloud > 20:
                        condition = "Partly Cloudy"
                    else:
                        condition = "Clear"
                    
                    weather_data['weather_condition'] = condition
                    logging.info(f"Added weather condition: {condition}")
            except Exception as e:
                logging.error(f"Error determining weather condition: {str(e)}")
            
            logging.info(f"Final weather data contains {len(weather_data)} metrics: {list(weather_data.keys())}")
            return weather_data
        
         # Step 1: Download and process taxi data
        logging.info("Downloading taxi data...")
        url = "https://github.com/dotnet/machinelearning/raw/refs/heads/main/test/data/taxi-fare-train.csv"
        
        response = requests.get(url)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        logging.info(f"Downloaded taxi data with {len(df)} rows")
        
        # Since the dataset doesn't have pickup_datetime, we'll create synthetic dates
        start_date = datetime(2019, 1, 1)  # Arbitrary start date
        
        # Create synthetic dates distributed over a year for demonstration
        num_rows = len(df)
        date_range = [start_date + timedelta(hours=i % (24*365)) for i in range(num_rows)]
        df['pickup_datetime'] = date_range
        
        # Extract date components - ensure they are stored as integers
        df['year'] = df['pickup_datetime'].dt.year.astype(int)
        df['month'] = df['pickup_datetime'].dt.month.astype(int)
        df['day'] = df['pickup_datetime'].dt.day.astype(int)
        df['hour'] = df['pickup_datetime'].dt.hour.astype(int)
        
        # Step 2: Process ERA5 data
        era5_base_path = "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.sfc"
        
        # Determine unique year-months in taxi data
        unique_dates = df[['year', 'month']].drop_duplicates().values
        logging.info(f"Found {len(unique_dates)} unique year-months to process")
        
        # Initialize results
        weather_impacts = []
        
        # NYC coordinates
        # Convert from standard longitude (-74.0) to 0-360 format
        nyc_lat = 40.7  # NYC latitude
        nyc_lon = 360 - 74.0  # NYC longitude converted to 0-360 range (should be around 286)
        
        logging.info(f"NYC coordinates: lat={nyc_lat}, lon={nyc_lon} (0-360 format)")
        
        # Process each year-month - limit to first few for testing
        for year, month in unique_dates[:1]:  # Just process the first month for testing
            try:
                # Ensure year and month are integers
                year_val = int(year)
                month_val = int(month)
                
                logging.info(f"Processing year-month: {year_val}-{month_val:02d}")
                
                # Find appropriate ERA5 files for this month
                era5_files = find_era5_files(year_val, month_val, era5_base_path)
                
                if not era5_files:
                    logging.warning(f"No ERA5 files found for {year_val}-{month_val:02d}, skipping")
                    continue
                
                logging.info(f"Found {len(era5_files)} ERA5 files for {year_val}-{month_val:02d}")
                
                # Open each dataset
                datasets = {}
                for var_name, file_path in era5_files.items():
                    try:
                        logging.info(f"Opening dataset for {var_name} from {file_path}")
                        datasets[var_name] = xr.open_dataset(file_path)
                        logging.info(f"Successfully opened dataset for {var_name}")
                    except Exception as e:
                        logging.warning(f"Failed to open dataset {var_name} from {file_path}: {str(e)}")
                
                # Get taxi data for this month
                taxi_month = df[(df['year'] == year_val) & (df['month'] == month_val)]
                logging.info(f"Found {len(taxi_month)} taxi records for {year_val}-{month_val:02d}")
                
                # Aggregate data by day and hour
                logging.info("Aggregating taxi data by day and hour")
                daily_stats = taxi_month.groupby(['day', 'hour']).agg({
                    'fare_amount': ['mean', 'count'],
                    'trip_distance': ['mean'],
                    'trip_time_in_secs': ['mean']
                }).reset_index()
                
                logging.info(f"Aggregated data has {len(daily_stats)} day-hour combinations")
                
                # Process each day and hour 
                for idx, row in daily_stats.iterrows():
                    try:
                        # Extract values using iloc to avoid FutureWarning
                        if isinstance(row['day'], pd.Series):
                            day = int(row['day'].iloc[0])
                            hour = int(row['hour'].iloc[0])
                        else:
                            day = int(safe_extract(row['day']))
                            hour = int(safe_extract(row['hour']))
                        
                        logging.info(f"Processing data for {year_val}-{month_val:02d}-{day:02d} {hour:02d}:00")
                        
                        # Format datetime for ERA5 selection
                        date_str = f"{year_val}-{month_val:02d}-{day:02d}T{hour:02d}:00:00"
                        
                        # Get weather data for this hour at NYC location
                        weather_data = extract_era5_weather_data(datasets, date_str, nyc_lat, nyc_lon)
                        
                        # Extract taxi metrics
                        try:
                            # Properly extract values from the aggregated data
                            # Handle MultiIndex columns from groupby.agg()
                            if isinstance(row.index, pd.MultiIndex) or ('fare_amount', 'count') in row:
                                ride_count = float(safe_extract(row[('fare_amount', 'count')]))
                                avg_fare = float(safe_extract(row[('fare_amount', 'mean')]))
                                avg_distance = float(safe_extract(row[('trip_distance', 'mean')]))
                                avg_trip_time = float(safe_extract(row[('trip_time_in_secs', 'mean')]))
                            else:
                                # Try to find columns by name
                                ride_count = float(safe_extract(row.get('fare_amount_count')))
                                avg_fare = float(safe_extract(row.get('fare_amount_mean')))
                                avg_distance = float(safe_extract(row.get('trip_distance_mean')))
                                avg_trip_time = float(safe_extract(row.get('trip_time_in_secs_mean')))
                            
                            taxi_metrics = {
                                'ride_count': ride_count,
                                'avg_fare': avg_fare,
                                'avg_distance': avg_distance,
                                'avg_trip_time': avg_trip_time
                            }
                            
                            logging.info(f"Extracted taxi metrics: {taxi_metrics}")
                            
                        except Exception as e:
                            logging.warning(f"Error extracting taxi metrics: {str(e)}")
                            # Fallback - inspect the columns and extract manually
                            logging.warning(f"Row columns: {row.index}")
                            
                            taxi_metrics = {
                                'ride_count': 0,
                                'avg_fare': 0,
                                'avg_distance': 0,
                                'avg_trip_time': 0
                            }
                            
                            # Try to extract metrics from column names
                            for col in row.index:
                                col_name = str(col)
                                if 'count' in col_name.lower():
                                    taxi_metrics['ride_count'] = float(safe_extract(row[col]))
                                elif 'fare' in col_name.lower() and 'mean' in col_name.lower():
                                    taxi_metrics['avg_fare'] = float(safe_extract(row[col]))
                                elif 'distance' in col_name.lower():
                                    taxi_metrics['avg_distance'] = float(safe_extract(row[col]))
                                elif 'time' in col_name.lower():
                                    taxi_metrics['avg_trip_time'] = float(safe_extract(row[col]))
                        
                        # Create readable date string
                        date_time_str = f"{year_val}-{month_val:02d}-{day:02d} {hour:02d}:00"
                        
                        # Combine with weather data
                        result = {
                            'year': year_val,
                            'month': month_val,
                            'day': day,
                            'hour': hour,
                            'taxi': taxi_metrics,
                            'weather': weather_data,
                            'date_str': date_time_str
                        }
                        
                        weather_impacts.append(result)
                        logging.info(f"Added result for {date_time_str} with {len(weather_data)} weather metrics")
                        
                    except Exception as e:
                        logging.warning(f"Error processing data for {year_val}-{month_val}-{day} {hour}:00: {str(e)}")
                        import traceback
                        logging.warning(traceback.format_exc())
                        continue
                
                # Close datasets to free memory
                for ds in datasets.values():
                    ds.close()
                
            except Exception as e:
                logging.warning(f"Error processing ERA5 files for {year}-{month}: {str(e)}")
                import traceback
                logging.warning(traceback.format_exc())
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
                        'ride_count': r['taxi'].get('ride_count', 0),
                        'avg_fare': r['taxi'].get('avg_fare', 0),
                        'avg_distance': r['taxi'].get('avg_distance', 0),
                        'avg_trip_time': r['taxi'].get('avg_trip_time', 0)
                    }
                    # Add weather variables
                    for k, v in r.get('weather', {}).items():
                        record[f"weather_{k}"] = v
                    
                    impact_records.append(record)
                
                impact_df = pd.DataFrame(impact_records)
                
                # Calculate correlations
                taxi_columns = ['ride_count', 'avg_fare', 'avg_distance', 'avg_trip_time']
                weather_columns = [col for col in impact_df.columns if col.startswith('weather_')]
                
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
                            logging.warning(f"Error calculating correlation for {taxi_col} vs {weather_col}: {str(e)}")
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
                for metric in ['ride_count', 'avg_fare', 'avg_distance', 'avg_trip_time']:
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
                logging.error(f"Error in correlation analysis: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
        
        # Apply the conversion function to ensure all NumPy types are converted
        weather_impacts = convert_numpy_types(weather_impacts)
        correlations = convert_numpy_types(correlations)
        
        # Return more informative results
        result = {
            'weather_impact_samples': weather_impacts[:48],  # First 15 samples
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
        logging.info("Analysis complete")
        return result
    
    except Exception as exc:
        logging.error(f"Error in taxi-weather analysis: {str(exc)}")
        import traceback
        logging.error(traceback.format_exc())
        raise self.retry(exc=exc, countdown=60*5)  # Retry after 5 minutes