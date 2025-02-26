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
        
        # Function to find relevant ERA5 NetCDF files
        def find_era5_files(year, month, base_path="/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.sfc"):
            """
            Find relevant ERA5 NetCDF files for weather analysis.
            
            Args:
                year (int): Year to find data for
                month (int): Month to find data for
                base_path (str): Base path to ERA5 data
                
            Returns:
                dict: Dictionary mapping variable names to file paths
            """
            # Format the directory path for the year and month
            year_month_dir = f"{base_path}/{year:04d}{month:02d}"
            
            # If directory doesn't exist, find the closest available directory
            if not os.path.exists(year_month_dir):
                logging.warning(f"Directory {year_month_dir} not found, looking for alternatives")
                
                # List all directories in the base path
                available_dirs = [d for d in os.listdir(base_path) 
                                if os.path.isdir(os.path.join(base_path, d)) and d.isdigit() and len(d) == 6]
                
                if not available_dirs:
                    raise FileNotFoundError(f"No valid year-month directories found in {base_path}")
                
                # Convert to integers for numeric comparison
                available_ym = [int(d) for d in available_dirs]
                
                # Find the closest year-month
                target_ym = year * 100 + month
                closest_ym = min(available_ym, key=lambda ym: abs(ym - target_ym))
                
                # Update directory path
                year_month_dir = f"{base_path}/{closest_ym}"
                logging.info(f"Using alternative directory: {year_month_dir}")
            
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
            
            files = os.listdir(year_month_dir)
            nc_files = [f for f in files if f.endswith('.nc')]
            
            for var, pattern in target_variables.items():
                matching_files = [f for f in nc_files if pattern in f]
                
                if matching_files:
                    # Sort by file size (larger files might have more complete data)
                    matching_files.sort(key=lambda f: os.path.getsize(os.path.join(year_month_dir, f)), reverse=True)
                    found_files[var] = os.path.join(year_month_dir, matching_files[0])
                    logging.info(f"Found file for {var}: {matching_files[0]}")
                else:
                    logging.warning(f"No file found for variable {var} with pattern {pattern}")
            
            if not found_files:
                raise FileNotFoundError(f"No suitable ERA5 files found in {year_month_dir}")
            
            return found_files
        
        # Function to extract and format weather data from ERA5 dataset
        def extract_era5_weather_data(datasets, date_str, lat_slice, lon_slice):
            """
            Extract and format weather data from ERA5 datasets for a specific date and region.
            
            Args:
                datasets (dict): Dictionary of variable names to xarray datasets
                date_str (str): Date string in format YYYY-MM-DDThh:mm:ss
                lat_slice (slice): Latitude slice for the region of interest
                lon_slice (slice): Longitude slice for the region of interest
                
            Returns:
                dict: Weather data for the specified date and region
            """
            weather_data = {}
            
            # Process each variable dataset
            for var_name, ds in datasets.items():
                try:
                    # Identify coordinate names (they can vary between datasets)
                    lat_names = [coord for coord in ds.coords if coord.lower() in ('latitude', 'lat')]
                    lon_names = [coord for coord in ds.coords if coord.lower() in ('longitude', 'lon')]
                    
                    if lat_names and lon_names:
                        lat_name, lon_name = lat_names[0], lon_names[0]
                        
                        # Create selection dict for the region
                        selection = {
                            lat_name: lat_slice,
                            lon_name: lon_slice
                        }
                        
                        # Select region first, then time
                        regional_ds = ds.sel(**selection)
                        var_data = regional_ds.sel(time=date_str, method="nearest")
                        
                        # Get the data variable (usually has the same name as the file)
                        if var_name in var_data.data_vars:
                            data_var = var_data[var_name]
                        else:
                            # Try to find the most relevant data variable
                            data_vars = list(var_data.data_vars.keys())
                            if len(data_vars) > 0:
                                data_var = var_data[data_vars[0]]
                            else:
                                continue
                        
                        # Calculate mean value for the region
                        if hasattr(data_var, 'mean'):
                            mean_val = float(data_var.mean().values)
                            
                            # Skip NaN values
                            if not np.isnan(mean_val):
                                # Process values based on variable type
                                if var_name == 't2m' or var_name == 'd2m' or var_name == 'skt':
                                    # Convert from Kelvin to Celsius
                                    mean_val = mean_val - 273.15
                                elif var_name == 'sp' or var_name == 'msl':
                                    # Convert from Pa to hPa
                                    mean_val = mean_val / 100.0
                                elif var_name == 'tcc':
                                    # Convert from 0-1 to percentage
                                    mean_val = mean_val * 100.0
                                
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
                    
                except Exception as e:
                    logging.warning(f"Error processing variable {var_name}: {e}")
            
            # Calculate derived metrics if possible
            try:
                if 'wind_u_component_10m_ms' in weather_data and 'wind_v_component_10m_ms' in weather_data:
                    u = weather_data['wind_u_component_10m_ms']
                    v = weather_data['wind_v_component_10m_ms']
                    # Calculate wind speed
                    weather_data['wind_speed_ms'] = round(np.sqrt(u**2 + v**2), 2)
                    # Calculate wind direction (meteorological convention)
                    weather_data['wind_direction_degrees'] = round(np.degrees(np.arctan2(-u, -v)), 2) % 360
            except Exception as e:
                logging.warning(f"Error calculating wind metrics: {e}")
            
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
            except Exception as e:
                logging.warning(f"Error determining weather condition: {e}")
            
            return weather_data
        
        # Step 1: Download and process taxi data
        url = "https://github.com/dotnet/machinelearning/raw/refs/heads/main/test/data/taxi-fare-train.csv"
        
        response = requests.get(url)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        
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
        
        # Step 2: Process ERA5 data
        era5_base_path = "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.sfc"
        
        # Determine unique year-months in taxi data
        unique_dates = df[['year', 'month']].drop_duplicates().values
        
        # Initialize results
        weather_impacts = []
        
        # NYC bounding box for weather data
        nyc_lat_slice = slice(40.5, 41.0)
        nyc_lon_slice = slice(-74.3, -73.7)
        
        # Process each year-month
        for year, month in unique_dates:
            try:
                # Find appropriate ERA5 files for this month
                era5_files = find_era5_files(year, month, era5_base_path)
                logging.info(f"Found {len(era5_files)} ERA5 files for {year}-{month}")
                
                # Open each dataset
                datasets = {}
                for var_name, file_path in era5_files.items():
                    try:
                        datasets[var_name] = xr.open_dataset(file_path)
                    except Exception as e:
                        logging.warning(f"Failed to open dataset {var_name} from {file_path}: {e}")
                
                # Get taxi data for this month
                taxi_month = df[(df['year'] == year) & (df['month'] == month)]
                
                # Aggregate data by day and hour
                daily_stats = taxi_month.groupby(['day', 'hour']).agg({
                    'fare_amount': ['mean', 'count'],
                    'trip_distance': ['mean'],
                    'trip_time_in_secs': ['mean']
                }).reset_index()
                
                # Process each day and hour
                for _, row in daily_stats.iterrows():
                    try:
                        day = row['day']
                        hour = row['hour']
                        
                        # Format datetime for ERA5 selection
                        date_str = f"{year}-{month:02d}-{day:02d}T{hour:02d}:00:00"
                        
                        # Get weather data for this hour
                        weather_data = extract_era5_weather_data(datasets, date_str, nyc_lat_slice, nyc_lon_slice)
                        
                        # Extract taxi metrics - handle MultiIndex columns from groupby.agg()
                        taxi_metrics = {
                            'ride_count': row[('fare_amount', 'count')],
                            'avg_fare': row[('fare_amount', 'mean')],
                            'avg_distance': row[('trip_distance', 'mean')],
                            'avg_trip_time': row[('trip_time_in_secs', 'mean')]
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
                        logging.warning(f"Error processing data for {year}-{month}-{day} {hour}:00: {e}")
                        continue
                
                # Close datasets to free memory
                for ds in datasets.values():
                    ds.close()
                
            except Exception as e:
                logging.warning(f"Error processing ERA5 files for {year}-{month}: {e}")
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