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
        import math
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
        
        # Function to extract and format weather data from ERA5 dataset
        def extract_era5_weather_data(datasets, time_str, lat_slice, lon_slice):
            """
            Extract weather data from multiple ERA5 datasets for a specific time and location.
            
            Args:
                datasets (dict): Dictionary of xarray datasets for different variables
                time_str (str): Time string in format YYYY-MM-DDThh:00:00
                lat_slice (slice): Latitude slice
                lon_slice (slice): Longitude slice
                
            Returns:
                dict: Weather data
            """
            weather_data = {}
            
            # Process each dataset
            for var_name, ds in datasets.items():
                try:
                    # Get the actual variable name in the dataset
                    # This might require some mapping or inspection of ds.variables
                    if var_name == 'temperature':
                        data_var = 't2m' if 't2m' in ds.variables else '2t'
                    elif var_name == 'wind_u':
                        data_var = 'u10' if 'u10' in ds.variables else '10u'
                    elif var_name == 'wind_v':
                        data_var = 'v10' if 'v10' in ds.variables else '10v'
                    elif var_name == 'cloud_cover':
                        data_var = 'tcc'
                    elif var_name == 'pressure':
                        data_var = 'msl'
                    elif var_name == 'precipitation':
                        data_var = 'tp'
                    else:
                        # Try to find the actual variable
                        potential_vars = list(ds.variables.keys())
                        # Skip coordinate variables
                        potential_vars = [v for v in potential_vars if v not in ['time', 'latitude', 'longitude', 'lat', 'lon']]
                        if potential_vars:
                            data_var = potential_vars[0]
                        else:
                            logging.warning(f"No suitable variable found in dataset for {var_name}")
                            continue
                    
                    # Select data for the specific time and region
                    try:
                        data = ds[data_var].sel(time=time_str, latitude=lat_slice, longitude=lon_slice)
                        
                        # Calculate average for the region
                        avg_value = float(data.mean().values)
                        weather_data[var_name] = avg_value
                        
                    except Exception as e:
                        logging.warning(f"Error extracting {var_name} data: {e}")
                        # Try alternate time dimension name if initial selection fails
                        try:
                            if 'time' not in ds.dims and 'step' in ds.dims:
                                # Some ERA5 files use 'step' instead of 'time'
                                data = ds[data_var].sel(step=time_str, latitude=lat_slice, longitude=lon_slice)
                                avg_value = float(data.mean().values)
                                weather_data[var_name] = avg_value
                        except Exception as e2:
                            logging.warning(f"Also failed with alternate time dimension: {e2}")
                            weather_data[var_name] = None
                
                except Exception as e:
                    logging.warning(f"Error processing {var_name}: {e}")
                    weather_data[var_name] = None
            
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
        
        # Function to find the closest ERA5 file
        def find_closest_era5_file(year, month, variables):
            """
            Find ERA5 files for specific variables for a given year and month.
            
            Args:
                year (int): The year
                month (int): The month
                variables (dict): Dictionary mapping variable names to their ERA5 codes
                                e.g. {'temperature': '2t', 'wind_u': '10u'}
            
            Returns:
                dict: Dictionary mapping variable names to file paths
            """
            # Format year and month as directory
            year_month_dir = f"{era5_base_path}/{year:04d}{month:02d}/"
            
            # Check if directory exists
            if not os.path.exists(year_month_dir):
                # Find available years (assuming YYYYMM format directories)
                all_dirs = os.listdir(era5_base_path)
                available_dirs = [d for d in all_dirs if d.isdigit() and len(d) == 6]
                
                if not available_dirs:
                    raise FileNotFoundError(f"No valid data directories found in {era5_base_path}")
                
                # Find closest year-month
                target_ym = year * 100 + month
                available_yms = [int(d) for d in available_dirs]
                closest_ym = min(available_yms, key=lambda ym: abs(ym - target_ym))
                
                year_month_dir = f"{era5_base_path}/{closest_ym:06d}/"
                logging.info(f"Using closest available data from {closest_ym:06d} instead of {year:04d}{month:02d}")
            
            # Dictionary to store files for each variable
            variable_files = {}
            
            # List all files in the directory
            try:
                files = os.listdir(year_month_dir)
            except OSError as e:
                logging.error(f"Error accessing directory {year_month_dir}: {e}")
                return variable_files
            
            # Match files for each variable
            for var_name, var_code in variables.items():
                matching_files = [f for f in files 
                                if f"_{var_code}." in f or f"_{var_code}_" in f 
                                and f.endswith(".nc")]
                
                if matching_files:
                    # Sort to ensure consistent selection and take the most appropriate file
                    # Using the highest resolution file (usually the largest)
                    matching_files.sort(key=lambda x: os.path.getsize(os.path.join(year_month_dir, x)), reverse=True)
                    variable_files[var_name] = os.path.join(year_month_dir, matching_files[0])
                    logging.info(f"Found {var_name} file: {matching_files[0]}")
                else:
                    logging.warning(f"No file found for {var_name} (code: {var_code}) in {year_month_dir}")
            
            return variable_files
        
        # Determine unique year-months in taxi data
        unique_dates = df[['year', 'month']].drop_duplicates().values
        
        # Initialize results
        weather_impacts = []
        
        # NYC bounding box for weather data
        nyc_lat_slice = slice(40.5, 41.0)
        nyc_lon_slice = slice(-74.3, -73.7)
        # Define variables needed for your analysis with their ERA5 codes
        required_variables = {
            'temperature': '2t',       # 2m temperature
            'wind_u': '10u',           # 10m U wind component 
            'wind_v': '10v',           # 10m V wind component
            'cloud_cover': 'tcc',      # Total cloud cover
            'pressure': 'msl',         # Mean sea level pressure
            'precipitation': 'tp'      # Total precipitation (if available)
        }

        # Process each year-month
        for year, month in unique_dates:
            # Find appropriate ERA5 file
            try:
                era5_file = find_closest_era5_file(year, month, required_variables)
                logging.info(f"Processing ERA5 file: {era5_file}")
                
                # Open the dataset
                ds = xr.open_dataset(era5_file)
                
                # Get taxi data for this month
                taxi_month = df[(df['year'] == year) & (df['month'] == month)]
                
                # Aggregate data
                daily_stats = taxi_month.groupby(['day', 'hour']).agg({
                    'fare_amount': ['mean', 'count'],
                    'trip_distance': ['mean'],
                    'trip_time_in_secs': ['mean']
                }).reset_index()
                
                # Process each day
                for day, day_group in daily_stats.groupby('day'):
                    try:
                        # Format date for xarray selection
                        date_str = f"{year}-{month:02d}-{day:02d}"
                        
                        # Process each hour
                        for hour, hour_data in day_group.groupby('hour'):
                            try:
                                # Extract weather data for this day
                                # Format datetime for ERA5 selection
                                hour_date_str = f"{date_str}T{hour:02d}:00:00"
                                
                                # Get weather data for this hour
                                weather_data = extract_era5_weather_data(ds, hour_date_str, nyc_lat_slice, nyc_lon_slice)
                                
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
                                except Exception as e:
                                    logging.warning(f"Error extracting taxi metrics: {e}")
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
                                logging.warning(f"Error processing hour {hour}: {e}")
                                continue
                        
                    except Exception as e:
                        logging.warning(f"Error processing day {day}: {e}")
                        continue
                
                # Close the dataset to free memory
                ds.close()
                
            except Exception as e:
                logging.warning(f"Error processing ERA5 file for {year}-{month}: {e}")
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