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
                return tuple(convert_numpy_types(i) for i in i)
            else:
                return obj
        
        # Rest of your code remains the same...
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
        def find_closest_era5_file(year, month):
            # Format year and month as directories
            year_month_dir = f"{era5_base_path}/{year:04d}{month:02d}/"
            
            # If the exact directory doesn't exist, find the closest one
            if not os.path.exists(year_month_dir):
                # Find available years
                available_years = [int(d) for d in os.listdir(era5_base_path) if d.isdigit() and len(d) == 6]
                if not available_years:
                    raise FileNotFoundError(f"No valid data directories found in {era5_base_path}")
                
                # Find closest year-month
                closest_ym = min(available_years, key=lambda ym: abs(ym - (year*100 + month)))
                year_month_dir = f"{era5_base_path}/{closest_ym}/"
            
            # Find temperature or precipitation file
            for suffix in ["_t2m", "_tp", ""]:
                for file in os.listdir(year_month_dir):
                    if suffix in file and file.endswith(".nc"):
                        return os.path.join(year_month_dir, file)
            
            raise FileNotFoundError(f"No suitable ERA5 file found in {year_month_dir}")
        
        # Determine unique year-months in taxi data
        unique_dates = df[['year', 'month']].drop_duplicates().values
        
        # Initialize results
        weather_impacts = []
        
        # Process each year-month
        for year, month in unique_dates:
            # Find appropriate ERA5 file
            try:
                era5_file = find_closest_era5_file(year, month)
                
                # Open the dataset
                ds = xr.open_dataset(era5_file)
                
                # Extract variables
                variables = list(ds.data_vars)
                
                # Get taxi data for this month
                taxi_month = df[(df['year'] == year) & (df['month'] == month)]
                
                # Aggregate data
                daily_stats = taxi_month.groupby(['day', 'hour']).agg({
                    'fare_amount': ['mean', 'count'],
                    'trip_distance': ['mean'],
                    'trip_time_in_secs': ['mean']
                }).reset_index()
                
                # Add a count column if it doesn't exist
                if 'count' not in taxi_month.columns:
                    daily_stats[('count', 'sum')] = daily_stats[('fare_amount', 'count')]
                
                # Process each day
                day_results = []
                for day, day_group in daily_stats.groupby('day'):
                    try:
                        # Format date for xarray selection
                        date_str = f"{year}-{month:02d}-{day:02d}"
                        
                        # Extract weather data for this day
                        day_weather = ds.sel(time=date_str, method="nearest")
                        
                        # Try to get NYC region data - handle different coordinate naming
                        try:
                            # Identify actual coordinate names in the dataset
                            lat_names = [coord for coord in ds.coords if coord.lower() in ('latitude', 'lat')]
                            lon_names = [coord for coord in ds.coords if coord.lower() in ('longitude', 'lon')]
                            
                            if lat_names and lon_names:
                                lat_name, lon_name = lat_names[0], lon_names[0]
                                
                                # Approximate NYC bounding box
                                nyc_bounds = {
                                    lat_name: slice(40.5, 41.0),
                                    lon_name: slice(-74.3, -73.7)
                                }
                                
                                # Subset to NYC region
                                nyc_weather = day_weather.sel(**nyc_bounds)
                                
                                # Calculate weather means
                                weather_means = {}
                                for var in variables:
                                    if var in nyc_weather and hasattr(nyc_weather[var], 'mean'):
                                        mean_val = float(nyc_weather[var].mean().values)
                                        # Only add non-NaN values
                                        if not math.isnan(mean_val):
                                            weather_means[var] = mean_val
                                        else:
                                            # Option 1: Skip this value
                                            # pass
                                            # Option 2: Replace with None (will become null in JSON)
                                            weather_means[var] = None
                                            # Option 3: Replace with a default value
                                            # weather_means[var] = 0
                            else:
                                # Fallback to global means
                                weather_means = {var: float(day_weather[var].mean().values) 
                                               for var in variables if var in day_weather and hasattr(day_weather[var], 'mean')}
                        except Exception:
                            # Fallback to global means
                            weather_means = {var: float(day_weather[var].mean().values) 
                                           for var in variables if var in day_weather and hasattr(day_weather[var], 'mean')}
                        
                        # Process each hour
                        for hour, hour_data in day_group.groupby('hour'):
                            # Extract taxi metrics with careful handling
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
                                        elif col[0] == 'count' and col[1] == 'sum':
                                            columns_dict['total_count'] = hour_data[col].values[0]
                                
                                taxi_metrics = {
                                    'ride_count': columns_dict.get('ride_count', 0),
                                    'avg_fare': columns_dict.get('avg_fare', 0),
                                    'avg_distance': columns_dict.get('avg_distance', 0),
                                    'avg_trip_time': columns_dict.get('avg_trip_time', 0),
                                    'total_count': columns_dict.get('total_count', columns_dict.get('ride_count', 0))
                                }
                            except Exception:
                                # Fallback with empty metrics
                                taxi_metrics = {
                                    'ride_count': 0,
                                    'avg_fare': 0,
                                    'avg_distance': 0,
                                    'avg_trip_time': 0,
                                    'total_count': 0
                                }
                            
                            # Combine with weather data
                            result = {
                                'year': int(year),  # Convert numpy types to native Python types
                                'month': int(month),
                                'day': int(day),
                                'hour': int(hour),
                                'taxi': taxi_metrics,
                                'weather': weather_means
                            }
                            
                            day_results.append(result)
                    except Exception:
                        continue
                
                weather_impacts.extend(day_results)
                
                # Close the dataset to free memory
                ds.close()
                
            except Exception:
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
                        except Exception:
                            continue
            except Exception:
                pass
        
        # Apply the conversion function to ensure all NumPy types are converted
        weather_impacts = convert_numpy_types(weather_impacts)
        correlations = convert_numpy_types(correlations)
        
        # Return results with all NumPy types converted to Python native types
        return {
            'weather_impact_samples': weather_impacts[:100],  # Limit to first 100 for reasonable response size
            'correlations': correlations,
            'total_days_analyzed': len(weather_impacts)
        }
        
    except Exception as exc:
        logging.error(f"Error in taxi-weather analysis: {exc}")
        raise self.retry(exc=exc, countdown=60*5)  # Retry after 5 minutes