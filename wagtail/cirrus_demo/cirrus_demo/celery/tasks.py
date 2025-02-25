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
        
        # Step 1: Download and process taxi data
        url = "https://github.com/dotnet/machinelearning/raw/refs/heads/main/test/data/taxi-fare-train.csv"
        
        response = requests.get(url)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        
        # Since the dataset doesn't have pickup_datetime, we'll create synthetic dates
        # This is a workaround since we don't have actual dates in the data
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
        
        # Step 2: Process ERA5 data (this will be computationally intensive)
        # Define path to ERA5 data
        era5_base_path = "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.sfc"
        
        # Function to find the closest ERA5 file for a given date
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
                
                # This operation is very computationally intensive
                ds = xr.open_dataset(era5_file)
                
                # Extract variables (temperature, precipitation, etc.)
                variables = list(ds.data_vars)
                
                # For each day in this month
                taxi_month = df[(df['year'] == year) & (df['month'] == month)]
                
                # Modify aggregation to use actual available columns
                daily_stats = taxi_month.groupby(['day', 'hour']).agg({
                    'fare_amount': ['mean', 'count'],
                    'trip_distance': ['mean'],
                    'trip_time_in_secs': ['mean'],
                    'count': ['sum']  # Using 'count' instead of passenger_count
                }).reset_index()
                
                # For each day, correlate with weather data
                # This nested loop with xarray operations is computationally intensive
                day_results = []
                for day, day_group in daily_stats.groupby('day'):
                    try:
                        # Extract weather data for this day
                        day_weather = ds.sel(time=f"{year}-{month:02d}-{day:02d}", method="nearest")
                        
                        # Calculate average weather metrics across NYC region
                        # Approximate NYC bounding box
                        nyc_bounds = {
                            'latitude': slice(40.5, 41.0),
                            'longitude': slice(-74.3, -73.7)
                        }
                        
                        # Expensive operation: subsetting and averaging over region
                        try:
                            nyc_weather = day_weather.sel(**nyc_bounds)
                            weather_means = {var: float(nyc_weather[var].mean().values) 
                                            for var in variables if 'time' in nyc_weather[var].dims}
                        except:
                            # Fallback if coordinate names are different
                            weather_means = {var: float(day_weather[var].mean().values) 
                                           for var in variables if 'time' in day_weather[var].dims}
                        
                        # Computationally intensive analysis:
                        # For each hour of the day, analyze relationship between
                        # weather conditions and taxi patterns
                        for hour, hour_data in day_group.groupby('hour'):
                            # Extract taxi metrics for this hour, using available columns
                            taxi_metrics = {
                                'ride_count': hour_data[('fare_amount', 'count')].values[0],
                                'avg_fare': hour_data[('fare_amount', 'mean')].values[0],
                                'avg_distance': hour_data[('trip_distance', 'mean')].values[0],
                                'avg_trip_time': hour_data[('trip_time_in_secs', 'mean')].values[0],
                                'total_count': hour_data[('count', 'sum')].values[0]  # Using count instead of passengers
                            }
                            
                            # Combine with weather data
                            result = {
                                'year': year,
                                'month': month,
                                'day': day,
                                'hour': hour,
                                'taxi': taxi_metrics,
                                'weather': weather_means
                            }
                            
                            day_results.append(result)
                    except Exception as e:
                        logging.warning(f"Error processing weather data for {year}-{month:02d}-{day:02d}: {e}")
                        continue
                
                weather_impacts.extend(day_results)
                # Close the dataset to free memory
                ds.close()
                
            except (FileNotFoundError, Exception) as e:
                logging.warning(f"Could not process ERA5 data for {year}-{month:02d}: {e}")
                continue
        
        # Calculate correlations between weather and taxi metrics
        correlations = {}
        
        if weather_impacts:
            # Convert results to dataframe for analysis, using the actual available metrics
            impact_df = pd.DataFrame([
                {
                    'year': r['year'],
                    'month': r['month'],
                    'day': r['day'],
                    'hour': r['hour'],
                    'ride_count': r['taxi']['ride_count'],
                    'avg_fare': r['taxi']['avg_fare'],
                    'avg_distance': r['taxi']['avg_distance'],
                    'avg_trip_time': r['taxi']['avg_trip_time'],
                    'total_count': r['taxi']['total_count'],
                    **{f"weather_{k}": v for k, v in r['weather'].items()}
                }
                for r in weather_impacts
            ])
            
            # Calculate correlations (computationally intensive)
            # Updated to use actual available taxi metrics
            taxi_columns = ['ride_count', 'avg_fare', 'avg_distance', 'avg_trip_time', 'total_count']
            weather_columns = [col for col in impact_df.columns if col.startswith('weather_')]
            
            # This is computationally intensive for large datasets
            for taxi_col in taxi_columns:
                for weather_col in weather_columns:
                    if impact_df[taxi_col].notna().sum() > 0 and impact_df[weather_col].notna().sum() > 0:
                        corr = impact_df[[taxi_col, weather_col]].corr().iloc[0, 1]
                        correlations[f"{taxi_col}_vs_{weather_col}"] = corr
        
        return {
            'weather_impact_samples': weather_impacts[:100],  # Limit to first 100 for reasonable response size
            'correlations': correlations,
            'total_days_analyzed': len(weather_impacts)
        }
        
    except Exception as exc:
        logging.error(f"Error in taxi-weather analysis: {exc}")
        raise self.retry(exc=exc, countdown=60*5)  # Retry after 5 minutes