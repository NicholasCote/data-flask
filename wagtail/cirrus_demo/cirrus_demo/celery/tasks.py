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

@shared_task(bind=True, max_retries=3, soft_time_limit=1800, time_limit=2100)
def taxi_weather_analysis(self):
    """
    A longer-running task that correlates taxi ride data with ERA5 weather data.
    Uses real ERA5 data for weather analysis.
    
    Soft time limit is set to 30 minutes (1800s)
    Hard time limit is set to 35 minutes (2100s)
    """
    try:
        import numpy as np
        import xarray as xr
        import pandas as pd
        import logging
        import os
        import requests
        from io import StringIO
        from datetime import datetime, timedelta
        
        # Set up detailed logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info("Starting taxi-weather analysis with ERA5 data")
        
        # Step 1: Download and process taxi data
        url = "https://github.com/dotnet/machinelearning/raw/refs/heads/main/test/data/taxi-fare-train.csv"
        
        try:
            logger.info(f"Downloading taxi data from {url}")
            response = requests.get(url)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text))
            logger.info(f"Successfully loaded CSV with {len(df)} rows")
            logger.info(f"Columns available: {', '.join(df.columns)}")
        except Exception as e:
            logger.error(f"Error downloading or parsing CSV: {str(e)}")
            raise self.retry(exc=e, countdown=300)
        
        # Since the dataset doesn't have pickup_datetime, we'll create synthetic dates
        # But use dates we know have ERA5 data
        logger.info("Creating synthetic dates for analysis")
        start_date = datetime(2018, 1, 1)  # Use 2018 data which should be available in ERA5
        
        # Create synthetic dates distributed over 2 months (Jan-Feb 2018)
        # This increases chance of finding ERA5 data while providing enough variety
        num_rows = len(df)
        date_range = [(start_date + timedelta(days=i % 59)) for i in range(num_rows)]
        df['pickup_datetime'] = date_range
        
        # Extract date components
        df['year'] = df['pickup_datetime'].dt.year
        df['month'] = df['pickup_datetime'].dt.month
        df['day'] = df['pickup_datetime'].dt.day
        df['hour'] = df['pickup_datetime'].dt.hour
        
        # Step 2: Process ERA5 data
        # Define path to ERA5 data - use the path you've verified exists
        era5_base_path = "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.sfc"
        
        # Double-check if the path exists
        if not os.path.exists(era5_base_path):
            logger.error(f"ERA5 base path does not exist: {era5_base_path}")
            # Try to list parent directories to help diagnose the issue
            parent_dir = os.path.dirname(era5_base_path)
            if os.path.exists(parent_dir):
                logger.info(f"Parent directory exists. Contents: {os.listdir(parent_dir)}")
            
            # If the specified path doesn't exist, try to find an alternative path
            alternative_paths = [
                "/glade/collections/rda/data/ds633.0/e5.oper.an.sfc",
                "/glade/campaign/collections/rda/ds633.0/e5.oper.an.sfc",
                "/glade/data/collections/rda/data/ds633.0",
                "/glade/data/rda/ds633.0"
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    logger.info(f"Found alternative ERA5 path: {alt_path}")
                    era5_base_path = alt_path
                    break
            else:
                # If no alternative path is found, log available directories
                logger.error("No alternative ERA5 path found")
                try:
                    logger.info(f"Directories in /glade: {os.listdir('/glade')}")
                except:
                    pass
                
                # Raise exception to retry later - the path might be temporarily unavailable
                raise self.retry(exc=FileNotFoundError(f"ERA5 path not found: {era5_base_path}"), countdown=600)
        
        # Function to find ERA5 files for a given date
        def find_era5_file(year, month, day=None):
            """Find ERA5 files for the given date"""
            logger.info(f"Looking for ERA5 data for {year}/{month}/{day or 'any'}")
            
            # Check several possible directory structures
            possible_paths = [
                f"{era5_base_path}/{year:04d}{month:02d}/",  # YYYYMM format
                f"{era5_base_path}/{year:04d}/{month:02d}/",  # YYYY/MM format
                f"{era5_base_path}/{year:04d}/"  # YYYY format (month files inside)
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    logger.info(f"Found valid ERA5 directory: {path}")
                    
                    # If we're looking for a specific day, try to find a file for that day
                    if day:
                        day_pattern = f"{year:04d}{month:02d}{day:02d}"
                        for file in os.listdir(path):
                            if day_pattern in file and file.endswith(".nc"):
                                file_path = os.path.join(path, file)
                                logger.info(f"Found ERA5 file for specific day: {file_path}")
                                return file_path
                    
                    # If no specific day or no file found for specific day, return any NetCDF file
                    for file in os.listdir(path):
                        if file.endswith(".nc"):
                            file_path = os.path.join(path, file)
                            logger.info(f"Found ERA5 file: {file_path}")
                            return file_path
            
            # If no file is found, try to find any ERA5 file to use as fallback
            logger.warning(f"No ERA5 file found for {year}/{month}. Searching for any file...")
            for root, dirs, files in os.walk(era5_base_path, topdown=True, followlinks=True):
                for file in files:
                    if file.endswith(".nc"):
                        file_path = os.path.join(root, file)
                        logger.info(f"Found fallback ERA5 file: {file_path}")
                        return file_path
                
                # Limit directory traversal to avoid excessive scanning
                if len(files) > 0 or len(dirs) > 10:
                    break
            
            raise FileNotFoundError(f"No ERA5 files found for {year}/{month} or any valid fallback")
        
        # Determine unique year-months in taxi data
        unique_dates = df[['year', 'month']].drop_duplicates().values
        logger.info(f"Found {len(unique_dates)} unique year-months in the taxi data")
        
        # Initialize results
        weather_impacts = []
        
        # Process each year-month
        for year, month in unique_dates:
            try:
                # Find appropriate ERA5 file
                try:
                    era5_file = find_era5_file(year, month)
                    logger.info(f"Processing ERA5 file: {era5_file}")
                    
                    # Load ERA5 dataset - this is computationally intensive
                    # Use dask for lazy loading to improve performance
                    ds = xr.open_dataset(era5_file, chunks={'time': 100})
                    
                    # Extract variables (temperature, precipitation, etc.)
                    variables = list(ds.data_vars)
                    logger.info(f"ERA5 variables available: {', '.join(variables[:10])}...")
                    
                except Exception as e:
                    logger.error(f"Error accessing ERA5 data for {year}-{month:02d}: {str(e)}")
                    # Skip to next month instead of raising exception
                    continue
                
                # Extract unique days for this month
                days_in_month = df[(df['year'] == year) & (df['month'] == month)]['day'].unique()
                logger.info(f"Processing {len(days_in_month)} days for {year}-{month:02d}")
                
                # Process a subset of days to ensure the task completes
                # This makes the task run longer but still complete
                days_to_process = days_in_month[:min(10, len(days_in_month))]
                
                # For each selected day in this month
                for day in days_to_process:
                    taxi_day = df[(df['year'] == year) & (df['month'] == month) & (df['day'] == day)]
                    
                    # Aggregate by hour
                    try:
                        daily_stats = taxi_day.groupby('hour').agg({
                            'fare_amount': ['mean', 'count'],
                            'trip_distance': ['mean'],
                            'trip_time_in_secs': ['mean'],
                            'count': ['sum']
                        }).reset_index()
                        
                        logger.info(f"Found {len(daily_stats)} hourly data points for {year}-{month:02d}-{day:02d}")
                    except Exception as e:
                        logger.error(f"Error aggregating taxi data for {year}-{month:02d}-{day:02d}: {str(e)}")
                        continue
                    
                    try:
                        # Try to find weather data for this specific day
                        day_time = f"{year}-{month:02d}-{day:02d}"
                        
                        # This operation is computationally intensive
                        # First check if the timestamp exists in the dataset
                        if hasattr(ds, 'time') and day_time in ds.time.astype(str).values:
                            day_weather = ds.sel(time=day_time, method="nearest")
                        else:
                            # If exact timestamp not found, use the first timestamp as fallback
                            first_time = ds.time.values[0]
                            day_weather = ds.sel(time=first_time)
                            logger.warning(f"Exact time {day_time} not found in ERA5 data, using {first_time}")
                        
                        # Extract weather variables using a computationally intensive approach
                        weather_means = {}
                        
                        # Find important climate variables if available
                        for var_name in ['t2m', 'tp', 'u10', 'v10', 'sp']:
                            if var_name in variables:
                                # Get mean value of this variable
                                try:
                                    # Try with broadcasting - computationally intensive but robust
                                    weather_means[var_name] = float(day_weather[var_name].mean().values)
                                except:
                                    # Fallback for complex variable structures
                                    try:
                                        val = day_weather[var_name]
                                        if hasattr(val, 'values'):
                                            # Average over all dimensions
                                            weather_means[var_name] = float(np.nanmean(val.values))
                                        else:
                                            weather_means[var_name] = float(val)
                                    except Exception as ve:
                                        logger.warning(f"Could not process variable {var_name}: {ve}")
                                        weather_means[var_name] = 0.0
                        
                        # If no important variables found, add a few generic ones
                        if len(weather_means) < 3:
                            for var_name in variables[:5]:  # Take first 5 available variables
                                try:
                                    weather_means[var_name] = float(day_weather[var_name].mean().values)
                                except:
                                    try:
                                        weather_means[var_name] = float(np.nanmean(day_weather[var_name].values))
                                    except:
                                        continue
                        
                        # For each hour, create an impact record
                        for _, hour_data in daily_stats.iterrows():
                            hour = hour_data['hour']
                            
                            # Extract taxi metrics for this hour
                            taxi_metrics = {
                                'ride_count': float(hour_data[('fare_amount', 'count')]),
                                'avg_fare': float(hour_data[('fare_amount', 'mean')]),
                                'avg_distance': float(hour_data[('trip_distance', 'mean')]),
                                'avg_trip_time': float(hour_data[('trip_time_in_secs', 'mean')]),
                                'total_count': float(hour_data[('count', 'sum')])
                            }
                            
                            # Create result record
                            result = {
                                'year': int(year),
                                'month': int(month),
                                'day': int(day),
                                'hour': int(hour),
                                'taxi': taxi_metrics,
                                'weather': {k: float(v) for k, v in weather_means.items()}
                            }
                            
                            weather_impacts.append(result)
                            
                    except Exception as e:
                        logger.error(f"Error processing weather data for {year}-{month:02d}-{day:02d}: {str(e)}")
                        continue
                
                # Close the dataset to free memory
                ds.close()
                
            except Exception as e:
                logger.error(f"Error processing month {year}-{month:02d}: {str(e)}")
                continue
        
        # Calculate correlations
        logger.info(f"Calculating correlations based on {len(weather_impacts)} data points")
        correlations = {}
        
        if weather_impacts:
            # Convert results to dataframe for analysis
            try:
                # Ensure all dictionary values are serializable
                clean_impacts = []
                for impact in weather_impacts:
                    try:
                        # Convert any numpy types to Python natives
                        for key in ['taxi', 'weather']:
                            for k, v in impact[key].items():
                                if hasattr(v, 'item'):
                                    impact[key][k] = v.item()  # Convert numpy type to native Python
                        clean_impacts.append(impact)
                    except Exception as e:
                        logger.warning(f"Error cleaning impact data: {e}")
                
                # Now create the DataFrame
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
                    for r in clean_impacts
                ])
                
                # Calculate correlations
                taxi_columns = ['ride_count', 'avg_fare', 'avg_distance', 'avg_trip_time', 'total_count']
                weather_columns = [col for col in impact_df.columns if col.startswith('weather_')]
                
                for taxi_col in taxi_columns:
                    for weather_col in weather_columns:
                        if impact_df[taxi_col].notna().sum() > 0 and impact_df[weather_col].notna().sum() > 0:
                            corr = impact_df[[taxi_col, weather_col]].corr().iloc[0, 1]
                            # Convert numpy float to Python float for JSON serialization
                            correlations[f"{taxi_col}_vs_{weather_col}"] = float(corr)
                            
                logger.info(f"Successfully calculated {len(correlations)} correlations")
                
            except Exception as e:
                logger.error(f"Error during correlation calculation: {str(e)}")
        else:
            logger.warning("No weather impacts found to calculate correlations")
        
        # Return results
        unique_days = len({(r['year'], r['month'], r['day']) for r in weather_impacts})
        logger.info(f"Analysis complete: {len(weather_impacts)} samples across {unique_days} days")
        
        return {
            'weather_impact_samples': weather_impacts[:100],  # Limit to first 100 for reasonable response size
            'correlations': correlations,
            'total_days_analyzed': unique_days
        }
        
    except Exception as exc:
        logger.error(f"Unexpected error in taxi-weather analysis: {exc}")
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=300)