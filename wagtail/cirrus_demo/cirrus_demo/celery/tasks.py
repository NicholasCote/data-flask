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
def taxi_weather_analysis_debug(self):
    """
    Enhanced version of taxi_weather_analysis with detailed debugging.
    """
    # Set up logging
    import logging
    logging.basicConfig(level=logging.DEBUG, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("taxi_weather_debug")
    logger.info("Starting taxi weather analysis task")
    
    try:
        import numpy as np
        import xarray as xr
        import os
        import requests
        import pandas as pd
        from io import StringIO
        from datetime import datetime, timedelta
        import traceback
        
        # Log system information
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"Python executable: {os.sys.executable}")
        
        # Step 1: Download and process taxi data
        logger.info("Downloading taxi data")
        url = "https://github.com/dotnet/machinelearning/raw/refs/heads/main/test/data/taxi-fare-train.csv"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            logger.info(f"Successfully downloaded taxi data: {len(response.text)} bytes")
        except Exception as e:
            logger.error(f"Error downloading taxi data: {e}")
            raise
        
        try:
            df = pd.read_csv(StringIO(response.text))
            logger.info(f"Loaded taxi data: {len(df)} rows, columns: {df.columns.tolist()}")
            logger.debug(f"Taxi data sample: {df.head(3).to_dict()}")
        except Exception as e:
            logger.error(f"Error parsing taxi data: {e}")
            raise
        
        # Create synthetic dates
        logger.info("Creating synthetic dates")
        start_date = datetime(2019, 1, 1)
        num_rows = len(df)
        date_range = [start_date + timedelta(hours=i % (24*365)) for i in range(num_rows)]
        df['pickup_datetime'] = date_range
        
        # Extract date components
        df['year'] = df['pickup_datetime'].dt.year
        df['month'] = df['pickup_datetime'].dt.month
        df['day'] = df['pickup_datetime'].dt.day
        df['hour'] = df['pickup_datetime'].dt.hour
        
        logger.info(f"Date range: {df['pickup_datetime'].min()} to {df['pickup_datetime'].max()}")
        logger.info(f"Unique year-month combinations: {df[['year', 'month']].drop_duplicates().values.tolist()}")
        
        # Step 2: Check ERA5 data path
        era5_base_path = "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.sfc"
        logger.info(f"Checking ERA5 base path: {era5_base_path}")
        
        if not os.path.exists(era5_base_path):
            logger.error(f"ERA5 base path does not exist: {era5_base_path}")
            raise FileNotFoundError(f"ERA5 base path not found: {era5_base_path}")
        
        # List contents of ERA5 base directory
        try:
            base_contents = os.listdir(era5_base_path)
            logger.info(f"ERA5 base directory contains {len(base_contents)} items")
            logger.debug(f"First 10 items: {base_contents[:10]}")
        except Exception as e:
            logger.error(f"Error listing ERA5 base directory: {e}")
            raise
        
        # Function to find the closest ERA5 file with detailed logging
        def find_closest_era5_file(year, month):
            logger.debug(f"Finding ERA5 file for {year}-{month:02d}")
            
            # Format year and month as directories
            year_month_dir = f"{era5_base_path}/{year:04d}{month:02d}/"
            logger.debug(f"Checking directory: {year_month_dir}")
            
            # If the exact directory doesn't exist, find the closest one
            if not os.path.exists(year_month_dir):
                logger.warning(f"Directory not found: {year_month_dir}")
                
                # Find available years
                try:
                    available_dirs = os.listdir(era5_base_path)
                    logger.debug(f"Available directories (first 10): {available_dirs[:10]}")
                    
                    # Filter for YYYYMM format directories
                    available_years = [int(d) for d in available_dirs if d.isdigit() and len(d) == 6]
                    logger.debug(f"Parsed year-months: {available_years[:10] if len(available_years) > 10 else available_years}")
                    
                    if not available_years:
                        logger.error(f"No valid data directories found in {era5_base_path}")
                        raise FileNotFoundError(f"No valid data directories found in {era5_base_path}")
                    
                    # Find closest year-month
                    closest_ym = min(available_years, key=lambda ym: abs(ym - (year*100 + month)))
                    logger.info(f"Using closest year-month directory: {closest_ym} instead of {year*100 + month}")
                    year_month_dir = f"{era5_base_path}/{closest_ym}/"
                    
                except Exception as e:
                    logger.error(f"Error finding alternative directory: {e}")
                    raise
            
            # Check if the directory exists and is accessible
            if not os.path.exists(year_month_dir):
                logger.error(f"Selected directory still doesn't exist: {year_month_dir}")
                raise FileNotFoundError(f"Cannot find suitable directory: {year_month_dir}")
            
            # List contents of the directory
            try:
                dir_contents = os.listdir(year_month_dir)
                logger.debug(f"Directory {year_month_dir} contains {len(dir_contents)} files")
                logger.debug(f"First 10 files: {dir_contents[:10]}")
            except Exception as e:
                logger.error(f"Error listing directory {year_month_dir}: {e}")
                raise
            
            # Find temperature or precipitation file
            for suffix in ["_t2m", "_tp", ""]:
                logger.debug(f"Looking for files with suffix: {suffix}")
                matching_files = [f for f in dir_contents if suffix in f and f.endswith(".nc")]
                
                if matching_files:
                    selected_file = matching_files[0]
                    logger.info(f"Found matching file: {selected_file}")
                    return os.path.join(year_month_dir, selected_file)
            
            logger.error(f"No suitable ERA5 file found in {year_month_dir}")
            raise FileNotFoundError(f"No suitable ERA5 file found in {year_month_dir}")
        
        # Determine unique year-months in taxi data
        unique_dates = df[['year', 'month']].drop_duplicates().values
        logger.info(f"Processing {len(unique_dates)} unique year-month combinations")
        
        # Initialize results
        weather_impacts = []
        
        # Process each year-month
        for i, (year, month) in enumerate(unique_dates):
            logger.info(f"Processing year-month {i+1}/{len(unique_dates)}: {year}-{month:02d}")
            
            # Find appropriate ERA5 file
            try:
                start_time = datetime.now()
                era5_file = find_closest_era5_file(year, month)
                logger.info(f"Found ERA5 file: {era5_file}")
                
                # Attempt to open the file and check its contents
                try:
                    logger.info(f"Opening ERA5 file with xarray")
                    ds = xr.open_dataset(era5_file)
                    logger.info(f"Successfully opened ERA5 file")
                    logger.debug(f"ERA5 dataset dimensions: {ds.dims}")
                    logger.debug(f"ERA5 dataset variables: {list(ds.data_vars)}")
                    logger.debug(f"ERA5 dataset coordinates: {list(ds.coords)}")
                except Exception as e:
                    logger.error(f"Error opening ERA5 file: {e}")
                    logger.error(traceback.format_exc())
                    continue
                
                # Extract variables
                variables = list(ds.data_vars)
                logger.info(f"Processing {len(variables)} variables: {variables}")
                
                # Get taxi data for this month
                taxi_month = df[(df['year'] == year) & (df['month'] == month)]
                logger.info(f"Found {len(taxi_month)} taxi records for {year}-{month:02d}")
                
                if len(taxi_month) == 0:
                    logger.warning(f"No taxi data for {year}-{month:02d}, skipping")
                    continue
                
                # Log column names to ensure proper aggregation
                logger.debug(f"Taxi month columns: {taxi_month.columns.tolist()}")
                
                # Try to perform the aggregation with proper error handling
                try:
                    daily_stats = taxi_month.groupby(['day', 'hour']).agg({
                        'fare_amount': ['mean', 'count'],
                        'trip_distance': ['mean'],
                        'trip_time_in_secs': ['mean']
                    }).reset_index()
                    
                    # If 'count' isn't a separate column, adapt
                    if 'count' not in taxi_month.columns:
                        logger.warning("'count' column not found, using fare_amount count as a substitute")
                        # This creates a MultiIndex column, so we need to access it correctly later
                        
                    logger.info(f"Aggregated to {len(daily_stats)} day-hour combinations")
                    logger.debug(f"Daily stats columns: {daily_stats.columns.tolist()}")
                except Exception as e:
                    logger.error(f"Error in aggregation: {e}")
                    logger.error(traceback.format_exc())
                    continue
                
                # Process each day
                day_results = []
                unique_days = daily_stats.index.get_level_values('day').unique() if isinstance(daily_stats.index, pd.MultiIndex) else daily_stats['day'].unique()
                logger.info(f"Processing {len(unique_days)} unique days in {year}-{month:02d}")
                
                for day in unique_days:
                    logger.debug(f"Processing day {day}")
                    try:
                        # Get the day's data
                        if isinstance(daily_stats.index, pd.MultiIndex):
                            day_group = daily_stats.loc[(day,)]
                        else:
                            day_group = daily_stats[daily_stats['day'] == day]
                        
                        # Format date for xarray selection
                        date_str = f"{year}-{month:02d}-{day:02d}"
                        logger.debug(f"Selecting ERA5 data for date: {date_str}")
                        
                        # Check if this date exists in the dataset
                        if 'time' in ds.coords:
                            time_values = ds.coords['time'].values
                            logger.debug(f"Dataset time range: {time_values[0]} to {time_values[-1]}")
                            
                            # Check if the date is in range
                            try:
                                # Extract weather data for this day
                                day_weather = ds.sel(time=date_str, method="nearest")
                                nearest_time = day_weather.coords['time'].values
                                logger.info(f"Selected weather data for {date_str}, nearest time: {nearest_time}")
                            except Exception as e:
                                logger.error(f"Error selecting date {date_str}: {e}")
                                logger.error(traceback.format_exc())
                                continue
                        else:
                            logger.error(f"No 'time' coordinate found in dataset")
                            continue
                        
                        # Try to get NYC region data
                        try:
                            # Identify actual coordinate names in the dataset
                            lat_names = [coord for coord in ds.coords if coord.lower() in ('latitude', 'lat')]
                            lon_names = [coord for coord in ds.coords if coord.lower() in ('longitude', 'lon')]
                            
                            if lat_names and lon_names:
                                lat_name, lon_name = lat_names[0], lon_names[0]
                                logger.info(f"Using coordinates: {lat_name}, {lon_name}")
                                
                                # Approximate NYC bounding box
                                nyc_bounds = {
                                    lat_name: slice(40.5, 41.0),
                                    lon_name: slice(-74.3, -73.7)
                                }
                                
                                # Check if coordinates are in range
                                lat_range = ds[lat_name].values
                                lon_range = ds[lon_name].values
                                logger.debug(f"Dataset lat range: {min(lat_range)} to {max(lat_range)}")
                                logger.debug(f"Dataset lon range: {min(lon_range)} to {max(lon_range)}")
                                
                                # Try to subset
                                nyc_weather = day_weather.sel(**nyc_bounds)
                                logger.info(f"Successfully subsetted NYC region, shape: {nyc_weather.dims}")
                                
                                # Calculate means for each variable
                                weather_means = {}
                                for var in variables:
                                    if var in nyc_weather and hasattr(nyc_weather[var], 'mean'):
                                        try:
                                            mean_val = float(nyc_weather[var].mean().values)
                                            weather_means[var] = mean_val
                                            logger.debug(f"Mean {var} for NYC: {mean_val}")
                                        except Exception as e:
                                            logger.warning(f"Could not calculate mean for {var}: {e}")
                            else:
                                logger.warning(f"Could not identify lat/lon coordinates. Available: {list(ds.coords)}")
                                # Fallback: use global means
                                weather_means = {var: float(day_weather[var].mean().values) 
                                               for var in variables if var in day_weather}
                                logger.info(f"Using global means as fallback")
                        except Exception as e:
                            logger.warning(f"Error getting NYC region data: {e}")
                            # Fallback to global means
                            try:
                                weather_means = {var: float(day_weather[var].mean().values) 
                                               for var in variables if var in day_weather}
                                logger.info(f"Using global means as fallback")
                            except Exception as e2:
                                logger.error(f"Error calculating global means: {e2}")
                                continue
                        
                        # Process each hour
                        for hour_idx, hour_data in day_group.groupby('hour' if 'hour' in day_group.columns else day_group.index.get_level_values('hour')):
                            logger.debug(f"Processing hour {hour_idx}")
                            
                            # Extract taxi metrics with careful handling of potentially multi-level columns
                            try:
                                if isinstance(hour_data.columns, pd.MultiIndex):
                                    taxi_metrics = {
                                        'ride_count': float(hour_data[('fare_amount', 'count')].values[0]) if ('fare_amount', 'count') in hour_data.columns else 0,
                                        'avg_fare': float(hour_data[('fare_amount', 'mean')].values[0]) if ('fare_amount', 'mean') in hour_data.columns else 0,
                                        'avg_distance': float(hour_data[('trip_distance', 'mean')].values[0]) if ('trip_distance', 'mean') in hour_data.columns else 0,
                                        'avg_trip_time': float(hour_data[('trip_time_in_secs', 'mean')].values[0]) if ('trip_time_in_secs', 'mean') in hour_data.columns else 0,
                                    }
                                else:
                                    # For flat columns
                                    fare_count_col = next((col for col in hour_data.columns if 'count' in col and 'fare' in col.lower()), None)
                                    fare_mean_col = next((col for col in hour_data.columns if 'mean' in col and 'fare' in col.lower()), None)
                                    distance_mean_col = next((col for col in hour_data.columns if 'mean' in col and 'distance' in col.lower()), None)
                                    time_mean_col = next((col for col in hour_data.columns if 'mean' in col and 'time' in col.lower()), None)
                                    
                                    taxi_metrics = {
                                        'ride_count': float(hour_data[fare_count_col].values[0]) if fare_count_col else 0,
                                        'avg_fare': float(hour_data[fare_mean_col].values[0]) if fare_mean_col else 0,
                                        'avg_distance': float(hour_data[distance_mean_col].values[0]) if distance_mean_col else 0,
                                        'avg_trip_time': float(hour_data[time_mean_col].values[0]) if time_mean_col else 0,
                                    }
                                
                                logger.debug(f"Taxi metrics for hour {hour_idx}: {taxi_metrics}")
                            except Exception as e:
                                logger.error(f"Error extracting taxi metrics for hour {hour_idx}: {e}")
                                logger.error(f"Hour data columns: {hour_data.columns.tolist()}")
                                logger.error(traceback.format_exc())
                                continue
                            
                            # Combine with weather data
                            result = {
                                'year': year,
                                'month': month,
                                'day': day,
                                'hour': hour_idx,
                                'taxi': taxi_metrics,
                                'weather': weather_means
                            }
                            
                            day_results.append(result)
                            logger.debug(f"Added result for {year}-{month:02d}-{day:02d} hour {hour_idx}")
                    
                    except Exception as e:
                        logger.error(f"Error processing day {day}: {e}")
                        logger.error(traceback.format_exc())
                        continue
                
                end_time = datetime.now()
                weather_impacts.extend(day_results)
                logger.info(f"Processed {year}-{month:02d}: added {len(day_results)} records in {(end_time-start_time).total_seconds():.2f} seconds")
                
                # Close the dataset to free memory
                ds.close()
                
            except Exception as e:
                logger.error(f"Error processing month {year}-{month:02d}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        # Report progress
        logger.info(f"Weather impact analysis complete. Total records: {len(weather_impacts)}")
        
        # Calculate correlations between weather and taxi metrics
        correlations = {}
        
        if weather_impacts:
            logger.info("Calculating correlations")
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
                logger.info(f"Created correlation dataframe with {len(impact_df)} rows and {len(impact_df.columns)} columns")
                logger.debug(f"Columns: {impact_df.columns.tolist()}")
                
                # Calculate correlations
                taxi_columns = ['ride_count', 'avg_fare', 'avg_distance', 'avg_trip_time']
                weather_columns = [col for col in impact_df.columns if col.startswith('weather_')]
                
                for taxi_col in taxi_columns:
                    for weather_col in weather_columns:
                        try:
                            if impact_df[taxi_col].notna().sum() > 0 and impact_df[weather_col].notna().sum() > 0:
                                corr = impact_df[[taxi_col, weather_col]].corr().iloc[0, 1]
                                correlations[f"{taxi_col}_vs_{weather_col}"] = corr
                                logger.debug(f"Correlation {taxi_col}_vs_{weather_col}: {corr}")
                            else:
                                logger.warning(f"Not enough valid data for correlation between {taxi_col} and {weather_col}")
                        except Exception as e:
                            logger.error(f"Error calculating correlation for {taxi_col} vs {weather_col}: {e}")
            
            except Exception as e:
                logger.error(f"Error in correlation calculation: {e}")
                logger.error(traceback.format_exc())
        
        logger.info(f"Analysis complete. Found {len(correlations)} correlations.")
        
        # Return results
        result = {
            'weather_impact_samples': weather_impacts[:100],  # Limit to first 100 for reasonable response size
            'correlations': correlations,
            'total_days_analyzed': len(weather_impacts)
        }
        
        # Log summary of results
        logger.info(f"Returning results with {len(result['weather_impact_samples'])} samples, "
                   f"{len(result['correlations'])} correlations, "
                   f"{result['total_days_analyzed']} days analyzed")
        
        return result
        
    except Exception as exc:
        logger.error(f"Fatal error in taxi-weather analysis: {exc}")
        logger.error(traceback.format_exc())
        raise self.retry(exc=exc, countdown=60*5)  # Retry after 5 minutes