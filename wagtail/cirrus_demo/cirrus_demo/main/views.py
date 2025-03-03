from ..celery.tasks import analyze_taxi_data, max_taxi_fare, total_taxi_fare, taxi_weather_analysis, get_glade_picture_task
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import logging
import os
from pathlib import Path

def home(request):
    return render(request, 'base.html')

def trigger_analysis(request):
    task = analyze_taxi_data.delay()
    return JsonResponse({'task_id': str(task.id)})

def trigger_max_analysis(request):
    task = max_taxi_fare.delay()
    return JsonResponse({'task_id': str(task.id)})

def trigger_sum_analysis(request):
    task = total_taxi_fare.delay()
    return JsonResponse({'task_id': str(task.id)})

def trigger_weather_analysis(request):
    task = taxi_weather_analysis.delay()
    return JsonResponse({'task_id': str(task.id)})

def check_task_status(request, task_id):
    task = analyze_taxi_data.AsyncResult(task_id)
    print(f"Task ID: {task_id}")
    print(f"Task state: {task.state}")
    print(f"Task ready: {task.ready()}")
    print(f"Task backend: {task.backend}")
    if task.ready():
        result = task.get()
        print(f"Task result: {result}")
        return JsonResponse({'status': task.state, 'result': result})
    return JsonResponse({'status': task.state})

def check_max_task_status(request, task_id):
    task = max_taxi_fare.AsyncResult(task_id)
    print(f"Task state: {task.state}")  # Debug print
    if task.ready():
        result = task.get()
        print(f"Task result: {result}")  # Debug print
        return JsonResponse({'status': task.state, 'result': result})
    return JsonResponse({'status': task.state})

def check_total_task_status(request, task_id):
    task = total_taxi_fare.AsyncResult(task_id)
    print(f"Task state: {task.state}")  # Debug print
    if task.ready():
        result = task.get()
        print(f"Task result: {result}")  # Debug print
        return JsonResponse({'status': task.state, 'result': result})
    return JsonResponse({'status': task.state})

def analysis_view(request):
    return render(request, 'analysis.html')

def glade_image(request):
    """
    View for the GLADE image visualization page.
    """
    return render(request, 'image.html')

@csrf_exempt
def trigger_glade_analysis(request):
    """
    View for starting the GLADE picture task with custom parameters.
    """
    if request.method == 'POST':
        try:
            # Parse JSON data from request body
            data = json.loads(request.body)
            zip_code = data.get('zipCode', '82001')  # Default to Cheyenne if not provided
            start_date = data.get('startDate', '2021-01')
            end_date = data.get('endDate', '2021-02')
            
            # Get lat/lon from zip code
            location_data = get_location_from_zip(zip_code)
            
            if not location_data:
                return JsonResponse({
                    'error': f'Could not find location for zip code {zip_code}'
                }, status=400)
            
            # Extract year and month values
            start_year, start_month = start_date.split('-')
            end_year, end_month = end_date.split('-')
            
            # Start the Celery task with the provided parameters
            task = get_glade_picture_task.delay(
                location_data['lat'], 
                location_data['lon'],
                f"{start_year}{start_month}",
                f"{end_year}{end_month}"
            )
            
            return JsonResponse({'task_id': task.id})
        
        except Exception as e:
            logging.error(f"Error triggering GLADE analysis: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    else:
        # For backwards compatibility, handle GET requests with default params
        task = get_glade_picture_task.delay(
            41.14,            # Cheyenne latitude
            360 - 104.82,     # Cheyenne longitude
            '202101',         # Start period
            '202102'          # End period
        )
        return JsonResponse({'task_id': task.id})

def check_glade_task(request, task_id):
    """
    View for checking the status of the GLADE picture task.
    """
    task_result = get_glade_picture_task.AsyncResult(task_id)
    
    response_data = {'task_id': task_id}
    
    if task_result.successful():
        response_data['status'] = 'SUCCESS'
        response_data['result'] = task_result.result
    elif task_result.failed():
        response_data['status'] = 'FAILURE'
        response_data['error'] = str(task_result.result)
    elif task_result.state == 'PROGRESS':
        response_data['status'] = 'PROGRESS'
        if task_result.info:
            response_data['progress'] = task_result.info.get('progress', 0)
            response_data['status_message'] = task_result.info.get('status', '')
    else:
        response_data['status'] = task_result.state
        
    return JsonResponse(response_data)

def directory_browser(request):
    # Base directory to start browsing from
    base_path = '/glade/campaign/'
    
    # Get the current path from query parameters, default to base_path
    current_path = request.GET.get('path', base_path)
    
    # Ensure the path is within the base directory (security measure)
    if not Path(current_path).resolve().is_relative_to(Path(base_path).resolve()):
        current_path = base_path
    
    try:
        # Get directories and files
        items = []
        for item in os.scandir(current_path):
            items.append({
                'name': item.name,
                'is_dir': item.is_dir(),
                'path': os.path.join(current_path, item.name)
            })
        
        # Sort items (directories first, then files)
        items.sort(key=lambda x: (not x['is_dir'], x['name'].lower()))
        
        # Get parent directory
        parent_path = str(Path(current_path).parent)
        if not Path(parent_path).resolve().is_relative_to(Path(base_path).resolve()):
            parent_path = None
            
    except PermissionError:
        items = []
        parent_path = str(Path(current_path).parent)
        error_message = "Permission denied to access this directory"
    except Exception as e:
        items = []
        parent_path = str(Path(current_path).parent)
        error_message = f"Error accessing directory: {str(e)}"
    
    context = {
        'current_path': current_path,
        'items': items,
        'parent_path': parent_path,
        'error_message': error_message if 'error_message' in locals() else None
    }
    
    return render(request, 'directory_browser.html', context)