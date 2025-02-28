from django.shortcuts import render, redirect
from django.http import JsonResponse
import os
from pathlib import Path
from ..celery.tasks import analyze_taxi_data, max_taxi_fare, total_taxi_fare, taxi_weather_analysis, get_glade_picture_task
from .glade_functions import get_glade_picture

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
    return render(request, 'glade_image.html')

def trigger_glade_analysis(request):
    """
    View for starting the GLADE picture task.
    """
    task = get_glade_picture_task.delay()
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