from django.shortcuts import render, redirect
from django.http import JsonResponse
from ..celery.tasks import analyze_taxi_data

def home(request):
    return render(request, 'base.html')

def trigger_analysis(request):
    task = analyze_taxi_data.delay()
    return JsonResponse({'task_id': str(task.id)})

def check_task_status(request, task_id):
    task = analyze_taxi_data.AsyncResult(task_id)
    if task.ready():
        return JsonResponse({'status': 'completed', 'result': task.get()})
    return JsonResponse({'status': 'pending'})

def analysis_view(request):
    return render(request, 'analysis.html')
