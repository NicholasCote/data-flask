from flask import render_template, request, session, redirect, url_for
from app import app, api
from app.main.stratus_py import list_all_buckets, list_bucket_objs, user_download_fileobj
from app.celery_setup.tasks import download_object
from app.nacordex.get_data import get_glade_picture
import os

@app.route('/')
def home():
    try:
        og_dir = 'glade'
        glade = os.listdir('/glade')
        return render_template('home.html', glade=glade, og_dir=og_dir)
    except:
        return render_template('home.html', glade=['No GLADE mount present'])
    
@app.route('/listdir/<og_dir>/<dir>')
def list_dir(dir, og_dir):
    try:
        og_dir = og_dir + '-' + dir
        glade = os.listdir('/' + og_dir.replace('-','/'))
        return render_template('home.html', glade=glade, og_dir=og_dir)
    except:
        return render_template('home.html')

@app.route('/templates/header.html')
def header():
    return render_template('header.html')

@app.route('/templates/navbar.html')
def navbar():
    return render_template('navbar.html')

@app.route('/stratus/login', methods=['POST'])
def stratus_login():
    session['endpoint'] = request.form['endpoint']
    session['access_id'] = request.form['access_id']
    session['secret_key'] = request.form['secret_key']
    return redirect(url_for('home'))


@app.route('/stratus/logout', methods=['POST'])
def stratus_logout():
    session.pop('access_id')
    session.pop('secret_key')
    return redirect(url_for('home'))

@app.route('/stratus/list_all', methods=['GET'])
def stratus_list_all_buckets():
    try:
        og_dir = 'glade'
        glade = os.listdir('/glade')
        return render_template('home.html', og_dir=og_dir, glade=glade, buckets = list_all_buckets(session['endpoint'], session['access_id'], session['secret_key']))
    except (TypeError, FileNotFoundError) as e:
        return render_template('home.html', glade= ['No GLADE mount present'], buckets = list_all_buckets(session['endpoint'], session['access_id'], session['secret_key']))
    
@app.route('/stratus/list_bucket_objs', methods=['GET'])
def stratus_list_all_bucket_objs():
    try:
        og_dir = 'glade'
        glade = os.listdir('/glade')
        bucket_name = request.args.get('bucket_name')
        bucket_objs = list_bucket_objs(session['endpoint'], bucket_name, session['access_id'], session['secret_key'])
        return render_template('home.html', og_dir=og_dir, glade=glade, bucket_objs = bucket_objs, bucket_name = bucket_name)
    except:
        bucket_name = request.args.get('bucket_name')
        bucket_objs = list_bucket_objs(session['endpoint'], bucket_name, session['access_id'], session['secret_key'])
        return render_template('home.html', glade=['No GLADE mount present'], bucket_objs = bucket_objs, bucket_name = bucket_name)

@app.route('/stratus/download/<bucket_name>', methods=['GET', 'POST'])
def stratus_download(bucket_name):
    if request.method == 'POST':
        selected_list = request.form.getlist('selected_object')
        for object in selected_list:
             download_object.delay(session['endpoint'], object, bucket_name, session['access_id'], session['secret_key'])
        return render_template('home.html')
    else:
        bucket_objs = list_bucket_objs(session['endpoint'], bucket_name, session['access_id'], session['secret_key'])
        return render_template('download.html', bucket_objs = bucket_objs, bucket_name = bucket_name)
    
@app.route('/stratus/download/<bucket_name>/<object>')
def object_download(bucket_name, object):
    return user_download_fileobj(session['endpoint'], object, bucket_name, session['access_id'], session['secret_key'])

@app.route('/stratus/download/<bucket_name>/filter', methods=['GET', 'POST'])
def stratus_download_filter(bucket_name):
    if request.method == 'POST':
        return render_template('home.html')
    else:
        filter = request.args.get('filter')
        bucket_objs = list_bucket_objs(session['endpoint'], bucket_name, session['access_id'], session['secret_key'])
        filtered = []
        for obj in bucket_objs:
            if filter in obj:
                filtered.append(obj)
        return render_template('download.html', bucket_objs = filtered, bucket_name = bucket_name)
    
@app.route('/glade/picture')
def glade_image():
    image = get_glade_picture()
    return render_template('image.html', image=image)