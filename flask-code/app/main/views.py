from flask import render_template, request, session, redirect, url_for, flash
from app import app, github, User, db
from app.main.stratus_py import list_all_buckets, list_bucket_objs
from app.nacordex.get_data import get_glade_picture
from git import Repo
import os, shutil
import fileinput
from distutils.dir_util import copy_tree

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/object_browser')
def object_browser():
    return render_template('object_browser.html')

@app.route('/templates/header.html')
def header():
    return render_template('header.html')

@app.route('/templates/navbar.html')
def navbar():
    return render_template('navbar.html')

@app.route('/github/login')
def github_login():
    return github.authorize()

@app.route('/github-callback')
@github.authorized_handler
def authorized(oauth_token):
    next_url = request.args.get('next') or url_for('home')
    if oauth_token is None:
        flash("Authorization failed.")
        return redirect(next_url)

    user = User.query.filter_by(github_access_token=oauth_token).first()
    if user is None:
        user = User(oauth_token)
        db.add(user)

    user.github_access_token = oauth_token
    db.commit()
    return redirect(next_url)

@app.route('/stratus/login', methods=['POST'])
def stratus_login():
    session['endpoint'] = request.form['endpoint']
    session['access_id'] = request.form['access_id']
    session['secret_key'] = request.form['secret_key']
    return redirect(url_for('object_browser'))


@app.route('/stratus/logout', methods=['POST'])
def stratus_logout():
    session.pop('access_id')
    session.pop('secret_key')
    return redirect(url_for('object_browser'))

@app.route('/stratus/list_all', methods=['GET'])
def stratus_list_all_buckets():
    try:
        og_dir = 'glade'
        glade = os.listdir('/glade')
        return render_template('object_browser.html', og_dir=og_dir, glade=glade, buckets = list_all_buckets(session['endpoint'], session['access_id'], session['secret_key']))
    except (TypeError, FileNotFoundError) as e:
        return render_template('object_browser.html', glade= ['No GLADE mount present'], buckets = list_all_buckets(session['endpoint'], session['access_id'], session['secret_key']))
    
@app.route('/stratus/list_bucket_objs', methods=['GET'])
def stratus_list_all_bucket_objs():
    try:
        og_dir = 'glade'
        glade = os.listdir('/glade')
        bucket_name = request.args.get('bucket_name')
        bucket_objs = list_bucket_objs(session['endpoint'], bucket_name, session['access_id'], session['secret_key'])
        return render_template('object_browser.html', og_dir=og_dir, glade=glade, bucket_objs = bucket_objs, bucket_name = bucket_name)
    except:
        bucket_name = request.args.get('bucket_name')
        bucket_objs = list_bucket_objs(session['endpoint'], bucket_name, session['access_id'], session['secret_key'])
        return render_template('object_browser.html', glade=['No GLADE mount present'], bucket_objs = bucket_objs, bucket_name = bucket_name)
    
@app.route('/glade/picture')
def glade_image():
    image = get_glade_picture()
    return render_template('image.html', image=image)

@app.route('/addGH', methods=['GET', 'POST'])
def add_gh():
    if request.method == 'POST':
        git_repo = request.form['git_repo']
        app_name = request.form['app_name']
        app_path = request.form['app_path']
        containerfile_dir = request.form['containerfile_dir']
        containerfile_name = request.form['containerfile_name']
        harbor_project = request.form['harbor_project']
        harbor_robot = request.form['harbor_robot']
        containerimg_name = request.form['containerimg_name']
        port_no = request.form['port_no']
        mem = request.form['mem']
        cpu = request.form['cpu']
        replicas = request.form['replicas']
        return render_template('add_gh_confirm.html', git_repo=git_repo, app_name=app_name, app_path=app_path, containerfile_dir=containerfile_dir, containerfile_name=containerfile_name, harbor_project=harbor_project, harbor_robot=harbor_robot, containerimg_name=containerimg_name, port_no=port_no, mem=mem, cpu=cpu, replicas=replicas)
    else:
        return render_template('add_gh.html')
    
@app.route('/addGHconfirm', methods=['POST'])
def add_gh_confirm():
    git_repo = request.form['git_repo']
    app_name = request.form['app_name']
    app_path = request.form['app_path']
    containerfile_dir = request.form['containerfile_dir']
    containerfile_name = request.form['containerfile_name']
    harbor_project = request.form['harbor_project']
    harbor_robot = request.form['harbor_robot']
    containerimg_name = request.form['containerimg_name']
    port_no = request.form['port_no']
    mem = request.form['mem']
    cpu = request.form['cpu']
    replicas = request.form['replicas']
    git_template_url = "https://github.com/NicholasCote/GHA-helm-template.git"
    temp_dir = app_name + "_temp"
    user_temp_dir = app_name + "_user"
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
    else:
        os.makedirs(user_temp_dir)
    if os.path.isdir(user_temp_dir):
        shutil.rmtree(user_temp_dir)
        os.makedirs(user_temp_dir)
    else:
        os.makedirs(user_temp_dir)
    temp_repo = Repo.clone_from(git_template_url, temp_dir)
    user_repo = Repo.clone_from(git_repo, user_temp_dir)
    with fileinput.FileInput(app_name + "_temp/app-helm-chart/values.yaml", inplace=True) as values:
        for line in values:
            if "<app_name>" in line:
                new_line = line.replace("<app_name>", app_name)
                print(new_line, end='')
            elif "<app_path>" in line:
                new_line = line.replace("<app_path>", app_path)
                print(new_line, end='')
            elif "<replicas>" in line:
                new_line = line.replace("<replicas>", replicas)
                print(new_line, end='')
            elif "<container_image>" in line:
                new_line = line.replace("<container_image>", "hub.k8s.ucar.edu/" + harbor_project + "/" + containerimg_name)
                print(new_line, end='')
            elif "<container_port>" in line:
                new_line = line.replace("<container_port>", port_no)
                print(new_line, end='')
            elif "<memory>" in line:
                new_line = line.replace("<memory>", mem)
                print(new_line, end='')
            elif "<CPU>" in line:
                new_line = line.replace("<CPU>", cpu)
                print(new_line, end='')
            else:
                print(line, end='')
    with fileinput.FileInput(app_name + "_temp/.github/workflows/build-push-img.yaml", inplace=True) as values:
        for line in values:
            if "<container_file_path>" in line:
                new_line = line.replace("<container_file_path>", containerfile_name)
                new_line = new_line.replace("<harbor_project>", harbor_project)
                new_line = new_line.replace("<container_image_name>", containerimg_name)
                if containerfile_dir == "/":
                    new_line = new_line.replace("<container_dir_path>", "")
                else:
                    new_line = new_line.replace("<container_dir_path>", containerfile_dir)
                print(new_line, end='')
            elif "<harbor_project>" in line:
                new_line = line.replace("<harbor_project>", harbor_project)
                new_line = new_line.replace("<container_image_name>", containerimg_name)
                print(new_line, end='')
            elif "<harbor_robot_account>" in line:
                new_line = line.replace("<harbor_robot_account>", harbor_robot)
                print(new_line, end='')
            else:
                print(line, end='')
    copy_tree(temp_dir + "/app-helm-chart", user_temp_dir + "/app-helm-chart")
    copy_tree(temp_dir + "/.github", user_temp_dir + "/.github")
    user_repo.git.add(all=True)
    user_repo.index.commit("Add custom Helm chart and GitHub Action from template")
    user_repo.git.push()
    shutil.rmtree(temp_dir)
    shutil.rmtree(user_temp_dir)
    return render_template('home.html')