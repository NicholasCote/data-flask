from flask import render_template, request, session, flash
from app import app
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
    github_username = request.form['github_username']
    github_access_token = request.form['github_access_token']
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
    remote = "https://" + github_username + ":" + github_access_token + "@" + git_repo.replace('https://','')
    session
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
    try:
        copy_tree(temp_dir + "/app-helm-chart", user_temp_dir + "/app-helm-chart")
        copy_tree(temp_dir + "/.github", user_temp_dir + "/.github")
        user_repo.git.add(all=True)
        user_repo.index.commit("Add custom Helm chart and GitHub Action from template")
        user_repo = Repo(user_temp_dir)
        origin = user_repo.remote(name="origin")
        origin.push()
        shutil.rmtree(temp_dir)
        shutil.rmtree(user_temp_dir)
        return render_template('home.html')
    except Exception as e:
        flash(e)
        shutil.rmtree(temp_dir)
        shutil.rmtree(user_temp_dir)
        return render_template('home.html')
