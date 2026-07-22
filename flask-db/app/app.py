import os
from flask import Flask, request, jsonify, render_template, abort

app = Flask(__name__)

# The NFS home mount root inside the pod. Everything the browser can see is
# confined to this directory -- see safe_resolve() below. Set via the
# BROWSE_ROOT env in the Rollout (defaults to /home).
BROWSE_ROOT = os.environ.get('BROWSE_ROOT', '/home')


def safe_resolve(rel_path):
    """Resolve a user-supplied relative path against BROWSE_ROOT and refuse
    anything that escapes it. Returns an absolute path guaranteed to be inside
    BROWSE_ROOT, or raises ValueError.

    realpath() resolves symlinks too, so a symlink inside the mount that points
    out of it (e.g. into / or another export) also gets rejected -- important on
    a real home dir where dotfile symlinks are common."""
    root = os.path.realpath(BROWSE_ROOT)
    # Treat the incoming path as relative regardless of leading slashes.
    candidate = os.path.realpath(os.path.join(root, rel_path.lstrip('/')))
    if candidate != root and not candidate.startswith(root + os.sep):
        raise ValueError('path escapes browse root')
    return candidate


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/whoami')
def whoami():
    """Report the process identity as seen inside the container. Pair this with
    the uid/gid columns in /browse: if the pod runs as uid 41188 but files show
    as 'nobody', that's an idmapd (Domain = ucar.edu) issue on the node, not a
    securityContext one."""
    try:
        groups = os.getgroups()
    except OSError:
        groups = []
    return jsonify({
        'uid': os.getuid(),
        'gid': os.getgid(),
        'groups': groups,
        'browse_root': BROWSE_ROOT,
    })


@app.route('/browse')
def browse():
    rel = request.args.get('path', '')
    try:
        target = safe_resolve(rel)
    except ValueError:
        abort(403)

    if not os.path.exists(target):
        abort(404)
    if not os.path.isdir(target):
        abort(400, description='Not a directory')

    entries = []
    try:
        with os.scandir(target) as it:
            for e in it:
                try:
                    st = e.stat(follow_symlinks=False)
                    entries.append({
                        'name': e.name,
                        'is_dir': e.is_dir(follow_symlinks=False),
                        'is_link': e.is_symlink(),
                        'size': st.st_size,
                        'uid': st.st_uid,
                        'gid': st.st_gid,
                        'mode': oct(st.st_mode & 0o777),
                    })
                except OSError:
                    # Entry we can't stat (permissions, broken symlink) -- list
                    # the name so the UID/GID story is still visible, mark it
                    # unreadable.
                    entries.append({
                        'name': e.name,
                        'is_dir': False,
                        'is_link': False,
                        'size': None,
                        'uid': None,
                        'gid': None,
                        'mode': None,
                    })
    except PermissionError:
        abort(403, description='Permission denied reading directory')

    entries.sort(key=lambda x: (not x['is_dir'], x['name'].lower()))

    # Relative path from root, for display and breadcrumb "up" navigation.
    root = os.path.realpath(BROWSE_ROOT)
    rel_display = '' if target == root else os.path.relpath(target, root)
    parent = None
    if target != root:
        parent = os.path.relpath(os.path.dirname(target), root)
        if parent == '.':
            parent = ''

    return jsonify({
        'root': BROWSE_ROOT,
        'path': rel_display,
        'parent': parent,
        'entries': entries,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)