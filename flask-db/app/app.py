from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix

# TEMP: stripped down to a source-IP echo service while this app sits behind
# traefik-external to test the mlc1 LoadBalancer fix. The NFS home browser
# (/browse, /whoami) is removed so nothing from the home mount is exposed
# publicly -- restore it from git history once the test is done.

app = Flask(__name__)
# Trust one X-Forwarded-For hop from traefik so the access log /
# request.remote_addr reflect the client IP traefik forwards. The header
# reaching the pod has a single entry, so x_for=1 is correct (x_for>1
# overshoots to the raw TCP peer).
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)


def ip_info():
    return {
        'x_forwarded_for': request.headers.get('X-Forwarded-For'),
        'x_real_ip': request.headers.get('X-Real-IP'),
        'forwarded': request.headers.get('Forwarded'),
        'remote_addr': request.remote_addr,
        'access_route': list(request.access_route),
    }


@app.route('/')
def index():
    return jsonify(ip_info())


@app.route('/debug/ip')
def debug_ip():
    return jsonify(ip_info())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
