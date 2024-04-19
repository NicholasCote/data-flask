from app import app
from .get_data import get_glade_picture
from flask import render_template
    
@app.route('/glade/picture')
def glade_image():
    image = get_glade_picture()
    return render_template('image.html', image=image)