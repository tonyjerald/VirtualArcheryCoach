from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response
from werkzeug.utils import secure_filename
import os
import cv2
import PoseModule

app = Flask(__name__)

# Configure file upload
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Route to the home page
@app.route('/')
def index():


    return render_template('index.html')
@app.route('/<name>')
def redirect(name):
    return render_template('index.html')

# Route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    index()
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # print(filepath)
        file.save(filepath)

        draw = request.form['draw']
        colour = request.form['colour']
        frames_per_second = int(request.form['frames_per_second'])
        position_id1 = int(request.form['position_id1'])
        position_id2 = int(request.form['position_id2'])
        position_id3 = int(request.form['position_id3'])

        return Response(PoseModule.main(filepath,colour, draw, position_id1, position_id2, position_id3, frames_per_second), mimetype='multipart/x-mixed-replace; boundary=frame')


# Route to serve the uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
