from flask import Flask, render_template, Response, make_response, url_for, request
import cv2
import hollistic1
import pose_milaune 
import os
import uuid

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')


"bhujangasana" , "padmasana" , "shavasana","tadasana","Virbhadrasana","vrikshyasana" 
@app.route('/Classify')

def run_script():
    res = hollistic1.open_camera()
    if res == "bhujangasana":
        return render_template('bhuj.html')
    elif res == "padmasana":
        return render_template('pad.html',text=res)
    elif res == "shavasana":
        return render_template('shav.html')
    elif res == "tadasana":
        return render_template('tad.html')
    elif res == "Virbhadrasana":
        return render_template('vribha.html')
    elif res == "vrikshyasana":
        return render_template('vrik.html')
    else:
        return "Unknown Pose"


@app.route('/static/statics/bhuj.html')

def bhu_practise():
    return render_template('bhuj.html')

@app.route('/practiseBhuj')
def practiseBhuj():
    return Response(pose_milaune.feedback_bhuj(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/statics/pad.html')

def pad_practise():
    return render_template('pad.html')

@app.route('/practisePad')
def practisePad():
    return Response(pose_milaune.feedback_padmasan(),mimetype='multipart/x-mixed-replace; boundary=frame')
    

@app.route('/static/statics/tad.html')

def tad_practise():
    return render_template('tad.html')

@app.route('/practiseTad')
def practiseTad():
    return Response(pose_milaune.feedback_tadasana(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/static/statics/virbha.html')

def vrib_practise():
    return render_template('vribha.html')

@app.route('/practiseVribha')
def practiseVribha():
    return Response(pose_milaune.feedback_virbhadra(),mimetype='multipart/x-mixed-replace; boundary=frame')
    


@app.route('/static/statics/vrik.html')

def vrik_practise():
    return render_template('vrik.html')

@app.route('/practiseVrik')
def practiseVrik():
    return Response(pose_milaune.feedback_vrikasana(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/statics/shav.html')

def shav_practise():
    return render_template('shav.html')

@app.route('/practiseShav')
def practiseShav():
    return Response(pose_milaune.feedback_shavasana(),mimetype='multipart/x-mixed-replace; boundary=frame')

app.config['UPLOAD_FOLDER'] = '/upload'
@app.route('/upload_file', methods=['POST'])
def upload_file():
    # Get the uploaded file
    file = request.files['file']
    filename = file.filename

    # Save the file to disk
    # Save the file to disk
    # file_path = os.path.join(app.config['UPLOAD_FOLDER'])
    # file.save(file_path)
    file_path = os.path.join('static/upload',filename)
    file.save(file_path)

    # Process the video and make predictions
    res = hollistic1.predict_single_action(file_path, 20)

    # Return the prediction
    if res == "bhujangasana":
        return render_template('bhuj.html')
    elif res == "padmasana":
        return render_template('pad.html')
    elif res == "shavasana":
        return render_template('shav.html')
    elif res == "tadasana":
        return render_template('tad.html')
    elif res == "Virbhadrasana":
        return render_template('vribha.html')
    elif res == "vrikshyasana":
        return render_template('vrik.html')
    else:
        return "Unknown Pose"

    # Do something with the file, e.g. read its contents or process it

    # Return a response to the client






if __name__ == '__main__':
    app.run(debug=True)