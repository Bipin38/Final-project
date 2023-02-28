from flask import Flask, render_template, Response
import cv2
import hollistic1
import pose_milaune 

app = Flask(__name__)

cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_pose')
def classify_pose():
    return render_template('try.html')

@app.route('/static/statics/bhuj.html#practice')

def run_script():
    # return Response(hollistic1.open_camera(),mimetype='multipart/x-mixed-replace; boundary=frame')
    res = hollistic1.open_camera()
    return res

@app.route('/static/statics/bhuj.html')

def bhu_practise():
    return Response(pose_milaune.feedback_bhuj(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/statics/pad.html')

def pad_practise():
    return Response(pose_milaune.feedback_padmasan(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/statics/tad.html')

def tad_practise():
    return Response(pose_milaune.feedback_tadasana(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/statics/vribha.html')

def vribha_practise():
    return Response(pose_milaune.feedback_virbhadra(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/statics/vrik.html')

def vrik_practise():
    return Response(pose_milaune.feedback_vrikasana(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/statics/shav.html')

def shav_practise():
    return Response(pose_milaune.feedback_shavasana(),mimetype='multipart/x-mixed-replace; boundary=frame')






if __name__ == '__main__':
    app.run(debug=True)