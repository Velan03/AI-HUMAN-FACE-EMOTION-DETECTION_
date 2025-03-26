# Import the necessary libraries
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from urllib.request import Request, urlopen
from flask import Flask, render_template, Response, request, redirect, flash

# Import the required Classes/Functions from Modules defined
from camera import VideoCamera
from Graphical_Visualisation import Emotion_Analysis

# Initialize the Flask app
app = Flask(__name__)

# Global configuration settings
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Utility function for file extension checking
def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

# Streaming generator function for webcam feed
def gen(camera):
    """Generator function to stream frames from the camera to the server."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Helper functions to generate responses based on the predicted emotion
def mood(result):
    """Generate a response based on the detected emotion."""
    responses = {
        "Happy": "Since you are happy, let's keep up the good mood with some amazing music!",
        "Sad": "It seems you're having a bad day, let's cheer you up with some music!",
        "Disgust": "Something has got you feeling disgusted. Let's improve your mood with some great music!",
        "Neutral": "It seems like a normal day. Let's turn it into a great one with some amazing music!",
        "Fear": "You seem scared. Some music might help!",
        "Angry": "You seem angry. Music will surely help you calm down!",
        "Surprise": "You seem surprised! Hopefully, it's good news. Let's celebrate with some music!"
    }
    return responses.get(result, 'Let’s enjoy some music!')

def provide_url(result):
    """Provide a URL based on the detected emotion."""
    urls = {
        "Happy": 'https://open.spotify.com/playlist/1BVPSd4dynzdlIWehjvkPj',
        "Sad": 'https://www.writediary.com/',
        "Disgust": 'https://open.spotify.com',
        "Neutral": 'https://www.netflix.com/',
        "Fear": 'https://www.youtube.com/watch?v=KWt2-lUpg-E',
        "Angry": 'https://www.onlinemeditation.org/',
        "Surprise": 'https://www.google.com/search?q=hotels+near+me'
    }
    return urls.get(result, '#')

def activities(result):
    """Generate activity suggestions based on the detected emotion."""
    activities_dict = {
        "Happy": '• Try out some dance moves',
        "Sad": '• Write in a journal',
        "Disgust": '• Listen to soothing music',
        "Neutral": '• Watch your favourite movie',
        "Fear": '• Get a good sleep',
        "Angry": '• Do meditation',
        "Surprise": '• Give yourself a treat'
    }
    return activities_dict.get(result, '• Take some time to relax.')

# Flask route for the homepage
@app.route('/')
def start():
    """Renders the home page."""
    return render_template('Start.html')

# Route to render the options page
@app.route('/option')
def option():
    """Renders the option page."""
    return render_template('option.html')

# Route to stream the video feed
@app.route('/video_feed')
def video_feed():
    """Returns a streamed response from the webcam."""
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for real-time emotion analysis from the webcam
@app.route('/RealTime', methods=['POST'])
def real_time():
    """Renders the RealTime page with webcam streaming."""
    return render_template('RealTime.html')

# Route to capture and analyze an image from the webcam
@app.route('/takeimage', methods=['POST'])
def take_image():
    """Captures image from the webcam, performs emotion analysis, and renders the result."""
    v = VideoCamera()
    _, frame = v.video.read()
    save_to = "static/"
    cv2.imwrite(save_to + "capture.jpg", frame)

    result = Emotion_Analysis("capture.jpg")

    # If no face is detected
    if len(result) == 1:
        return render_template('NoDetection.html', orig=result[0])

    # Process the emotion analysis result
    sentence = mood(result[3])
    activity = activities(result[3])
    link = provide_url(result[3])
    
    return render_template('Visual.html', orig=result[0], pred=result[1], bar=result[2], 
                           music=result[3], sentence=sentence, activity=activity, image=result[3], link=link)

# Route for manual image upload page
@app.route('/ManualUpload', methods=['POST'])
def manual_upload():
    """Renders the manual upload page."""
    return render_template('ManualUpload.html')

# Route to handle uploaded image from the user
@app.route('/uploadimage', methods=['POST'])
def upload_image():
    """Handles image upload, performs emotion analysis, and renders the result."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            result = Emotion_Analysis(filename)

            if len(result) == 1:
                return render_template('NoDetection.html', orig=result[0])

            # Process the emotion analysis result
            sentence = mood(result[3])
            activity = activities(result[3])
            link = provide_url(result[3])

            return render_template('Visual.html', orig=result[0], pred=result[1], bar=result[2], 
                                   music=result[3], sentence=sentence, activity=activity, image=result[3], link=link)

# Route to analyze image from URL provided by the user
@app.route('/imageurl', methods=['POST'])
def image_url():
    """Fetches an image from a URL, performs emotion analysis, and renders the result."""
    url = request.form['url']
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    # Fetch, decode and save the image
    webpage = urlopen(req).read()
    arr = np.asarray(bytearray(webpage), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    save_to = "static/"
    cv2.imwrite(save_to + "url.jpg", img)

    result = Emotion_Analysis("url.jpg")

    if len(result) == 1:
        return render_template('NoDetection.html', orig=result[0])

    # Process the emotion analysis result
    sentence = mood(result[3])
    activity = activities(result[3])
    link = provide_url(result[3])

    return render_template('Visual.html', orig=result[0], pred=result[1], bar=result[2], 
                           music=result[3], sentence=sentence, activity=activity, image=result[3], link=link)

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
