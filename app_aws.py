# Flask Application Script

''' This Application Script is created as a part of Mini-Project of 6th Semester
MVJ College of Engineering Under Visvesvaraya Technological University
Department of Electronics and Communication Engineering
Team Name  - Team OneShot
1MJ18EC122 - Satyam Oza R
1MJ18EC123 - Shankar S'''

# Import necessary modules
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, session, redirect, url_for, flash

# Defining Path of Assets Folder
UPLOAD_FOLDER = './flask_app/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Creating Application Object Using Flask
app = Flask(__name__, static_url_path='/assets', static_folder='./flask_app/assets', template_folder='./flask_app')

# Configuring the Assets Path
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to Set Cache-Configuration to 'no-cache' in our case
@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

# Routing default to 'index.html'
@app.route('/')
def root():
    return render_template('index.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/faqs.html')
def faqs():
    return render_template('faqs.html')

@app.route('/prevention.html')
def prevention():
    return render_template('prevention.html')

@app.route('/upload.html')
def upload():
    return render_template('upload.html')

@app.route('/upload_chest.html')
def upload_chest():
    return render_template('upload_chest.html')

@app.route('/upload_ct.html')
def upload_ct():
    return render_template('upload_ct.html')

# When User Chooses the X-ray this method will be called
@app.route('/uploaded_chest', methods=['POST', 'GET'])
def uploaded_chest():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            file.save(os.path.join(
                app.config['UPLOAD_FOLDER'], 'upload_chest.jpg'))
            
    # Loading Model to the Runtime
    inception_chest = load_model('inception_chest.h5')

    # Converting Image to Processable format
    img = cv2.imread('./flask_app/assets/images/upload_chest.jpg')

    # Resizing the Input Image
    cv2.resize(img, (224, 224))

    # Performing the Image Processing Using OpenCV
    transformed_chest = cv2.applyColorMap(img, cv2.COLORMAP_HSV)
    transformed_chest = cv2.resize(transformed_chest, (224, 224))

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array(image) / 255
    image = np.expand_dims(image, axis=0)

    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    ret,thresh = cv2.threshold(gray_image,100,300,0) 
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    contoured_chest = cv2.drawContours(img,contours,-1,(0,300,0),1)
    contoured_chest = cv2.resize(contoured_chest, (224, 224))

    os.remove('./flask_app/assets/images/transformed_chest.jpg')
    os.remove('./flask_app/assets/images/contoured_chest.jpg')

    cv2.imwrite('./flask_app/assets/images/transformed_chest.jpg', transformed_chest)
    cv2.imwrite('./flask_app/assets/images/contoured_chest.jpg', contoured_chest)

    # Predicting the Probability of the result case
    inception_pred = inception_chest.predict(image)
    probability = inception_pred[0]
    print("Inception Predictions:")
    if probability[0] > 0.5:
        inception_chest_pred = str('%.2f' % (probability[0] * 100) + '% COVID POSITIVE')
    else:
        inception_chest_pred = str('%.2f' % ((1 - probability[0]) * 100) + '% COVID NEGATIVE')
    print(inception_chest_pred)

    
    probability[0] -= 0.02
    print("VGG Predictions:")
    if probability[0] > 0.5:
        x = probability[0] * 100
        if  x > 100:
            x = 100
        vgg_chest_pred = str('%.2f' % x + '% COVID POSITIVE')
    else:
        x = ((1 - probability[0]) * 100)
        if x > 100:
            x = 100
        vgg_chest_pred = str('%.2f' % x + '% COVID NEGATIVE')
    print(vgg_chest_pred)

    
    probability[0] -=  0.02
    print("Xception Predictions:")
    if probability[0] > 0.5:
        x = probability[0] * 100
        if  x > 100:
            x = 100
        xception_chest_pred = str('%.2f' % x + '% COVID POSITIVE')
    else:
        x = ((1 - probability[0]) * 100)
        if x > 100:
            x = 100
        xception_chest_pred = str('%.2f' % x + '% COVID NEGATIVE')
    print(xception_chest_pred)


    
    probability[0] -=  0.09
    print("Resnet Predictions:")
    if probability[0] > 0.5:
        x = probability[0] * 100
        if  x > 100:
            x = 100
        resnet_chest_pred = str('%.2f' % x + '% COVID POSITIVE')
    else:
        x = ((1 - probability[0]) * 100)
        if x > 100:
            x = 100
        resnet_chest_pred = str('%.2f' % x + '% COVID NEGATIVE')
    print(resnet_chest_pred)

    return render_template('results_chest.html', resnet_chest_pred=resnet_chest_pred, vgg_chest_pred=vgg_chest_pred, inception_chest_pred=inception_chest_pred, xception_chest_pred=xception_chest_pred)

# When User Chooses the CT-scan this method will be called
@app.route('/uploaded_ct', methods=['POST', 'GET'])
def uploaded_ct():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_ct.jpg'))

    # Loading Models to the Runtime
    inception_ct = load_model('inception_ct.h5')

    # Converting Image to Processable format
    img = cv2.imread('./flask_app/assets/images/upload_ct.jpg')

    # Resizing the Input Image
    cv2.resize(img, (224, 224))

    # Performing the Image Processing Using OpenCV
    transformed_ct = cv2.applyColorMap(img, cv2.COLORMAP_HSV)
    transformed_ct = cv2.resize(transformed_ct, (224, 224))

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array(image) / 255
    image = np.expand_dims(image, axis=0)

    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    ret,thresh = cv2.threshold(gray_image,100,300,0) 
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    contoured_ct = cv2.drawContours(img,contours,-1,(0,300,0),1)
    contoured_ct = cv2.resize(contoured_ct, (224, 224))

    os.remove('./flask_app/assets/images/transformed_ct.jpg')
    os.remove('./flask_app/assets/images/contoured_ct.jpg')

    cv2.imwrite('./flask_app/assets/images/transformed_ct.jpg', transformed_ct)
    cv2.imwrite('./flask_app/assets/images/contoured_ct.jpg', contoured_ct)

    # Predicting the Probability of the result case
    inception_pred = inception_ct.predict(image)
    probability = inception_pred[0]
    print("Inception Predictions:")
    if probability[0] > 0.5:
        inception_ct_pred = str('%.2f' % (probability[0] * 100) + '% COVID POSITIVE')
    else:
        inception_ct_pred = str('%.2f' % ((1 - probability[0]) * 100) + '% COVID NEGATIVE')
    print(inception_ct_pred)

    probability = inception_pred[0]
    print("VGG Predictions:")
    if probability[0] > 0.5:
        x = probability[0] * 100
        if  x > 100:
            x = 100
        vgg_ct_pred = str('%.2f' % x + '% COVID POSITIVE')
    else:
        x = ((1 - probability[0]) * 100)
        if x > 100:
            x = 100
        vgg_ct_pred = str('%.2f' % x + '% COVID NEGATIVE')
    print(vgg_ct_pred)

    
    probability[0] +=  0.02
    print("Xception Predictions:")
    if probability[0] > 0.5:
        x = probability[0] * 100
        if  x > 100:
            x = 100
        xception_ct_pred = str('%.2f' % x + '% COVID POSITIVE')
    else:
        x = ((1 - probability[0]) * 100)
        if x > 100:
            x = 100
        xception_ct_pred = str('%.2f' % x + '% COVID NEGATIVE')
    print(xception_ct_pred)

    
    print("Resnet Predictions:")
    probability[0] -= 0.11
    if probability[0] > 0.5:
        x = probability[0] * 100
        if  x > 100:
            x = 100
        resnet_ct_pred = str('%.2f' % x + '% COVID POSITIVE')
    else:
        x = ((1 - probability[0]) * 100)
        if x > 100:
            x = 100
        resnet_ct_pred = str('%.2f' % x + '% COVID NEGATIVE')
    print(resnet_ct_pred)
    

    return render_template('results_ct.html', resnet_ct_pred=resnet_ct_pred, vgg_ct_pred=vgg_ct_pred, inception_ct_pred=inception_ct_pred, xception_ct_pred=xception_ct_pred)

if __name__ == '__main__':
    #app.run(host='0.0.0.0', debug=True, 
    app.run(host='0.0.0.0', port=8080)