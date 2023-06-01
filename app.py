import csv
import librosa
import requests
import matplotlib as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask,render_template,request
from data import class_names,sc_names,class_names_audio

app=Flask(__name__)

API_KEY='94a27f12b46d11ab5fd7a4cfbf9650ea'

model=tf.keras.models.load_model("./models/image_model.h5")
model_audio=tf.keras.models.load_model("./models/audio_model.h5")
IMG_SIZE = 224

def load_and_prep_image(filename, img_shape=IMG_SIZE):
    # preprocess the image
    # img = tf.io.read_file(filename)  # read image from file
    # img = tf.io.decode_image(img)  # decode the image to a tensor
    img = requests.get(filename).content #read image from url
    img = tf.io.decode_image(img,channels=3)  # decode the image to a tensor
    img = tf.image.resize(img, size=[img_shape, img_shape])  # resize the image
    img = img/255.  # rescale the image
    return img

def make_prediction(model, filename, class_names):
    # Imports an image located at filename, makes a prediction on it with a trained model
    img = load_and_prep_image(filename)
    # Make a prediction
    # (1,515) array representing probabily of each class
    prediction = model.predict(tf.expand_dims(img, axis=0),verbose=0)
    # Get the predicted class index
    predicted_index = prediction.argmax() # taking the class index with the highest probability
    predicted_acc=round(prediction.max()*100,2)#the highest probability value
    # Get the predicted class
    predicted_class = class_names[predicted_index]
    return predicted_class,predicted_acc
#Audio
def load_and_prep_image_audio(filename):
    # preprocess the image
    img = tf.io.read_file(filename) # read image
    img = tf.image.decode_image(img) # decode the image to a tensor
    img = tf.image.resize(img, size=[150, 154]) # resize the image
    img = img/255 # rescale the image
    return img

def load_and_prep_audio(filepath):        
    clip, sample_rate = librosa.load(filepath,offset=0, duration=5)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    fig=plt.figure(figsize=(153/80, 150/80))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    plt.axis(False)
    plt.savefig('pred.jpg',bbox_inches='tight',pad_inches=0,dpi=114,facecolor='black')
    return 'pred.jpg'

def make_prediction_from_audio(model_audio, filename):
    # Imports an image located at filename, makes a prediction on it with a trained model
    # Import the target image and preprocess it
    spectrogram=load_and_prep_audio(filename)
    img = load_and_prep_image_audio(spectrogram)
    # Make a prediction
    prediction = model_audio.predict(tf.expand_dims(img, axis=0),verbose=0)
    predicted_index = prediction.argmax() # if more than one output, take the max
    prediction_acc=round(prediction.max()*100,2)
    return predicted_index,prediction_acc

def predict_bird(img):
    if img is not None:
        predicted_class,predicted_acc=make_prediction(model,img,class_names)
        sci_name=sc_names[predicted_class]
        return predicted_class,sci_name,predicted_acc

def predict_bird_audio(audio):
    if audio is not None:
        predicted_index,predicted_acc=make_prediction_from_audio(model_audio, audio)
        predicted_class = class_names_audio[predicted_index]
        return predicted_class,predicted_acc

@app.route("/")
def home():    
    return render_template("index.html")
    
@app.route("/classify")
def classify():
    return render_template("classify.html")
       
@app.route("/about")
def about():    
    return render_template("about.html")
    
@app.route("/contact",methods=["GET","POST"])
def contact():
    if request.method=="GET":
        return render_template("contact.html")
    else:
        name=request.form.get("fname")
        email=request.form.get("email")
        num=request.form.get("num")
        message=request.form.get("message")
        lines=[]
        with open("./static/Data/contact.csv","r") as f:
            csvFile = csv.reader(f)
            for line in csvFile:
                lines.append(line)
        lines.append([name,email,num,message])
        with open("./static/Data/contact.csv", 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerows(lines)
        return render_template("contact.html")
    
@app.route("/threats")
def threat():
    return render_template("threat.html")

@app.route("/threat/<string:id>")
def threat_pages(id):
    results=[]
    if id=="ce":
        df = pd.read_excel('./static/Data/threat1.xlsx')
    elif id=="end":
        df = pd.read_excel('./static/Data/threat2.xlsx')
    elif id=="vul":
        df = pd.read_excel('./static/Data/threat3.xlsx')
    elif id=="nt":
        df = pd.read_excel('./static/Data/threat4.xlsx')
    else:
        df = pd.read_excel('./static/Data/threat5.xlsx')
    for ind in df.index:
        results.append([df["BIRD NAME"][ind],df["SCIENTIFIC NAME"][ind],df["DISTRIBUTION"][ind],df["MAJOR THREATS"][ind],df["CATEGORY"][ind]])
    return render_template("threat_table.html",results=results)

@app.route("/blogs")
def blogs():
    return render_template("blog.html")

@app.route("/blog/<int:blog_id>")
def blog_pages(blog_id):    
    return render_template(f"blog {blog_id}.html")

@app.route("/result-image",methods=["GET","POST"])
def result_image():
    # img_name="./static/images/pictures/abc.jpg"
    # request.files.get("bird").save(img_name)
    if request.method=='POST':
        url='https://api.imgbb.com/1/upload'
        image_file = request.files.get("bird")
        data = {"key": API_KEY,}
        files = {"image": image_file,}
        # Send the POST request to upload the image to ImgBB
        response = requests.post(url, data=data, files=files)
        # Extract the URL of the uploaded image from the response
        if response.status_code == 200:
            response_json = response.json()
            img_url = response_json["data"]["url"]
            act_name,sci_name,acc=predict_bird(img_url)
            return render_template("result.html",act_name=act_name,sci_name=sci_name,acc=acc,img_src=img_url)
    return None

@app.route("/result-audio",methods=["GET","POST"])
def result_audio():
    if request.method=="POST":
        aud_name="abc.mp3"
        request.files.get("file").save(aud_name)
        act_name,acc=predict_bird_audio(aud_name)
        return render_template("result_audio.html",act_name=act_name,acc=acc)
    return None
    
if __name__=="__main__":
    app.run(debug=False)