import librosa
import requests
import numpy as np
from PIL import Image
# import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data import class_names,sc_names,class_names_audio

model=load_model("./models/image_model.h5")
model_audio=load_model("./models/audio_model.h5")
IMG_SIZE = 224

def load_and_prep_image(file, img_shape):
    # preprocess the image
    file = requests.get(file, stream=True).raw #read image from url
    img=np.asarray(Image.open(file).resize((img_shape,img_shape)))
    img = img/255.  # rescale the image
    img=np.expand_dims(img, axis=0)
    return img

def make_prediction(model, filepath, class_names):
    # Imports an image located at filename, makes a prediction on it with a trained model
    img = load_and_prep_image(filepath,IMG_SIZE)
    # Make a prediction
    # (1,515) array representing probabily of each class
    prediction = model.predict(img,verbose=0)
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

def predict_bird_image(img):
    if img is not None:
        predicted_class,predicted_acc=make_prediction(model,img,class_names)
        sci_name=sc_names[predicted_class]
        return predicted_class,sci_name,predicted_acc

def predict_bird_audio(audio):
    if audio is not None:
        predicted_index,predicted_acc=make_prediction_from_audio(model_audio, audio)
        predicted_class = class_names_audio[predicted_index]
        return predicted_class,predicted_acc
