import os
import csv
import requests
import pandas as pd
from werkzeug.utils import secure_filename
from flask import Flask,render_template,request
from utility import predict_bird_image,predict_bird_audio


UPLOAD_FOLDER = './static/images/uploads/'
API_KEY='94a27f12b46d11ab5fd7a4cfbf9650ea'

app=Flask(__name__,static_folder='static')

app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

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

@app.route("/result-image",methods=["POST"])
def result_image():
    if request.method=='POST':
        #Saving the image of the bird and then predicting
        if 'bird' not in request.files:
            return render_template("classify.html")
        
        file = request.files['bird']
        if file.filename == '':
            return render_template("classify.html")
        
        if file:
            filename = secure_filename(file.filename)  #Use this werkzeug method to secure filename. 
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            
            img_path=os.path.join(app.config['UPLOAD_FOLDER'],filename)
            # request.files.get("bird").save(img_path)
            act_name,sci_name,acc=predict_bird_image(img_path)
            return render_template("result.html",act_name=act_name,sci_name=sci_name,acc=acc,img_src=img_path)

        #Uploading the image in the cloud and getting the url
        # url='https://api.imgbb.com/1/upload'
        # image_file = request.files.get("bird")
        # data = {"key": API_KEY,}
        # files = {"image": image_file,}
        # # Send the POST request to upload the image to ImgBB
        # response = requests.post(url, data=data, files=files)
        # # Extract the URL of the uploaded image from the response
        # if response.status_code == 200:
        #     response_json = response.json()
        #     img_url = response_json["data"]["url"]
        #     act_name,sci_name,acc=predict_bird_image(img_url)
        #     return render_template("result.html",act_name=act_name,sci_name=sci_name,acc=acc,img_src=img_url)
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