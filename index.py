from flask import Flask,render_template,request
from utility import predict_bird,predict_bird_audio
import pandas as pd
import csv
app=Flask(__name__)

@app.route("/classify",methods=["GET"])
def classify():
    return render_template("classify.html")
       

@app.route("/",methods=["GET"])
def home():
    if request.method=="GET":
        return render_template("index.html")
    
@app.route("/about",methods=["GET"])
def about():
    if request.method=="GET":
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
    
@app.route("/threats",methods=["GET"])
def threat():
    if request.method=="GET":
        return render_template("threat.html")

@app.route("/threat/<string:id>",methods=["GET"])
def threat_pages(id):
    
    if request.method=="GET":
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

@app.route("/blogs",methods=["GET"])
def blogs():
    if request.method=="GET":
        return render_template("blog.html")

@app.route("/blog/<int:blog_id>",methods=["GET"])
def blog_pages(blog_id):
    if request.method=="GET":
        return render_template(f"blog {blog_id}.html")

@app.route("/result-image",methods=["POST"])
def result_image():
    img_name="./static/images/pictures/abc.jpg"
    request.files.get("bird").save(img_name)
    act_name,sci_name,acc=predict_bird(img_name)
    img_src="./static/images/pictures/abc.jpg"
    return render_template("result.html",act_name=act_name,sci_name=sci_name,acc=acc,img_src=img_src)

@app.route("/result-audio",methods=["POST"])
def result_audio():
    aud_name="abc.mp3"
    request.files.get("file").save(aud_name)
    act_name,acc=predict_bird_audio(aud_name)
    return render_template("result_audio.html",act_name=act_name,acc=acc)

if __name__=="__main__":
    app.run(debug=False,host='0.0.0.0')

