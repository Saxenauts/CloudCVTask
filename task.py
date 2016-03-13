from flask import Flask, render_template, request, redirect, jsonify
from flask_bootstrap import Bootstrap

app = Flask(__name__)
bootstrap = Bootstrap(app)

@app.route('/', methods = ['GET'])
def initial():
    return render_template("index.html")


@app.route('/image', methods = ['POST'])
def image():
    print (request.form['src'])
    return render_template("index.html")
    
@app.route('/ques', methods = ['POST'])
def ques():
    question = request.form['ques']
    print ("the ques is: ")
    print (question)
    return render_template("index.html")
    
if __name__ == '__main__':
    app.run(debug = True)
    
    
