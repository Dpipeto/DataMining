from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def home():
    a=1
    return "Hello, World!"
@app.route('/inicio/')
def index():
    return render_template('index.html')