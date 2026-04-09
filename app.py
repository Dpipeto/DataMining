from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/Generos/')
def generos():
    return render_template('generos.html')

@app.route('/Generos/Criticos/')
def criticos():
    return render_template('criticos.html')