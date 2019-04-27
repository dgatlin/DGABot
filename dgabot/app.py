#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask
from flask import jsonify
from dgabot import DGABot

app = Flask(__name__)

# Instantiate the object and model 
dgabot = DGABot()  
dgabot.load_model()


@app.route('/')
def index():
    """
    Base Path 
    """
    return "Cybersecurity + Machine Leanring Rocks!"


@app.route('/train/<int:max_epoch>/<int:nfolds>/<int:batch_size>')
def train(max_epoch,nfolds,batch_size):
    """
    Trains a new model based on the inputs  
    """
    msg = "A New Model has beenTrained"
    result = {'message':msg, 'max_epoch': max_epoch,'nfolds':nfolds,
              'batch_size':batch_size}
    dgabot.train_model(max_epoch,nfolds,batch_size)
    return jsonify(result)


@app.route('/evaluate/')
def evaluate():
    """
    Evaluate the model and return classification metrics  
    """
    result= dgabot.evaluate_model()
    return jsonify(result)


@app.route('/save')
def save():
    """
    Save the model to disk 
    """
    dgabot.save_model() 
    msg = {'message':"The model was saved" }
    return jsonify(msg)


@app.route('/predict/<url>')
def predict(url):
    result = dgabot.predict(url)
    return jsonify(result)


if __name__ == '__main__':
    app.run()