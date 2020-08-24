#!/usr/bin/env python
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.externals import joblib
import math
import traceback
import pandas as pd
import numpy as np

# Your API definition
app = Flask(__name__)
clf_entropy = joblib.load("minor_notpreprocessed_dt-backendbmp.pkl") # Load "minor_notpreprocessed_dt-backend.pkl"
print ('Model loaded')
clf_entropy = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
print ('Model columns loaded')

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods = ["POST"])
def predict():
     Age=float(request.form['age'])
     FSI=float(request.form['first Sexual Intercourse'])
     HC=float(request.form['hormonal Contraceptives'])
     P=float(request.form['pregnancies'])
     SP=float(request.form['sexual Partners'])
     YRSS= Age-FSI
     NSPP=int(math.ceil(SP/YRSS))
     HPA = int(math.ceil(HC/Age))
     NPA= int(math.ceil(P/Age))
     NSA= int(math.ceil(SP/Age))
     NYHC= int(math.ceil(Age-HC))
     APP =int(math.ceil(P/SP))
     if Age>30 or SP>2 or HC>5:
         return render_template('index.html', prediction_text='Test Result: Postive')
     
     final_features=[[Age,FSI,SP,P,NSPP,APP,HC,NSA,NYHC,YRSS,NPA,HPA]]
     
     new_output=clf_entropy.predict(final_features)
     if new_output == 0:
      return render_template('index.html', prediction_text='Test Result: Negative')
     else:
      return render_template('index.html', prediction_text='Test Result: Postive')
    
    


if __name__=="__main__":
    # For local development, set to True:
    
    # For public web serving:
    #app.run(debug=True)
    app.run( host='127.0.0.1',debug=True,port=5000)
    
#if __name__ == '__main__':
    #try:
     #   port = int(sys.argv[1]) # This is for a command-line input
    #except:
     #   port = 5000 # If you don't provide any port the port will be set to 12345

 #   clf_gini = joblib.load("minor_notpreprocessed_dt-backend.pkl") # Load "minor_notpreprocessed_dt-backend.pkl"
  #  print ('Model loaded')
   # model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    #print ('Model columns loaded')

    #app.run( debug=True)
