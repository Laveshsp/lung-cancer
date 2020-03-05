# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 02:46:21 2020

@author: Laveshsp
"""

import numpy as np
from flask import Flask,render_template,request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)

    #output = round(prediction[0], 2)
    return render_template('predict.html')
    #return render_template('prediction.html', prediction_text='Employee Salary should be $ {}'.format(output))

@app.route('/predict/predict_result',methods=['POST'])
def predict_result():
 #   '''
   # For direct API calls trought request
  #  '''
  
  int_features = [float(x) for x in request.form.values()]
  print(int_features)
  final_features = [np.array(int_features)]
  prediction = model.predict(final_features)
  output = prediction
  #data = request.get_json(force=True)
  #prediction = model.predict([np.array(list(data.values()))])
  if(output[0]==0):
      return render_template('predict.html', prediction_text='The patient will survive for a minimum span of one-year Post Thoracic Surgery '.format(output))
  else:
      return render_template('predict.html', prediction_text='The patient will not survive for a span of one-year Post Thoracic Surgery '.format(output))
 #   output = prediction[0]
   # return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)