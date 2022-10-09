# -*- coding: utf-8 -*-
# @Time : 09-10-2022 11:42
# @Author : rohan
# @File : app.py
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template

app = Flask(__name__)  # Start point of the flask app
regmodel = pickle.load(open('./artifacts/regmodel.pkl', 'rb'))
scalar = pickle.load(open('./artifacts/scaling.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    # print(data)
    # print(np.array(list(data.values())).reshape(1, -1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    # print(output[0])
    return jsonify(output[0])


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    # print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text="Predicted house price is {}".format(round(output, 2)))


if __name__ == "__main__":
    app.run(debug=True)
