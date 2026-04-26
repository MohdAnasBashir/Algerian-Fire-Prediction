import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask,request,jsonify,render_template, session


application=Flask(__name__)
app=application


app.secret_key = "supersecretkey"

## import regression and standard scaler pickle
regression_model=pickle.load(open("models/regression.pkl","rb"))
standard_scaler=pickle.load(open("models/scaler.pkl","rb"))

classification_model=pickle.load(open("models/logistic_model.pkl","rb"))
scaler_logistic=pickle.load(open("models/scaler_log.pkl","rb"))

##Home Page

@app.route("/")
def index():
    return render_template('index.html')

##now for prediction wit respect to classofiaction model
@app.route("/predictdataclass", methods=['GET','POST'])  
def predict_datapointclass():

    if request.method == 'POST':

        # Read inputs
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Region=float(request.form.get('Region'))

        # Scale + predict
        new_data_scaled = scaler_logistic.transform(
            [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Region]]
        )

        prediction = classification_model.predict(new_data_scaled)[0]

        if prediction == 1:
            session['form_data'] = request.form.to_dict()
            result = "Fire Detected"
        else:
            session.pop('form_data', None)
            result = "No Fire"

        #  PASS data back to template
        #It contains all the data sent from your HTML form
        return render_template('class.html', results=result, data=request.form)

    # For GET request
    return render_template('class.html', data={})


##now for prediction woth respect to my model
@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():

    data = session.get('form_data')

    # Block direct access
    if not data:
        return render_template('home.html', results="⚠️ Please predict fire first")
    if request.method == 'POST':
    # Extract values
        Temperature=float(data.get('Temperature'))
        RH=float(data.get('RH'))
        Ws=float(data.get('Ws'))
        Rain=float(data.get('Rain'))
        FFMC=float(data.get('FFMC'))
        DMC=float(data.get('DMC'))
        ISI=float(data.get('ISI'))
        Region=float(data.get('Region'))

        Classes = 1  # only when fire

        new_data_scaled = standard_scaler.transform(
            [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        )

        result = regression_model.predict(new_data_scaled)[0]

        return render_template('home.html', results=result, data=data)

    # 👉 GET request → only show form
    return render_template('home.html', data=data)


if __name__=="__main__":
    app.run(debug=True, port=5001)