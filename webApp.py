import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask,request,jsonify,render_template


application=Flask(__name__)
app=application

## import regression and standard scaler pickle
regression_model=pickle.load(open("models/regression.pkl","rb"))
standard_scaler=pickle.load(open("models/scaler.pkl","rb"))



##Home Page

@app.route("/")
def index():
    return render_template('index.html')


##now for prediction woth respect to my model
@app.route("/predictdata",methods=['GET','POST'])  
def predict_datapoint():
    if request.method=='POST':
        ##first of all we need it to read all the inputs
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))
        ##now we need to standard scaler tranform of new data
        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=regression_model.predict(new_data_scaled)
        ##now i will have to show result in home .html only

        return render_template('home.html',results=result[0])
    else:
       #in get return rendertempalte
       return render_template('home.html')

if __name__=="__main__":
    app.run(debug=True, port=5001)