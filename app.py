import joblib
from flask import Flask,request,render_template,redirect,session
import pandas as pd
import numpy as np


app = Flask(__name__)

car_models = joblib.load('car_models')
companies = joblib.load('companies')
model = joblib.load('model_rfr_57')
yrs = joblib.load('years')
fuel_type = joblib.load('fuel_type')
dummy_cols = joblib.load('dummy_cols')
l=list(range(len(companies)))


@app.route('/')
def index():
    return render_template('index.html',car_models=car_models,companies=companies,length=l,yrs=yrs,fuel_type=fuel_type)


@app.route('/oLcR3F8jjsDTpmWYmpeMtQunR4uqPurPcRuLyQRXkmFJDuFN4xxPHtzR4cpiJ49VBp8aQyFYBQVMEaUzGq1mHUUZgUJ9Kv1uoPVh')
def home():
    return render_template('home.html',car_models=car_models,companies=companies,length=l,yrs=yrs,fuel_type=fuel_type)

@app.route('/predict',methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('model')
    year = request.form.get('year')
    fl_tp = request.form.get('fuel_type')
    kms = request.form.get('kms')

    n = {'company':[company],'model':[car_model],'kms':[kms],'year':[year],'fuel_type':[fl_tp]}
    inp_dummy = pd.get_dummies(data=pd.DataFrame(n),columns=['company','model','year','fuel_type'],)
    input_df = pd.DataFrame(columns=dummy_cols)
    input_df = input_df.append(inp_dummy,sort=False)
    input_df.fillna(0,inplace=True)
    price = int(model.predict(input_df.values))

    return render_template('predict.html',car_models=car_models,companies=companies,length=l,yrs=yrs,fuel_type=fuel_type,price=price)

if __name__ == '__main__':
    app.run(debug=True) 