from flask import Flask, render_template, request
import pickle
import numpy as np  
import seaborn as sns 
import pandas as pd

model=pickle.load(open('xgb.pkl','rb'))

app= Flask(__name__)

@app.route('/')
def man():
    return render_template('Index.html')

@app.route('/predict', methods=['POST'])
def home():
    Loan_Amount = int(request.form['a'])
    Applicant_Income = int(request.form['b'])
    Coapplicant_Income = int(request.form['c'])
    Loan_Amount_Term = int(request.form['d'])
    Credit_History = int(request.form['e'])
    Property_Area = request.form['Area']
    if (Property_Area == 'Rural') :
        Property_Area_Rural = 1
        Property_Area_Semiurban = 0
        Property_Area_Urban = 0

    elif (Property_Area == 'Semiurban'):
        Property_Area_Rural = 0
        Property_Area_Semiurban = 1
        Property_Area_Urban = 0
    
    # elif (Area == 'Property_Area_Urban'):
    else :
        Property_Area_Rural = 0
        Property_Area_Semiurban = 0
        Property_Area_Urban = 1



    arr=np.array([[Loan_Amount,Applicant_Income,Coapplicant_Income,Loan_Amount_Term,
                Credit_History,Property_Area_Rural,Property_Area_Semiurban,Property_Area_Urban]])



    # arr= pd.DataFrame({'Loan_Amount':[Loan_Amount],'Applicant_Income':[Applicant_Income],
    #                     'Coapplicant_Income':[Coapplicant_Income],'Loan_Amount_Term':[Loan_Amount_Term],
    #                     'Credit_History':[Credit_History],
    #                     'Property_Area_Rural':[Property_Area_Rural],
    #                     'Property_Area_Semiurban':[Property_Area_Semiurban],
    #                     'Property_Area_Urban':[Property_Area_Urban]})



    pred=model.predict(arr)
    return render_template('Return.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)




