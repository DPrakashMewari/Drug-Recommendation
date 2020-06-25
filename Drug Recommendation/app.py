# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:07:32 2020

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:43:50 2020

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:19:54 2020

@author: hp
"""
import pickle
from flask import Flask,jsonify,request,render_template
import pandas as pd

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')



@app.route("/submit", methods=['GET','POST'])    
def predict():
    if(request.method == 'POST'):
             
                 
       
             
        data=request.values['Symtoms']
        condition=data
        print(condition)
               
        with open('nlp_data.pickle', 'rb') as f:
         
           dt = pickle.load(f)
        
        df_drugs=dt[dt['condition'].str.lower().str.contains(condition.lower())]
        
        output=df_drugs[['drugName','rating','usefulCount','Valuation']].groupby('drugName').agg({'rating':'median','usefulCount':'sum','Valuation':'median'}).sort_values(by='Valuation',ascending=False)[['rating','usefulCount']].head(10)
        
        if(output.size!=0):
             return  render_template('result.html',  tables=[output.to_html(classes='data', header="true")])
        else:
             return  render_template('result1.html')
             
               

        
    else:
        s='Please Try Again Later'
        return jsonify(s)
if __name__ == '__main__':
    app.run()