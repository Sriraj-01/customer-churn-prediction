Customer Churn Prediction Web App

An end-to-end Machine Learning web application that predicts whether a telecom customer is likely to churn.          
Built with Python, Flask, XGBoost, and Scikit-learn

Project Overview           
Customer churn is a major problem for telecom companies.           
This application allows users to enter customer details and instantly get:              
-> Churn probability                   
-> Risk level (Low / Medium / High)                
-> Visual risk indicator          
The model was trained on the Telco Customer Churn Dataset using feature engineering, preprocessing pipelines, and gradient boosting

Machine Learning Pipeline
: Raw Input
→ Data Cleaning
→ Feature Engineering
→ One-Hot Encoding + Scaling
→ XGBoost Model
→ Churn Probability
→ Flask API
→ Web Dashboard

Tech Stack
: Python
-> Flask
-> Scikit-learn
-> XGBoost
-> HTML & CSS
-> GitHub & Render

Project Structure
customer-churn-prediction/
app/ - Flask web app
src/ - ML pipeline
models/ - Trained model
notebooks/ - EDA & training
deployment/ - Deployment files

To Run Locally                   
git clone https://github.com/Sriraj-01/customer-churn-prediction.git           
cd customer-churn-prediction         
pip install -r requirements.txt          
python app/app.py             

Author             
Sriraj Yamana            
GitHub: https://github.com/Sriraj-01
