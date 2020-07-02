# Core Pkgs
import streamlit as st 

# EDA Pkgs
import pandas as pd 
import numpy as np 


# Data Viz Pkg
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns 


# ML Packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import xgboost as xgb

def main():
	"""Semi Automated ML App with Streamlit """
		st.subheader("ML Model Trainer With _Mr_AI")
	activities = ["Regressor Model","Classification Model","About"]	
	choice = st.sidebar.selectbox("Select Activities",activities)

	if choice == 'Regressor Model':
		st.subheader("Regressor ML Models")
		data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())


			# Model Building
			X = df.iloc[:,0:-1] 
			Y = df.iloc[:,-1]
			seed = 7
			# prepare models
			models = []
			models.append(('LinearReg', LinearRegression()))
			models.append(('Tree Reg', DecisionTreeRegressor()))
			models.append(('XGB Reg', xgb.XGBRegressor()))
			models.append(('KNN Reg', KNeighborsRegressor()))
			models.append(('NB', GaussianNB()))
			#models.append(('SVM', SVC()))
			# evaluate each model in turn
			
			model_names = []
			model_mean = []
			model_std = []
			all_models = []
			scoring = 'accuracy'
			for name, model in models:
				kfold = model_selection.KFold(n_splits=10, random_state=seed)
				cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
				model_names.append(name)
				model_mean.append(cv_results.mean())
				model_std.append(cv_results.std())
				
				accuracy_results = {"model name":name,"model_accuracy":cv_results.mean(),"standard deviation":cv_results.std()}
				all_models.append(accuracy_results)
				

			if st.checkbox("Metrics As Table"):
				st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Algo","Mean of Accuracy","Std"]))

			if st.checkbox("Metrics As JSON"):
				st.json(all_models)



	


	elif choice == 'Classification Model':
		st.subheader("Classifier ML Models")
		data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())


			# Model Building
			X = df.iloc[:,0:-1] 
			Y = df.iloc[:,-1]
			seed = 7
			# prepare models
			models = []
			models.append(('LogisticReg', LogisticRegression()))
			models.append(('XGB Class', xgb.XGBClassifier()))
			models.append(('KNNClass', KNeighborsClassifier()))
			models.append(('DecisionTreeClass', DecisionTreeClassifier()))
			models.append(('NB', GaussianNB()))
			models.append(('SVM', SVC()))
			# evaluate each model in turn
			
			model_names = []
			model_mean = []
			model_std = []
			all_models = []
			scoring = 'accuracy'
			for name, model in models:
				kfold = model_selection.KFold(n_splits=10, random_state=seed)
				cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
				model_names.append(name)
				model_mean.append(cv_results.mean())
				model_std.append(cv_results.std())
				
				accuracy_results = {"model name":name,"model_accuracy":cv_results.mean(),"standard deviation":cv_results.std()}
				all_models.append(accuracy_results)


			if st.checkbox("Metrics As Table"):
				st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Algo","Mean of Accuracy","Std"]))

			if st.checkbox("Metrics As JSON"):
				st.json(all_models)
	elif choice == 'About':
		st.subheader("Written By Mohammad Harati | تهیه شده توسط محمد هراتی")
		st.subheader("             Instagram : _Mr_AI           ")
		st.subheader("Thank's For Following me In Instagram |ممنون از اینکه من رو در اینستاگرام حمایت میکنید")       
		st.subheader("                  (: منتظر آپدیت ها و برنامه های دیگه باشین           ")		
                
                
            
		  



if __name__ == '__main__':
	main()