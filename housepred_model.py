#Step-1 Import Libraries
'''Importing the libraries required for the project like numpy for numerical calculations, Pandas for for data analysis,
sklearn for machine learning, etc'''
from typing import final

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder,FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
import matplotlib
matplotlib.use('MacOSX')



#Step-2 Load House Price Prediction Dataset using pandas read_csv function
df=pd.read_csv('AmesHousing.csv')

#Step-4 EDA
'''Getting insights like shape, null values and mean,std from House Price Prediction Dataset'''
# print(df.head())
# print(df.shape)
# print(df.describe())
# print(df.info())

# drop columns Order, PID as they are ID types
df=df.drop(['Order', 'PID'],axis=1)
# Step-4.1 Univariate and Multivariate Analysis

f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=(12, 8))
sns.histplot(df['SalePrice'],kde=True,color='blue',ax=ax_hist)
sns.boxplot(x=df['SalePrice'],ax=ax_box)
plt.suptitle("Ames Housing: SalePrice Analysis", fontsize=16)
plt.xlabel("Sale Price ($)")
plt.ylabel("Number of Houses")
# plt.show()
# plt.show()

print(df.shape)
print(df.columns[:10])
print(df.columns[-10:])
print(df.isnull().sum().sort_values(ascending=False))



correlation=df.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False).head(10).index
top_features=df[correlation].corr()

plt.figure(figsize=(12,8))
sns.heatmap(top_features,annot=True,cmap='Blues')
# plt.show()



#Step-5  Feature Engineering

df['TotalSF']=df['1st Flr SF']+df['2nd Flr SF'] +df['Total Bsmt SF']
df['HouseAge']=df['Yr Sold']-df['Year Built']
df['RemodelAge']=df['Yr Sold']-df['Year Remod/Add']
df['TotalBathrooms']=df['Full Bath']+0.5*df['Half Bath']+ df['Bsmt Full Bath']+0.5* df['Bsmt Half Bath']

#Splitting the data into input and output where SalePrice is our target variable which we want to predict

X=df.drop('SalePrice',axis=1)
X = df[['Overall Qual', 'TotalSF', 'TotalBathrooms', 'Garage Cars',
        'Year Built', 'Garage Area', 'Lot Area',
        'Overall Cond', 'HouseAge', 'Gr Liv Area']]
y=df['SalePrice']
y=np.log1p(y)

#Step-6
'''Splitting the data into training data and test data. Training data will be used to train the model and testing
data will be used to test our model '''

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=7)

# Step-7
'''Pipeline: To prevent Data Leakage and ensure smooth and clean workflow we use all preprocessing steps inside pipeline.'''

# Categorising the data into numerical and categorical to apply transformation acc to data types in the pipeline

numerical_features=X.select_dtypes(include=['int64','float64']).columns
categorical_features=X.select_dtypes(include=['object']).columns


numerical_pipeline = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_pipeline = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor_lr= ColumnTransformer(transformers=[

    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features),
])


# Step-8
'''BASELINE LOGISTIC REGRESSION PIPELINE 
- Constructing full workflow using preprocessor steps and defining model Linear Regression '''

clf_pipeline_lr= Pipeline(steps=[
    ('preprocessor', preprocessor_lr),
    ('classifier', LinearRegression())
])

clf_pipeline_ridge= Pipeline(steps=[
    ('preprocessor', preprocessor_lr),
    ('classifier', Ridge())
])

clf_pipeline_rf= Pipeline(steps=[
    ('preprocessor', preprocessor_lr),
    ('classifier', RandomForestRegressor())
])

clf_pipeline_gb= Pipeline(steps=[
    ('preprocessor', preprocessor_lr),
    ('classifier', GradientBoostingRegressor())
])


clf_pipeline_lr.fit(X_train,y_train)
y_pred=clf_pipeline_lr.predict(X_test)

clf_pipeline_ridge.fit(X_train,y_train)
y_pred_ridge=clf_pipeline_ridge.predict(X_test)

clf_pipeline_rf.fit(X_train,y_train)
y_pred_rf=clf_pipeline_rf.predict(X_test)

clf_pipeline_gb .fit(X_train,y_train)
y_pred_gb=clf_pipeline_gb .predict(X_test)

#Step-7 Calculating RMSE and R2 Score


r2_score_lr=r2_score(y_test,y_pred)
r2_score_ridge=r2_score(y_test,y_pred_ridge)
r2_score_rf=r2_score(y_test,y_pred_rf)
r2_score_gb=r2_score(y_test,y_pred_gb)


results = pd.DataFrame({

'Model':['Linear Regression','Random Forest','Gradient Boosting','Ridge'],
'R2 Score':[r2_score_lr, r2_score_rf, r2_score_gb, r2_score_ridge]


})

print(results)

#Step-8 Hyperparameter Tunnuing

param_space=[
    {

        'classifier__n_estimators':[100,200,300],
        'classifier__learning_rate':[0.01,0.05,0.1],
        'classifier__max_depth':[5,10]

    }
]

gridsearch_gb=GridSearchCV(

    estimator=clf_pipeline_gb,
    param_grid=param_space,
    scoring='r2',
    cv=3,
    n_jobs=-1
)

gridsearch_gb.fit(X_train,y_train)
print('Best parameters:',gridsearch_gb.best_params_)
print('Best score:',gridsearch_gb.best_score_)

best_model = gridsearch_gb.best_estimator_

y_pred_best = best_model.predict(X_test)
final_price=np.expm1(y_pred_best)

print("Tuned R2:", r2_score(y_test, y_pred_best))

#Step-9 Feature Importance extracted from Gradient Boosting to understand dominate predictors

model = best_model.named_steps['classifier']
preprocessor = best_model.named_steps['preprocessor']
feature_names = preprocessor.get_feature_names_out()
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
})

top_features = feature_importance_df.sort_values(by='importance', ascending=False).head(10)

plt.figure(figsize=(10,6))

sns.barplot(
    x=top_features['importance'],
    y=top_features['feature']
)

plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.tight_layout()
plt.show()

#step-10 Saving the model

import joblib
joblib.dump(best_model, "house_price_model.pkl")