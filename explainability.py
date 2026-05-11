# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 22:55:05 2026

@author: savin
"""

import pandas as pd
import numpy as np

df = pd.read_excel("C:/Users/savin/OneDrive/Desktop/Ratul sir's/Oak North company loan default/for_python.xlsx")

# Putting feature variable to X
X = df.drop('y',axis=1)
# Putting response variable to y
y = df['y']


# now lets split the data into train and test
from sklearn.model_selection import train_test_split

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
X_train.shape, X_test.shape


#Random Forest ------------------------------------------
from sklearn.ensemble import RandomForestClassifier

classifier_rf = RandomForestClassifier(random_state=42, max_depth=5,
                                       n_estimators=100)

 
classifier_rf.fit(X_train, y_train)



train_proba_rf = classifier_rf.predict_proba(X_train)
test_proba_rf = classifier_rf.predict_proba(X_test)

#----------------------------------------------------------
#------------------------------------------------------------
#Explainability start

#SHAPE code

from shap import TreeExplainer
from shap import summary_plot

explainer = TreeExplainer(classifier_rf)

shap_values = np.array(explainer.shap_values(X_train))

X_train.columns

first = shap_values[0]
first_df = pd.DataFrame(first, columns=["A", "B"])
second = shap_values[1]

#6th customer
sixth = shap_values[5]

#13th customer
thirteenth = shap_values[12]

X_train.columns
#Explainability end
#------------------------------------------------------
#Logistic regression --------------------------------------

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

train_proba_logistic = model.predict_proba(X_train) 
test_proba_logistic = model.predict_proba(X_test)

#-----------------------------------------------------------

#XGBoost----------------------------------------------------

from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

train_proba_xgboost = model.predict_proba(X_train)
test_proba_xgboost = model.predict_proba(X_test)
#----------------------------------------------------------