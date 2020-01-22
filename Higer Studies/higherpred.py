# Import dependencies
import pandas as pd
import numpy as np

# Load the dataset in a dataframe object and include only four features as mentioned

df = pd.read_csv(r"C:\Users\SHUBHAM\Documents\Flask\Higer Studies\higherstudies.csv")
include = ['student ID', 'Gender', 'overall rating', 'Salary', 'higher studies'] # Only four features
df_ = df[include]

# Data Preprocessing
categoricals = []
for col, col_type in df_.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     else:
          df_[col].fillna(0, inplace=True)

df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

# Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
dependent_variable = 'higher studies'
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]
lr = LogisticRegression()
lr.fit(x, y)

# Save your model
from sklearn.externals import joblib
joblib.dump(lr, 'higherpred.pkl')
print("Model dumped!")

# Load the model that you just saved
lr = joblib.load('higherpred.pkl')

# Saving the data columns from training
model_columns = list(x.columns)
joblib.dump(model_columns, 'higher_prediction.pkl')
print("Models columns dumped!")
