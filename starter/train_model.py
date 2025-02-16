# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import os
from joblib import dump

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv("../data/census.csv")
#print(data.columns)


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False, 
    encoder=encoder, lb=lb)

# Proces the test data with the process_data function.

dump(encoder_test, '../model/encoder.joblib')
dump(lb_test, '../model/lb.joblib')
model = train_model(X_train, y_train)
dump(model, '../model/model.joblib')
predictions = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, predictions)
print(precision)

