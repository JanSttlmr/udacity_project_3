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
#Trim
data.columns = data.columns.str.strip()
data = data.map(lambda x: x.strip() if isinstance(x, str) else x)

data.drop("fnlgt", axis="columns", inplace=True)
data.drop("education-num", axis="columns", inplace=True)
data.drop("capital-gain", axis="columns", inplace=True)
data.drop("capital-loss", axis="columns", inplace=True)


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
# Process the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False, 
    encoder=encoder, lb=lb)

def model_slicing():
    """
    Calculate performance on model slices
    """
    result_collection = []

    for cat in cat_features:
        result_collection.append("Category: " + cat)
        for val in set(test[cat]):
            result_collection.append("Slice: " + val)
            df_val = test[test[cat] == val]
            X_test_val, y_test_val, _, _ = process_data(
                df_val, categorical_features=cat_features,
                label="salary", encoder=encoder, lb=lb, training=False)
            y_predictions = model.predict(X_test_val)
            precision_val, recall_val, fbeta_val = compute_model_metrics(
                y_test_val, y_predictions)
            precision_string = "Precision: " + str(precision_val) 
            recall_string = "Recall: " + str(recall_val) 
            fbeta_string = "Fbeta: " + str(fbeta_val) 
            result_collection.append(precision_string)
            result_collection.append(recall_string)
            result_collection.append(fbeta_string)
            result_collection.append("\n")
        result_collection.append("\n\n")  
        

    with open('slice_model_output.txt', 'w') as out:
        for row in result_collection:
            out.write(row + '\n')

dump(encoder_test, '../model/encoder.joblib')
dump(lb_test, '../model/lb.joblib')
model = train_model(X_train, y_train)
dump(model, '../model/model.joblib')
predictions = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, predictions)
model_slicing()
print(precision)
print(recall)
print(fbeta)

