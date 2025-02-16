# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a simple random forest, with the default settings.

## Intended Use
The model predicts if someone makes more or less than 50k

## Training Data
80% of original data. Used label encoding and one-hot-encoder

## Evaluation Data
20% of original data. Used label encoding and one-hot-encoder

## Metrics
Precision: 65%
Recall: 56%
Fbeta: 60%

## Ethical Considerations
census data are very personal. We should be careful

## Caveats and Recommendations
The performance metrics could be improved. The model is very basic
