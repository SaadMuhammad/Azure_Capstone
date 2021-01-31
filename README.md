

# Azure Capstone Project: <ins> Predicting Defaulters Based on Previous Payments </ins>


# Dataset


### Overview
The dataset used for this project is taken from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) which is an excellent source for opensource datasets to experiment with. The dataset is of credit card clients who defaulted on their regular payments based on payments made during past 6 months. The dataset contain 24 features from **Education**,**Age** **Gender** to **payments history of past months** with **30000** instances and no missing values. More details on dataset can be found [**here**](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

### Task

This is a classification problem to find if the clients will default on their next payments. To classify this we have 30 feautres with details as follows:

Column Names | Details
------------ | -------------
ID | Client ID's
LIMIT_BAL | Amount of the given credit (NT dollar)
SEX | Gender (1 = male; 2 = female).
EDUCATION | (1 = graduate school; 2 = university; 3 = high school; 4 = others).
MARRIAGE | (1 = married; 2 = single; 3 = others).
AGE | Age (year)
PAY_0 | the repayment status in September, 2005
PAY_2 | the repayment status in August, 2005
PAY_3 | the repayment status in July, 2005
PAY_4 | the repayment status in June, 2005
PAY_5 | the repayment status in May, 2005
PAY_6 | the repayment status in April, 2005
BILL_AMT1 | amount of bill statement in September, 2005
BILL_AMT2 | amount of bill statement in August, 2005
BILL_AMT3 | amount of bill statement in July, 2005
BILL_AMT4 | amount of bill statement in June, 2005
BILL_AMT5 | amount of bill statement in May, 2005
BILL_AMT6 | amount of bill statement in May, 2005
PAY_AMT1 | amount paid in September, 2005
PAY_AMT2 | amount paid in Augustr, 2005
PAY_AMT3 | amount paid in July, 2005
PAY_AMT4 | amount paid in June, 2005
PAY_AMT5 | amount paid in May, 2005
PAY_AMT6 | amount paid in April, 2005
default payment next month | Target Variable (to be predicted)

### Access

The CSV file has been added to Azure Storage via **TabularDatasetFactory** Class of Azure ML. It is also available and can be accessed here: [**My Github Link**](https://raw.githubusercontent.com/SaadMuhammad/Azure_Capstone/main/default_credit_clients1.csv)

## Automated ML


Configrations used for **Auto ML** are as follows:

Configration | Details
------------ | -------------
task | This is a classification task to predict defaulters for payments in next month
compute_target | Experiment is being run on amlcluster namely **SaadCompute**
label_column_name | **'default payment next month'** is the target coulmn
enable_onnx_compatible_models | enable_onnx_compatible_models is set to **True** because i have converted the best automl to **ONNX Format**
n_cross_validations | No of cross validations is set to 4


Also settings were as follows:

Setting | Details
------------ | -------------
experiment_timeout_minutes | Set the experiment to finsih model selection and experiment in **25 mins**
max_concurrent_iterations | **5** iterations were allowded to run in parallel
primary_metric | **Accuracy** is being selected to be the primary metric to be improved/selected 

### Results

The AutoML model namely **Voting Ensemble** scored the best accuracy with **0.82256667**. 

The parameters for **Voting Ensemble** were: 
boosting_type' as 'gbdt', 'colsample_bytree' set at 1.0, 'importance_type' was 'split', 'learning_rate' was set to 0.1,
'min_child_samples'were set to 20, 'min_child_weight'is set to 0.001, 'n_estimators' were 100, 'n_jobs' were 1, 'num_leaves'were 31,
'silent': True, 'subsample' was set to 1.0, 'subsample_for_bin' were 200000, 'subsample_freq' was 0, 'verbose' was -10.

**Improvements for AutoML**

1. Addition of model stacking and blending techniques in AutoML will be a very valuable addition. As this will provide independence in stacking or blending top performing models to further improve the mertics

2. Option to select/create multi layer deeplearning model or even pre-trained model to be utilized (transfer learning techniques) will be of great use here.

3.Custom imputations to fill missing values based on data to data scenario would be a game changer i.e. Imputations that fill missing values based on feature importance, correlations and other statistcal knowledge will enhance model performance surprisingly.

**Screenshots for the RunWidget & Model**
![Screenshot](https://github.com/SaadMuhammad/Azure_Capstone/blob/main/Screenshots/automl_runs.PNG)

![Screenshot](https://github.com/SaadMuhammad/Azure_Capstone/blob/main/Screenshots/auto_acc.PNG)

![Screenshot](https://github.com/SaadMuhammad/Azure_Capstone/blob/main/Screenshots/auto_best.PNG)


## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
