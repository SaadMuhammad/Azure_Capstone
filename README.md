

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

3. Custom imputations to fill missing values based on data to data scenario would be a game changer i.e. Imputations that fill missing values based on feature importance, correlations and other statistcal knowledge will enhance model performance surprisingly.

**Screenshots for the RunWidget & Model**

![Screenshot](https://github.com/SaadMuhammad/Azure_Capstone/blob/main/Screenshots/automl_runs.PNG)

![Screenshot](https://github.com/SaadMuhammad/Azure_Capstone/blob/main/Screenshots/auto_acc.PNG)

![Screenshot](https://github.com/SaadMuhammad/Azure_Capstone/blob/main/Screenshots/auto_best.PNG)


## Hyperparameter Tuning


Hyper Drive configration consist of compute target created in Azure. a pyhton script (**train.py** for our case), **Logistic Regression** as the classification algorithm and inverse regulaization and mat_iteratons as hyper paramters

2 paramters were used here, Inverse of regularization strength **'--C'** and **'max_iters'** **'--C'** is a trade of paramter of logistic regression and is used to apply penalty on magnitude of paramters to reduce overfitting higher values of C correspond to less regularizationand wise versa and **'max_iters'** is the number of iterations model will take to converge to a minima. We have employed **RandomSampling** as our paramter sampling because the biggest benefit of RandomSampling is it choose hyperparamters randmoly thus reducing the computational time and complexity. Other options we had were Grid sampling and Bayesian sampling both of them had their own importance like Grid sampling confines the search within a given set of paramters while Bayesian sampling select the next set of hyperparameters based on how the previous hyperparams performed. For Extensive search we can employ RandomSampling as a starting point and then using those hyperparameters do a Grid or Bayesian sampling as per our understanding of the problem and time.

Early stopping policy used here is **BanditPolicy**, its biggest benifit is that this policy terminates any runs early where the primary metric (accuray in our case) is not within the selected/pre-defined slack factor with respect to the best performing training run.

### Results

The best Accuracy with which logistic regression performed was **0.7834444444444445** with Regularization Strength at **1.0** and Max Iterations at **170**.
 
**HyperDrive Improvements**

1. Using different primary metric like AUC, ROC, or precision and recall as sometimes accuracy alone is not enough to ensure model's success.

2. Use a combination of Grid/Bayesian with Random Sampling as ensembling or boosting like Grid sampling confines the search within a given set of paramters while Bayesian sampling select the next set of hyperparameters based on how the previous hyperparams performed. For Extensive search we can employ RandomSampling as a starting point and then using those hyperparameters do a Grid or Bayesian sampling as per our understanding of the problem and time

### HyperDrive Screenshots

![Screenshot](https://github.com/SaadMuhammad/Azure_Capstone/blob/main/Screenshots/hyper_runs1.PNG)

![Screenshot](https://github.com/SaadMuhammad/Azure_Capstone/blob/main/Screenshots/hyper_best.PNG)

![Screenshot](https://github.com/SaadMuhammad/Azure_Capstone/blob/main/Screenshots/Hyper_img.PNG)

![Screenshot](https://github.com/SaadMuhammad/Azure_Capstone/blob/main/Screenshots/hyper_cord.PNG)


## Model Deployment

The best model is deployed using **Azure Containce Instance (ACI Service)** with 2 cpu_cores and 1 gb of memory_gb. Ofcourse this deployment wont be possible without scoring file **'scoreScript.py'** which was saved in outputs via **best_run** alongside **'envFile.yml'**. Scoring file was important for creating Inference Configration for ACI.

After ACI service creation and getting a Swagger and Scoring URL we create a same to test in **JSON** format and read the response from the deployed model. As shown below


![Screenshot](https://github.com/SaadMuhammad/Azure_Capstone/blob/main/Screenshots/response.PNG)


and this is the deployed model screenshot:

![Screenshot](https://github.com/SaadMuhammad/Azure_Capstone/blob/main/Screenshots/deployed_model.PNG)


## Screen Recording

A detail tour of complete project can be viewed [**here**](https://youtu.be/MJARE2u-Plw)


## Standout Suggestions

Apart from above work i have also converted my best model in an **Open Neural Network Exchange (ONNX)** format. The reason for doing is because ONNX is a high-performance inference engine for deploying models to production. It's optimized for both cloud and edge and works on Linux, Windows, and Mac. ONNX Runtime is used in high-scale Microsoft services such as Bing, Office, and Azure Cognitive Services. Performance gains are dependent on a number of factors, but these Microsoft services have seen an average 2x performance gain on CPU.

Another reason is models from many frameworks including TensorFlow, PyTorch, SciKit-Learn, Keras, Chainer, MXNet, MATLAB, and SparkML can be exported or converted to the standard ONNX format. Once the models are in the ONNX format, they can be run on a variety of platforms and devices.

Here's how it can be done:

![Screenshot](https://github.com/SaadMuhammad/Azure_Capstone/blob/main/Screenshots/ONNX.PNG)

ONNX model can be viewed and downloaded from [**here**](https://github.com/SaadMuhammad/Azure_Capstone/blob/main/saad_onnx_model.onnx)
