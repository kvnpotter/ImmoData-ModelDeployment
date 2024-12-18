# **ImmoData - Model deployment using Streamlit**

[Introduction](#Introduction)    |    [Description](#Description)    |    [Installation-Environment setup](#Installation-Environment-setup)    |    [Usage](#Usage)    |    [Contributors](#Contributors)    |    [Timeline](#Timeline)

## **Introduction**

This repo contains the fourth and final project in a series aimed at completing a data science workflow from start (data collection) to finish (modelling using machine learning and deployment) during my AI and Data science bootcamp training at BeCode (Brussels, Belgium). The final goal is to create a machine learning model capable of predicting real estate prices in Belgium.

The specific aims for this project are :
1. Being able to deploy a machine learning model
2. Create a streamlit app that can handle a machine learning model

Specifications for the final app are:
- the app is capable of predicting a price, given user input for features retained in the model

## **Description**

### Remarks

- The model used for deployment was selected in a previous, modelling-specific, project.
- The model itself was not included in the repo due to size constraints for files on GitHub

### Model specifications

As a reminder, in the previous project a KNN regression model, allowing to predict property prices, was constructed using predictors scraped from a Belgian real estate website (ImmoWeb). The subset of predictor variables to be included in the model were chosen based on their correlation (for numerical features ; Spearman's ρ), or strength of interaction (for categorical features ; Kruskal-Wallis test) to the response variable. While some multicollinearity was detected amongst numerical predictors, decorrelation using PCA did not significantly improve the model's performance. The choice was made to make use of the raw predictors to allow better interpretation of the model. Other issues with the model, features and dataset are detailed in the previous project which can be found [here](https://github.com/kvnpotter/ImmoData-Modelling).

The retained features for price prediction are:
- **Postal code** ; categorical feature describing locality, but transformed to mean taxable income/locality (continuous). The transformation to a numerical predictor does not substantially improve the model, however it does remove possible issues with extrapolation (predictions of price using a postal code outside of the range of the training dataset)
- **Subtype of property** ; categorical feature with subdivisions as available on the ImmoWeb website
- **State of the building** ; categorical feature with subdivisions as available on the ImmoWeb website
- **Number of facades** ; integer, number of free facades to the property
- **Number of rooms** ; integer, number of bedrooms
- **Surface area of the plot of land** ; float, surface area of the plot of land in $m^2$
- **Living area** ; float, surface area of indoor spaces in $m^2$

The model itself was constructed using following parameters obtained using 5-fold cross-validation grid-search, with RMSE as optimization metric:
- **Distance metric** : Gower distance
- **n neighbours** : 17
- **weights** : inverse distance

**Caveat** : After fitting the model, substantial overfitting was detected (using $R^2$, MAE, RMSE, MAPE). This is clearly visible in the graph depicting actual and predicted price values, ordered by price. The best method found for dealing with the issue, given the dataset, was to restrict the price range, thereby excluding extreme values. The retained range is 200,000€ to 600,000€. **As a result, the model (and the app) should not be used to predict property prices lying (far) outside this range.** In addition, examination of the graph also indicates a systematic bias, where prices under the mean value are often overestimated, and prices over the mean are most often underestimated. This is taken to be a limitation of KNN regression, where averaging prices over 17 neighbours to make predictions results in values more close to the mean price.


![Price vs. rank ; training dataset ; actual value and predicted](./graphs/resid_train.png)
Graph 1: Actual (black dot ; training dataset) and predicted (green line ; training dataset) prices, ordered by increasing actual price. The x axis represents the observation rank. The sigmoid form of the curve indicates few extreme values at the low and high price values. Predictions closely follow these extreme values, indicating overfitting. A systematic bias, where prices under the mean value are often overestimated, and prices over the mean are most often underestimated is visible

### Streamlit app code and data flow

## Modelling real estate data

Modelling occurs by instantiating a Modeller object. Methods available for this class allow to model the data using KNN regression and a combination of following parameters:
- n_neighbors : varying number of nearest neighbors to take into account
- metric : which distance metric to use for NN calculation (Euclidean, cosine or Gower)
- weights : which weighting to attribute to neighbors (uniform, inverste distance)

Methods available for Modeller objects allow setting model parameters (including adding cross-validation grid search or not), getting the model (a method permitting to send the data to the appropriate model, with or without CV gridsearch, based on selected parameters).

CV gridsearch, using 5 CV folds and based on RMSE scoring was used for hyperparameter tuning.
Gower distance was included in modelling, despite not being available natively in sklearn, due to the fact that Gower distance is assumed to be better suited to datasets mixing categorical and numerical data.

## Model evaluation

For model evaluation, an object of Evaluator class is instantiated, allowing to calculate, and store, model evaluation metrics such as R2, MAE, RMSE, MAPE, etc.

The choice was made to exclude R2 adjusted, in favour of simple R2 since the number of observations in the dataset is much larger than the number of features.

## Visualisation

A number of visualiser classes were constructed to group all functionality pertaining to the building of graphs.

## Main

The main script contains all iterations of modelling (8 total) on a combination of parameters (using CV gridsearch) and different combinations of predictors.
In particular:
- Using the original postal code data vs. replacing it with tax data
- Using the original (correlated) predictors at property level vs. observation scores on the first two axes of PCA on these variables

The results of each iteration were automatically recorded in the ./Results-Graphs/model_metrics.csv file

Finally, the script stores the best model (see evalutation_report.md) as a pickle file.

   ## **Installation-Environment setup**

You can create a virtual environment for the script using venv.
```shell
python -m venv C:\path\to\new\virtual\environment
```

Or using conda.
```shell
conda create --name <my-env>
conda activate <my-env>
```

Included in the repository is a cross-platform environment.yml file, allowing to create a copy of the one used for this project. The environment name is given in the first line of the file.
```shell
conda env create -f environment.yml
conda activate wikipedia_scraper_env
conda env list #verify the environment was installed correctly
```

## **Usage**

The repository contains following files and directories
- Model and Testing jupyter notebooks : Notebooks detailing modelling from loading the data to calculating evaluation metrics (to be completed)
- main.py : Main script, as described above, performing modelling, recording evaluation metrics and pickling the best model.
- (environment, license and readme files)
- test_scripts directory : containing an attempt at writing a function for Gower distance calculation (function from the gower package used in final modelling)
- Data directory : contains clean property data obtained from EDA project and income/tax data obtained from the belgian government (FOD Financiën/SPF Finances)
- Results-Graphs directory : contains all outputs from visualisation and table creation methods, including evaluation metrics for 8 models
- classes : contains separate modules for data preprocessing, modelling, evaluation and visualisation
- evaluation_report.md : a report evaluating the selected model

# Contributors 
This project was completed by:
1. [Kevin](https://github.com/kvnpotter)
