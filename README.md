# Alphabet Soup Funding Prediction

In this project, I'll use a neural network model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. I will be using machine learning and neural networks to analyze a CSV containing over 34,000 organizations that have received funding from Alphabet Soup, using metadata columns to predict the success of future applicants.

## Overview of the Analysis

Following is the overview of the analysis performed for Alphabet Soup Funding Prediction project:

  * The purpose of the analysis is to train and evaluate a model to predict if an Alphabet Soup-funded organiztion wll be successful based on the features in dataset. I used a historical dataset of organizations funded by Alphbet Soup foundation.
  * To answer the questions regarding features and target variables : 'IS_SUCCESSFUL' is the target i tried to predict and 'APPLICATION_TYPE', 'AFFILIATION', 'CLASSIFICATION', 'USE_CASE',	'ORGANIZATION',	'STATUS',	'INCOME_AMT',	'SPECIAL_CONSIDERATIONS' and 'ASK_AMT' are the features. 'EIN' and 'NAME' are neither the atrget nor the features, hence they are dropped while training the model.
  * Then, I tried models with different number of layers, different number of perceptrons per layer, different type of activation, using PCA to try and optimize the model.

## Data preprocessing

This section describes the step taken to preprocess the data for neural network model:

  * The data was read into Pandas DataFrame from a CSV file, and 'EIN' and 'NAME' columns were dropped as they don't serve any purpose in this context in all the attempts.
  * At first, in AplhabetSoupCharity.ipynb, Reduced occurrences that are rare in 'APPLICATION_TYPE' and 'CLASSIFICATION' columns and binning them under 'Others'. Then, get the dummy variables for categorical features to have a dataset with all numerical values. Then, data was scaled using StandardScaler with 43 features.
  * Data processing for Optimizing model:
      * While attempting to optimize the model for better accuracy, in AlpabetSoupCharityOptimization.ipynb, For the first attempt, mostly the processing was done same as above mentioned but i added some more rare occurrences 'APPLICATION_TYPE' and 'CLASSIFICATION' columns in 'Others' bin. Then, using PCA data was described into 4 features/components, which then was scaled and resulted in 3 features.
      * For the second attempt, the data used was same as previous attempt.
      * For the thrid attempt, the data was used without performing PCA with 39 features as the 1st optimization attempt.

## Model Architecture and Results

This section emphasis on the number of Dense layers in a model, number of perceptron(neurons/units) used in each layer, type of activation function and results of the model. The model architecture consists of three layers: an input layer for receiving data, followed by two hidden layers responsible for processing information, and finally an output layer for generating predictions or outputs.
  * Firstly, in AplhabetSoupCharity.ipynb, The model architecture consists an input layer with 80 units with 'relu' as an activation function , followed by a hidden layer with 30 units wih 'relu' activation, and finally an output layer with 1 unit with activation function as 'sigmoid'. In this neural network, the accuracy score was 72.90%.
  * While trying to Optimize the model in AplhabetSoupCharity_Optimization.ipynb,, following attempts were made:
      * For the first attempt, The model architecture consists an input layer with 50 units and 'relu' activation, followed by a hidden layer with 20 units with 'relu' activation and output layer with 1 unit with 'sigmoid' activation function which results an accuracy score of 71.70%.
      * For the second attempt, This model has one input layer with 30 units, 2 hidden layers with units 20 and 10 respectively with 'relu' activation and finally output layer with 1 unit and activation as 'sigmoid'. This model results has the accuracy score of 71.80%.
      * For final attempt, I used keras tuner to find the best architecture for model that can produce better accuracy score achieve or exceed the target accuracy of 75%. As per keras tuner, best model has 'tanh' activation, 9 units in first layer, three hidden layers with units 9, 1 and 5 in each hidden layer and then an output layer, that will result in a model with accuracy score of 73.64%. I was not able to achieve the target accuracy of 75%.
   
## Summary
