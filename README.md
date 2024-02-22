# Singapore Resale Flat Prices Predicting

## About

This project is about using the data provided by the government to predict the resale prices of the singapore flatts. The was downloaded from the website www.data.gov.in. Then the data was pre processed and used for a regression machine learning model building. With help of the machine learning model we can predict the resale price. The model was deployed into a streamlit app for predicting the price.

## Problem Statement

The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.

## Technologies Used

1. Python
2. Streamlit
3. Scikit Learn
4. XGBoost
5. Pandas
6. Numpy
7. Matplotlib
8. Plotly Express
9. Regular Expression
10. Pickle
11. Seaborn
12. Statistics

## Deliverables

1. A well-trained machine learning model for resale price prediction.
2. A user-friendly web application (built with Streamlit) deployed on Any Cloud Platform
3. Documentation and instructions for using the application.
4. A project report summarizing the data analysis, model development, and deployment process.

## Pre Processing Machine Learning Model Building.

Step-1:
First the data set has to be downloaded from the official website.

Step-2:
The data has to be preprocessed using python scripting, using pandas. The data has to be wrangled since the give data is divided into five csv files namely 1990-1999, 2000-2012, 2012-2014, 2015-2017, 2017-current. All the 5 data sets to be wrangled into one and the the excessive data has to be handled. We need to check for the null values also.

Step-3:
Checking and changing the data types. Then check the data set for feature selection. The selected features has to be in the format of float of int, So we are using statistics and some encoders like Ordinal encoder for encoding the string type data to float type.

Step-4:
Checking the data for skewness and outliers, There is wellfound skewness and outliers in the combined dataset. So we have to use some methods for handling skewness and outliers. I used Z-Score methor for handling outliers since there is lesser anount of outliers are there the Z-score method will eliminate the outliers and I used Log transformation for the skewness.

Step-5:
After Preparing the dataset we have for select the training and testing data and check with few Regressor models to select the best Regressor for Machine learning model. I used Decision Tree Regressor, Extra Trees Regressor, Random Forest Regressor, Gradiant Boosting Regressor and XGboosting Regrssors.

Step-6:
Selecting the best Regreesor. Here The XGBoosting regressor got the higherst R-square value so I selected that regressor and used Parameter grid search to select the best model among the testing datasets.

Step-6:
Fnal step in model building is checking the model for prediction and the saving the model in a pickle file. The pickle file is used to save the machine learning model for fure use. It reduces the model testing time on each time we need to use the model.

## Streamlit App Building.

I have created the streamlit app for deployig the machine learning model. Given the code in the repository as well.

## Steps for using the Streamlit App:

1. First install the requires libraries given in the main file.
2. The run the code in your IDE terminal or Command prompt with the code line "streamlit run location_of_the_file/main_app.py".
3. When entering the app there is 3 sections "Home", "Analysis", "Predict Resale price", 
4. The EDA process is displayed in the Analysis page.
5. The price prediction is done in the Predict Resale price page, You need to enter the details to predict the price.

## Conclusion
  This app will be really usefull for the preople who are going to buy the flats in Sigapore.
  
**App link** : [https://singaporeresaleflatpriceprediction-iyfnegtqke3keuuyb5bwyx.streamlit.app/]

## Thank You.
