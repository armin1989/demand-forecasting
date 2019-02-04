# demand-forecasting
A repo containing my work on a demand forecasting challenge

This challenge is hosted on Kaggle (url: https://www.kaggle.com/c/demand-forecasting-kernels-only)
It involves forecasting sales amounts of certain items across several stores over a period of 3 months given 5 years of previous sales data.

Description of files:

- eda.ipynb : An EDA of the time series data (incomplete)

- sarima.py : Script that fits a SARIMA model to the series data, this has some nice visualizations
              (the code has room for improvment)

- cnn.py : Script that fits a convolutional neural network to the time series data 
          (the code has room for improvment)
          
I am currently working on applying holt-winters models as well as RNNs for the forecasting task.