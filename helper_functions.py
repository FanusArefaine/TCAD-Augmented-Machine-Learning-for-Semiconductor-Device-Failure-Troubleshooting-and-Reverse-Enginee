import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.model_selection._split import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score





# Base Model

# Split into training and validations subsets

def base_linear_model(dataframe, device_parameter):

    X_train, X_validation, y_train, y_validation = train_test_split(dataframe, device_parameter, test_size = 0.25, random_state = 42, shuffle = True)

    # Fitting the data into linear regression model 
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # Evaluation of base linear regression model
    t_pred = lin_reg.predict(X_train)
    y_pred = lin_reg.predict(X_validation)
  
    # Calculating mean square error on training and testing 
    train_mse = mean_squared_error(y_train, t_pred)
    test_mse = mean_squared_error(y_validation, y_pred)

    print("Training mean squared error: ", train_mse)
    print("Testing mean squared error: ", test_mse)


    # Plotting results of linear regression base model
    fig, ax = plt.subplots()
    ax.scatter(y_pred, y_validation, edgecolors=(0,0,1))
    ax.plot([y_validation.min(), y_validation.max()], [y_validation.min(), y_validation.max()], 'r--', lw=3) 
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.show()
    


def check_skewness(df):

    for col in df.columns:
        print("Feature {} skewness: {}".format(col, df[col].skew()))
