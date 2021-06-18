#USER INPUT
#MAKE SURE year and value HAVE THE SAME NUMBER OF INPUTS
year = [2010, 2015, 2021]
value = [32940, 24145, 8637]
startingYear = 2010
numberOfYearsToPredict = 20
#DO NOT CHANGE ANYTHING PAST THIS POINT UNLESS YOU ARE THE DEVELOPER

print("Please Wait...\n\n")

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

#This is the actual code for the machine learning (W3schools, n.d.)
mymodel = np.poly1d(np.polyfit(year, value, 3))

#keep this variable out of for loop or else all predictions will be the same
#This variable should be set to the year before 
years = startingYear - 1

#In computer science you count 0,1,2,3... instead of 1,2,3...
#This line adds one to this variable because of the above statement
numberOfYearsToPredictPlusOne = numberOfYearsToPredict + 1

#Keep this out of for loop
df1 = pd.DataFrame()

print("\nYear", " Value\n===================")

#For loop (W3schools, n.d.)
for i in range (numberOfYearsToPredictPlusOne):
    years= years + 1
    valuePrediction = mymodel(years)

    #When rounding with numpy, use np.round() isntead of round() (The SciPy community, 2021)
    valuePredictionRounded = np.round(valuePrediction, 2)

    #Use df.append for writing to csv in a for loop using pandas (Saucoide, 2018)
    df0 = pd.DataFrame([valuePredictionRounded])
    df1 = df1.append(df0)
    print(years, "|", valuePredictionRounded)

print("\n\nDONE!\n\n")
df1.to_csv("Results.csv")

# References APA 7th Edition


# Saucoide. (2018, March 20). Pandas DataFrames in a loop, df.to_csv(). Stack Overflow. https://stackoverflow.com/questions/49395052/pandas-dataframes-in-a-loop-df-to-csv

# The SciPy community. (2021, January 31). Numpy.around — NumPy v1.20 manual. NumPy. https://numpy.org/doc/stable/reference/generated/numpy.around.html

# W3schools. (n.d.). Python for loops. W3Schools Online Web Tutorials. https://www.w3schools.com/python/python_for_loops.asp

# W3schools. (n.d.). Python machine learning polynomial regression. W3Schools Online Web Tutorials. https://www.w3schools.com/python/python_ml_polynomial_regression.asp