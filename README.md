# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
VIJAYASHREE B  
212223040238
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='green')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='green')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:



To Read Head and Tail Files

![image](https://github.com/user-attachments/assets/638536cb-8626-4e75-91be-0c58ad9e3596)

![image](https://github.com/user-attachments/assets/1a3e3b74-47e9-44cc-8377-a18d61746d2d)

Compare Dataset

![image](https://github.com/user-attachments/assets/330f0bc4-db31-4988-8a01-b0660ae56a05)

Predicted Value

![image](https://github.com/user-attachments/assets/c8affce4-c634-4be6-a08c-7390aa1694bf)

Graph For Training Set

![image](https://github.com/user-attachments/assets/7aa33d1d-9896-473b-969d-e3422db3d632)

Graph For Testing Set

![image](https://github.com/user-attachments/assets/c85296fa-3425-40bb-8da9-e6f8c578d8d7)

Error

![image](https://github.com/user-attachments/assets/799b77fa-3162-45e0-a89d-eedaf5f71958)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
