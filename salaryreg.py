import numpy
import pandas as pd
from sklearn.linear_model import LinearRegression
data=pd.read_csv('salary_data.csv')
y=data['Salary']
x=data['YearsExperience']
x = x.values
x=x.reshape(-1,1)
model=LinearRegression()
model.fit(x,y)
inp= float(input("Enter the Years of Experience:"))
output= model.predict([[inp]])
print(output)






