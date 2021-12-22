import re
from numpy.core.fromnumeric import shape
from numpy.lib.polynomial import poly, polyfit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("test.csv")


X = df["x"]
y = df["y"]

X = X.values
y = y.values

def func(x,a,b,c):
    return a*np.exp(b*x)+c


    
popt, pcov = curve_fit(func,X,y)
print(popt)
plt.plot(X, func(X, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

plt.show()
# def func1(x,a,b):
#     return a * np.exp(-b*x)

# def custom_curve_fit(xdata, ydata): 
#     popt1, pcov1 = curve_fit(func1, xdata, ydata) 
#     # r^2 계산 
#     residuals1 = ydata - func1(xdata, *popt1) 
#     ss_res1 = np.sum(residuals1**2) 
#     ss_tot = np.sum((ydata-np.mean(ydata))**2) 
#     r1 = 1 - (ss_res1 / ss_tot) 
#     return popt1, r1


# plt.scatter(X, y)
# plt.show()

# p1, r1 = custom_curve_fit(X, y) 
# print(f'coef : {r1}')
# print(f'R^2 : {r1}')



# from scipy.optimize import curve_fit 
# import matplotlib.pyplot as plt 

# def func1(x,a,b,c):
#     return a * np.exp(-b*x)+c

# def func2(x,a,b,c):
#     return a * pow(2.7182, -b*x)+c


# def custom_curve_fit(xdata,ydata):
#     popt1, pcov1 = curve_fit(func1,xdata,ydata)
    
    
#     residuals1 = ydata - func1(xdata, *popt1)
#     ss_res1 = np.sum(residuals1**2)
#     ss_tot = np.sum((ydata-np.mean(ydata))**2)
#     r1 = 1 - (ss_res1 / ss_tot)

#     return popt1, r1

# xdata = np.linspace(0, 4, 50) 
# y = func1(xdata, 2.5, 1.3, 0.5) 
# np.random.seed(1729) 
# y_noise = 0.2 * np.random.normal(size=xdata.size) 
# ydata = y + y_noise 
# plt.scatter(xdata, ydata, 'b-', label='data')


# plt.show()