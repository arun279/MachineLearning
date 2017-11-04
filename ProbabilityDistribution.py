############################################################################
# title: Probability Distribution on University Data
# authors: Arun Krishnamurthy and Nitin Nataraj
# email: arunkris@buffalo.edu
# coding: utf-8
############################################################################

import pandas as pd
import matplotlib.pyplot as plt
import math
from functools import reduce
from scipy.stats import norm
from scipy.stats import multivariate_normal

import numpy as np

# Read the dataset in Excel format
get_ipython().magic('matplotlib inline')
xlsx_file = pd.ExcelFile('./DataSet/university data.xlsx')
df = xlsx_file.parse("university_data")

# Drop row number 51 in the excel sheet(or 49 in this case because Row 0 has the headers)
df = df.iloc[0:49,:]

# UBitName and personNumber
UBitName = "nitinnat, arunkris"
personNumber = "50246850, 50247445"
print ("UBitName = ", UBitName)
print ("personNumber: ", personNumber)

# Create a new data frame with just the first 4 columns
df2 = df.iloc[:,2:6]
labels = df.iloc[:,0]

# Mean 
means = [df2["CS Score (USNews)"].mean(), df2["Research Overhead %"].mean(), 
         df2["Admin Base Pay$"].mean(), df2["Tuition(out-state)$"].mean()]

mu1, mu2, mu3, mu4 = means
print('mi1 = ', str(mu1))
print('mu2 = ', str(mu2))
print('mu3 = ', str(mu3))
print('mu3 = ', str(mu4))

# Variance
variances = [df2["CS Score (USNews)"].var(), df2["Research Overhead %"].var(), 
             df2["Admin Base Pay$"].var(), df2["Tuition(out-state)$"].var()]

var1,var2,var3,var4 = variances
print("var1 = ", str(var1))
print("var2 = ", str(var2))
print("var3 = ", str(var3))
print("var4 = ", str(var4))

# Standard deviation
stds = [df2["CS Score (USNews)"].std(), df2["Research Overhead %"].std(), 
         df2["Admin Base Pay$"].std(), df2["Tuition(out-state)$"].std()]

sigma1, sigma2, sigma3, sigma4 = stds
print("sigma1 = ", str(sigma1))
print("sigma2 = ", str(sigma2))
print("sigma3 = ", str(sigma3))
print("sigma4 = ", str(sigma4))

# df3 = (df2 - df2.mean()) / df2.std()
df3 = df2

# Covariance matrix
covarianceMat = np.array(df3.cov())
print ("covarianceMat = \n", str(covarianceMat))

# Correlation matrix
correlationMat = np.array(df3.corr())
print ("correlationMat = \n", str(correlationMat))

# Probability density function
pdfs = [norm.pdf(df3["CS Score (USNews)"],df3["CS Score (USNews)"].mean() , df3["CS Score (USNews)"].std()),
        norm.pdf(df3["Research Overhead %"], df3["Research Overhead %"].mean(), df3["Research Overhead %"].std()),
        norm.pdf(df3["Admin Base Pay$"],df3["Admin Base Pay$"].mean(),df3["Admin Base Pay$"].std() ),
        norm.pdf(df3["Tuition(out-state)$"],df3["Tuition(out-state)$"].mean() , df3["Tuition(out-state)$"].std())
        ]

pdfs = np.array(pdfs).T

# Calculating the log likelihood
logLikelihood = sum(np.log([reduce(lambda x,y: x*y, p) for p in pdfs]))
print ("logLikelihood = ", str(logLikelihood))

# Calculating conditional log likelihood
conditionalLogLikelihood = sum(multivariate_normal.logpdf(np.array(df3), mean=df3.mean(), cov = covarianceMat, 
                                                          allow_singular=True))
print ("conditionalLogLikelihood = ", str(conditionalLogLikelihood))