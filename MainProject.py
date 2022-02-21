#GPU libraries
'''
import pynvml
import shutil
from numba import jit, cuda
import cudf
from cuml import train_test_split
from cuml import LogisticRegression as cuml_Logistic
from cuml import RandomForestClassifier as cuml_RandomForestClassifier
#rest
'''
#tensorFlow:
'''
from keras.optimizers import SGD
from tensorflow.keras.layers import Dense
from keras.models import Sequential
'''
from sklearn.manifold import TSNE
from pyDeepInsight import ImageTransformer, LogScaler
from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
#rest

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import svm
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import BaggingRegressor
import pandas as pd
import scipy.spatial as sci
import timeit
import math
import numpy as np
from sklearn import model_selection
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.backends.backend_pdf import PdfPages
import itertools
import logging
import os
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
#evaluation metrics
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import ROCAUC
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import log_loss

# =============================================================================
'''
Logger class to track code output
'''

# Creating Logger class and helper methods
class Transcript(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, 'w')
        
    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        
    def flush(self):
        pass
    
def start(filename):
    sys.stdout = Transcript(filename)
    
def stop():
    sys.stdout.logfile.close()
    sys.stdout = sys.stdout.terminal
    
# =============================================================================
'''
Main console to execute code
'''
def main():
    
    #Initiate system output
    #start('initialSysOut.txt')
    #create logger
    #logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'), filename="initial.txt", force = True)
    #logger = logging.getLogger('FinalSeminar')
    
    
    #Read the original data files
    trainDF = pd.read_csv("train.csv")
    testDF = pd.read_csv("test.csv")
        
    #demonstrateHelpers(trainDF)
    
    #print(corrTest(trainDF))

    #trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF)
    #trainInput, testInput, trainOutput, testIDs, predictors = beginPreprocessing(trainDF, testDF)

    
    #doExperiment(trainInput, trainOutput, predictors)
    
    trainInput, testInput, trainOutput, testIDs, predictors = beginPreprocessing(trainDF, testDF)
    
    #stackerTest1(trainInput, trainOutput, predictors)
    
    #stackerTest(trainInput, trainOutput, predictors)
    
    #sys.stdout = open('hyperParam.csv', 'w')
    #hyperparam(trainInput, trainOutput)
    #sys.stdout.close()
    
    #xgbTest(trainInput, trainOutput, testInput, predictors)

    #tacker(trainInput, trainOutput, predictors)
    
    #aucCheck(trainInput, trainOutput, predictors)
    
    #doExperiment(trainInput, trainOutput, predictors)
    
    #confMat(trainInput, trainOutput)
    
    #hypParamTest(trainInput, trainOutput, predictors)
    
    doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors)
    
    #investigate(trainDF, testDF)

    #close system output

    #stop()

    
    
# ===============================================================================
'''
Reading data from csv files
'''

def readData(numRows = None):
    trainDF = pd.read_csv("data/train.csv")
    testDF = pd.read_csv("data/test.csv")
    
    outputCol = ['SalePrice']
    
    return trainDF, testDF, outputCol



# ===============================================================================
'''
Formal Pre-Processing
'''
def beginPreprocessing (trainDF, testDF):
    
    '''
    Divide the data frames into testing and training sets
    '''
    fullDF = trainDF
    trainInput = trainDF.iloc[:, :127]
    testInput = testDF.iloc[:, :]
    
    trainOutput = trainDF.loc[:, 'Response']
    testIDs = testDF.loc[:, 'Id']
    
    #to with readability:
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 200)

    
    '''
    Making predictors
    '''
    
    s = "Product_Info_1 Product_Info_2 	Product_Info_3	Product_Info_4	Product_Info_5	Product_Info_6	Product_Info_7	Ins_Age	Ht	Wt	BMI	Employment_Info_1	Employment_Info_2	Employment_Info_3	Employment_Info_4	Employment_Info_5	Employment_Info_6	InsuredInfo_1	InsuredInfo_2	InsuredInfo_3	InsuredInfo_4	InsuredInfo_5	InsuredInfo_6	InsuredInfo_7	Insurance_History_1	Insurance_History_2	Insurance_History_3	Insurance_History_4	Insurance_History_5	Insurance_History_7	Insurance_History_8	Insurance_History_9	Family_Hist_1	Family_Hist_2	Family_Hist_3	Family_Hist_4	Family_Hist_5	Medical_History_1	Medical_History_2	Medical_History_3	Medical_History_4	Medical_History_5	Medical_History_6	Medical_History_7	Medical_History_8	Medical_History_9	Medical_History_10	Medical_History_11	Medical_History_12	Medical_History_13	Medical_History_14	Medical_History_15	Medical_History_16	Medical_History_17	Medical_History_18	Medical_History_19	Medical_History_20	Medical_History_21	Medical_History_22	Medical_History_23	Medical_History_24	Medical_History_25	Medical_History_26	Medical_History_27	Medical_History_28	Medical_History_29	Medical_History_30	Medical_History_31	Medical_History_32	Medical_History_33	Medical_History_34	Medical_History_35	Medical_History_36	Medical_History_37	Medical_History_38	Medical_History_39	Medical_History_40	Medical_History_41	Medical_Keyword_1	Medical_Keyword_2	Medical_Keyword_3	Medical_Keyword_4	Medical_Keyword_5	Medical_Keyword_6	Medical_Keyword_7	Medical_Keyword_8	Medical_Keyword_9	Medical_Keyword_10	Medical_Keyword_11	Medical_Keyword_12	Medical_Keyword_13	Medical_Keyword_14	Medical_Keyword_15	Medical_Keyword_16	Medical_Keyword_17	Medical_Keyword_18	Medical_Keyword_19	Medical_Keyword_20	Medical_Keyword_21	Medical_Keyword_22	Medical_Keyword_23	Medical_Keyword_24	Medical_Keyword_25	Medical_Keyword_26	Medical_Keyword_27	Medical_Keyword_28	Medical_Keyword_29	Medical_Keyword_30	Medical_Keyword_31	Medical_Keyword_32	Medical_Keyword_33	Medical_Keyword_34	Medical_Keyword_35	Medical_Keyword_36	Medical_Keyword_37	Medical_Keyword_38	Medical_Keyword_39	Medical_Keyword_40	Medical_Keyword_41	Medical_Keyword_42	Medical_Keyword_43	Medical_Keyword_44	Medical_Keyword_45	Medical_Keyword_46	Medical_Keyword_47	Medical_Keyword_48 Wt*BMI bmiHtCond Agehigh productInfoLetter	productInfoInt	medKeyCount	binaryMean	binaryMin	binaryMax"
    predictors = s.split()    


    '''        
    Missing values
    '''
    #Using our missing value plot from the investigate method, I identified missing values.
    #I use correlation with the response variable to determine whether filling missing values
    #in a particular way is efficient or not. I first calculate correlation before any processing, and then after fillna()
    #Since all of the missing predictors are either continous or discrete, I can replace them easily. 
    #My correlation tests show that predictors with a >20% missing values do well when 
    #the missing values are replaced with 0. For lower percentage missed values, I use the mean.
    
    #trainInput:
    #<20% missing value predictors:
    trainInput.loc[:, 'Employment_Info_1'].fillna(trainInput.loc[:, 'Employment_Info_1'].mean(), inplace=True)
    trainInput.loc[:, 'Employment_Info_4'].fillna(trainInput.loc[:, 'Employment_Info_4'].mean(), inplace=True) 
    trainInput.loc[:, 'Employment_Info_6'].fillna(trainInput.loc[:, 'Employment_Info_6'].mean(), inplace=True)  
    trainInput.loc[:, 'Medical_History_1'].fillna(trainInput.loc[:, 'Medical_History_1'].mean(), inplace=True)  
    
    #>20% missing value predictors:
    trainInput.loc[:, 'Medical_History_32'].fillna(0, inplace=True)
    trainInput.loc[:, 'Medical_History_24'].fillna(0, inplace=True)
    trainInput.loc[:, 'Medical_History_15'].fillna(0, inplace=True)
    trainInput.loc[:, 'Medical_History_10'].fillna(0, inplace=True)    
    trainInput.loc[:, 'Family_Hist_5'].fillna(0, inplace=True)
    trainInput.loc[:, 'Family_Hist_4'].fillna(0, inplace=True)  
    trainInput.loc[:, 'Family_Hist_3'].fillna(0, inplace=True)
    trainInput.loc[:, 'Family_Hist_2'].fillna(0, inplace=True)        
    trainInput.loc[:, 'Insurance_History_5'].fillna(0, inplace=True)   
    
    #testInput:
    #<20% missing value predictors:
    testInput.loc[:, 'Employment_Info_1'].fillna(testInput.loc[:, 'Employment_Info_1'].mean(), inplace=True)
    testInput.loc[:, 'Employment_Info_4'].fillna(testInput.loc[:, 'Employment_Info_4'].mean(), inplace=True) 
    testInput.loc[:, 'Employment_Info_6'].fillna(testInput.loc[:, 'Employment_Info_6'].mean(), inplace=True)  
    testInput.loc[:, 'Medical_History_1'].fillna(testInput.loc[:, 'Medical_History_1'].mean(), inplace=True)  
    
    #>20% missing value predictors:
    testInput.loc[:, 'Medical_History_32'].fillna(0, inplace=True)
    testInput.loc[:, 'Medical_History_24'].fillna(0, inplace=True)
    testInput.loc[:, 'Medical_History_15'].fillna(0, inplace=True)
    testInput.loc[:, 'Medical_History_10'].fillna(0, inplace=True)    
    testInput.loc[:, 'Family_Hist_5'].fillna(0, inplace=True)
    testInput.loc[:, 'Family_Hist_4'].fillna(0, inplace=True)  
    testInput.loc[:, 'Family_Hist_3'].fillna(0, inplace=True)
    testInput.loc[:, 'Family_Hist_2'].fillna(0, inplace=True)        
    testInput.loc[:, 'Insurance_History_5'].fillna(0, inplace=True)           


    #normalization is not needed since data is already normal
    

    '''
    Wt*BMI
    '''
    #Looking at the scaterplots from investigate, it is apparent that BMI and weight
    #are significant determinators of the response variable. 
    #Thus, I will make a couple of dummy variables to account for this.
    
    #create new predictor = Wt*BMI
    trainInput['Wt*BMI'] = trainInput['Wt']*trainInput['BMI']
    testInput['Wt*BMI'] = testInput['Wt']*trainInput['BMI']
    
    
    '''
    bmihtCond
    '''
    #The scatter plot and further analysis in Excel tells us that BMI >0.85 and Wt>0.4 
    #are predominately classified as high risk (level1, level2). Lets add the dummy variable
    trainInput['bmiHtCond'] = np.where((trainInput['BMI']>0.85) & (trainInput['Wt']>0.4), 1, 0)
    testInput['bmiHtCond'] = np.where((testInput['BMI']>0.85) & (testInput['Wt']>0.4), 1, 0)    
    
    
    '''
    Ins_Age
    '''
    #According to hist plot, proportion of high risk classification drastically increases on Ins_Age>0.7
    trainInput['Agehigh'] = np.where((trainInput['Ins_Age']>0.7), 1, 0)
    testInput['Agehigh'] = np.where((testInput['Ins_Age']>0.7), 1, 0)    
    
    
    '''
    Product_Info_2
    '''
    #Product Info 2  variable is comprised of a character and number, we can seperate these
    #to make two new variabels, and then apply label encoding. 
    
    trainInput['productInfoLetter'] = trainInput.Product_Info_2.str[0]
    trainInput['productInfoInt'] = trainInput.Product_Info_2.str[1]
    testInput['productInfoLetter'] = testInput.Product_Info_2.str[0]
    testInput['productInfoInt'] = testInput.Product_Info_2.str[1]
    
    trainInput['productInfoLetter'] = LabelEncoder().fit_transform(trainInput.productInfoLetter)
    trainInput['productInfoInt'] = LabelEncoder().fit_transform(trainInput.productInfoInt)
    #product info 2 also included to provide more features to the model
    trainInput['Product_Info_2'] = LabelEncoder().fit_transform(trainInput.Product_Info_2)
    testInput['productInfoLetter'] = LabelEncoder().fit_transform(testInput.productInfoLetter)
    testInput['productInfoInt'] = LabelEncoder().fit_transform(testInput.productInfoInt)
    testInput['Product_Info_2'] = LabelEncoder().fit_transform(testInput.Product_Info_2)
    
    
    '''
    MedicalKeywords
    '''
    #The medical keywords are all dummy variables. The correlation heatmap in investigate
    #shows that there is no correlation between these words. Consequently, Im making a 
    #new predictor that sums the medical keywords.
    
    medKeyCols = 'Medical_Keyword_1	Medical_Keyword_2	Medical_Keyword_3	Medical_Keyword_4	Medical_Keyword_5	Medical_Keyword_6	Medical_Keyword_7	Medical_Keyword_8	Medical_Keyword_9	Medical_Keyword_10	Medical_Keyword_11	Medical_Keyword_12	Medical_Keyword_13	Medical_Keyword_14	Medical_Keyword_15	Medical_Keyword_16	Medical_Keyword_17	Medical_Keyword_18	Medical_Keyword_19	Medical_Keyword_20	Medical_Keyword_21	Medical_Keyword_22	Medical_Keyword_23	Medical_Keyword_24	Medical_Keyword_25	Medical_Keyword_26	Medical_Keyword_27	Medical_Keyword_28	Medical_Keyword_29	Medical_Keyword_30	Medical_Keyword_31	Medical_Keyword_32	Medical_Keyword_33	Medical_Keyword_34	Medical_Keyword_35	Medical_Keyword_36	Medical_Keyword_37	Medical_Keyword_38	Medical_Keyword_39	Medical_Keyword_40	Medical_Keyword_41	Medical_Keyword_42	Medical_Keyword_43	Medical_Keyword_44	Medical_Keyword_45	Medical_Keyword_46	Medical_Keyword_47	Medical_Keyword_48'
    medKeyColsList= medKeyCols.split()
    trainInput['medKeyCount'] = trainInput.loc[:, medKeyColsList].sum(axis=1)
    testInput['medKeyCount'] = testInput.loc[:, medKeyColsList].sum(axis=1)
    
    
    '''
    Mean target encoding
    '''
    #features have a non-linear target dependency, so let us apply mean encoding!
    #to avoid overfitting, the categorical data will be encoded using k-folds.
    
    trainInput['Product_Info_1'], testInput['Product_Info_1'] = targetEncoding(trainDF, testInput, ['Product_Info_1'], 'Response')
    trainInput['Product_Info_2'], testInput['Product_Info_2'] = targetEncoding(trainDF, testInput, ['Product_Info_2'], 'Response')
    trainInput['Product_Info_3'], testInput['Product_Info_3'] = targetEncoding(trainDF, testInput, ['Product_Info_3'], 'Response')
    trainInput['Product_Info_5'], testInput['Product_Info_5'] = targetEncoding(trainDF, testInput, ['Product_Info_5'], 'Response')
    trainInput['Product_Info_6'], testInput['Product_Info_6'] = targetEncoding(trainDF, testInput, ['Product_Info_6'], 'Response')
    trainInput['Product_Info_7'], testInput['Product_Info_7'] = targetEncoding(trainDF, testInput, ['Product_Info_7'], 'Response')   
    trainInput['Employment_Info_2'], testInput['Employment_Info_2'] = targetEncoding(trainDF, testInput, ['Employment_Info_2'], 'Response')      
    trainInput['Employment_Info_3'], testInput['Employment_Info_3'] = targetEncoding(trainDF, testInput, ['Employment_Info_3'], 'Response')          
    trainInput['Employment_Info_5'], testInput['Employment_Info_5'] = targetEncoding(trainDF, testInput, ['Employment_Info_5'], 'Response')          
    trainInput['InsuredInfo_1'], testInput['InsuredInfo_1'] = targetEncoding(trainDF, testInput, ['InsuredInfo_1'], 'Response')       
    trainInput['InsuredInfo_2'], testInput['InsuredInfo_2'] = targetEncoding(trainDF, testInput, ['InsuredInfo_2'], 'Response')           
    trainInput['InsuredInfo_3'], testInput['InsuredInfo_3'] = targetEncoding(trainDF, testInput, ['InsuredInfo_3'], 'Response')       
    trainInput['InsuredInfo_4'], testInput['InsuredInfo_4'] = targetEncoding(trainDF, testInput, ['InsuredInfo_4'], 'Response')      
    trainInput['InsuredInfo_5'], testInput['InsuredInfo_5'] = targetEncoding(trainDF, testInput, ['InsuredInfo_5'], 'Response')       
    trainInput['InsuredInfo_6'], testInput['InsuredInfo_6'] = targetEncoding(trainDF, testInput, ['InsuredInfo_6'], 'Response')      
    trainInput['InsuredInfo_7'], testInput['InsuredInfo_7'] = targetEncoding(trainDF, testInput, ['InsuredInfo_7'], 'Response')       
    trainInput['Insurance_History_1'], testInput['Insurance_History_1'] = targetEncoding(trainDF, testInput, ['Insurance_History_1'], 'Response')      
    trainInput['Insurance_History_2'], testInput['Insurance_History_2'] = targetEncoding(trainDF, testInput, ['Insurance_History_2'], 'Response')      
    trainInput['Insurance_History_3'], testInput['Insurance_History_3'] = targetEncoding(trainDF, testInput, ['Insurance_History_3'], 'Response')      
    trainInput['Insurance_History_4'], testInput['Insurance_History_4'] = targetEncoding(trainDF, testInput, ['Insurance_History_4'], 'Response')      
    trainInput['Insurance_History_7'], testInput['Insurance_History_7'] = targetEncoding(trainDF, testInput, ['Insurance_History_7'], 'Response')      
    trainInput['Insurance_History_8'], testInput['Insurance_History_8'] = targetEncoding(trainDF, testInput, ['Insurance_History_8'], 'Response')      
    trainInput['Insurance_History_9'], testInput['Insurance_History_9'] = targetEncoding(trainDF, testInput, ['Insurance_History_9'], 'Response')      
    trainInput['Family_Hist_1'], testInput['Family_Hist_1'] = targetEncoding(trainDF, testInput, ['Family_Hist_1'], 'Response')      
    trainInput['Medical_History_2'], testInput['Medical_History_2'] = targetEncoding(trainDF, testInput, ['Medical_History_2'], 'Response')      
    trainInput['Medical_History_3'], testInput['Medical_History_3'] = targetEncoding(trainDF, testInput, ['Medical_History_3'], 'Response')      
    trainInput['Medical_History_4'], testInput['Medical_History_4'] = targetEncoding(trainDF, testInput, ['Medical_History_4'], 'Response')      
    trainInput['Medical_History_5'], testInput['Medical_History_5'] = targetEncoding(trainDF, testInput, ['Medical_History_5'], 'Response')      
    trainInput['Medical_History_6'], testInput['Medical_History_6'] = targetEncoding(trainDF, testInput, ['Medical_History_6'], 'Response')      
    trainInput['Medical_History_7'], testInput['Medical_History_7'] = targetEncoding(trainDF, testInput, ['Medical_History_7'], 'Response')      
    trainInput['Medical_History_8'], testInput['Medical_History_8'] = targetEncoding(trainDF, testInput, ['Medical_History_8'], 'Response')      
    trainInput['Medical_History_9'], testInput['Medical_History_9'] = targetEncoding(trainDF, testInput, ['Medical_History_9'], 'Response')      
    trainInput['Medical_History_11'], testInput['Medical_History_11'] = targetEncoding(trainDF, testInput, ['Medical_History_11'], 'Response')      
    trainInput['Medical_History_12'], testInput['Medical_History_12'] = targetEncoding(trainDF, testInput, ['Medical_History_12'], 'Response')      
    trainInput['Medical_History_13'], testInput['Medical_History_13'] = targetEncoding(trainDF, testInput, ['Medical_History_13'], 'Response')      
    trainInput['Medical_History_14'], testInput['Medical_History_14'] = targetEncoding(trainDF, testInput, ['Medical_History_14'], 'Response')      
    trainInput['Medical_History_16'], testInput['Medical_History_16'] = targetEncoding(trainDF, testInput, ['Medical_History_16'], 'Response')      
    trainInput['Medical_History_17'], testInput['Medical_History_17'] = targetEncoding(trainDF, testInput, ['Medical_History_17'], 'Response')      
    trainInput['Medical_History_18'], testInput['Medical_History_18'] = targetEncoding(trainDF, testInput, ['Medical_History_18'], 'Response')      
    trainInput['Medical_History_19'], testInput['Medical_History_19'] = targetEncoding(trainDF, testInput, ['Medical_History_19'], 'Response')      
    trainInput['Medical_History_20'], testInput['Medical_History_20'] = targetEncoding(trainDF, testInput, ['Medical_History_20'], 'Response')      
    trainInput['Medical_History_21'], testInput['Medical_History_21'] = targetEncoding(trainDF, testInput, ['Medical_History_21'], 'Response')      
    trainInput['Medical_History_22'], testInput['Medical_History_22'] = targetEncoding(trainDF, testInput, ['Medical_History_22'], 'Response')      
    trainInput['Medical_History_23'], testInput['Medical_History_23'] = targetEncoding(trainDF, testInput, ['Medical_History_23'], 'Response')      
    trainInput['Medical_History_25'], testInput['Medical_History_25'] = targetEncoding(trainDF, testInput, ['Medical_History_25'], 'Response')      
    trainInput['Medical_History_26'], testInput['Medical_History_26'] = targetEncoding(trainDF, testInput, ['Medical_History_26'], 'Response')      
    trainInput['Medical_History_27'], testInput['Medical_History_27'] = targetEncoding(trainDF, testInput, ['Medical_History_27'], 'Response')      
    trainInput['Medical_History_28'], testInput['Medical_History_28'] = targetEncoding(trainDF, testInput, ['Medical_History_28'], 'Response')      
    trainInput['Medical_History_29'], testInput['Medical_History_29'] = targetEncoding(trainDF, testInput, ['Medical_History_29'], 'Response')      
    trainInput['Medical_History_30'], testInput['Medical_History_30'] = targetEncoding(trainDF, testInput, ['Medical_History_30'], 'Response')      
    trainInput['Medical_History_31'], testInput['Medical_History_31'] = targetEncoding(trainDF, testInput, ['Medical_History_31'], 'Response')      
    trainInput['Medical_History_33'], testInput['Medical_History_33'] = targetEncoding(trainDF, testInput, ['Medical_History_33'], 'Response')      
    trainInput['Medical_History_34'], testInput['Medical_History_34'] = targetEncoding(trainDF, testInput, ['Medical_History_34'], 'Response')      
    trainInput['Medical_History_35'], testInput['Medical_History_35'] = targetEncoding(trainDF, testInput, ['Medical_History_35'], 'Response')      
    trainInput['Medical_History_36'], testInput['Medical_History_36'] = targetEncoding(trainDF, testInput, ['Medical_History_36'], 'Response')      
    trainInput['Medical_History_37'], testInput['Medical_History_37'] = targetEncoding(trainDF, testInput, ['Medical_History_37'], 'Response')      
    trainInput['Medical_History_38'], testInput['Medical_History_38'] = targetEncoding(trainDF, testInput, ['Medical_History_38'], 'Response')      
    trainInput['Medical_History_39'], testInput['Medical_History_39'] = targetEncoding(trainDF, testInput, ['Medical_History_39'], 'Response')      
    trainInput['Medical_History_40'], testInput['Medical_History_40'] = targetEncoding(trainDF, testInput, ['Medical_History_40'], 'Response')      
    trainInput['Medical_History_41'], testInput['Medical_History_41'] = targetEncoding(trainDF, testInput, ['Medical_History_41'], 'Response')      
   
    
    '''
    Drop ID Variable
    '''
    testInput.drop(['Id'], axis=1, inplace=True)
    trainInput.drop(['Id'], axis =1, inplace=True)
    
    
    '''
    binaryVariables
    '''
    #There are 48 binary variables in the dataset, lets apply a version of target mean encoding to these variables
    
    s = "Medical_Keyword_1	Medical_Keyword_2	Medical_Keyword_3	Medical_Keyword_4	Medical_Keyword_5	Medical_Keyword_6	Medical_Keyword_7	Medical_Keyword_8	Medical_Keyword_9	Medical_Keyword_10	Medical_Keyword_11	Medical_Keyword_12	Medical_Keyword_13	Medical_Keyword_14	Medical_Keyword_15	Medical_Keyword_16	Medical_Keyword_17	Medical_Keyword_18	Medical_Keyword_19	Medical_Keyword_20	Medical_Keyword_21	Medical_Keyword_22	Medical_Keyword_23	Medical_Keyword_24	Medical_Keyword_25	Medical_Keyword_26	Medical_Keyword_27	Medical_Keyword_28	Medical_Keyword_29	Medical_Keyword_30	Medical_Keyword_31	Medical_Keyword_32	Medical_Keyword_33	Medical_Keyword_34	Medical_Keyword_35	Medical_Keyword_36	Medical_Keyword_37	Medical_Keyword_38	Medical_Keyword_39	Medical_Keyword_40	Medical_Keyword_41	Medical_Keyword_42	Medical_Keyword_43	Medical_Keyword_44	Medical_Keyword_45	Medical_Keyword_46	Medical_Keyword_47	Medical_Keyword_48"
    v = s.split()
    
    binaryTrain, binaryTest = binaryManipulation(trainDF, testInput, v, 'Response')

    trainInput = pd.concat([trainInput, binaryTrain], axis=1, sort=False)
    testInput = pd.concat([testInput, binaryTest], axis=1, sort=False)
    

    #dropping features did not improve model accuracy. Therefore, choosing to keep features to help with model.
    '''
    feature selection / dimensionality reduction
    
    featureSelectionCat(trainInput, trainOutput, testInput)
    featureSelectionMIF(trainInput, trainOutput, testInput)
    featureSelectionRFE()
    
    #dropping variables
    testInput.drop(['Product_Info_6', 'Product_Info_7', 'InsuredInfo_4', 'Insurance_History_1', 'Insurance_History_3', 'Insurance_History_8', 'Medical_History_3', 'Medical_History_11','Medical_History_12', 'Medical_History_14', 'Medical_History_25', 'Medical_History_35', 'Medical_Keyword_5', 'Medical_Keyword_6','Medical_Keyword_7', 'Medical_Keyword_28','Medical_Keyword_29'], axis=1, inplace=True)
    trainInput.drop(['Product_Info_6', 'Product_Info_7', 'InsuredInfo_4', 'Insurance_History_1', 'Insurance_History_3', 'Insurance_History_8', 'Medical_History_3', 'Medical_History_11','Medical_History_12', 'Medical_History_14', 'Medical_History_25', 'Medical_History_35', 'Medical_Keyword_5', 'Medical_Keyword_6','Medical_Keyword_7', 'Medical_Keyword_28','Medical_Keyword_29'], axis =1, inplace=True)
    '''
    
    
    '''
    Binarization
    '''
    #adding 8 new features to teh data set that contain binary classification probabilities
    
    trainDF = pd.concat([trainInput, trainOutput], axis=1)
    
    #we know there are 8 classes, so we can specify the output probability dataframe
    outProbDF = pd.DataFrame(index=range(trainDF.shape[0]), columns = range(8))
  

    #filling in values for different classifications so that each column contains data regarding 
    #a specific label or class
    outProbDF.iloc[:,0] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==1) else 0)
    outProbDF.iloc[:,1] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==2) else 0)
    outProbDF.iloc[:,2] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==3) else 0)
    outProbDF.iloc[:,3] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==4) else 0)
    outProbDF.iloc[:,4] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==5) else 0)
    outProbDF.iloc[:,5] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==6) else 0)
    outProbDF.iloc[:,6] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==7) else 0)
    outProbDF.iloc[:,7] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==8) else 0)    
    
    #making output columns for dataframe
    probTrain = pd.DataFrame(index=range(trainDF.shape[0]), columns = range(8))
    probTest = pd.DataFrame(index=range(testInput.shape[0]), columns = range(8))
    
    trainInDF = trainInput
    
    #using eXtreme Gradient Boosting for binary classification 
    alg = xgb.XGBClassifier()
    
    #dimensions for stratifiedkfold
    skfX = trainInput
    skfY = outProbDF.iloc[:,0]
    
    #loop through each column
    for counter in range(0,8):
        #runnig K folds to prevent overfitting
        skf = StratifiedKFold(n_splits = 5)
        for trIndex, cvIndex in skf.split(skfX,skfY):
            
            #defining folds
            trInput = trainInDF.iloc[trIndex]
            trOutput =  outProbDF.iloc[:,counter][trIndex]
            
            alg.fit(trInput, trOutput)
            
            #extracting probability of specific classification
            predictions = alg.predict_proba(trainInDF.iloc[cvIndex])[:,1]
            #adding probabilities to the output data frame
            probTrain.iloc[cvIndex, counter] = predictions
            
        #testing data
        alg.fit(trainInDF, outProbDF.iloc[:,counter])
        predictions = alg.predict_proba(testInput)[:,1]
        probTest.iloc[:,counter] = predictions
    
    #cols = ['prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7', 'prob8']
    probTrain = probTrain.set_axis(['prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7', 'prob8'], axis=1)  
    probTest = probTest.set_axis(['prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7', 'prob8'], axis=1) 


    #converting the reported probabilities to numeric so it isn't viewed as an 'object' and cause errors in our models
    probTrain['prob1']= pd.to_numeric(probTrain['prob1'], errors='coerce').fillna(0, downcast='infer')
    probTrain['prob2']= pd.to_numeric(probTrain['prob2'], errors='coerce').fillna(0, downcast='infer')
    probTrain['prob3']= pd.to_numeric(probTrain['prob3'], errors='coerce').fillna(0, downcast='infer')
    probTrain['prob4']= pd.to_numeric(probTrain['prob4'], errors='coerce').fillna(0, downcast='infer')
    probTrain['prob5']= pd.to_numeric(probTrain['prob5'], errors='coerce').fillna(0, downcast='infer')
    probTrain['prob6']= pd.to_numeric(probTrain['prob6'], errors='coerce').fillna(0, downcast='infer')
    probTrain['prob7']= pd.to_numeric(probTrain['prob7'], errors='coerce').fillna(0, downcast='infer')
    probTrain['prob8']= pd.to_numeric(probTrain['prob8'], errors='coerce').fillna(0, downcast='infer')
    
    #checking the newly created features
    print(probTrain.dtypes)
    print(probTest.dtypes)    

    print('probTrain:')
    print(probTrain.head)
    print('probTest:')
    print(probTest.head)


    #add features to the training and test set
    trainInput = pd.concat([trainInput, probTrain], axis =1)
    testInput = pd.concat([testInput, probTest], axis=1)

    #verifying data sets
    print(trainInput.head)
    print(testInput.head)

    

    
    print('preprocessingComplete')
    
    
    #return cleaned data    
    return trainInput, testInput, trainOutput, testIDs, predictors



# ===============================================================================     
'''
Target encoding helper function used in preprocessing
'''
def targetEncoding(trainInput, testInput, cols, target, n_folds = 10):
    
    trainCopy , testCopy = trainInput.copy(), testInput.copy()
    
    #defining folds
    kf = KFold(n_splits = n_folds, random_state=(1500), shuffle=True)
    
    #loop over each column
    for col in cols:
        
        #calculate a mean value for categories that equal zero due to their low number (prevents NaN)
        missingMean = trainCopy[target].mean()
        
        #make dictionary to keep track of fold results
        Dictionary = {}
        
        i=1
        
        for trIn, testIn in kf.split(trainInput):
            
            #calculates target mean
            targetMean = trainCopy.iloc[trIn].groupby(col)[target].mean()
            
            #add to dictionary
            Dictionary['fold'+ str(i)]=targetMean
            
            #to keep track
            i+=1
            
            #creating column
            trainCopy.loc[testIn, col+'Enc'] = trainCopy.loc[testIn, col].map(targetMean)
        
        #replacing values
        trainCopy[col+'Enc'].fillna(missingMean, inplace=True)
        
        #test set + dictionary
        fold1 = Dictionary.get('fold1')
        fold2 = Dictionary.get('fold2')
        fold3 = Dictionary.get('fold3')
        fold4 = Dictionary.get('fold4')
        fold5 = Dictionary.get('fold5')
        fold6 = Dictionary.get('fold6')
        fold7 = Dictionary.get('fold7')
        fold8 = Dictionary.get('fold8')
        fold9 = Dictionary.get('fold9')
        fold10 = Dictionary.get('fold10')       
        
        #applying mean of all folds to test dataset
        folds = pd.concat([fold1, fold2, fold3, fold4, fold5, fold6,fold7, fold8, fold9, fold10], axis=1)

        foldsMean = folds.mean(axis=1)         
        
        testCopy[col+'Enc'] = testCopy[col].map(foldsMean)
        testCopy[col+'Enc'].fillna(missingMean, inplace=True)
    
    
    #filter out relevant columns        
    trainCopy = trainCopy.filter(like = 'Enc', axis=1)
    testCopy = testCopy.filter(like = 'Enc', axis=1)
    
    
    return trainCopy, testCopy

    

# ===============================================================================     
'''
Target encoding applied to binary predictors used in preprocessing
'''
def binaryManipulation(trainInput, testInput, cols, target, n_folds = 10):
    
    trainCopy , testCopy = trainInput.copy(), testInput.copy()
    
    #defining folds
    kf = KFold(n_splits = n_folds, random_state=(1500), shuffle=True)
    
    #iterate
    for col in cols:

        #make dictionary to keep track of fold results
        Dictionary = {}
        i=0
        
        for trIn, testIn in kf.split(trainInput):
            
            #calculates target mean
            targetMean = trainCopy.iloc[trIn].groupby(col)[target].mean()
            
            #add att to dictionary
            Dictionary[str(col)+'enc'+str(i)]=targetMean
            
            #tracking
            i+=1
            
            #creating column
            trainCopy.loc[testIn, col+'Enc'] = trainCopy.loc[testIn, col].map(targetMean)
        
        
        #test set + dictionary
        fold1 = Dictionary.get(str(col)+'enc'+str(0))
        fold2 = Dictionary.get(str(col)+'enc'+str(1))
        fold3 = Dictionary.get(str(col)+'enc'+str(2))
        fold4 = Dictionary.get(str(col)+'enc'+str(3))
        fold5 = Dictionary.get(str(col)+'enc'+str(4))
        fold6 = Dictionary.get(str(col)+'enc'+str(5))
        fold7 = Dictionary.get(str(col)+'enc'+str(6))
        fold8 = Dictionary.get(str(col)+'enc'+str(7))
        fold9 = Dictionary.get(str(col)+'enc'+str(8))
        fold10 = Dictionary.get(str(col)+'enc'+str(9))       
        
        #applying mean of all folds to test dataset
        folds = pd.concat([fold1, fold2, fold3, fold4, fold5, fold6,fold7, fold8, fold9, fold10], axis=1)
        
        '''
        print('printing folds')
        print(folds)
        '''
        
        foldsMean = folds.mean(axis=1)                
        
        #creating column in test data set
        testCopy[col+'Enc'] = testCopy[col].map(foldsMean)
    
    #filtering        
    trainCopy = trainCopy.filter(like = 'Enc', axis=1)
    
    
    #creating 3 key attributes with the probabilities :
    trainCopy['binaryMean'] = trainCopy.mean(axis=1)
    trainCopy['binaryMin'] = trainCopy.min(axis=1)
    trainCopy['binaryMax'] = trainCopy.max(axis=1)
    
    #check
    print('printingMean')
    print(trainCopy['binaryMean'])
   
    print('printingMax')
    print(trainCopy['binaryMax'])
    
    #repeate for test set
    testCopy = testCopy.filter(like = 'Enc', axis=1)
    testCopy['binaryMean'] = testCopy.mean(axis=1)
    testCopy['binaryMin'] = testCopy.min(axis=1)
    testCopy['binaryMax'] = testCopy.max(axis=1)
    
    #filter for relevant columns
    trainCopy = trainCopy.filter(like = 'binary', axis=1)
    testCopy = testCopy.filter(like = 'binary', axis=1)
    
    
    
    return trainCopy, testCopy    



# ===============================================================================     
'''
Binarization helper function
'''
def xgbClass(trainDF, testDF):
    
    #we know there are 8 classes, so we can specify the output probability dataframe
    outProbDF = pd.DataFrame(index=range(trainDF.shape[0]), columns = range(8))
  
    
    #filling in values for different classifications
    outProbDF.iloc[:,0] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==1) else 0)
    outProbDF.iloc[:,1] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==2) else 0)
    outProbDF.iloc[:,2] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==3) else 0)
    outProbDF.iloc[:,3] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==4) else 0)
    outProbDF.iloc[:,4] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==5) else 0)
    outProbDF.iloc[:,5] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==6) else 0)
    outProbDF.iloc[:,6] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==7) else 0)
    outProbDF.iloc[:,7] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==8) else 0)    
    
    
    
    #making output columns for dataframe
    probTrain = pd.DataFrame(index=range(trainDF.shape[0]), columns = range(8))
    probTest = pd.DataFrame(index=range(testDF.shape[0]), columns = range(8))
    
    trainInDF = trainDF.drop('Response', 1)
    
    #using xgbclassifier
    alg = xgb.XGBClassifier()
    
    #defining dimensions for stratified kfold
    skfX = trainDF.drop('Response', 1)
    skfY = outProbDF.iloc[:,0]
    
    for counter in range(0,8):
        #stratified kfold
        skf = StratifiedKFold(n_splits = 5)
        for trIndex, cvIndex in skf.split(skfX,skfY):
            
            trainInput = trainInDF.iloc[trIndex]
            trainOutput =  outProbDF.iloc[:,counter][trIndex]
            
            
            alg.fit(trainInput, trainOutput)
            predictions = alg.predict_proba(trainInDF.iloc[cvIndex])
            probTrain.iloc[cvIndex, counter] = predictions
            
            
        #testing set next
        alg.fit(trainInDF, outProbDF.iloc[:,counter])
        predictions = alg.predict_proba(testDF)[:,1]
        probTest.iloc[:,counter] = predictions
    
    
    return probTrain, probTest



# ===============================================================================     
'''
Testing XGBBoost algorithm for binary probabilities on the dataset   
'''
def xgbTest(trainInput, trainOutput, testInput, predictors):
    
    
    trainDF= pd.concat([trainInput, trainOutput], axis =1)
    testDF= testInput
    
    #calling helper xgbClass to apply binarization function
    probTrain, probTest = xgbClass(trainDF,testDF)
    
    #splitting df
    trainInput = trainDF.drop('Response', 1)
    testInput = testDF
    trainOutput = trainDF.loc[:, 'Response']
    
    #combining new columns
    trainInput = pd.concat([trainInput, probTrain], axis =1)
    testInput = pd.concat([testInput, probTest], axis =1)
    
    
    #defining algorithms
    alg = xgb.XGBClassifier()
    
    #defining specific columns for analysis
    s = "Product_Info_1 Product_Info_2 	Product_Info_3	Product_Info_4	Product_Info_5	Product_Info_6	Product_Info_7	Ins_Age	Ht	Wt	BMI	Employment_Info_1	Employment_Info_2	Employment_Info_3	Employment_Info_4	Employment_Info_5	Employment_Info_6	InsuredInfo_1	InsuredInfo_2	InsuredInfo_3	InsuredInfo_4	InsuredInfo_5	InsuredInfo_6	InsuredInfo_7	Insurance_History_1	Insurance_History_2	Insurance_History_3	Insurance_History_4	Insurance_History_5	Insurance_History_7	Insurance_History_8	Insurance_History_9	Family_Hist_1	Family_Hist_2	Family_Hist_3	Family_Hist_4	Family_Hist_5	Medical_History_1	Medical_History_2	Medical_History_3	Medical_History_4	Medical_History_5	Medical_History_6	Medical_History_7	Medical_History_8	Medical_History_9	Medical_History_10	Medical_History_11	Medical_History_12	Medical_History_13	Medical_History_14	Medical_History_15	Medical_History_16	Medical_History_17	Medical_History_18	Medical_History_19	Medical_History_20	Medical_History_21	Medical_History_22	Medical_History_23	Medical_History_24	Medical_History_25	Medical_History_26	Medical_History_27	Medical_History_28	Medical_History_29	Medical_History_30	Medical_History_31	Medical_History_32	Medical_History_33	Medical_History_34	Medical_History_35	Medical_History_36	Medical_History_37	Medical_History_38	Medical_History_39	Medical_History_40	Medical_History_41	Medical_Keyword_1	Medical_Keyword_2	Medical_Keyword_3	Medical_Keyword_4	Medical_Keyword_5	Medical_Keyword_6	Medical_Keyword_7	Medical_Keyword_8	Medical_Keyword_9	Medical_Keyword_10	Medical_Keyword_11	Medical_Keyword_12	Medical_Keyword_13	Medical_Keyword_14	Medical_Keyword_15	Medical_Keyword_16	Medical_Keyword_17	Medical_Keyword_18	Medical_Keyword_19	Medical_Keyword_20	Medical_Keyword_21	Medical_Keyword_22	Medical_Keyword_23	Medical_Keyword_24	Medical_Keyword_25	Medical_Keyword_26	Medical_Keyword_27	Medical_Keyword_28	Medical_Keyword_29	Medical_Keyword_30	Medical_Keyword_31	Medical_Keyword_32	Medical_Keyword_33	Medical_Keyword_34	Medical_Keyword_35	Medical_Keyword_36	Medical_Keyword_37	Medical_Keyword_38	Medical_Keyword_39	Medical_Keyword_40	Medical_Keyword_41	Medical_Keyword_42	Medical_Keyword_43	Medical_Keyword_44	Medical_Keyword_45	Medical_Keyword_46	Medical_Keyword_47	Medical_Keyword_48 productInfoLetter	productInfoInt	medKeyCount	binaryMean	binaryMin	binaryMax"

    
    v = s.split()
    predictors = v
    
    #cvmean score calculator
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    
    
    print("XGB Average Score:", cvMeanScore)



# ===============================================================================    
'''
Correlation checking helper function
'''
def corrTest(df, colNames, test):
    
    corrDF = df.loc[:, colNames]
    
    corrFigures = corrDF.apply(lambda col: col.corr(test) ,axis =0)
    
    return corrFigures



# ===============================================================================    
'''
Dedicated function to visualize classification report
'''
def classificationReport(trainInput, trainOutput):
    
    #creating test from training data
    xTrain, xTest, yTrain, yTest = train_test_split(trainInput, trainOutput, test_size = 0.2, random_state = 1)
    
    
    #defining base models
    baseModels = [("mlp", MLPClassifier(alpha = 0.006, hidden_layer_sizes = 100)), 
                  ("xgbC", xgb.XGBClassifier(gamma = 0.6, max_depth = 3)), ("rForest", RandomForestClassifier(max_features= 0.9, n_estimators = 125 ))]
    
    #second level model
    metaModel = LogisticRegression(multi_class = 'ovr', solver='liblinear')

    #models to be evaluated
    models = [LogisticRegression(multi_class = 'ovr', solver='liblinear'), RandomForestClassifier(), StackingClassifier(estimators = baseModels, final_estimator = metaModel)]
    
    #define helper function for repeated iterations
    def classVis(alg):
        visualizer = ClassificationReport(alg, support=True)
        visualizer.fit(xTrain, yTrain)  
        visualizer.score(xTest, yTest)       
        return visualizer.poof()
    
    #apply and plot to all models
    for i in models:
        ax = plt.subplot(2,2,2)
        classVis(i)
 
    

# ===============================================================================    
'''
Dedicated function to evaluate and visualize classification error in algorithm predictions
'''        
def classificationError(trainInput, trainOutput):
    
    #creating test data
    xTrain, xTest, yTrain, yTest = train_test_split(trainInput, trainOutput, test_size = 0.2, random_state = 1)
    
    
    #defining base models
    baseModels = [("mlp", MLPClassifier(alpha = 0.006, hidden_layer_sizes = 100)), 
                  ("xgbC", xgb.XGBClassifier(gamma = 0.6, max_depth = 3)), ("rForest", RandomForestClassifier(max_features= 0.9, n_estimators = 125 ))]
    
    #second level model
    metaModel = LogisticRegression(multi_class = 'ovr', solver='liblinear')

    #models that will be passed
    models = [ LogisticRegression(multi_class = 'ovr', solver='liblinear'), RandomForestClassifier(), StackingClassifier(estimators = baseModels, final_estimator = metaModel)]
    
    #define helper function for repeated iterations    
    def errorVis(alg):
        visualizer = ClassPredictionError(alg)
        visualizer.fit(xTrain, yTrain)  
        visualizer.score(xTest, yTest)        
        return visualizer.poof()

    #apply and plot to all models    
    for i in models:
        ax = plt.subplot(2,2,2)
        errorVis(i)   



# ===============================================================================    
'''
Dedicated function to calculate and plot ROC-AUC curve
'''        
def rocAucCurve(trainInput, trainOutput):
    
    #creating test data
    xTrain, xTest, yTrain, yTest = train_test_split(trainInput, trainOutput, test_size = 0.2, random_state = 1)
    
    
    #defining base models
    baseModels = [("mlp", MLPClassifier(alpha = 0.006, hidden_layer_sizes = 100)), 
                  ("xgbC", xgb.XGBClassifier(gamma = 0.6, max_depth = 3)), ("rForest", RandomForestClassifier(max_features= 0.9, n_estimators = 125 ))]
    
    #second level model
    metaModel = LogisticRegression(multi_class = 'ovr', solver='liblinear')

    #models that will be passed    
    models = [ LogisticRegression(multi_class = 'ovr', solver='liblinear'), RandomForestClassifier(), StackingClassifier(estimators = baseModels, final_estimator = metaModel)]
    
    #define helper function for repeated iterations    
    def rocAucVis(alg):
        visualizer = ROCAUC(alg)
        visualizer.fit(xTrain, yTrain)  
        visualizer.score(xTest, yTest)        
        return visualizer.poof()

    #apply and plot to all models     
    for i in models:
        ax = plt.subplot(2,2,2)
        rocAucVis(i)   
        


# ===============================================================================    
'''
Dedicated function to calculate hamming loss and cohen kappa score
'''                
def additionalMetrics(trainInput, trainOutput):

    #creating test data
    xTrain, xTest, yTrain, yTest = train_test_split(trainInput, trainOutput, test_size = 0.2, random_state = 1)
    
    
    #defining base models
    baseModels = [("mlp", MLPClassifier(alpha = 0.006, hidden_layer_sizes = 100)), 
                  ("xgbC", xgb.XGBClassifier(gamma = 0.6, max_depth = 3)), ("rForest", RandomForestClassifier(max_features= 0.9, n_estimators = 125 ))]
   
    #secondary model
    metaModel = LogisticRegression(multi_class = 'ovr', solver='liblinear')

    #relevant models
    models = [ LogisticRegression(multi_class = 'ovr', solver='liblinear'), RandomForestClassifier(), StackingClassifier(estimators = baseModels, final_estimator = metaModel)]
    
    #define helper function for repeated iterations  
    def addMetrics(alg):
        alg.fit(xTrain, yTrain)
        prediction = alg.predict(xTest)
        
        #these try and except methods were taken from online documentation
        try: 
            prob = alg.predict_proba(xTest)
            log_metric = log_loss(yTest,prob)
        except:
            prob = "Not probablistic"
            log_metric = 0
        else:
            prediction = alg.predict(xTest)
        
        #define and calculate the metrics
        cks=cohen_kappa_score(yTest,prediction)
        hl=hamming_loss(yTest,prediction)  
        
        print('Cohen Kappa Score & Hamming Loss: ')
        print(cks)
        print(hl)
   
    #to iterate over multiple models 
    for i in models:
        print(str(i))
        addMetrics(i)
        print("Next Model Below")



# ===============================================================================
'''
Chi2 classification feature selection method
'''
def featureSelectionCat(trainInput, trainOutput, testInput):
    
     #define algorithm and fit
     alg = SelectKBest(score_func=chi2, k='all')
     alg.fit(trainInput, trainOutput)
     
     #printoutputs
     for i in range(len(alg.scores_)):
         print('Predictor %d: %f' % (i, alg.scores_[i]))
        
     pyplot.bar([i for i in range(len(alg.scores_))], alg.scores_)
     pyplot.show()
         
     
     
# ===============================================================================        
'''
Mutual Information Feature Selection Method
'''
def featureSelectionMIF(trainInput, trainOutput, testInput):
    
     alg = SelectKBest(score_func=mutual_info_classif, k='all')
     alg.fit(trainInput, trainOutput)

     #printoutputs
     for i in range(len(alg.scores_)):
         print('Predictor %d: %f' % (i, alg.scores_[i]))
    
     pyplot.bar([i for i in range(len(alg.scores_))], alg.scores_)
     pyplot.show()
     
     
     
# ===============================================================================        
'''
Wrapper method for RFE wrapper
'''     
def featureSelectionRFE(trainInput, trainOutput):
    
    #define rfe object
    rfe = RFE(estimator = RandomForestClassifier(), n_features_to_select = 110)
    
    rfe.fit(trainInput, trainOutput)
    
    for i in range(trainInput.shape[1]):
        print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))


# ===============================================================================
'''
Rapids library code (utilized in Google Colab to allow for GPU utilization)
'''
def rapidsCode():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    device_name = pynvml.nvmlDeviceGetName(handle)

    if device_name != b'Tesla T4':
        raise Exception("""
                        Unfortunately this instance does not have a T4 GPU.
                        
                        Please make sure you've configured Colab to request a GPU instance type.
                        
                        Sometimes Colab allocates a Tesla K80 instead of a T4. Resetting the instance.
                        
                        If you get a K80 GPU, try Runtime -> Reset all runtimes...
                        """)
    
    else:
        print('Woo! You got the right kind of GPU!')    
    
    sys.path.append('/usr/local/lib/python3.6/site-packages/')
    os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
    os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice/'

    # copy .so files to current working dir
    for fn in ['libcudf.so', 'librmm.so']:
        shutil.copy('/usr/local/lib/'+fn, os.getcwd())
        
        
        
# ===============================================================================
'''
Helper function for pre-processing visualizations and trial/error investigative work
'''
def preprocessHelper(trainDF, testDF):
    
    '''
    Initializing data
    '''
    fullDF = trainDF
    trainInput = trainDF.iloc[:, :127]
    testInput = testDF.iloc[:, :]
    
    trainOutput = trainDF.loc[:, 'Response']
    testIDs = testDF.loc[:, 'Id']
    
    
    '''
    Checking missing values
    '''
    
    missingPercent = trainInput.isnull().sum()/len(trainInput)
    missingPercent = missingPercent[missingPercent>0]
    
    
    '''
    Bar plot to visualize missing values
    '''
    
    missingCols = ['Employment_Info_1','Employment_Info_4','Employment_Info_6','Insurance_History_5','Family_Hist_2','Family_Hist_3','Family_Hist_4','Family_Hist_5','Medical_History_1','Medical_History_10','Medical_History_15','Medical_History_24','Medical_History_32']
    x=range(len(missingPercent))
    fig, ax = plt.subplots()
    
    ax.barh(x, missingPercent, align='center', color='red')
    ax.set_yticks(x)
    ax.set_yticklabels(missingCols)
    ax.invert_yaxis()
    ax.set_xlabel('Percentage')
    ax.set_title('Percentage of missing values on training data')
    for i,v in enumerate(missingPercent):
        ax.text(v, i+.2, str(v), color='black', fontweight='bold')
    
    
    '''
    Exploring classification imbalance through visualization
    '''
    # As this is a multi-classification problem, classification of training data is an important indicator
    # to further understand the data
    
    sns.set_color_codes()
    plt.figure(figsize=(12,12))
    sns.countplot(trainOutput, color='blue').set_title('Count of Output Class')
    
    
    '''
    Exploring key variables through visualization
    '''
    #Weight
    f, ax = plt.subplots(1,2, figsize=(14,7))
    sns.boxplot(x='Wt', data=trainInput, orient='v', ax=ax[0])
    sns.displot(trainInput['Wt'], ax=ax[1])
    
    #Age
    f, ax = plt.subplots(1,2, figsize=(14,7))
    sns.boxplot(x='Ins_Age', data=trainInput, orient='v', ax=ax[0])
    sns.displot(trainInput['Ins_Age'], ax=ax[1])
    
    #BMI
    f, ax = plt.subplots(1,2, figsize=(14,7))
    sns.boxplot(x='BMI', data=trainInput, orient='v', ax=ax[0])
    sns.displot(trainInput['BMI'], ax=ax[1])
    
    #Height
    f, ax = plt.subplots(1,2, figsize=(14,7))
    sns.boxplot(x='Ht', data=trainInput, orient='v', ax=ax[0])
    sns.displot(trainInput['Ht'], ax=ax[1])
    
    
    '''
    Histograms with Hue
    '''
    #Distributions and boxplots are helpful, but we could get more insight using histograms that show classification associated with variable values'
    #Lets use our histogram and pdf helper function to draw visualizations for key predictors
    
    histData = fullDF
    histData = histData.rename(columns={'Response':'label'})
    
    items = {'Medical_History_1	','Medical_History_2',	'Medical_History_3',	'Medical_History_4',	'Medical_History_5',	'Medical_History_6',	'Medical_History_7',	'Medical_History_8',	'Medical_History_9',	'Medical_History_10',	'Medical_History_11', 'Medical_History_12',	'Medical_History_13',	'Medical_History_14',	'Medical_History_15',	'Medical_History_16',	'Medical_History_17',	'Medical_History_18',	'Medical_History_19',	'Medical_History_20',	'Medical_History_21',	'Medical_History_22',	'Medical_History_23',	'Medical_History_24',	'Medical_History_25',	'Medical_History_26',	'Medical_History_27',	'Medical_History_28',	'Medical_History_29',	'Medical_History_30',	'Medical_History_31',	'Medical_History_32','Medical_History_33',	'Medical_History_34',	'Medical_History_35',	'Medical_History_36',	'Medical_History_37',	'Medical_History_38',	'Medical_History_39',	'Medical_History_40',	'Medical_History_41'}
    pdfHist('TestHistogram.pdf', histData, items)
    
    
    '''
    Scatterplots
    '''
    #scatterplots plots will help us to view relationship between predictors
    
    histData = fullDF
    histData = histData.rename(columns={'Response':'label'})
    
    i = "Employment_Info_1	Employment_Info_2	Employment_Info_3	Employment_Info_4	Employment_Info_5	Employment_Info_6 InsuredInfo_1	 InsuredInfo_2	InsuredInfo_3	InsuredInfo_4	InsuredInfo_5	InsuredInfo_6	InsuredInfo_7"
    l = i.split()
    itemsList= list(itertools.permutations(l,2))
    #items = {'Id',	'Product_Info_1',	'Product_Info_2',	'Product_Info_3',	'Product_Info_4',	'Product_Info_5',	'Product_Info_6',	'Product_Info_7',	'Ins_Age',	'Ht',	'Wt',	'BMI'}
    #itemsList = list(itertools.permutations(items,2))
    
    pdfScatter('employmentInsuredScatter.pdf', histData, itemsList)
    
    
    '''
    initial correlation analysis
    '''    
    
    s = "Id	Product_Info_1	Product_Info_3	Product_Info_4	Product_Info_5	Product_Info_6	Product_Info_7	Ins_Age	Ht	Wt	BMI	Employment_Info_1	Employment_Info_2	Employment_Info_3	Employment_Info_4	Employment_Info_5	Employment_Info_6	InsuredInfo_1	InsuredInfo_2	InsuredInfo_3	InsuredInfo_4	InsuredInfo_5	InsuredInfo_6	InsuredInfo_7	Insurance_History_1	Insurance_History_2	Insurance_History_3	Insurance_History_4	Insurance_History_5	Insurance_History_7	Insurance_History_8	Insurance_History_9	Family_Hist_1	Family_Hist_2	Family_Hist_3	Family_Hist_4	Family_Hist_5	Medical_History_1	Medical_History_2	Medical_History_3	Medical_History_4	Medical_History_5	Medical_History_6	Medical_History_7	Medical_History_8	Medical_History_9	Medical_History_10	Medical_History_11	Medical_History_12	Medical_History_13	Medical_History_14	Medical_History_15	Medical_History_16	Medical_History_17	Medical_History_18	Medical_History_19	Medical_History_20	Medical_History_21	Medical_History_22	Medical_History_23	Medical_History_24	Medical_History_25	Medical_History_26	Medical_History_27	Medical_History_28	Medical_History_29	Medical_History_30	Medical_History_31	Medical_History_32	Medical_History_33	Medical_History_34	Medical_History_35	Medical_History_36	Medical_History_37	Medical_History_38	Medical_History_39	Medical_History_40	Medical_History_41	Medical_Keyword_1	Medical_Keyword_2	Medical_Keyword_3	Medical_Keyword_4	Medical_Keyword_5	Medical_Keyword_6	Medical_Keyword_7	Medical_Keyword_8	Medical_Keyword_9	Medical_Keyword_10	Medical_Keyword_11	Medical_Keyword_12	Medical_Keyword_13	Medical_Keyword_14	Medical_Keyword_15	Medical_Keyword_16	Medical_Keyword_17	Medical_Keyword_18	Medical_Keyword_19	Medical_Keyword_20	Medical_Keyword_21	Medical_Keyword_22	Medical_Keyword_23	Medical_Keyword_24	Medical_Keyword_25	Medical_Keyword_26	Medical_Keyword_27	Medical_Keyword_28	Medical_Keyword_29	Medical_Keyword_30	Medical_Keyword_31	Medical_Keyword_32	Medical_Keyword_33	Medical_Keyword_34	Medical_Keyword_35	Medical_Keyword_36	Medical_Keyword_37	Medical_Keyword_38	Medical_Keyword_39	Medical_Keyword_40	Medical_Keyword_41	Medical_Keyword_42	Medical_Keyword_43	Medical_Keyword_44	Medical_Keyword_45	Medical_Keyword_46	Medical_Keyword_47	Medical_Keyword_48"
    v = s.split()
    corrTable = corrTest(trainInput, v, trainOutput)
    y = corrTable
    
    #plot barplot    
    plt.figure(figsize= (25,15))
    ax = corrTable.plot(kind='bar')
    ax.set_title('Correlation with Response')
    ax.set_xlabel('Predictor')
    ax.set_ylabel('Corr')
    ax.set_xticklabels(v)
    
    plotLabels(ax)
    
    plt.savefig("corrFigureCont0")
    

    '''
    missing values
    '''
    
    trainInput.loc[:, 'Medical_History_32'].fillna(0, inplace=True)
    trainInput.loc[:, 'Medical_History_24'].fillna(0, inplace=True)
    trainInput.loc[:, 'Medical_History_15'].fillna(0, inplace=True)
    trainInput.loc[:, 'Medical_History_10'].fillna(0, inplace=True)
    trainInput.loc[:, 'Medical_History_1'].fillna(0, inplace=True)
    #continuous
    trainInput.loc[:, 'Family_Hist_5'].fillna(0, inplace=True)
    trainInput.loc[:, 'Family_Hist_4'].fillna(0, inplace=True)  
    trainInput.loc[:, 'Family_Hist_3'].fillna(0, inplace=True)
    trainInput.loc[:, 'Family_Hist_2'].fillna(0, inplace=True)    
    trainInput.loc[:, 'Employment_Info_1'].fillna(0, inplace=True)
    trainInput.loc[:, 'Employment_Info_4'].fillna(0, inplace=True)  
    trainInput.loc[:, 'Employment_Info_6'].fillna(0, inplace=True)
    trainInput.loc[:, 'Insurance_History_5'].fillna(0, inplace=True)    

    #Checking correlations after filling missing values discrete variable
    s = "Id	Product_Info_1	Product_Info_3	Product_Info_4	Product_Info_5	Product_Info_6	Product_Info_7	Ins_Age	Ht	Wt	BMI	Employment_Info_1	Employment_Info_2	Employment_Info_3	Employment_Info_4	Employment_Info_5	Employment_Info_6	InsuredInfo_1	InsuredInfo_2	InsuredInfo_3	InsuredInfo_4	InsuredInfo_5	InsuredInfo_6	InsuredInfo_7	Insurance_History_1	Insurance_History_2	Insurance_History_3	Insurance_History_4	Insurance_History_5	Insurance_History_7	Insurance_History_8	Insurance_History_9	Family_Hist_1	Family_Hist_2	Family_Hist_3	Family_Hist_4	Family_Hist_5	Medical_History_1	Medical_History_2	Medical_History_3	Medical_History_4	Medical_History_5	Medical_History_6	Medical_History_7	Medical_History_8	Medical_History_9	Medical_History_10	Medical_History_11	Medical_History_12	Medical_History_13	Medical_History_14	Medical_History_15	Medical_History_16	Medical_History_17	Medical_History_18	Medical_History_19	Medical_History_20	Medical_History_21	Medical_History_22	Medical_History_23	Medical_History_24	Medical_History_25	Medical_History_26	Medical_History_27	Medical_History_28	Medical_History_29	Medical_History_30	Medical_History_31	Medical_History_32	Medical_History_33	Medical_History_34	Medical_History_35	Medical_History_36	Medical_History_37	Medical_History_38	Medical_History_39	Medical_History_40	Medical_History_41	Medical_Keyword_1	Medical_Keyword_2	Medical_Keyword_3	Medical_Keyword_4	Medical_Keyword_5	Medical_Keyword_6	Medical_Keyword_7	Medical_Keyword_8	Medical_Keyword_9	Medical_Keyword_10	Medical_Keyword_11	Medical_Keyword_12	Medical_Keyword_13	Medical_Keyword_14	Medical_Keyword_15	Medical_Keyword_16	Medical_Keyword_17	Medical_Keyword_18	Medical_Keyword_19	Medical_Keyword_20	Medical_Keyword_21	Medical_Keyword_22	Medical_Keyword_23	Medical_Keyword_24	Medical_Keyword_25	Medical_Keyword_26	Medical_Keyword_27	Medical_Keyword_28	Medical_Keyword_29	Medical_Keyword_30	Medical_Keyword_31	Medical_Keyword_32	Medical_Keyword_33	Medical_Keyword_34	Medical_Keyword_35	Medical_Keyword_36	Medical_Keyword_37	Medical_Keyword_38	Medical_Keyword_39	Medical_Keyword_40	Medical_Keyword_41	Medical_Keyword_42	Medical_Keyword_43	Medical_Keyword_44	Medical_Keyword_45	Medical_Keyword_46	Medical_Keyword_47	Medical_Keyword_48"
    v = s.split()
    corrTable = corrTest(trainInput, v, trainOutput)
    y = corrTable

    #plot barplot    
    plt.figure(figsize= (25,15))
    ax = corrTable.plot(kind='bar')
    ax.set_title('Correlation with Response')
    ax.set_xlabel('Predictor')
    ax.set_ylabel('Corr')
    ax.set_xticklabels(v)
    
    plotLabels(ax)
    
    plt.savefig("corrFigureCont0")
    
    
    '''
    Heatmaps
    '''
    #heatmaps to check correlation between the numerous dummy variables
    
    cols = 'Medical_Keyword_1	Medical_Keyword_2	Medical_Keyword_3	Medical_Keyword_4	Medical_Keyword_5	Medical_Keyword_6	Medical_Keyword_7	Medical_Keyword_8	Medical_Keyword_9	Medical_Keyword_10	Medical_Keyword_11	Medical_Keyword_12	Medical_Keyword_13	Medical_Keyword_14	Medical_Keyword_15	Medical_Keyword_16	Medical_Keyword_17	Medical_Keyword_18	Medical_Keyword_19	Medical_Keyword_20	Medical_Keyword_21	Medical_Keyword_22	Medical_Keyword_23	Medical_Keyword_24	Medical_Keyword_25	Medical_Keyword_26	Medical_Keyword_27	Medical_Keyword_28	Medical_Keyword_29	Medical_Keyword_30	Medical_Keyword_31	Medical_Keyword_32	Medical_Keyword_33	Medical_Keyword_34	Medical_Keyword_35	Medical_Keyword_36	Medical_Keyword_37	Medical_Keyword_38	Medical_Keyword_39	Medical_Keyword_40	Medical_Keyword_41	Medical_Keyword_42	Medical_Keyword_43	Medical_Keyword_44	Medical_Keyword_45	Medical_Keyword_46	Medical_Keyword_47	Medical_Keyword_48'
    colList= cols.split()
    correlation = trainInput.loc[:,colList].corr()
    plt.figure(figsize=(50,50))
    sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
    plt.title('Correlation between Medical_Keywords')



# ===============================================================================     
'''
Dedicated function for bar plots
'''
def plotLabels (ax, spacing =4):
    for bar in ax.patches:
        yValue = bar.get_height()
        xValue = bar.get_x() + bar.get_width()/2
        
        va='bottom'
        
        if yValue <0:
            spacing=-1
            va='top'
        label ='{:.3f}'.format(yValue)
        
        ax.annotate(label, (xValue, yValue), xytext=(0,spacing), textcoords='offset points', ha='center', va=va)



# ===============================================================================     
'''
Histogram helper function
'''
def histHelper (ax, data, x, hue):
    
    xValues = data[x]
    dataHue = data[hue]
    hueLabels = sorted(dataHue.unique())
    
    
    #plotting
    values = []
    color = sns.color_palette(palette='icefire', n_colors=len(hueLabels))
    
    for i in hueLabels:
        try:
            add = np.array(dataHue == i)
            values.append(np.array(xValues[add]))
            
        except KeyError:
            values.append(np.array([]))
            
    ax.hist(x = values, color = color, bins=10, histtype='barstacked', label=hueLabels)
    ax.legend()
    ax.set(xlabel=x)
    
    
    
# ===============================================================================     
'''
Saves histograms to pdfs    
'''
def pdfHist(fileName, data, items):
    plt.close("all")
    
    with PdfPages(fileName) as pdf:
        nRows, nCols = 2 , 2
        axes = nRows*nCols
        
        for i, z in enumerate(items):
            nAxis = i % axes
            r = nAxis// nRows
            c = nAxis % nRows

            if nAxis == 0:
                f, ax = plt.subplots(nRows, nCols, figsize = (15,10))
                
            histHelper(ax[r,c], data, z, 'label')
                
            print('drawing{}'.format(z))
                
            if (i % axes == axes-1) or (i == len(items)-1):
                f.tight_layout()
                pdf.savefig(f)
                plt.close('all')
                print('pdf histplot saved')
                


# ===============================================================================     
'''
Helper function to make scatter plots               
'''
def scatterHelper(ax, data, x, y, hue):
    
    xValues = data[x]
    yValues = data[y]
    hueValues = data[hue]
    hueLabels=sorted(hueValues.unique())
    
    #random sample with set seed
    np.random.seed(0)
    length = len(data)
    sampleSelection = np.random.choice(length, np.min([5000,length]), replace=False)
    
    xSample = xValues[sampleSelection]
    ySample = yValues[sampleSelection]
    hueSample = hueValues[sampleSelection]
    #plotting
    for i, z in enumerate(hueLabels):
        try:
            add = np.array(hueSample == z)
            labelSampleX = xSample[add]
            labelSampleY = ySample[add]
            ax.scatter(labelSampleX, labelSampleY, s=10, color=sns.color_palette(palette='icefire', n_colors=len(hueLabels))[i], alpha = .8, marker='+', edgecolors='none', label = z, rasterized = True)
        except KeyError:
            print("Key error {}".format(z))
            
    ax.legend()
    ax.set(xlabel=x, ylabel=y)
 
    
 
# ===============================================================================     
'''
Saves scatterplots to pdfs    
'''
def pdfScatter(fileName, data, items):
    plt.close("all")
    
    with PdfPages(fileName) as pdf:
        nRows, nCols = 2 , 2
        axes = nRows*nCols
        
        for i, z in enumerate(items):
            nAxis = i % axes
            r = nAxis// nRows
            c = nAxis % nRows

            if nAxis == 0:
                f, ax = plt.subplots(nRows, nCols, figsize = (15,10))
                
            x, y = z
            scatterHelper(ax[r,c], data, x, y, 'label')
                
            print('drawing{}'.format(z))
                
            if (i % axes == axes-1) or (i == len(items)-1):
                f.tight_layout()
                pdf.savefig(f)
                plt.close('all')
                print('pdf scatter saved')       



# ===============================================================================    
'''
Function for set of experiments regarding different models
'''
def doExperiment(trainInput, trainOutput, predictors):
    
    
    cv = 10
    
    
    """
    Adding Multi class Logistic Regression (One vs Rest Algorithm)
    """
    
    alg = LogisticRegression(multi_class = 'ovr', solver='liblinear')
    
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=cv, scoring='accuracy', n_jobs=-1).mean()
    print("Logistic Regression Average Score:", cvMeanScore)
    
    
    """
    Adding Random Forest Classifier
    """
    
    alg = RandomForestClassifier()
    
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=cv, scoring='accuracy', n_jobs=-1).mean()
    print("Random Forest Classifier Average Score:", cvMeanScore)
    
    
    """
    Adding Random Forest Regressor
    """
    
    alg = RandomForestRegressor()
    
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=cv, scoring='r2', n_jobs=-1).mean()
    print("Random Forest Regressor Average Score:", cvMeanScore)
    
    
    """
    Adding Extra Trees Classifier
    """
    
    alg = ExtraTreesClassifier()
    
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=cv, scoring='accuracy', n_jobs=-1).mean()
    print("Extra Trees Classifier Average Score:", cvMeanScore)
    
    
    """
    Adding KNeighborsClassifier 
    """
    
    alg = KNeighborsClassifier()
    
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=cv, scoring='accuracy', n_jobs=-1).mean()
    print("KNeighborsClassifier Average Score:", cvMeanScore)
    
    
    """
    Adding Neural Network
    """
    
    alg = MLPClassifier()    
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=cv, scoring='accuracy', n_jobs=-1).mean()
    print("MLP Classifier Average Score:", cvMeanScore)    
    
    
    """
    Adding Extra Trees Regressor
    """
    
    alg = ExtraTreesRegressor()
    
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=cv, scoring='r2', n_jobs=-1).mean()
    print("Extra Trees Regressor Average Score:", cvMeanScore)
    


# ===============================================================================    
'''
Function for set of experiments regarding initial models
'''    
def initialExperiments(trainInput, trainOutput, predictors):    
    
    '''
    GradientBoostingRegressor
    '''
    
    alg1 = GradientBoostingRegressor()
    alg1.fit(trainInput.loc[:,predictors], trainOutput)
    cvScores = model_selection.cross_val_score(alg1, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')
    print("Accuracy: ", cvScores.mean())
    
    
    '''
    Ridge Regression
    '''

    alg2 = Ridge(alpha=250)
    alg2.fit(trainInput.loc[:,predictors], trainOutput)
    cvScores = model_selection.cross_val_score(alg2, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')
    print("Accuracy: ", cvScores.mean())
    

    '''
    Lasso Regression
    '''
    
    alg3 = Lasso(alpha=800)
    alg3.fit(trainInput.loc[:,predictors], trainOutput)
    cvScores = model_selection.cross_val_score(alg3, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')
    print("Accuracy: ", cvScores.mean())
    
    
    '''
    Bagging meta-estimator
    '''
    
    alg4 = BaggingRegressor(KNeighborsClassifier(), max_samples = 0.85, max_features=0.85)
    alg4.fit(trainInput.loc[:,predictors], trainOutput)
    cvScores = model_selection.cross_val_score(alg4, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')
    print("Accuracy: ", cvScores.mean())



# ===============================================================================     
'''
Testing regressors on binary probabilities 
'''
def regressionTest(trainInput, trainOutput, testInput, predictors):
    
    trainDF= pd.concat([trainInput, trainOutput], axis =1)
    testDF= testInput
    
    
    #calling rowProb helper function
    probTrain, probTest = rowProb(trainDF,testDF)
    
        
    trainInput = trainDF.drop('Response', 1)
    testInput = testDF
    trainOutput = trainDF.loc[:, 'Response']
    
    
    #combining new columns
    trainInput = pd.concat([trainInput, probTrain], axis =1)
    testInput = pd.concat([testInput, probTest], axis =1)
    
    
    #making predictors
    s = "Product_Info_1 Product_Info_2 	Product_Info_3	Product_Info_4	Product_Info_5	Product_Info_6	Product_Info_7	Ins_Age	Ht	Wt	BMI	Employment_Info_1	Employment_Info_2	Employment_Info_3	Employment_Info_4	Employment_Info_5	Employment_Info_6	InsuredInfo_1	InsuredInfo_2	InsuredInfo_3	InsuredInfo_4	InsuredInfo_5	InsuredInfo_6	InsuredInfo_7	Insurance_History_1	Insurance_History_2	Insurance_History_3	Insurance_History_4	Insurance_History_5	Insurance_History_7	Insurance_History_8	Insurance_History_9	Family_Hist_1	Family_Hist_2	Family_Hist_3	Family_Hist_4	Family_Hist_5	Medical_History_1	Medical_History_2	Medical_History_3	Medical_History_4	Medical_History_5	Medical_History_6	Medical_History_7	Medical_History_8	Medical_History_9	Medical_History_10	Medical_History_11	Medical_History_12	Medical_History_13	Medical_History_14	Medical_History_15	Medical_History_16	Medical_History_17	Medical_History_18	Medical_History_19	Medical_History_20	Medical_History_21	Medical_History_22	Medical_History_23	Medical_History_24	Medical_History_25	Medical_History_26	Medical_History_27	Medical_History_28	Medical_History_29	Medical_History_30	Medical_History_31	Medical_History_32	Medical_History_33	Medical_History_34	Medical_History_35	Medical_History_36	Medical_History_37	Medical_History_38	Medical_History_39	Medical_History_40	Medical_History_41	Medical_Keyword_1	Medical_Keyword_2	Medical_Keyword_3	Medical_Keyword_4	Medical_Keyword_5	Medical_Keyword_6	Medical_Keyword_7	Medical_Keyword_8	Medical_Keyword_9	Medical_Keyword_10	Medical_Keyword_11	Medical_Keyword_12	Medical_Keyword_13	Medical_Keyword_14	Medical_Keyword_15	Medical_Keyword_16	Medical_Keyword_17	Medical_Keyword_18	Medical_Keyword_19	Medical_Keyword_20	Medical_Keyword_21	Medical_Keyword_22	Medical_Keyword_23	Medical_Keyword_24	Medical_Keyword_25	Medical_Keyword_26	Medical_Keyword_27	Medical_Keyword_28	Medical_Keyword_29	Medical_Keyword_30	Medical_Keyword_31	Medical_Keyword_32	Medical_Keyword_33	Medical_Keyword_34	Medical_Keyword_35	Medical_Keyword_36	Medical_Keyword_37	Medical_Keyword_38	Medical_Keyword_39	Medical_Keyword_40	Medical_Keyword_41	Medical_Keyword_42	Medical_Keyword_43	Medical_Keyword_44	Medical_Keyword_45	Medical_Keyword_46	Medical_Keyword_47	Medical_Keyword_48 productInfoLetter	productInfoInt	medKeyCount	binaryMean	binaryMin	binaryMax"
    v = s.split()
    predictors = v
    
    
    #running extra Trees Regressor 
    alg = ExtraTreesRegressor()
    
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("Extra Trees Regressor Average Score:", cvMeanScore)
    
    #running LogisticRegression
    alg = LogisticRegression(multi_class = 'ovr', solver='liblinear')
    
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("Logistic Regression Average Score:", cvMeanScore)



# ===============================================================================     
'''
This method is a helper function for the regression test method
Creates new columns containing binary probabilities
'''
def rowProb(trainDF, testDF):
    
    #we know there are 8 classes, so we can specify the output probability dataframe
    outProbDF = pd.DataFrame(index=range(trainDF.shape[0]), columns = range(8))
  

    #filling in values for different classifications so that each column contains data regarding 
    #a specific label or class
    outProbDF.iloc[:,0] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==1) else 0)
    outProbDF.iloc[:,1] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==2) else 0)
    outProbDF.iloc[:,2] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==3) else 0)
    outProbDF.iloc[:,3] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==4) else 0)
    outProbDF.iloc[:,4] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==5) else 0)
    outProbDF.iloc[:,5] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==6) else 0)
    outProbDF.iloc[:,6] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==7) else 0)
    outProbDF.iloc[:,7] = trainDF.loc[:, 'Response'].map(lambda val: 1 if(val==8) else 0)    
    
    #making output columns for dataframe
    probTrain = pd.DataFrame(index=range(trainDF.shape[0]), columns = range(8))
    probTest = pd.DataFrame(index=range(testDF.shape[0]), columns = range(8))
    
    trainInDF = trainDF.drop('Response', 1)
    
    alg = RandomForestClassifier()
    
    #dimensions for stratifiedkfold
    skfX = trainDF.drop('Response', 1)
    skfY = outProbDF.iloc[:,0]
    
    #loop through each column
    for counter in range(0,8):

        skf = StratifiedKFold(n_splits = 5)
        for trIndex, cvIndex in skf.split(skfX,skfY):
            
            trainInput = trainInDF.iloc[trIndex]
            trainOutput =  outProbDF.iloc[:,counter][trIndex]
            
            alg.fit(trainInput, trainOutput)
            
            #extracting probability of specific classification
            predictions = alg.predict_proba(trainInDF.iloc[cvIndex])[:,1]
            probTrain.iloc[cvIndex, counter] = predictions
            
        #testing data
        alg.fit(trainInDF, outProbDF.iloc[:,counter])
        predictions = alg.predict_proba(testDF)[:,1]
        probTest.iloc[:,counter] = predictions
    
    
    return probTrain, probTest



# ===============================================================================     
'''
This method is used for stacking of models    
'''
def stacker(trainInput, trainOutput, predictors):
    'defining base models'
    baseModels = [("LogR", LogisticRegression(multi_class = 'ovr', solver='liblinear')), 
                  ("Neigh", KNeighborsClassifier()), ("rForest", RandomForestClassifier())]
    metaModel = LogisticRegression(multi_class = 'ovr', solver='liblinear')
    
    alg = StackingClassifier(estimators = baseModels, final_estimator = metaModel)
    
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], 
                                                  trainOutput, cv=10, scoring='accuracy', n_jobs=-1).mean()
    print("Stacking Classifier Average Score:", cvMeanScore)    



# ===============================================================================     
'''
This method tests different stacked combination accuracies   
'''
def stackerTest(trainInput, trainOutput, predictors):
    
    '''
    first model
    '''
    baseModels = [("logReg", ExtraTreesClassifier()), 
                  ("xgbC", xgb.XGBClassifier()), ("rForest", RandomForestClassifier(max_features= 0.9, n_estimators = 300 ))]
    metaModel = xgb.XGBClassifier()
    
    alg = StackingClassifier(estimators = baseModels, final_estimator = metaModel)
    
    cvMeanScore = model_selection.cross_val_score(alg, trainInput, 
                                                  trainOutput, cv=10, scoring='accuracy', n_jobs=-1).mean()
    print("Stacking Classifier1 Average Score:", cvMeanScore)    
    
    '''
    second model
    '''
    
    
    baseModels = [("extraTree", ExtraTreesClassifier()), 
                  ("xgbC", xgb.XGBClassifier()), ("rForest", RandomForestClassifier(max_features= 0.9, n_estimators = 300 ))]
    metaModel = LogisticRegression(multi_class = 'multinomial', solver='lbfgs')
    
    alg = StackingClassifier(estimators = baseModels, final_estimator = metaModel)
    
    cvMeanScore = model_selection.cross_val_score(alg, trainInput, 
                                                  trainOutput, cv=10, scoring='accuracy', n_jobs=-1).mean()
    print("Stacking Classifier2 Average Score:", cvMeanScore) 
    


# ===============================================================================   
'''
This method is used for stacking of models    
'''
def stackerTest1(trainInput, trainOutput, predictors):
    
    #defining base models
    baseModels = [("mlp", MLPClassifier(alpha = 0.006, hidden_layer_sizes = 100)), 
                  ("xgbC", xgb.XGBClassifier(gamma = 0.6, max_depth = 3)), ("rForest", RandomForestClassifier(max_features= 0.9, n_estimators = 125 ))]
    
    #secondary model for stacking classifier
    metaModel = LogisticRegression(multi_class = 'ovr', solver='liblinear')
    
    alg = StackingClassifier(estimators = baseModels, final_estimator = metaModel)
    
    cvMeanScore = model_selection.cross_val_score(alg, trainInput, 
                                                  trainOutput, cv=10, scoring='accuracy', n_jobs=-1).mean()
    
    
    print("rowProb Stacking Classifier Average Score:", cvMeanScore)    
    
    
    
# ===============================================================================     
'''
Testing xgbBoost algorithm
'''
def xgbExperiment(trainInput, trainOutput, predictors):
    
    cv = 10

    
    """
    XGB Classifier
    """    
    
    alg = xgb.XGBClassifier()
    kFold = KFold(n_splits = cv, random_state=10)
    score = cross_val_score(alg, trainInput, trainOutput, cv = kFold).mean()
    
    print("XGB Classifier score:", score)



    """
    XGB Regressor
    """
    
    alg = xgb.XGBRegressor()
    kFold = KFold(n_splits = cv, random_state=10)
    score = cross_val_score(alg, trainInput, trainOutput, cv = kFold).mean()
    
    print("XGB Regressor score:", score)
    


    
# ===============================================================================    
'''
Implementation of vote based ensemble
'''                
def ensembleClassifier(trainInput, trainOutput):
    
    #defining base models
    baseModels = [("mlp", MLPClassifier(alpha = 0.006, hidden_layer_sizes = 100)), 
                  ("xgbC", xgb.XGBClassifier(gamma = 0.6, max_depth = 3)), ("logR", LogisticRegression(multi_class = 'ovr', solver='liblinear')), ("rForest", RandomForestClassifier(max_features= 0.9, n_estimators = 125 ))]
    
    #voting classifier from sklearn
    alg = VotingClassifier(estimators = baseModels, voting = 'hard')
    
    cvMeanScore = model_selection.cross_val_score(alg, trainInput, 
                                                  trainOutput, cv=10, scoring='accuracy', n_jobs=-1).mean()
    
    
    print("Voting Ensemble Classifier Average Score:", cvMeanScore)       



# ===============================================================================     
'''
This function is an implementation of the MLP I used in Tensor Flow with Keras API
'''    
def nnMLP(trainInput, trainOutput):
    
    #input shape for model
    inputShape = trainInput.shape[1]
    
    #defining model
    model = Sequential()
    model.add(Dense(40, activation = 'relu', kernel_initializer = 'he_normal', 
                    input_shape=(inputShape)))
    model.add(Dense(30, activation = 'relu', kernel_initializer = 'he_normal'))    
    model.add(Dense(20, activation = 'relu', kernel_initializer = 'he_normal'))   
    model.add(Dense(16, activation = 'relu', kernel_initializer = 'he_normal'))       
    model.add(Dense(8, activation='softmax'))
    
    #optimizer
    lrate = 0.2
    decay = lrate/80
    sgd = SGD(lr=lrate, decay=decay)
    
    #compiling the model
    model.compile(optimizer = sgd, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy', 'auc'])
    
    #fitting model
    model.fit(trainInput, trainOutput, epochs = 100, batch_size = 200, verbose = 1)
    
    #model evaluation
    loss, accur = model.evaluate(trainInput, trainOutput, verbose = 1)
    
    #print results
    print('nnMLP accuracy: ')
    print(accur)



def imageTransf(trainInput, trainOutput):
    

    trainInput, testInput, trainOutput, testOutput = train_test_split(trainInput, trainOutput, test_size = 0.2, stratify = trainOutput)
    
    
    ln = LogScaler()
    normTrain = ln.fit_transform(trainInput)
    normTest = ln.fit_transform(testInput)
    it = ImageTransformer(feature_extractor = 'tsne', pixels = 50, n_jobs = -1)
    plt.figure(figsize = (5,5))
    _ = it.fit(normTrain, plot = True)
    
    imageTrain = it.fit_transform(normTrain)
    imageTest = it.transform(normTest)
    
    fig, ax = plt.subplots(1, 4, figsize=(25, 7))
    for i in range(0,2):
        ax[i].imshow(imageTrain[i])
        ax[i].title.set_text("Train[{}] - class '{}'".format(i, trainOutput[i]))
        plt.tight_layout()
    
    
    model = models.Sequential()
    model.add(layers.Conv2D(50, (1,1), activation = 'relu', input_shape = (50,50,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(100,(3,3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2,2)))    
    model.add(layers.Conv2D(100,(3,3), activation = 'relu'))    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(10))
    
    model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  

    
    model.fit(imageTrain, trainOutput, epochs = 20, batch_size = 1600, verbose = 1)
    loss, acc = model.evaluate(imageTest, testOutput, batch_size = 1600, verbose = 1)
    print('Convolutional Neural Network Accuracy: ')
    print(acc)


# ===============================================================================     
'''
Implementation of a model that relies on caculated binary probabilities only to create output
'''
def binaryModel(trainInput, trainOutput, testInput):
    
    #concatenating to obtain dimensions for next step
    df = pd.concat([trainInput, trainOutput], axis = 1)
    
    #defining DF that will contain binary values pertaining to specific classes
    probDF = pd.DataFrame(index=range(df.shape[0]), columns = range(8))
    
    #filling misssing values
    probDF.iloc[:,0] = df.loc[:, 'Response'].map(lambda val: 1 if(val==1) else 0)
    probDF.iloc[:,1] = df.loc[:, 'Response'].map(lambda val: 1 if(val==2) else 0)
    probDF.iloc[:,2] = df.loc[:, 'Response'].map(lambda val: 1 if(val==3) else 0)
    probDF.iloc[:,3] = df.loc[:, 'Response'].map(lambda val: 1 if(val==4) else 0)
    probDF.iloc[:,4] = df.loc[:, 'Response'].map(lambda val: 1 if(val==5) else 0)
    probDF.iloc[:,5] = df.loc[:, 'Response'].map(lambda val: 1 if(val==6) else 0)
    probDF.iloc[:,6] = df.loc[:, 'Response'].map(lambda val: 1 if(val==7) else 0)
    probDF.iloc[:,7] = df.loc[:, 'Response'].map(lambda val: 1 if(val==8) else 0)    

    #creating probability dataframe
    probTrain = pd.DataFrame(index=range(df.shape[0]), columns = range(8))
    probTest = pd.DataFrame(index=range(df.shape[0]), columns = range(8))
        
    trainInDF = trainInput
    
    #alg with hyperoptimized params
    alg = xgb.XGBClassifier(gamma = 0.6, max_depth = 3)
    
    #dimensions for kFold
    kfX = trainInput
    kfY = probDF.iloc[:,0]
    
    #loop through each column
    for counter in range(0,8):
        
        #creating folds
        kf = KFold(n_splits = 5)
        for trIndex, cvIndex in kf.split(kfX, kfY):
            
            #defining specific folds for the iteration
            trInput = trainInDF.iloc[trIndex]
            trOutput =  probDF.iloc[:,counter][trIndex]
            
            #applying model
            alg.fit(trInput, trOutput)
            
            #extracting probability of specific classification
            predictions = alg.predict_proba(trainInDF.iloc[cvIndex])[:,1]
            probTrain.iloc[cvIndex, counter] = predictions
            
        #testing data
        alg.fit(trainInDF, probDF.iloc[:,counter])
        predictions = alg.predict_proba(testInput)[:,1]
        probTest.iloc[:,counter] = predictions
        
    #renaming columns
    probTrain = probTrain.set_axis(['prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7', 'prob8'], axis=1)  
    probTest = probTest.set_axis(['prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7', 'prob8'], axis=1) 
    
    #converting probabilities to numerical from 'object' type.
    probTrain['prob1']= pd.to_numeric(probTrain['prob1'], errors='coerce').fillna(0, downcast='infer')
    probTrain['prob2']= pd.to_numeric(probTrain['prob2'], errors='coerce').fillna(0, downcast='infer')
    probTrain['prob3']= pd.to_numeric(probTrain['prob3'], errors='coerce').fillna(0, downcast='infer')
    probTrain['prob4']= pd.to_numeric(probTrain['prob4'], errors='coerce').fillna(0, downcast='infer')
    probTrain['prob5']= pd.to_numeric(probTrain['prob5'], errors='coerce').fillna(0, downcast='infer')
    probTrain['prob6']= pd.to_numeric(probTrain['prob6'], errors='coerce').fillna(0, downcast='infer')
    probTrain['prob7']= pd.to_numeric(probTrain['prob7'], errors='coerce').fillna(0, downcast='infer')
    probTrain['prob8']= pd.to_numeric(probTrain['prob8'], errors='coerce').fillna(0, downcast='infer')
    
    #Applied multinomial LogisticRegression to the results
    secondAlg = LogisticRegression(multi_class = 'multinomial', solver='lbfgs')
    
    cvMeanScore = model_selection.cross_val_score(secondAlg, probTrain, 
                                                  trainOutput, cv=10, scoring='accuracy', n_jobs=-1).mean()
    
    
    
    print("Binary Multinomial Logistic Average Score:", cvMeanScore)    
    
    

         
        
       
        







                
                
    
# ===============================================================================     
'''
Function for hyperparameter tuning (gradient boosting regressor used here)  
'''
def hypParamTest(trainInput, trainOutput, predictors):
    
    '''
    Gradient Boosting Regressor
    '''    
   
    #define depth list
    depthList = pd.Series([2,2.5,3,3.5,4,4.5,5])
    
    #calculate accuracies
    accuracies = depthList.map(lambda x: (model_selection.cross_val_score(GradientBoostingRegressor(max_depth = x), trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')).mean())
    print(accuracies)
    
    #plot
    plt.plot(depthList, accuracies)
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    
    #print most efficient
    print("Most Efficient: ", depthList.loc[accuracies.idxmax()])
    
    
    '''
    Ridge
    '''
    alphaList = pd.Series([50,100,150,200,250,300])
    
    accuracies = alphaList.map(lambda x: (model_selection.cross_val_score(Ridge(alpha = x), trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')).mean())
    print(accuracies)
    
    
    plt.plot(alphaList, accuracies)
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    
    print("Most Efficient: ", alphaList.loc[accuracies.idxmax()])
    
    
    '''
    Lasso
    '''
    alphaList = pd.Series([50,100,200,300,400,500])
    
    accuracies = alphaList.map(lambda x: (model_selection.cross_val_score(Lasso(alpha = x), trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')).mean())
    print(accuracies)
    
    
    plt.plot(alphaList, accuracies)
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    
    
    print("Most Efficient: ", alphaList.loc[accuracies.idxmax()])


    
# ===============================================================================     
'''
This function calculates AUC scores as it performs better in high class imbalance unlike the more common 'accuracy' scoring
'''
def aucCheck(trainInput, trainOutput, predictors):
 
    #splitting trainInput and trainOutput to get a testing set from within our training set
    trInput = trainInput.iloc[0:50000, :]
    trOutput = trainOutput.iloc[0:50000]
    tsInput = trainInput.iloc[50000:,:]
    tsOutput = trainOutput.iloc[50000:]
 
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=1)
    
    """
    Multi class Logistic Regression (One vs Rest Algorithm)
    """
    
    alg = LogisticRegression(multi_class = 'ovr', solver='liblinear')

    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    print("Multi class Logistic Regression AUC Score:", cvMeanScore)
  
    
    """
    Adding Extra Trees Classifier
    """
    
    alg = ExtraTreesClassifier()
    
    model =alg.fit(trInput, trOutput)
    algPredict = model.predict(tsInput)
    print(metrics.roc_auc_score(tsOutput, algPredict, multi_class='ovo'))
    
    #cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    #print("Extra Classifier AUC Score:", cvMeanScore)
    

    """
    Adding Random Forest Classifier
    """
    
    alg = RandomForestClassifier()
    
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    print("Random Forest Classifier AUC Score:", cvMeanScore)
 
    
    
# ===============================================================================     
'''
Creates confusion matrix
'''
def confMat(trainInput, trainOutput):
    
    #splitting trainInput and trainOutput to get a testing set from within our training set
    
    trInput, tsInput, trOutput, tsOutput = train_test_split(trainInput, trainOutput, test_size = 0.15)
    
    #defining algorithms
    algs = [KNeighborsClassifier(), MLPClassifier()]
    
    
    #iterate over algorithms and plot  
    for i in algs:
        
        model = i.fit(trInput, trOutput)
        algPredict = model.predict(tsInput)
    
        matrix = confusion_matrix(tsOutput, algPredict)
        matrix = matrix.astype('float')/matrix.sum(axis=1)[:,np.newaxis]
    
        results = list(set(tsOutput))
    
        matrixDF = pd.DataFrame(matrix, index = results, columns = results)
    
        plt.figure(figsize=(12,10))
        sns.heatmap(matrixDF, annot=True, cmap="Blues")
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

    
# ===============================================================================     
'''
Helper function
'''        
def optimizePlot(gridObject, dictObject):

    paramA = dictObject['max_features']
    paramB = dictObject['n_estimators']
    
    scores = gridObject.cv_results_['mean_test_score']
    scores = np.array(scores).reshape(len(paramA), len(paramB))
    
    _, ax = plt.subplots(1,1)

    for row, i in enumerate(paramB):
        ax.plot(paramA, scores[row, :], label= 'n_estimators: '+ str(i))
    
    ax.tick_params(axis = 'x', rotation=0)
    ax.set_title('ExtraTreesClassifier Grid Search Scores')
    ax.set_xlabel('max_features')
    ax.set_ylabel('Score')
    ax.legend(loc='best')
    ax.grid('on')
    plt.show()
 

# ===============================================================================     
'''
specific use case function
'''        
def xgbStackConfMat(trainInput, trainOutput):
    
    'splitting trainInput and trainOutput to get a testing set from within our training set'
    print(trainOutput.head)
    trInput, tsInput, trOutput, tsOutput = train_test_split(trainInput, trainOutput, test_size = 0.15)
    print(trOutput.head)
    xgbclass = xgb.XGBClassifier()
    
    baseModels = [("mlp", MLPClassifier()), 
                  ("xgbC", xgbclass), ("rForest", RandomForestClassifier(max_features= 0.9, n_estimators = 300 ))]
    metaModel = LogisticRegression(multi_class = 'ovr', solver='liblinear')
    
    alg = StackingClassifier(estimators = baseModels, final_estimator = metaModel)
    
    alg.fit(trInput, trOutput)
    algPredict = alg.predict(tsInput)
    
    matrix = confusion_matrix(tsOutput, algPredict)
    matrix = matrix.astype('float')/matrix.sum(axis=1)[:,np.newaxis]
    
    results = list(set(tsOutput))
    
    matrixDF = pd.DataFrame(matrix, index = results, columns = results)
    
    plt.figure(figsize=(12,10))
    sns.heatmap(matrixDF, annot=True, cmap="Blues")
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    

# ===============================================================================     
'''
gridSearch for MLPClassifier()
'''
def hyperparamMLP(trainInput, trainOutput):
    
    cv = 3
    
    model = MLPClassifier()
    
    trainDF = pd.concat([trainInput, trainOutput], axis = 1)
    trainDF = trainDF.sample(30000)
    
    trainInput = trainDF.drop(['Response'], axis=1)
    trainOutput = trainDF.loc[:,'Response']
    
    
    #defining parameter dictionary
    paramGrid = {'hidden_layer_sizes': [150, 300], 'alpha': [0.008, 0.02]}
    
    #initilzing gridsearch object
    gridAlg = GridSearchCV(estimator = model, param_grid = paramGrid, scoring = 'accuracy', cv = cv, verbose = 10)
    
    gridAlg.fit(trainInput, trainOutput)
    
    
    print(gridAlg.cv_results_)
    
    results = pd.DataFrame(gridAlg.cv_results_)
    print(results)
    
    #call to helper function for plotting (need to specify parameter names in helper function)  
    optimizePlot(gridAlg, paramGrid)
    
 
    
# ===============================================================================     
'''
gridSearch for ExtraTreesClassifier()
'''   
def hyperparamTrees(trainInput, trainOutput):

    cv = 3
    
    model = ExtraTreesClassifier()
    
    trainDF = pd.concat([trainInput, trainOutput], axis = 1)
    trainDF = trainDF.sample(30000)
    
    trainInput = trainDF.drop(['Response'], axis=1)
    trainOutput = trainDF.loc[:,'Response']
    
    
    #defining parameter dictionary
    paramGrid = {'max_features': [8, 12 , 20], 'n_estimators': [50,150,250]}
    
    gridAlg = GridSearchCV(estimator = model, param_grid = paramGrid, scoring = 'accuracy', cv = cv, verbose = 10)
    
    gridAlg.fit(trainInput, trainOutput)
    
    
    print(gridAlg.cv_results_)
    
    results = pd.DataFrame(gridAlg.cv_results_)
    
    print(results)

    #call to helper function for plotting (need to specify parameter names in helper function)    
    optimizePlot(gridAlg, paramGrid)    
    


# ===============================================================================     
'''
gridSearch for RandomForestClassifier()
'''       
def hyperparamForest(trainInput, trainOutput):

    cv = 3
    
    model = RandomForestClassifier(random_state = 1)
    
    trainDF = pd.concat([trainInput, trainOutput], axis = 1)
    trainDF = trainDF.sample(30000)
    
    trainInput = trainDF.drop(['Response'], axis=1)
    trainOutput = trainDF.loc[:,'Response']
    
    
    #defining parameter dictionary
    paramGrid = {'max_features': ['sqrt',0.3, 0.8], 'n_estimators': [50,150,250]}
    
    gridAlg = GridSearchCV(estimator = model, param_grid = paramGrid, scoring = 'accuracy', cv = cv, verbose = 10)
    
    gridAlg.fit(trainInput, trainOutput)
    
    
    print(gridAlg.cv_results_)
    
    results = pd.DataFrame(gridAlg.cv_results_)
    
    print(results)

    #call to helper function for plotting (need to specify parameter names in helper function)      
    optimizePlot(gridAlg, paramGrid)
 


# ===============================================================================     
'''
SVC hyperparameterization through HyperOpt
'''    
def svcHyperOpt(trainInput,trainOutput):
    
    #defining params and ranges
    space = {
        'C': hp.quniform('C', 0,10),
        'gamma': hp.uniform('gamma', 4,15)
        }
    
    #scoring
    def test(params):
        alg = svm.SVC(**params)
        return cross_val_score(alg, trainInput, trainOutput).mean()
    
    #evaluation
    def score(params):
        acc = test(params)
        return {'loss': -acc, 'status': STATUS_OK}
    
    trials = Trials()
    best = fmin(score, space, algo=tpe.suggest, max_evals = 80, trials = trials)
    print ('best: ')
    print(best)



# ===============================================================================     
'''
MLP hyperparameterization through HyperOpt
'''    
def mlpHyperOpt(trainInput, trainOutput):
    
    space = {}
    space['hidden_layer_sizes'] = 10 + hp.randint('hidden_layer_sizes', 40)
    space['alpha'] = hp.loguniform('alpha', -5*np.log(10), 5*np.log(10))
    
    def test(params):
        alg = MLPClassifier(**params)
        return cross_val_score(alg, trainInput, trainOutput).mean()
    
    def f(params):
        acc = test(params)
        return {'loss': -acc, 'status': STATUS_OK}
    
    trials = Trials()
    best = fmin(f, space, algo = tpe.suggest, max_evals = 100, trials = trials)
    print('best: ')
    print(best)



# ===============================================================================     
'''
SVC hyperparameterization through gridsearch
'''    
def SVChyperparam(trainInput, trainOutput):
   
    cv = 3
   
    model = SVC()
    
    trainDF = pd.concat([trainInput, trainOutput], axis = 1, sort= False)
    trainDF = trainDF.sample(30000)
   
    trainInput = trainDF.drop(['Response'], axis=1)
    trainOutput = trainDF.loc[:,'Response']
   
    #defining parameter dictionary
    #paramGrid = {'max_depth': [2,3,5], 'n_estimators': [x for x in range(300,2000,3000)] }
    paramGrid = {'gamma': [1,4,8], 'C': [5,15,25] }
   
    gridAlg = GridSearchCV(estimator = model, param_grid = paramGrid, scoring = 'accuracy', cv = cv, verbose = 10)
   
    gridAlg.fit(trainInput, trainOutput)
    
    print(gridAlg.cv_results_)
   
    results = pd.DataFrame(gridAlg.cv_results_)
   
    print(results)
    optimizePlot(gridAlg, paramGrid)    
    
# ===============================================================================     
'''
XGB hyperparameterization through HyperOpt
'''        
def xgbHyperOpt(trainInput,trainOutput):
    
    #sampling to reduce computational time
    trainDF = pd.concat([trainInput, trainOutput], axis = 1)
    trainDF = trainDF.sample(30000)
    
    trainInput = trainDF.drop(['Response'], axis = 1)
    trainOutput = trainDF.loc['Response']   
    
    #defining params and ranges
    space = {
        'max_depth': hp.quniform('max_depth', 3,18),
        'gamma': hp.uniform('gamma', 4,15)
        }
    #scoring
    def test(params):
        alg = xgb.XGBClassifier(**params)
        return cross_val_score(alg, trainInput, trainOutput).mean()
    
    #evaluation    
    def score(params):
        acc = test(params)
        return {'loss': -acc, 'status': STATUS_OK}
    
    trials = Trials()
    best = fmin(score, space, algo=tpe.suggest, max_evals = 80, trials = trials)
    print ('best: ')
    print(best)       



# ===============================================================================     
'''
random Forest hyperparameterization through HyperOpt
'''        
def randomForestHyperOpt(trainInput, trainOutput):
    
    space = {}
    space['max_depth'] = hp.choice('max_depth', range(1,20))
    space['max_features'] = hp.choice('max_features', range(1,5))
    
    def test(params):
        alg = RandomForestClassifier(**params)
        return cross_val_score(alg, trainInput, trainOutput).mean()
    
    best = 0 
    
    def f(params):
        global best
        acc = test(params)
        if acc > best:
            best = acc
        print ('new best: ', best, params)
        return {'loss': -acc, 'status': STATUS_OK}
    
    trials = Trials()
    best = fmin(f, space, algo = tpe.suggest, max_evals = 80, trials = trials)
    print('best: ')
    print(best)
    


# ===============================================================================
'''
Kaggle experiment to evaluate binary model + stack classifier
'''
def kaggleExperiment1(trainInput, testInput, trainOutput, testIDs, predictors):
  
    
    #concatenating to obtain dimensions for next step
    df = pd.concat([trainInput, trainOutput], axis = 1)
    
    #defining DF that will contain binary values pertaining to specific classes
    probDF = pd.DataFrame(index=range(df.shape[0]), columns = range(8))
    
    #filling misssing values
    probDF.iloc[:,0] = df.loc[:, 'Response'].map(lambda val: 1 if(val==1) else 0)
    probDF.iloc[:,1] = df.loc[:, 'Response'].map(lambda val: 1 if(val==2) else 0)
    probDF.iloc[:,2] = df.loc[:, 'Response'].map(lambda val: 1 if(val==3) else 0)
    probDF.iloc[:,3] = df.loc[:, 'Response'].map(lambda val: 1 if(val==4) else 0)
    probDF.iloc[:,4] = df.loc[:, 'Response'].map(lambda val: 1 if(val==5) else 0)
    probDF.iloc[:,5] = df.loc[:, 'Response'].map(lambda val: 1 if(val==6) else 0)
    probDF.iloc[:,6] = df.loc[:, 'Response'].map(lambda val: 1 if(val==7) else 0)
    probDF.iloc[:,7] = df.loc[:, 'Response'].map(lambda val: 1 if(val==8) else 0)    

    #creating probability dataframe
    probTrain = pd.DataFrame(index=range(df.shape[0]), columns = range(8))
    probTest = pd.DataFrame(index=range(testInput.shape[0]), columns = range(8))
        
    trainInDF = trainInput
    
    #alg with hyperoptimized params
    alg = xgb.XGBClassifier(gamma = 0.6, max_depth = 3)
    
    #dimensions for kFold
    kfX = trainInput
    kfY = probDF.iloc[:,0]
    
    #loop through each column
    for counter in range(0,8):
        
        #creating folds
        kf = KFold(n_splits = 5)
        for trIndex, cvIndex in kf.split(kfX, kfY):
            
            #defining specific folds for the iteration
            trInput = trainInDF.iloc[trIndex]
            trOutput =  probDF.iloc[:,counter][trIndex]
            
            #applying model
            alg.fit(trInput, trOutput)
            
            #extracting probability of specific classification
            predictions = alg.predict_proba(trainInDF.iloc[cvIndex])[:,1]
            probTrain.iloc[cvIndex, counter] = predictions
            
        #testing data
        alg.fit(trainInDF, probDF.iloc[:,counter])
        predictions = alg.predict_proba(testInput)[:,1]
        probTest.iloc[:,counter] = predictions
        
    #renaming columns
    probTrain = probTrain.set_axis(['prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7', 'prob8'], axis=1)  
    probTest = probTest.set_axis(['prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7', 'prob8'], axis=1) 
    
    #converting probabilities to numerical from 'object' type.
    probTrain['prob1']= pd.to_numeric(probTrain['prob1'], errors='coerce').fillna(0, downcast='infer')
    probTrain['prob2']= pd.to_numeric(probTrain['prob2'], errors='coerce').fillna(0, downcast='infer')
    probTrain['prob3']= pd.to_numeric(probTrain['prob3'], errors='coerce').fillna(0, downcast='infer')
    probTrain['prob4']= pd.to_numeric(probTrain['prob4'], errors='coerce').fillna(0, downcast='infer')
    probTrain['prob5']= pd.to_numeric(probTrain['prob5'], errors='coerce').fillna(0, downcast='infer')
    probTrain['prob6']= pd.to_numeric(probTrain['prob6'], errors='coerce').fillna(0, downcast='infer')
    probTrain['prob7']= pd.to_numeric(probTrain['prob7'], errors='coerce').fillna(0, downcast='infer')
    probTrain['prob8']= pd.to_numeric(probTrain['prob8'], errors='coerce').fillna(0, downcast='infer')
    
    baseModels = [("xgbC", xgb.XGBClassifier(gamma = 0.6, max_depth = 3)), ("rForest", RandomForestClassifier(max_features = 0.9, n_estimators = 125))]
    
    metaModel = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')
    
    alg = StackingClassifier(estimators = baseModels, final_estimator = metaModel)
    
    alg.fit(probTrain, trainOutput)
    
    predictions = alg.predict(probTest)
    
    submission = pd.DataFrame({ "Id": testIDs, "Response": predictions })

    # Prepare CSV
    submission.to_csv('data/testResults1.csv', index=False)



# ===============================================================================
'''
Kaggle experiment to evaluate multinomial logistic regression
'''
def kaggleExperiment2(trainInput, testInput, trainOutput, testIDs, predictors):
  
    
    'defining base models'
    baseModels = [("mlp", MLPClassifier(alpha = 0.006, hidden_layer_sizes = 100)), 
                  ("xgbC", xgb.XGBClassifier(gamma = 0.6, max_depth = 3)), ("rForest", RandomForestClassifier(max_features= 0.9, n_estimators = 125 ))]
    metaModel = LogisticRegression(multi_class = 'multinomial', solver='lbfgs')
    
    
    alg = StackingClassifier(estimators = baseModels, final_estimator = metaModel)
    

    # Train the algorithm using all the training data
    alg.fit(trainInput, trainOutput)

    # Make predictions on the test set.
    predictions = alg.predict(testInput)

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "Response": predictions
    })

    # Prepare CSV
    submission.to_csv('data/testResults.csv', index=False)

    
# ===============================================================================
'''
Runs the algorithm on the testing set and writes the results to a csv file.
'''
def doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors):
  
    
    'defining base models'
    baseModels = [("mlp", MLPClassifier(alpha = 0.006, hidden_layer_sizes = 100)), 
                  ("xgbC", xgb.XGBClassifier(gamma = 0.6, max_depth = 3)), ("rForest", RandomForestClassifier(max_features= 0.9, n_estimators = 125 ))]
    metaModel = LogisticRegression(multi_class = 'ovr', solver='liblinear')
    
    
    alg = StackingClassifier(estimators = baseModels, final_estimator = metaModel)
    

    # Train the algorithm using all the training data
    alg.fit(trainInput, trainOutput)

    # Make predictions on the test set.
    predictions = alg.predict(testInput)

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "Response": predictions
    })

    # Prepare CSV
    submission.to_csv('data/testResults.csv', index=False)


    
# ===============================================================================
'''
Individual correlation calculator
'''    
def indCorr(df, colName):
    return df.loc[:, colName].corr(df.loc[:, 'SalePrice'])



# ===============================================================================
'''
Demonstrates some helper functions for intitial data exploration.
'''
def demonstrateHelpers(trainDF):
    print("Attributes with missing values:", getAttrsWithMissingValues(trainDF), sep='\n')
    
    numericAttrs = getNumericAttrs(trainDF)
    print("Numeric attributes:", numericAttrs, sep='\n')
    
    nonnumericAttrs = getNonNumericAttrs(trainDF)
    print("Non-numeric attributes:", nonnumericAttrs, sep='\n')

    print("Values, for each non-numeric attribute:", getAttrToValuesDictionary(trainDF.loc[:, nonnumericAttrs]), sep='\n')



# ===============================================================================
'''
Returns a dictionary mapping an attribute to the array of values for that attribute.
'''
def getAttrToValuesDictionary(df):
    attrToValues = {}
    for attr in df.columns.values:
        attrToValues[attr] = df.loc[:, attr].unique()

    return attrToValues



# ===============================================================================
'''
Returns the attributes with missing values.
'''
def getAttrsWithMissingValues(df):
    valueCountSeries = df.count(axis=0)  # 0 to count down the rows
    numCases = df.shape[0]  # Number of examples - number of rows in the data frame
    missingSeries = (numCases - valueCountSeries)  # A Series showing the number of missing values, for each attribute
    attrsWithMissingValues = missingSeries[missingSeries != 0].index
    return attrsWithMissingValues



# =============================================================================

'''
Returns the numeric attributes.
'''
def getNumericAttrs(df):
    return __getNumericHelper(df, True)



# =============================================================================
'''
Returns the non-numeric attributes.
'''
def getNonNumericAttrs(df):
    return __getNumericHelper(df, False)

def __getNumericHelper(df, findNumeric):
    isNumeric = df.applymap(np.isreal) # np.isreal is a function that takes a value and returns True (the value is real) or False
                                       # applymap applies the given function to the whole data frame
                                       # So this returns a DataFrame of True/False values indicating for each value in the original DataFrame whether it is real (numeric) or not

    isNumeric = isNumeric.all() # all: For each column, returns whether all elements are True
    attrs = isNumeric.loc[isNumeric==findNumeric].index # selects the values in isNumeric that are <findNumeric> (True or False)
    return attrs



# =============================================================================

if __name__ == "__main__":
    main()

