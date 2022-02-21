# Prudential-Insurance-Risk-Assessment

## Abstract
Today, most applicants apply for insurance policies on- line. Consequently, data science plays an important role in the insurance world. An insurance company’s success is greatly impacted by its ability to accurately and reli- ably assess risk associated with a particular policy, based on the specific applicant’s information and data.
This project uses an ensemble and stacking of ma- chine learning algorithms to solve a complex multi- classification problem on a life insurance data set. The real-life normalized data set is obtained form the Kaggle Prudential Life Insurance Assessment Challenge; this data set is used to tackle and explore complex data science techniques.
Keywords: Multi-Class Classification, Stacking, Extreme Gradient Boosting, Logistic Regression, Grid Optimiza- tion, Hyperopt.

## Data Description
The data set under consideration is a real life data set pro- vided by a life insurance company. Thus, the data is par- ticularly challenging to work on as it contains data that is normalized to maintain confidentiality. In fact, this is a prime reason why this data set was chosen for the purpose of this paper, as it proposes a problem that is difficult to decipher theoretically since the precise meaning behind the relative values assigned to variables is unknown and thus up for interpretation.
The data set is comprised of 128 variables that per- tain to attributes of life insurance applicants, and only 4 of these variables have a clear interpretation: Age, BMI, Height, and Weight. Additionally, the data set includes a variety of different types of variables:
• Over 50 categorical variables pertaining to informa- tion like the medical history of the applicant.
• More than 10 continuous variables containing infor- mation such as height and weight of the applicant.
• Many discrete variables: over 40 of these discrete
variables are dummy variables regarding the exis- tence of a ’Medical Keyword’ in the applicant’s ap- plication.
In addition to the high dimensionality of the data set, there are approximately 60,000 entries or observations which make the problem very computationally expensive. The task is to create a model that accurately predicts the ’Response’ variable for each ID/individual in the data set. The target variable is an ordinal measure of ’risk’ on a scale from 1-8, and thus presents a high dimensional multi-class classification problem.

## Running the Product
Relevant methods are executed through the ’main’ function. The methods of ’pd.read csv’ to read the data and begin processing will always want to be run, and are included in the main function of the project submission. Other important methods are ’stackertest1’ method, that runs the best per- forming model. Additionally, the ’doKaggleTest’ method is an implementation of the same method for Kaggle result.

## Using the Product
Using the product is fairly simple. The main submission file included contains a plethora of methods that test alter- native strategies. I made sure to remove redundant ’trial’ code, and the code included is primarily examples of dif- ferent kinds of techniques that I implemented, which can be tested out by including the relative method in the main function.
Some of the code in the project submission relies on unconventional libraries that might need to be installed on your computer. However, in most cases, these special li- braries only pertain to very specific methods that might not necessarily need to be run and thus these libraries could be commented out for simplicity - though this will probably show warnings in your IDE of choice as it would not recognize some methods in some particular functions.

## Files
For an in-depth report on the project, computational constraints, and data analysis, please see the included "ProjectReport.pdf". The main code is stored in the "MainProject.py" file. The data folder contains the given training and testing files provided by Prudential. The kaggle folder contains my submission file that yielded the results described in the project report. The Miscellaneous folder contains examples of the various visualizations and analyses produced by the functions in my code. 

If you have any suggestions or feedback, I welcome your insight and collaboration :) Thanks for checking out my project.


