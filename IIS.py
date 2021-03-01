
import pandas as pd
import os

"""The 'Data.csv' file was saved at the below location and hence the 
directory was changed to access the file , without throwing the FILE-NOT-FOUND
error"""
os.chdir('/Users/varishgrover/desktop/') 

#Reading the CSV file and loading it into the pandas data frame
myData=pd.read_csv('Data.csv')

#**********************************************************************************************

"""The given code block converts the categorical features into integers for
comparision. The catagorical features cannot be used for logistic regression
and hence they are first mapped to integers

Following catagorical features have been mapped :-

1. Gender
2. Degree
3. Specialization 
4. College State


TO FIND ALL THE ENTRY TYPES CORRESPONDING TO A PARTICULAR PARAMETER WE RUN THE
FOLLOWING COMMAND -

myData['Specialization'].value_counts()

"""


replacement = {"Gender":{"m": 0, "f": 1},
              "Degree":{"B.Tech/B.E.": 0,"MCA":1, "M.Tech./M.E.":2, "M.Sc. (Tech.)":3},
              "Specialization":{"electronics and communication engineering":1,
"computer science & engineering":2,
"information technology":3,                      
"computer engineering":4,
"computer application":5,                          
"mechanical engineering":6,                        
"electronics and electrical engineering":7,         
"electronics & telecommunications":8,              
"electrical engineering":9,              
"electronics & instrumentation eng":10,               
"civil engineering":11,                               
"information science engineering":12,              
"electronics and instrumentation engineering":13 , 
"instrumentation and control engineering":14 ,  
"electronics engineering":15 ,                 
"biotechnology":16,                                
"other":17,                                 
"industrial & production engineering":18,          
"chemical engineering":     19,                        
"applied electronics and instrumentation":  20,       
"computer science and technology":  21,                
"telecommunication engineering":22,                    
"mechanical and automation":23,                        
"automobile/automotive engineering": 24,               
"instrumentation engineering": 25,                    
"mechatronics":  26,                                   
"aeronautical engineering":27,                         
"electronics and computer engineering":28,             
"computer science":  29,                               
"industrial engineering": 30,                          
"metallurgical engineering": 31,                      
"biomedical engineering": 32,                          
"electrical and power engineering": 33,               
"information & communication technology": 34,       
"internal combustion engine":     35,               
"embedded systems technology":  36,                    
"information science":  37,                        
"power systems and automation":   38,            
"computer and communication engineering": 39,     
"polymer technology": 40,                             
"ceramic engineering": 41,            
"control and instrumentation engineering": 42,      
"industrial & management engineering": 43,          
"mechanical & production engineering":  44,           
"computer networking":   45,                      
"electronics":46 },
               "CollegeState":{
"Uttar Pradesh":0,
"Karnataka":1,
"Tamil Nadu":2,
"Telangana":3,
"Maharashtra":4,
"Andhra Pradesh":5,     
"West Bengal":6,
"Punjab":7,
"Madhya Pradesh":8,
"Haryana":9,
"Rajasthan":10,
"Orissa":11,
"Delhi":12,
"Uttarakhand":13,
"Kerala":14,
"Jharkhand":15,
"Chhattisgarh":16,
"Gujarat":17,
"Himachal Pradesh":18,
"Bihar":19,
"Jammu and Kashmir":20,
"Assam":21,
"Union Territory":22,
"Sikkim":23,
"Meghalaya":24,
"Goa":25,
               }           
             }

#**********************************************************************************************

#The below code block modifies the "DOB-Date of Birth" column in the given dataset

"""This code block calculates the AGE (in years) for all people from their 
date of birth . It then replaces the DOB  with the AGE , as AGE is a 
better means of comparison"""

(Rows, Columns)= myData.shape #(Number of rows , Number of columns)
for i in range(Rows):
    myData.at[i,"DOB"]=int(2020-int(myData.at[i,"DOB"][0:4]))  #Age calculate

convert_dict = {'DOB': int } #Changing the data-type of the column
myData = myData.astype(convert_dict) 

myData = myData.rename(columns = {"DOB":"Age"}) #Renaming the column

#**********************************************************************************************

#Replacing the categorical columns with their mapping as above
myData.replace(replacement, inplace=True) 

#Dropping a few columns - The dropped columns are changed again and again for experimentation
myData.drop(["ID","CollegeID","CollegeCityTier","CollegeState","12graduation","Degree"],axis=1,inplace = True)

#**********************************************************************************************

#Creating a parameters dataframe which consists of all input parameters
parameters=pd.DataFrame(myData.iloc[:,:-1].select_dtypes(include=['int64','float64'])) 

#Loading the expected output -Last Column in the .csv file  - into another dataframe
expectedOutput=pd.DataFrame(myData.iloc[:,-1])

#Splitting into training and testing data   
from sklearn.model_selection import train_test_split
parameters_test,parameters_train,expectedOutput_test,expectedOutput_train=train_test_split(parameters,expectedOutput,test_size=0.3,random_state=1)

#Creating a logistic regression object
from sklearn.linear_model import LogisticRegression
model= LogisticRegression()

#Training the "Training data set"
model.fit(parameters_train,expectedOutput_train)

#Testing the "Testing data set"
expectedOutput_predict=model.predict(parameters_test)

#**********************************************************************************************
#Accuracy

"""Accuracy is defined as :- Number of correct outputs / Total number of outputs"""

Accuracy=model.score(parameters_test,expectedOutput_test)

print("Accuracy:- ",Accuracy*100,"%")

#**********************************************************************************************
# Confusion Matrix

#Importing the confusion matrix object from the library
from sklearn.metrics import confusion_matrix

#Printing the confusion matrix 
print("Confusion Matrix :- ",'\n','\n', confusion_matrix(expectedOutput_test,expectedOutput_predict))

#**********************************************************************************************
# Class Wise Accuracy

#Extracting the elements of confusion matrix
TN, FP, FN, TP= confusion_matrix(expectedOutput_test,expectedOutput_predict).ravel()

#Applying formulae for class wise accuracy  :-

#Class "1" : Accuracy of the output of the class "1"
PositiveClassAccuracy= TP/(TP + FN) 

#Class "0" :  Accuracy of the output of the class "0"
NegativeClassAccuracy= TN/(TN + FP)

print("Positive class accuracy is :-  ",PositiveClassAccuracy*100,"%")
print("Negative class accuracy is :-  ",NegativeClassAccuracy*100,"%")

#**********************************************************************************************
