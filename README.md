# Salary Prediction System
The dataset (attached as ‘Data.xlsx’) provides anonymised bio data information along
with their respective skill scores. Specifically, the following information is available for
every engineer:
1. Scores on Aspiring Minds' AMCAT - a standardized test of job skills. The test
includes cognitive, domain and personality assessments.
2. Personal information like gender and date of birth.
3. Pre-university information like 10th and 12th grade marks, board of education
and 12th grade graduation year.
4. University information like GPA, college major, college reputation proxy,
graduation year and college location.
5. The outcome whether the engineer, at the end of graduation gets a “High Salary”
(labelled as 1) or “Low Salary” (labelled as 0) in the last column.

The task is to make a **ML-based prediction system** for students to get **"high salary" or
not**. 

1. The end goal at hand - Classifying whether the student will get a “high salary” or
not.
2. Divide the data into training and testing data.
3. Using the sklearn library, train a logistic regression model using the training data.
4. Predict the outcome for the testing data (Separated out in step 2).
5. Using the predicted labels and actual labels, find out accuracy, confusion matrix
and class-wise accuracies.
