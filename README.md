# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, drop unnecessary columns, and encode categorical variables. 
2.Define the features (X) and target variable (y). 
3.Split the data into training and testing sets.
4.Train the logistic regression model, make predictions, and evaluate using accuracy and other

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SATHISH S
RegisterNumber:212225040390
*/
import pandas as pd
import numpy as np

df = pd.read_csv("Placement_Data.csv")
df

df1 = df.copy()

df1 = df1.drop(['sl_no', 'salary'], axis=1)

df1.isnull().sum()
df1.duplicated().sum()

df1

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df1['gender'] = le.fit_transform(df1['gender'])
df1['ssc_b'] = le.fit_transform(df1['ssc_b'])
df1['hsc_b'] = le.fit_transform(df1['hsc_b'])
df1['hsc_s'] = le.fit_transform(df1['hsc_s'])
df1['degree_t'] = le.fit_transform(df1['degree_t'])
df1['workex'] = le.fit_transform(df1['workex'])
df1['specialisation'] = le.fit_transform(df1['specialisation'])
df1['status'] = le.fit_transform(df1['status'])

df1

x = df1.iloc[:, :-1]
x

y = df1['status']
y

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver="liblinear")

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("Accuracy Score:", accuracy)
print("\nConfusion Matrix:\n", confusion)
print("\nClassification Report:\n", cr)

from sklearn import metrics

cn_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=['true', 'false']
)

cn_display.plot()

```

## Output:
1)ACCURACY SCORE,CONFUSION MATRIX,CLASSIFICATION REPORT
<img width="731" height="346" alt="590853659-e98ece6e-2ca8-4d91-9f9a-7807de78a859" src="https://github.com/user-attachments/assets/f5994e43-1c54-4951-abe8-6f49d24b45cd" />
CONFUSION MATRIX
<img width="833" height="581" alt="590853794-30795d89-5e80-4fe0-ad58-5f6fbef4777a" src="https://github.com/user-attachments/assets/a400e77c-21f5-468b-bac8-eb53e3d9f432" />




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
