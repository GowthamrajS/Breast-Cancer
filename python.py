
"""
Dataset Information:
    
# # CASE STUDY: BREAST CANCER CLASSIFICATION

# # STEP #1: PROBLEM STATEMENT

# 
# - Predicting if the cancer diagnosis is benign or malignant based on several observations/features 
# - 30 features are used, examples:
#         - radius (mean of distances from center to points on the perimeter)
#         - texture (standard deviation of gray-scale values)
#         - perimeter
#         - area
#         - smoothness (local variation in radius lengths)
#         - compactness (perimeter^2 / area - 1.0)
#         - concavity (severity of concave portions of the contour)
#         - concave points (number of concave portions of the contour)
#         - symmetry 
#         - fractal dimension ("coastline approximation" - 1)
# 
# - Datasets are linearly separable using all 30 input features
# - Number of Instances: 569
# - Class Distribution: 212 Malignant, 357 Benign
# - Target class:
#          - Malignant
#          - Benign
    
"""
# Importing Data
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea

# Reviewing Data
df =load_breast_cancer()

df_pd = pd.DataFrame(np.c_[df["data"],df["target"]],columns = np.append(df["feature_names"],["target"]))

#  Checking NaN is Avaliable or Not

df_pd.isnull().sum()

# splitting data to X and Y

x = df_pd.drop(["target"],axis = 1)

y = df_pd["target"]


# Data Visualisation

sea.pairplot(df_pd,hue = "target" ,vars = ["mean radius","mean texture","mean perimeter","mean area"])

sea.pairplot(df_pd,hue = "target" ,vars = ["mean radius"])


sea.countplot(df_pd["target"])

# Dataset Splitting to TrainSet and TestSet

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8,random_state =0 )


# Feature Scaling data 
from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train,y_train)

x_test = scaler.transform(x_test)

# Importing Classifier

from sklearn.svm import SVC

svc = SVC()

svc.fit(x_train,y_train)

svc.score(x_train,y_train)

# correctly fitted 0.98 in svc

y_pred = svc.predict(x_test)


from sklearn.metrics import confusion_matrix

cm  = confusion_matrix(y_test , y_pred)

sea.heatmap(cm ,annot=True)







 