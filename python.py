
"""
Dataset Information:
    
                Breast Cancer Wisconsin (Diagnostic)
                
    Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. A few of the images can be found at [Web Link] 

    Separating plane described above was obtained using Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree Construction Via Linear Programming." Proceedings of the 4th Midwest Artificial Intelligence and Cognitive Science Society, pp. 97-101, 1992], a classification method which uses linear programming to construct a decision tree. Relevant features were selected using an exhaustive search in the space of 1-4 features and 1-3 separating planes. 

    The actual linear program used to obtain the separating plane in the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34]. 

    This database is also available through the UW CS ftp server: 
        ftp ftp.cs.wisc.edu 
        cd math-prog/cpo-dataset/machine-learn/WDBC/
    
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







 