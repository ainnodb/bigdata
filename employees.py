import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


Employees_Data = pd.read_csv('employees_dataset.csv')
Employees_Data.insert(5, 'enrolled', "Yes")
Employees_Data.insert(6, 'degree_clean', "")
Employees_Data["degree_clean"] = np.where(Employees_Data["degree"] == "bachelor", 1,
                                          np.where(Employees_Data["degree"] == "master", 2,
                                                   np.where(Employees_Data["degree"] == "phd", 3, 4)
                                                   )
                                          )
used_features = ["degree_clean", "education", "skills", "working_experience","position"]

X_train, X_test = train_test_split(Employees_Data, test_size=0.1, random_state=int(time.time()))
gnb = GaussianNB()
gnb.fit(X_train[used_features].values, X_train["enrolled"])
y_pred = gnb.predict(X_test[used_features])

print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(X_test.shape[0], (X_test["enrolled"] != y_pred).sum(),
              100 * (1 - (X_test["enrolled"] != y_pred).sum() / X_test.shape[0])))
