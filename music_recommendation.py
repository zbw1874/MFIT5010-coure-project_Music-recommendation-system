import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,roc_curve,auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn import svm
from xgboost import XGBClassifier,plot_importance
import matplotlib.pyplot as plt

## read datasets and merge them
data_row = pd.read_csv('dataset\\train.csv')
member = pd.read_csv('dataset\\members.csv')
song = pd.read_csv('dataset\\songs.csv')
song_extra = pd.read_csv('dataset\\song_extra_info.csv')

data = data_row.merge(member,on='msno',how='left')
data = data.merge(song,on='song_id',how='left')
data = data.merge(song_extra,on='song_id',how='left')

data=data.dropna()## drop NA

### transform object to category and encode them
for k in data.select_dtypes(include=['object']).columns:
    data.loc[:,k] = data.loc[:,k].astype('category')

for j in data.select_dtypes(include=['category']).columns:
    data.loc[:,j] = data.loc[:,j].cat.codes

## split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(data.drop(columns='target',axis=1),data['target'],test_size=0.3)

## define function plotting roc curve
def roc_curve_plot(roc_auc,fpr,tpr):
    plt.subplots(figsize=(7, 5.5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

# Decision tree
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
dt_predict = dt.predict(x_test)
dt_proba = dt.predict_proba(x_test)[:, 1] #calculate probability
print(classification_report(y_test, dt_predict))
print(confusion_matrix(y_test, dt_predict))
print(accuracy_score(y_test, dt_predict)) ###accuracy
dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_test, dt_proba)
dt_roc_auc = auc(dt_fpr, dt_tpr)
roc_curve_plot(dt_roc_auc, dt_fpr, dt_tpr)

# Random forest classifier
rf=RandomForestClassifier()#default parameter
rf.fit(x_train,y_train)
rf_predict = rf.predict(x_test)
rf_proba = rf.predict_proba(x_test)[:, 1] #calculate probability
print(classification_report(y_test, rf_predict))
print(confusion_matrix(y_test, rf_predict))
print(accuracy_score(y_test, rf_predict)) ###accuracy
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_proba)
rf_roc_auc = auc(rf_fpr, rf_tpr)
roc_curve_plot(rf_roc_auc, rf_fpr, rf_tpr)
## feature importance plot
importances = rf.feature_importances_
indices = np.argsort(importances)
y_ticks = np.arange(0, len(x_test.columns))
fig, ax = plt.subplots()
ax.barh(y_ticks, importances[indices])
ax.set_yticklabels(x_test.columns[indices])
ax.set_yticks(y_ticks)
ax.set_title("Feature importance of random forest")
fig.tight_layout()
plt.show()

'''
# SVM(rbf)
svm_rbf = svm.SVC(C=0.01, kernel='rbf',max_iter=-1)
svm_rbf.fit(x_train, y_train)
svm_rbf_predict = svm_rbf.predict(x_test)
print(classification_report(y_test, svm_rbf_predict))
print(confusion_matrix(y_test, svm_rbf_predict))
print(accuracy_score(y_test, svm_rbf_predict)) ###accuracy
'''

# SVM(poly)
svm_poly = svm.SVC(C=0.8, kernel='poly', gamma=20)
svm_poly.fit(x_train, y_train)
svm_poly_predict = svm_poly.predict(x_test)
print(classification_report(y_test, svm_poly_predict))
print(confusion_matrix(y_test, svm_poly_predict))
print(accuracy_score(y_test, svm_poly_predict)) ###accuracy

'''
# SVM(linear)
svm_linear = svm.SVC(C=0.1, kernel='linear')
svm_linear.fit(x_train, y_train)
svm_linear_predict = svm_linear.predict(x_test)
print(classification_report(y_test, svm_linear_predict))
print(confusion_matrix(y_test, svm_linear_predict))
print(accuracy_score(y_test, svm_linear_predict)) ###accuracy
'''

# XGBoost
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
xgb_predict = xgb.predict(x_test)
xgb_proba = xgb.predict_proba(x_test)[:, 1] #calculate probability
print(classification_report(y_test, xgb_predict))
print(confusion_matrix(y_test, xgb_predict))
print(accuracy_score(y_test, xgb_predict)) ###accuracy
xgb_fpr, xgb_tpr, xgb_thresholds = roc_curve(y_test, xgb_proba)
xgb_roc_auc = auc(xgb_fpr, xgb_tpr)
roc_curve_plot(xgb_roc_auc, xgb_fpr, xgb_tpr)# roc curve plot
plot_importance(xgb)# feature importance of xgb
plt.show()
