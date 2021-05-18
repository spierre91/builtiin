import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

df = pd.read_csv('telco_churn.csv')

print(df.head())

print(len(df.columns))
print(len(df))
df['Churn'] = np.where(df['Churn'] == 'Yes', 1, 0)



X = df[['tenure', 'MonthlyCharges']]
y = df['Churn']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


clf_model = LogisticRegression()
clf_model.fit(X_train, y_train)
y_pred = clf_model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))



conmat = confusion_matrix(y_test, y_pred)
print(conmat)
val = np.mat(conmat) 
print(val)

classnames = list(set(y_train))
df_cm = pd.DataFrame(
        val, index=classnames, columns=classnames, 
    )

print(df_cm)


print(len(y_test))

from collections import Counter

print(Counter(y_test))



import matplotlib.pyplot as plt
import seaborn as sns
df_cm = df_cm.astype('float') / df_cm.sum(axis=1)[:, np.newaxis]  
plt.figure()
heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues", fmt='g')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Churn Logistic Regression Model Results')
plt.show()

y_pred_proba = clf_model.predict_proba(np.array(X_test))[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


sns.set()
plt.plot(fpr, tpr)
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
AUROC = np.round(roc_auc_score(y_test, y_pred_proba), 2)
plt.title(f'Logistic Regression Model ROC curve; AUROC: {AUROC}');
plt.show()



from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

average_precision = average_precision_score(y_test, y_pred_proba)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision, marker='.', label='Logistic')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.title(f'Precision Recall Curve. AUPRC: {average_precision}')
plt.show()

