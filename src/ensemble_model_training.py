# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

def classification_report_train_test(y_train, y_train_pred, y_test, y_test_pred):

  print('='*60)
  print("\t\tCLASSICATION REPORT FOR TRAIN")
  print('='*60, '\n')
  print(classification_report(y_train, y_train_pred))
  print('='*60)
  print("\t\tCLASSICATION REPORT FOR TEST")
  print('='*60, '\n')
  print(classification_report(y_test, y_test_pred))

def plot_cm(actual, pred):

  cf_matrix = confusion_matrix(actual, pred)

  group_names = ['True Neg','False Pos','False Neg','True Pos']
  group_counts = ["{0:0.0f}".format(value) for value in
                  cf_matrix.flatten()]
  group_percentages = ["{0:.2%}".format(value) for value in
                      cf_matrix.flatten()/np.sum(cf_matrix)]
  labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
  labels = np.asarray(labels).reshape(2,2)
  sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues').set(xlabel = 'Predicted', ylabel='Actual')

  plt.show()

path = '/Users/sunithaadiraju/Desktop/ipma_ensemble/data/input/tagged_data_for_ensemble_model_training/'

df = pd.read_excel(path + 'full_filtered_results.xlsx')

df = df.drop_duplicates()

df = df[~df.sim_match.isnull()]#.drop(['ngram_length', 'riskword_length'], axis=1)

df_pn = pd.read_csv(path + 'positive_negative_phrases_scores.csv')


df_combined = df.append(df_pn, ignore_index=True)

# RETRAINING _ WORK
#df_combined = pd.read_excel('/Users/sunithaadiraju/Desktop/ipma_ensemble/TESTING_ DATA_tagged_data_for_ensemble_model_training/df_combined.csv')

cols = ['sim_match', 
        'universal_sentence_encoder',
        'glove6b', 
        'glove840b', 
        'komninos',
        'mpnet', 
        'paraphrase_mpnet']

df1 = df_combined[cols]

df1.isnull().sum()

df1.head()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = df1.drop('sim_match', axis=1)
y = df1.sim_match

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

"""### Model Building & Fine-tuning"""

# Decision Tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

dt = DecisionTreeClassifier()#class_weight='balanced'))

dt.fit(X_train, y_train)

y_train_pred_dt = dt.predict(X_train)
y_test_pred_dt = dt.predict(X_test)

# Random Forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(class_weight='balanced')

rf.fit(X_train, y_train)

y_train_pred_rf = rf.predict(X_train)
y_test_pred_rf = rf.predict(X_test)

import joblib
#joblib.dump(rf, 'random_forest_final_model.pkl')
#joblib.dump(rf, 'random_forest_final_model_sunitha.pkl')
joblib.dump(rf, 'random_forest_final_model_sunitha_v2.pkl')

# XGBoost

from xgboost import XGBClassifier

xgb = XGBClassifier(silent=False, 
                      scale_pos_weight=4,
                      learning_rate=0.01,  
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=1000, 
                      reg_alpha = 0.3,
                      max_depth=5, 
                      gamma=10)

xgb.fit(X_train, y_train)

y_train_pred_xgb = xgb.predict(X_train)
y_test_pred_xgb = xgb.predict(X_test)

# Light GBM

import lightgbm as lgb

d_train = lgb.Dataset(X_train, label=y_train)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10

lgbm = lgb.train(params, d_train, 100)

y_train_pred_lgbm = lgbm.predict(X_train)
y_test_pred_lgbm = lgbm.predict(X_test)

for i in range(0,len(y_test_pred_lgbm)):
    if y_test_pred_lgbm[i]>=.5:       # setting threshold to .5
       y_test_pred_lgbm[i]=1
    else:  
       y_test_pred_lgbm[i]=0


for i in range(0,len(y_train_pred_lgbm)):
    if y_train_pred_lgbm[i]>=.5:       # setting threshold to .5
       y_train_pred_lgbm[i]=1
    else:  
       y_train_pred_lgbm[i]=0

classification_report_train_test(y_train, y_train_pred_dt, y_test, y_test_pred_dt)

classification_report_train_test(y_train, y_train_pred_rf, y_test, y_test_pred_rf)

classification_report_train_test(y_train, y_train_pred_xgb, y_test, y_test_pred_xgb)

classification_report_train_test(y_train, y_train_pred_lgbm, y_test, y_test_pred_lgbm)

# cm for dt

plot_cm(y_test, y_test_pred_dt)

# cm for rf

plot_cm(y_test, y_test_pred_rf)

# cm for xgb

plot_cm(y_test, y_test_pred_xgb)

# cm for l

plot_cm(y_test, y_test_pred_lgbm)

"""### Feature Importance"""

import matplotlib.pyplot as plt

# get importance
importance = pd.DataFrame(dt.feature_importances_)

importance.set_index(X_train.columns).plot(kind='barh')

# get importance
importance = pd.DataFrame(rf.feature_importances_)

importance.set_index(X_train.columns).plot(kind='barh')

# get importance
importance = pd.DataFrame(xgb.feature_importances_)

importance.set_index(X_train.columns).plot(kind='barh')

lgb.plot_importance(lgbm)
plt.show()