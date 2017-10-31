# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:31:14 2017

@author: mingxia.huang
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

def gini(y, p):
    assert len(y) == len(p)
    a = np.asarray(np.c_[y, p, np.arange(len(y))], dtype = np.float)
    a = a[np.lexsort((a[:,2], -1 * a[:,1]))]
    loss = a[:,0].sum()
    gini = a[:,0].cumsum().sum() / loss
    gini -= (len(y) + 1) / 2
    return gini / len(y)

def gini_normalized(y, p):
    return gini(y, p) / gini(y, y)

def gini_xgb(p, dtrain):
    y=dtrain.get_label()
    score = gini_normalized(y, p)
    return 'gini', score

def gini_lgb(y, p):
    score = gini_normalized(y, p)
    return 'gini', score, True
    
def feature_engineering(df):
    df["nNa_ind"] = df[["ps_ind_02_cat", "ps_ind_04_cat", "ps_ind_05_cat"]].isnull().sum(axis = 1)
    df["nNa_car"] = df[["ps_car_01_cat", "ps_car_02_cat", "ps_car_03_cat", "ps_car_05_cat", "ps_car_07_cat", "ps_car_09_cat"]].isnull().sum(axis = 1)
    df["reg3_car13"] = df.ps_car_13 * df.ps_reg_03
    return df
    
train = pd.read_csv("./downloads/train.csv", na_values = -1)
test = pd.read_csv("./downloads/test.csv", na_values = -1)

# bayesian mean encoding features:
ps_ind_bin_features = ["ps_ind_06_bin", "ps_ind_07_bin", "ps_ind_08_bin", "ps_ind_09_bin", "ps_ind_16_bin", "ps_ind_17_bin", "ps_ind_18_bin"]
bayesian_features = {"ps_ind_bin_prob": ps_ind_bin_features}
feature_fold = 20
fekf = GroupKFold(n_splits = feature_fold)
j = 1
df_list = []

for t,v in fekf.split(train, train.target, groups = train.id):
    print "feature engineering folds #{} ...".format(j)
    for name in bayesian_features:
        smoother = train.iloc[t].target.mean()
        prob = train.iloc[t].groupby(bayesian_features[name]).target.mean().to_frame(name).reset_index()
        
        vdf = pd.merge(train.iloc[v], prob, how = "left")
        df_list.append(vdf)
    j += 1

train = pd.concat(df_list)
for name in bayesian_features:
    prob = train.groupby(bayesian_features[name]).target.mean().to_frame(name).reset_index()
    test = pd.merge(test, prob, how = "left")

# feature engineering
train = feature_engineering(train)
test = feature_engineering(test)

ind_group_id = pd.concat([train, test]).groupby(ps_ind_bin_features).target.size().to_frame("ind_group_id").reset_index()
train = pd.merge(train, ind_group_id, how = "left")
test = pd.merge(test, ind_group_id, how = "left")

# k_fold training
n_fold = 5
gkf = GroupKFold(n_splits = n_fold)

features = train.columns.difference(["id", "target", "benchmark"])
calc_features = ["ps_calc_01", "ps_calc_02", "ps_calc_03", "ps_calc_04", "ps_calc_05", "ps_calc_06", "ps_calc_07", 
                 "ps_calc_08", "ps_calc_09", "ps_calc_10", "ps_calc_11", "ps_calc_12", "ps_calc_13", "ps_calc_14", 
                 "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin", "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin"]
meaningless_features = ["ps_ind_11_bin", "ps_ind_13_bin"]
redundant_features = ["ps_ind_10_bin", "ps_ind_12_bin", "ps_car_10_cat", "ps_ind_14", "ind_group_id"]
features = features.difference(calc_features)
features = features.difference(meaningless_features)
features = features.difference(redundant_features)

scores = []
i = 1
test["target"] = 0
dtest = xgb.DMatrix(test[features])
for t, v in gkf.split(train[features], train.target, groups = train.id):
    print "cross - validation splits #{} ...".format(i)
    X_train = train[features].iloc[t]
    Y_train = train.target.iloc[t]
    X_valid = train[features].iloc[v]
    Y_valid = train.target.iloc[v]
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dvalid = xgb.DMatrix(X_valid, label=Y_valid)
    watchlist  = [(dtrain,'train'),(dvalid,'validation')]
    param = {'max_depth': 5, 'eta':0.05, 'silent':1, 'min_child_weight':3, 'subsample' : 0.8,
    "objective": "binary:logistic",'colsample_bytree':0.8}
    model = xgb.train(param, dtrain, 500, watchlist,early_stopping_rounds=50,feval=gini_xgb,maximize=True)
    valid_pred = model.predict(dvalid)
    score = gini_normalized(Y_valid, valid_pred)
    test["target"] = test["target"] + model.predict(dtest)
    print "fold {} local score:".format(i), score, "\n"
    scores.append(score)
    i += 1

# gini time line:
# all features - 0.2807
# remove all calc features - 0.2818
# remove meaningless features - 0.2818+
# set min_sum_hessian_in_leaf to 100 - 0.2833
# set min_sum_hessian_in_leaf to 230 - 0.2834
# add nNa_X - 0.2842
# add bayesian - 0.2860
# add multi - 0.2864

test["target"] = test["target"] * (1.0 / n_fold)
score = np.average(scores)
print "average score: ", score
print "standard deviation: ", np.std(scores)
test[["id", "target"]].to_csv("./submission/{}.csv".format(score), index = False)
