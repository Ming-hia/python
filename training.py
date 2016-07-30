# -*- coding:utf-8 -*-

import pandas as pd
import xgboost as xgb
import numpy as np
from util import RMSLE
from sklearn.cross_validation import train_test_split
import gc

INPUT_TRAIN_FILE = "tables/train_merged.csv"
INPUT_TEST_FILE = "tables/test_merged.csv"

sample_seed = 12573
nround = 500

def load_data():
    schema = {"id": np.int32, "Semana": np.int16, "channel": np.int16, "price": np.float64, "demand_week_1": np.float64, "demand_week_2": np.float64,"demand_week_3": np.float64, "demand_week_4": np.float64, "demand_week_5": np.float64, "demand_week_6": np.float64, "demand_week_7": np.float64, "weight": np.float64, "inch": np.float64, "piece": np.float64, "brand": np.int8, "is_drink": np.int8, "pct": np.float64, "has_choc": np.int8, "has_vanilla": np.int8, "has_multigrain": np.int8,"is_bread": np.int8, "is_lata": np.int8, "hot_dog": np.int8, "sandwich": np.int8, "State": np.int8, "popularity": np.int8, "NaN": np.int8, "NickName": np.int8, "NonName": np.int8, "Group": np.int8, "Grocery": np.int8, "SuperChain": np.int8, "Pharmacy": np.int8, "Education": np.int8, "Cafe": np.int8, "Restuarant": np.int8}

    train = pd.read_csv(INPUT_TRAIN_FILE, header = 0, dtype = schema)
    test = pd.read_csv(INPUT_TEST_FILE, header = 0, dtype = schema)
    return train, test

def rmsle(pred, DMatrix):
    target = DMatrix.get_label()
    pred[pred < 1] = 1.0
    return "RMSLE", RMSLE(target, pred)

def label_nan(col):
    return np.array(col.isnull(), dtype = np.int8)

def fill_nan(col):
    col[col.isnull()] = -1
    return col

if __name__ == "__main__":
    print ""
    print "loading data from according to the schema ..."
    print ""
    
    train,test = load_data()
    test["demand"] = np.nan
    
    for i in range(1,8):
        test["demand_week_" + str(i)][test["demand_week_" + str(i)].isnull()] = 0.0
    
    for i in range(1,7):
        test["demand_week_" + str(i)] = test["demand_week_" + str(i+1)]

    for fea in ["weight","inch","piece","pct"]:
        train[fea + "_isna"] = label_nan(train[fea])
        test[fea + "_isna"] = label_nan(test[fea])
        train[fea] = fill_nan(train[fea])
        test[fea] = fill_nan(test[fea])

    print "preparing for training ..."

    train = train[train["demand_week_7"] > 0]

    features = list(train.columns)
    features.remove("demand_week_7")
    features.remove("price")
    
    X_train, X_valid, Y_train, Y_valid = train_test_split(train[features], train["demand_week_7"], test_size = 0.3, random_state = sample_seed)

    del train
    gc.collect()
    
    dtrain = xgb.DMatrix(X_train, label = Y_train)
    dvalid = xgb.DMatrix(X_valid, label = Y_valid)
    dtest = xgb.DMatrix(test[features])

    param = {"max_depth":8, "eta":0.02, "silent":1, "min_child_weight":3, "subsample":0.7, "objective":"reg:linear", "colsample_bytree":0.8}
    watch_list = [(dtrain, "train"), (dvalid, "valid")]
    
    print "start xgboost training ..."

    model = xgb.train(param, dtrain, nround, watch_list, early_stopping_rounds = 20, feval = rmsle)

    test["demand_week_7"] = model.predict(dtest)
    test["demand_week_7"][test["demand_week_7"] < 0.0] = 0.0
    test["Demanda_uni_equil"][test["Semana"] == 10] = test["demand_week_7"]

    for i in range(1,7):
        test["demand_week_" + str(i)] = test["demand_week_" + str(i+1)]

    test["demand_week_7"] = model.predict(dtest)
    test["Demanda_uni_equil"][test["Semana"] == 11] = test["demand_week_7"]
    test["Demanda_uni_equil"][test["Demanda_uni_equil"] < 1.0] = 1.0

    print ""
    print "write the result to the csv file ..."
    print ""

    test[["id", "Demanda_uni_equil"]].to_csv("submissions/xgb_submission.csv", index = False)

    print ""
    print "complete."
    print ""
