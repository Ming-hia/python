# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 16:20:28 2017

@author: mingxia.huang
"""

import numpy as np
import pandas as pd

def f1_score_single(y_true, y_pred):
    y_true = set(y_true)
    y_pred = set(y_pred)
    cross_size = len(y_true & y_pred)
    if cross_size == 0: return 0.
    p = 1. * cross_size / len(y_pred)
    r = 1. * cross_size / len(y_true)
    return 2 * p * r / (p + r)
    
def f1_score(y_true, y_pred):
    return np.mean([f1_score_single(x, y) for x, y in zip(y_true, y_pred)])
     
def evaluate(actual, prediction):
    """
    actual format - valid[["user_id", "product_id", "reordered"]]
    predict format - [["user_id", "product_id", "pred"]]
    """
    reordered_cnts = actual.groupby("user_id").reordered.sum()
    reordered_cnts = reordered_cnts[reordered_cnts == 0].reset_index()
    reordered_cnts["products"] = ["None"]
    empty = reordered_cnts[["user_id", "products"]]
    actual = actual[actual.reordered == 1][["user_id", "product_id"]]
    actual = actual[["user_id","product_id"]].groupby("user_id")["product_id"].apply(list).to_frame("actual")
    actual = pd.concat([actual.reset_index(), empty])

    reordered_cnts = prediction.groupby("user_id").pred.sum()
    reordered_cnts = reordered_cnts[reordered_cnts == 0].reset_index()
    reordered_cnts["products"] = ["None"]
    empty = reordered_cnts[["user_id", "products"]]
    prediction = prediction[prediction.pred == 1][["user_id", "product_id"]]
    prediction = prediction[["user_id","product_id"]].groupby("user_id")["product_id"].apply(list).to_frame("pred")
    prediction = pd.concat([prediction.reset_index(), empty])
    df = pd.merge(actual[["user_id", "actual"]], prediction[["user_id", "pred"]], on = "user_id", how = "left")
    df = df.fillna("None")
    return f1_score(list(df.actual), list(df.pred))
    
if __name__ == "__main__":
    a = pd.DataFrame({"user_id": [1,1,3], "product_id":[1,2,3], "reordered":[1,0,0]})
    p = pd.DataFrame({"user_id": [1,1,3], "product_id":[1,2,3], "pred":[1,1,0]})
    print evaluate(a, p)