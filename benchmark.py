# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:22:06 2017

@author: mingxia.huang
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

orders = pd.read_csv("./downloads/orders.csv")
prior = pd.read_csv("./downloads/order_products__prior.csv")
submission = pd.read_csv("./downloads/sample_submission.csv")

prior = pd.merge(prior, orders)

def keep_k_products(df, k = 5):
    df["num_orders"] = df.groupby("user_id")["order_number"].transform(max)
    top_k_products = df[(df.num_orders - df.order_number) < k]
    return top_k_products
   
top_k_products = keep_k_products(prior)

train = orders[orders.eval_set == "train"]
test = orders[orders.eval_set == "test"]

prediction = pd.merge(train, top_k_products[["user_id", "product_id"]], on = "user_id", how = "left")[["order_id", "product_id"]]
prediction["pred"] = 1
prediction = pd.merge(prediction, prior[["order_id", "product_id", "reordered"]], on = ["order_id"], how = "outer")