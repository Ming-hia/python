# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:22:06 2017
    
@author: mingxia.huang
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

orders = pd.read_csv("./downloads/orders.csv")
prior = pd.read_csv("./downloads/order_products__prior.csv")
train = pd.read_csv("./downloads/order_products__train.csv")
submission = pd.read_csv("./downloads/sample_submission.csv")

prior = pd.merge(prior, orders)
train = pd.merge(train, orders)

def keep_k_orders(df, k = 10):
    df["num_orders"] = df.groupby("user_id")["order_number"].transform(max)
    top_k_orders = df[(df.num_orders - df.order_number) < k]
    return top_k_orders

top_k_orders = keep_k_orders(prior)

def keep_frac_items(top_k_orders, frac = 0.1):
    user_product_count = top_k_orders.groupby(["user_id", "product_id"]).order_id.size().to_frame('product_count')
    user_orders_count = top_k_orders.groupby('user_id').size().to_frame('order_count')
    user_product = user_product_count.join(user_orders_count)
    user_product['product_basket_percentage'] = user_product['product_count'] / user_product['order_count']
    
    return user_product[user_product['product_basket_percentage'] >= frac].reset_index()

top_k_products = keep_frac_items(top_k_orders, 0.025)
top_k_products["pred"] = 1

train_orders = orders[orders.eval_set == "train"]
test_orders = orders[orders.eval_set == "test"]

valid = pd.merge(train_orders[["order_id", "user_id"]], top_k_products[["user_id", "product_id", "pred"]])
valid = pd.merge(train[["order_id", "product_id", "reordered"]], valid[["order_id", "product_id", "pred"]], how = "outer")
valid.reordered = valid.reordered.fillna(0)
valid.pred = valid.pred.fillna(0)
print "local score: ", f1_score(valid.reordered, valid.pred)
print "confusion matrix: \n", confusion_matrix(valid.reordered, valid.pred)

prediction = pd.merge(test_orders[["order_id", "user_id"]], top_k_products[["user_id", "product_id"]], how = "outer")

prediction.product_id = prediction.product_id.fillna(-1)
prediction = prediction.dropna()
prediction.product_id = prediction.product_id.apply(int)
prediction.order_id = prediction.order_id.apply(int)
prediction["product_id"] = prediction["product_id"].apply(str)
prediction["product_id"][prediction["product_id"] == "-1"] = "None"

output = prediction[["order_id","product_id"]].groupby("order_id")["product_id"].apply(list).to_frame("products")
output["products"] = output["products"].apply(lambda x: " ".join(x))
output = output.reset_index()
output.to_csv("./outputs/prediction.csv", index = False)
print "complete."
