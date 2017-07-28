# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:56:34 2017

@author: mingxia.huang
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from random import sample
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def keep_k_orders(df, k = 10):
    df["num_orders"] = df.groupby("user_id")["order_number"].transform(max)
    top_k_orders = df[(df.num_orders - df.order_number) < k]
    return top_k_orders
    
def keep_frac_items(top_k_orders, frac = 0.1):
    user_product_count = top_k_orders.groupby(["user_id", "product_id"]).order_id.size().to_frame('product_count')
    user_orders_count = top_k_orders.groupby('user_id').size().to_frame('order_count')
    user_product = user_product_count.join(user_orders_count)
    user_product['product_basket_percentage'] = user_product['product_count'] / user_product['order_count']    
    return user_product[user_product['product_basket_percentage'] >= frac].reset_index()    

aisles = pd.read_csv("./downloads/aisles.csv")
print "finish reading aisles.csv;"
departments = pd.read_csv("./downloads/departments.csv")
print "finish reading departments.csv;"
prior = pd.read_csv("./downloads/order_products__prior.csv")
print "finish reading order_products__prior.csv;"
train = pd.read_csv("./downloads/order_products__train.csv")
print "finish reading order_products__train.csv;"
orders = pd.read_csv("./downloads/orders.csv")
print "finish reading orders.csv;"
products = pd.read_csv("./downloads/products.csv")
print "finish reading products.csv.\n"

products = pd.merge(products, departments, on = "department_id", how = "left")
products = pd.merge(products, aisles, on = "aisle_id", how = "left")

# extracting product features
print "extracting product features."
probs = pd.DataFrame()
probs["orders"] = prior.groupby("product_id").size() # orders per product in prior
probs["purchases"] = prior.groupby(["product_id", "order_id"]).add_to_cart_order.max().reset_index().groupby("product_id").add_to_cart_order.mean() # describes the relation of product and the number of the orders
probs["purchase_order"] = prior.groupby("product_id").add_to_cart_order.mean() # describes the importances of the product
probs["product_importances_rate"] = probs.purchase_order / probs.purchases
probs["reorders"] = prior.groupby("product_id").reordered.sum() # reordered number per product in prior
probs["reorder_rate"] = probs.reorders / probs.orders # reorder rate per product
probs = probs.reset_index()
products = pd.merge(products, probs, on = "product_id")
del probs

# extracting user-product features
prior = pd.merge(prior, orders)
top_k_orders = keep_k_orders(prior)
top_k_products = keep_frac_items(top_k_orders, 0.025)
top_k_products["is_top_k"] = 1
user_product_list = prior[["user_id","product_id"]].drop_duplicates()
user_product_list = pd.merge(user_product_list, top_k_products, how = "left")
user_product_list.is_top_k = user_product_list.is_top_k.fillna(0)

train_orders = orders[orders.eval_set == "train"]
test_orders = orders[orders.eval_set == "test"]
del orders
train_orders = pd.merge(train_orders, user_product_list, on = "user_id", how = "left")
test = pd.merge(test_orders, user_product_list, on = "user_id", how = "left")

train = pd.merge(train_orders, train, how = "left")
train.reordered = train.reordered.fillna(0)
train = pd.merge(train, products, on = "product_id")
test = pd.merge(test, products, on = "product_id")

features = ["days_since_prior_order", "order_dow", "order_hour_of_day", "department_id", "aisle_id", "reorder_rate", "orders", "reorders",
            "purchase_order", "purchases", "product_importances_rate",
            "is_top_k"]

# feature - f1_score timeline: 
# ["days_since_prior_order", "order_dow", "order_hour_of_day", "department_id", "aisle_id", "reorder_rate", "orders", "reorders"] - 0.242
# ["purchase_order", "purchases", "product_importances_rate"] - 0.243 (+ 0.001)



def k_fold(index, k = 5):
    n = len(index)
    index = set(index)
    pool = index
    s = n / k
    index_list = []
    for i in range(k):
        if i < (k - 1):
            ids = sample(pool, s)
        else:
            ids = pool
        index_list.append({"train": index.difference(ids), "valid": set(ids)})
        pool = pool.difference(ids)
    return index_list

n_fold = 5            
user_splits = k_fold(train.user_id.unique(), n_fold)

X_test = test[features]
test["pred"] = 0
avg_threshold = 0
scores = []

for i in range(n_fold):
    X_train = train[train.user_id.isin(user_splits[i]["train"])][features]
    Y_train = train[train.user_id.isin(user_splits[i]["train"])]["reordered"]
    X_valid = train[train.user_id.isin(user_splits[i]["valid"])][features]
    Y_valid = train[train.user_id.isin(user_splits[i]["valid"])]["reordered"]
    model = lgb.LGBMClassifier(objective='binary',
                        max_depth = 4,
                        learning_rate=0.05,
                        n_estimators=100)
    model.fit(X_train, Y_train, verbose = 10)
    valid_pred = pd.DataFrame(model.predict_proba(X_valid))[1]
    threshold = float(sum(Y_valid)) / len(Y_valid)
    avg_threshold += threshold
    valid_pred = np.where(valid_pred > threshold, 1, 0)
    score = f1_score(Y_valid, valid_pred)
    print "fold {} local score:".format(i), score
    scores.append(score)
    test["pred"] += pd.DataFrame(model.predict_proba(X_test))[1]

avg_threshold /= n_fold
test["pred"] /= n_fold
print "average score: ", np.average(scores)
print "average threshold: ", avg_threshold

test["pred"] = np.where(test.pred > avg_threshold, 1, 0)
prediction = test[["order_id", "product_id", "pred"]]
prediction.product_id = prediction.product_id.apply(str)
order_cnt = prediction.groupby("order_id").pred.sum()
order_cnt = order_cnt[order_cnt == 0].reset_index()
order_cnt["products"] = "None"
empty = order_cnt[["order_id", "products"]]
prediction = prediction[prediction.pred == 1][["order_id", "product_id"]]
output = prediction[["order_id","product_id"]].groupby("order_id")["product_id"].apply(list).to_frame("products")
output["products"] = output["products"].apply(lambda x: " ".join(x))
output = pd.concat([output.reset_index(), empty])
output.to_csv("./outputs/prediction.csv", index = False)
print "complete."
