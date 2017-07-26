# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 15:20:14 2017

@author: mingxia.huang
"""

import pandas as pd
import numpy as np
import lightgbm as lgb

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

# product features
print "extracting product features."
probs = pd.DataFrame()
probs["orders"] = prior.groupby("product_id").size()
probs["reorders"] = prior.groupby("product_id").reordered.sum()
probs["reorder_rate"] = probs.reorders / probs.orders
probs = probs.reset_index()
products = pd.merge(products, probs, on = "product_id")
del probs

prior = pd.merge(prior, orders, on = "order_id", how = "left")

# user features
print "extracting user features."
usr = pd.DataFrame()
usr["average_days_between_orders"] = orders.groupby("user_id").days_since_prior_order.mean()
usr["number_of_orders"] = orders.groupby("user_id").size()

user = pd.DataFrame()
user['total_items'] = prior.groupby("user_id").size()
user['all_products'] = prior.groupby('user_id')["product_id"].apply(set)
user['total_distinct_items'] = user.all_products.map(len)
#user['total_distinct_items'] = prior.groupby("user_id")["product_id"].aggregate(lambda x: x.unique().size)

user = user.join(usr)
del usr
user["average_basket"] = (user.total_items / user.number_of_orders)

# product_id X user_id ?
print "extracting user X product features.\n"
prior["user_product"] = prior.product_id + prior.user_id * 100000
dictionary = dict()
for row in prior.itertuples():
    z = row.user_product
    if z not in dictionary:
        dictionary[z] = (1, (row.order_number, row.order_id), row.add_to_cart_order)
    else:
        dictionary[z] = (dictionary[z][0] + 1,
                max(dictionary[z][1], (row.order_number, row.order_id)),
                dictionary[z][2] + row.add_to_cart_order)
del prior

# to dataframe with less memory
userXproduct = pd.DataFrame.from_dict(dictionary, orient='index')
userXproduct.columns = ['number_of_orders', 'last_order_id', 'sum_pos_in_cart']
userXproduct.last_order_id = userXproduct.last_order_id.map(lambda x: x[1])

# train - test - split
print("split orders : train, test")
test_orders = orders[orders.eval_set == "test"]
train_orders = orders[orders.eval_set == "train"]

def features(selected_orders, labels_given = False):
    print("build candidate list")
    order_list = []
    product_list = []
    labels = []
    i = 0
    for row in selected_orders.itertuples():
        i += 1
        if i % 10000 == 0:
            print('order row',i)
        order_id = row.order_id
        user_id = row.user_id
        user_products = user.all_products[user_id]
        product_list += user_products
        order_list += [order_id] * len(user_products)
        if labels_given:
            labels += [(order_id, product) in train.index for product in user_products]
        
    df = pd.DataFrame({'order_id':order_list, 'product_id':product_list})
    labels = np.array(labels)
    del order_list
    del product_list
    
    print('user related features')
    df['user_id'] = df.order_id.map(orders.user_id)
    df['user_total_orders'] = df.user_id.map(user.number_of_orders)
    df['user_total_items'] = df.user_id.map(user.total_items)
    df['total_distinct_items'] = df.user_id.map(user.total_distinct_items)
    df['user_average_days_between_orders'] = df.user_id.map(user.average_days_between_orders)
    df['user_average_basket'] =  df.user_id.map(user.average_basket)
    
    print('order related features')
    # df['dow'] = df.order_id.map(orders.order_dow)
    df['order_hour_of_day'] = df.order_id.map(orders.order_hour_of_day)
    df['days_since_prior_order'] = df.order_id.map(orders.days_since_prior_order)
    df['days_since_ratio'] = df.days_since_prior_order / df.user_average_days_between_orders
    
    print('product related features')
    df['aisle_id'] = df.product_id.map(products.aisle_id)
    df['department_id'] = df.product_id.map(products.department_id)
    df['product_orders'] = df.product_id.map(products.orders)
    df['product_reorders'] = df.product_id.map(products.reorders)
    df['product_reorder_rate'] = df.product_id.map(products.reorder_rate)

    print('user_X_product related features')
    df['z'] = df.user_id * 100000 + df.product_id
    df.drop(['user_id'], axis=1, inplace=True)
    df['UP_orders'] = df.z.map(userXproduct.number_of_orders)
    df['UP_orders_ratio'] = (df.UP_orders / df.user_total_orders)
    df['UP_last_order_id'] = df.z.map(userXproduct.last_order_id)
    df['UP_average_pos_in_cart'] = (df.z.map(userXproduct.sum_pos_in_cart) / df.UP_orders)
    df['UP_reorder_rate'] = (df.UP_orders / df.user_total_orders)
    df['UP_orders_since_last'] = df.user_total_orders - df.UP_last_order_id.map(orders.order_number)
    df['UP_delta_hour_vs_last'] = abs(df.order_hour_of_day - df.UP_last_order_id.map(orders.order_hour_of_day)).map(lambda x: min(x, 24-x))
    #df['UP_same_dow_as_last_order'] = df.UP_last_order_id.map(orders.order_dow) == \
    #                                              df.order_id.map(orders.order_dow)

    df.drop(['UP_last_order_id', 'z'], axis=1, inplace=True)
    return (df, labels)
    
df_train, labels = features(train_orders, labels_given = True)

f_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
       'user_average_days_between_orders', 'user_average_basket',
       'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
       'aisle_id', 'department_id', 'product_orders', 'product_reorders',
       'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
       'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last',
       'UP_delta_hour_vs_last'] # 'dow', 'UP_same_dow_as_last_order'


print('formating for lgb')
d_train = lgb.Dataset(df_train[f_to_use],
                      label=labels,
                      categorical_feature=['aisle_id', 'department_id'])  # , 'order_hour_of_day', 'dow'
del df_train

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 96,
    'max_depth': 10,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5
}
ROUNDS = 100

print('light GBM train :-)')
bst = lgb.train(params, d_train, ROUNDS)
# lgb.plot_importance(bst, figsize=(9,20))
del d_train

### build candidates list for test ###

df_test, _ = features(test_orders)

print('light GBM predict')
preds = bst.predict(df_test[f_to_use])

df_test['pred'] = preds

TRESHOLD = 0.22  # guess, should be tuned with crossval on a subset of train data

d = dict()
for row in df_test.itertuples():
    if row.pred > TRESHOLD:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

for order in test_orders.order_id:
    if order not in d:
        d[order] = 'None'

output = pd.DataFrame.from_dict(d, orient='index')

output.reset_index(inplace=True)
output.columns = ['order_id', 'products']
output.to_csv('outputs/benchmark.csv', index=False)

