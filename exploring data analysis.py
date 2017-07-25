# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:23:59 2017

@author: mingxia.huang
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

print "show schemas:"
print "aisles - ", ','.join(aisles.columns)
print "departments - ", ','.join(departments.columns)
print "order_products_prior - ", ','.join(prior.columns)
print "order_products_train - ", ','.join(train.columns)
print "orders - ", ','.join(orders.columns)
print "products - ", ','.join(products.columns), '\n'

#reorder rate for each product
probs = pd.DataFrame()
probs["reorder_rate"] = prior.groupby("product_id").reordered.sum() / prior.groupby("product_id").size()
probs = probs.reset_index()

#merge products,aisles & departments
products = pd.merge(products, probs, on = "product_id")
products_aisles = pd.merge(products, aisles, on = "aisle_id")
products_aisles_departments = pd.merge(products, departments, on = "department_id")

print "number of products unlikely to be reordered:", np.sum(products_aisles_departments.reorder_rate == 0)
print "most frequently ordered products:", products_aisles_departments.sort_values(by = "reorder_rate").tail(5)[["product_name", "reorder_rate"]], '\n'

num_of_products = prior.groupby("order_id").add_to_cart_order.max().value_counts()
plt.figure(0)
plt.xticks(rotation='vertical')
sns.barplot(num_of_products.index, num_of_products.values,palette="cubehelix")
plt.ylabel('Number of Occurrences Prior Data', fontsize=17)
plt.xlabel('Number of products ordered in the order_id', fontsize=13)
plt.show()

num_of_products = train.groupby("order_id").add_to_cart_order.max().value_counts()
plt.figure(1)
plt.xticks(rotation='vertical')
sns.barplot(num_of_products.index, num_of_products.values,palette="cubehelix")
plt.ylabel('Number of Occurrences Train Data', fontsize=17)
plt.xlabel('Number of products ordered in the order_id', fontsize=13)
plt.show()

print "number of null rows in orders", orders.isnull().sum(), '\n'

plt.figure(2)
order_interval = orders.days_since_prior_order.value_counts()
f, ax = plt.subplots(figsize=(12.5, 8))
sns.barplot(order_interval.index,order_interval.values,palette='GnBu_d')
plt.ylabel("Order Count", fontsize=12)
plt.xlabel("Order Interval (Day)", fontsize=12)
plt.show()

print "number of unique user_ids in each dataset", orders.groupby("eval_set").user_id.aggregate(lambda x: x.unique().size), '\n'

