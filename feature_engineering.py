# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 17:12:25 2017

@author: mingxia.huang
"""

import pandas as pd

print "reading data ..."
prior = pd.read_csv("./downloads/order_products__prior.csv")
orders = pd.read_csv("./downloads/orders.csv")
products = pd.read_csv("./downloads/products.csv")
print "done.\n"
print "exploring data analysis:"
print "prior contrains {} / {} orders;".format(prior.order_id.unique().size, orders.order_id.unique().size)
print "prior contrains {} / {} products;".format(prior.product_id.unique().size, products.product_id.unique().size)
prior = pd.merge(prior, orders)
print "prior contrains {} / {} users.".format(prior.user_id.unique().size, orders.user_id.unique().size)

print "product features extraction."
product_library = pd.DataFrame()
print "product_occur_times;"
product_library["product_occur_times"] = prior.groupby("product_id").order_id.size() # orders per product in prior
print "product_order_number_occur_times;"
product_library["product_order_number_occur_times"] = prior.groupby(["product_id","order_number"]).order_id.size().reset_index().groupby("product_id")[0].mean() # orders per product and order number, describes the hot trendency each product
print "product_urgent_avg;"
product_library["product_urgent_avg"] = prior.groupby("product_id").add_to_cart_order.mean() # whether a product is urgent or important
print "product_order_urgent_max_avg;"
product_library["product_order_urgent_max_avg"] = prior.groupby(["product_id", "order_id"]).add_to_cart_order.max().reset_index().groupby("product_id")[0].mean()
print "product_order_urgent_min_avg;"
product_library["product_order_urgent_min_avg"] = prior.groupby(["product_id", "order_id"]).add_to_cart_order.min().reset_index().groupby("product_id")[0].mean()
print "product_urgent_rate_max;"
product_library["product_urgent_rate_max"] = product_library.product_urgent_avg / product_library.product_order_urgent_max_avg
print "product_urgent_rate_min;"
product_library["product_urgent_rate_min"] = product_library.product_order_urgent_min_avg / product_library.product_urgent_avg
print "product_reorder_times;"
product_library["product_reorder_times"] = prior.groupby("product_id").reordered.sum() # reorders per product in prior
print "product_hot_hour;"
product_library["product_hot_hour"] = prior.groupby(["product_id", "order_hour_of_day", "order_number"]).order_id.size().reset_index().groupby("product_id")[0].max() # hot hour per product
print "product_hot_dow;"
product_library["product_hot_dow"] = prior.groupby(["product_id", "order_dow", "order_number"]).order_id.size().reset_index().groupby("product_id")[0].max() # hot dow per product
print "product_number_of_fans;"
product_library["product_number_of_fans"] = prior.groupby("product_id").user_id.agg(lambda x: len(x.unique())) # number of fans per product
print "product_first_occur"
product_library["product_first_occur"] = prior.groupby("product_id").order_number.min() # first occured order number
print "product_order_interval"
product_library["product_order_interval"] = prior.groupby("product_id").days_since_prior_order.mean() # interval days

print "product features finished.\nsaving ... "
product_library.reset_index().to_csv("./features/product_features.csv", index = False)
print "completed."


