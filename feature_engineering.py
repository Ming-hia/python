# -*- coding: utf-8 -*-
"""
    Created on Mon Jul 31 17:12:25 2017
    
    @author: mingxia.huang
    """

import pandas as pd

product_update_flag = 0
user_update_flag = 0
user_product_update_flag = 1

def keep_k_orders(df, k = 10):
    df["num_orders"] = df.groupby("user_id")["order_number"].transform(max)
    top_k_orders = df[(df.num_orders - df.order_number) < k]
    return top_k_orders

def keep_frac_items(top_k_orders, frac = 0.1):
    user_product_count = top_k_orders.groupby(["user_id", "product_id"]).order_id.size().to_frame('product_count')
    user_orders_count = top_k_orders.groupby('user_id').size().to_frame('order_count')
    user_product = user_product_count.join(user_orders_count)
    user_product['product_basket_percentage'] = user_product['product_count'] / user_product['order_count']
    return user_product[user_product['product_basket_percentage'] >= frac].reset_index()[["user_id", "product_id"]]

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
# order_id, product_id, add_to_cart_order, reordered, user_id, eval_set, order_number, order_dow, order_hour_of_day, days_since_prior_order

if product_update_flag:
    print "product features extraction ..."
    product_library = pd.DataFrame()
    print "product_occur_times;"
    product_library["product_occur_times"] = prior.groupby("product_id").order_id.size() # orders per product in prior
    print "product_order_number_occur_times;"
    product_library["product_order_number_occur_times"] = prior.groupby(["product_id","order_number"]).order_id.size().reset_index().groupby("product_id")[0].mean() # orders per product and order number, describes the hot trendency each product
    print "product_urgent_avg;"
    product_library["product_urgent_avg"] = prior.groupby("product_id").add_to_cart_order.mean() # whether a product is urgent or important
    print "product_order_urgent_max_avg;"
    product_library["product_order_urgent_max_avg"] = prior.groupby(["product_id", "order_id"]).add_to_cart_order.max().reset_index().groupby("product_id").add_to_cart_order.mean()
    print "product_order_urgent_min_avg;"
    product_library["product_order_urgent_min_avg"] = prior.groupby(["product_id", "order_id"]).add_to_cart_order.min().reset_index().groupby("product_id").add_to_cart_order.mean()
    print "product_urgent_rate_max;"
    product_library["product_urgent_rate_max"] = product_library.product_urgent_avg / product_library.product_order_urgent_max_avg
    print "product_urgent_rate_min;"
    product_library["product_urgent_rate_min"] = product_library.product_order_urgent_min_avg / product_library.product_urgent_avg
    print "product_reordered_times;"
    product_library["product_reordered_times"] = prior.groupby("product_id").reordered.sum() # reorders per product in prior
    product_library["product_reordered_rate"] = product_library.product_reordered_times / product_library.product_occur_times
    print "product_hot_hour;"
    product_library["product_hot_hour"] = prior.groupby(["product_id", "order_hour_of_day", "order_number"]).order_id.size().reset_index().groupby("product_id")[0].max() # hot hour per product
    print "product_hot_dow;"
    product_library["product_hot_dow"] = prior.groupby(["product_id", "order_dow", "order_number"]).order_id.size().reset_index().groupby("product_id")[0].max() # hot dow per product
    print "product_number_of_fans;"
    product_library["product_number_of_fans"] = prior.groupby("product_id").user_id.agg(lambda x: len(x.unique())) # number of fans per product
    print "product_first_occur;"
    product_library["product_first_occur"] = prior.groupby("product_id").order_number.min() # first occured order number
    print "product_order_interval."
    product_library["product_order_interval"] = prior.groupby("product_id").days_since_prior_order.mean() # interval days

    print "product features finished.\nsaving ... "
    product_library.reset_index().to_csv("./features/product_features.csv", index = False)
    print "completed.\n"

if user_update_flag:
    print "user features extraction ..."
    user_library = pd.DataFrame()
    print "user_occur_times;"
    user_library["user_occur_times"] = prior.groupby("user_id").order_id.size()
    print "user_order_times;"
    user_library["user_order_times"] = prior.groupby("user_id").order_number.max()
    print "user_order_avg;"
    user_library["user_order_avg"] = user_library.user_occur_times / user_library.user_order_times
    print "user_reordered_times;"
    user_library["user_reordered_times"] = prior.groupby("user_id").reordered.sum()
    print "user_reordered_rate;"
    user_library["user_reordered_rate"] = user_library.user_reordered_times / user_library.user_occur_times
    print "user_purchase_frequency;"
    user_library["user_purchase_frequency"] = prior.groupby("user_id").days_since_prior_order.mean()
    print "user_product_fans;"
    user_library["user_product_fans"] = prior.groupby("user_id").product_id.agg(lambda x: len(x.unique()))
    print "user_order_purchase_max."
    user_library["user_order_purchase_max"] = prior.groupby("user_id").add_to_cart_order.max() # purchase ability per user

    print "user features finished.\nsaving ... "
    user_library.reset_index().to_csv("./features/user_features.csv", index = False)
    print "completed.\n"

if user_product_update_flag:
    print "user product features extraction ..."
    print "find products in recent k orders;"
    top_k_orders = keep_k_orders(prior)
    top_k_products = keep_frac_items(top_k_orders, 0.025)
    top_k_products["is_top_k"] = 1
    print "keep products in last order;"
    last_order = keep_k_orders(prior, 1)
    recent_ordered_products = keep_frac_items(last_order, 0.0)
    recent_ordered_products["in_last_order"] = 1

    print "user product features extraction ..."
    user_product_library = pd.DataFrame()
    user_product_library["user_product_first_orders"] = prior.groupby(["user_id","product_id"]).order_number.min()
    user_product_library["user_product_last_orders"] = prior.groupby(["user_id","product_id"]).order_number.max()
    user_product_library["user_product_purchase_avg"] = prior.groupby(["user_id","product_id"]).add_to_cart_order.mean()
    user_product_library["user_product_occur_times"] = prior.groupby(["user_id","product_id"]).order_id.size()
    user_product_library["user_product_reordered_times"] = prior.groupby(["user_id","product_id"]).reordered.sum()
    user_product_library["user_product_reordered_rate"] = user_product_library.user_product_reordered_times / user_product_library.user_product_occur_times
    
    if not user_update_flag:
        user_library = pd.DataFrame()
        user_library["user_occur_times"] = prior.groupby("user_id").order_id.size()
    user_product_library = pd.merge(user_product_library.reset_index(), user_library.reset_index(), how = "left")
    user_product_library["user_product_order_rate"] = user_product_library.user_product_occur_times / user_product_library.user_occur_times
    user_product_library["user_product_orders_since_last_order "] = user_product_library.user_occur_times / user_product_library.user_product_last_orders
    user_product_library["user_product_rate_since_first_order"] = user_product_library.user_product_occur_times / (user_product_library.user_occur_times - user_product_library.user_product_first_orders + 1)
    
    user_product_library = pd.merge(user_product_library, top_k_products, how = "left")
    user_product_library = pd.merge(user_product_library, recent_ordered_products, how = "left")
    user_product_library.is_top_k = user_product_library.is_top_k.fillna(0)
    user_product_library.in_last_order = user_product_library.in_last_order.fillna(0)

    print "user product features finished.\nsaving ... "
    user_product_library.to_csv("./features/user_product_features.csv", index = False)
    print "completed."
