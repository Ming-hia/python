import pandas as pd
import numpy as np
from random import sample
import warnings
warnings.filterwarnings("ignore")

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

# merge
prior = pd.merge(prior, orders)

# extracting user features
print "extracting user features."
users = pd.DataFrame()
users["days_since_prior_order_avg"] = orders.groupby("user_id").days_since_prior_order.mean()
users["number_of_order_per_user"] = orders.groupby("user_id").size()

users_prior = pd.DataFrame()
users_prior["total_items"] = prior.groupby("user_id").size()
users_prior["all_products"] = prior.groupby("user_id")["product_id"].apply(set)
users_prior["total_distinct_items"] = users_prior.all_products.map(len)

users = users_prior.join(users)
del users_prior
users["average_basket"] = (users.total_items / users.number_of_order_per_user)
users = users.reset_index()

# extracting user - product features
print "extracting user - product features.\n"
top_k_orders = keep_k_orders(prior)
top_k_products = keep_frac_items(top_k_orders, 0.025)
top_k_products["is_top_k"] = 1
last_order = keep_k_orders(prior,1)
recent_ordered_products = keep_frac_items(last_order, 0.0)
recent_ordered_products["in_last_order"] = 1

user_product_list = prior[["user_id","product_id"]].drop_duplicates()
user_product_list = pd.merge(user_product_list, top_k_products, how = "left")
user_product_list = pd.merge(user_product_list, recent_ordered_products, how = "left")
user_product_list.is_top_k = user_product_list.is_top_k.fillna(0)
user_product_list.in_last_order = user_product_list.in_last_order.fillna(0)

user_product = pd.DataFrame()
grouped = prior.groupby(["user_id", "product_id"])
user_product["ordered_times"] = grouped.order_id.size()
user_product["reordered_times"] = grouped.reordered.sum()
user_product["reordered_rate_exact"] = user_product.reordered_times / user_product.ordered_times
user_product["days_since_prior_order_avg_exact"] = grouped.days_since_prior_order.mean()
user_product["purchase_order_avg"] = grouped.add_to_cart_order.mean()
user_product["purchase_order_std"] = grouped.add_to_cart_order.std()
user_product["purchase_order_max"] = prior.groupby(["user_id", "product_id", "order_id"]).add_to_cart_order.max().reset_index().groupby(["user_id", "product_id"]).add_to_cart_order.mean()
user_product["purchase_importance_rate"] = user_product.purchase_order_avg / user_product.purchase_order_max
user_product["order_hour_of_day_avg"] = grouped.order_hour_of_day.mean()
user_product["order_hour_of_day_std"] = grouped.order_hour_of_day.std()

user_product_list = pd.merge(user_product_list, user_product.reset_index())

# train - test split
print "split train / test set."
train_orders = orders[orders.eval_set == "train"]
test_orders = orders[orders.eval_set == "test"]
del orders

train_orders = pd.merge(train_orders, user_product_list, on = "user_id", how = "left")
test = pd.merge(test_orders, user_product_list, on = "user_id", how = "left")
del user_product_list

train = pd.merge(train_orders, train, how = "left")
train.reordered = train.reordered.fillna(0)
del train_orders
del test_orders

# merge all features
print "merge product features."
train = pd.merge(train, products, on = "product_id")
test = pd.merge(test, products, on = "product_id")
del products

print "merge user features."
train = pd.merge(train, users, on = "user_id")
test = pd.merge(test, users, on = "user_id")
del users

print "save features."
train.to_csv("./data/train.csv", index = False)
test.to_csv("./data/test.csv", index = False)
print "complete."