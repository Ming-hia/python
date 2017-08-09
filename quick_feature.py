import pandas as pd
import lightgbm as lgb
from F1Optimizer_numba import *
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

print "read data ..."
train = pd.read_csv("./downloads/order_products__train.csv")
orders = pd.read_csv("./downloads/orders.csv")
products = pd.read_csv("./downloads/products.csv")
departments = pd.read_csv("./downloads/departments.csv")
aisles = pd.read_csv("./downloads/aisles.csv")

print "read features ..."
product_features = pd.read_csv("./features/product_features.csv")
user_features = pd.read_csv("./features/user_features.csv")
user_product_features = pd.read_csv("./features/user_product_features.csv")
user_product_additional_features = pd.read_csv("./features/user_product_additional_features.csv")
department_hour_features = pd.read_csv("./features/department_hour_features.csv")[["department_id", "order_hour_of_day", "department_hour_reordered_rate"]]
user_department_features = pd.read_csv("./features/user_department_features.csv")[["user_id", "department_id", "user_department_reordered_rate", "department_reordered_rate"]]
user_aisle_features = pd.read_csv("./features/user_aisle_features.csv")[["user_id", "aisle_id", "user_aisle_reordered_rate", "aisle_reordered_rate"]]

print "merge data ..."
products = pd.merge(products, departments, on = "department_id", how = "left")
products = pd.merge(products, aisles, on = "aisle_id", how = "left")
product_features = pd.merge(products, product_features, on = "product_id", how = "inner")

print "merge features ..."
train_orders = orders[orders.eval_set == "train"]
train_orders = train_orders[train_orders.order_id <= train_orders.order_id.max() * 0.2]
del orders

train_orders = pd.merge(train_orders, user_product_features, on = "user_id", how = "left")
train_orders = pd.merge(train_orders, user_product_additional_features, on = ["user_id", "product_id"], how = "left")
del user_product_features

train = pd.merge(train_orders, train, how = "left")
del train_orders

train.reordered = train.reordered.fillna(0)

train = pd.merge(train, product_features, on = "product_id")
train = pd.merge(train, department_hour_features, on = ["department_id", "order_hour_of_day"], how = "left")
train = pd.merge(train, user_department_features, on = ["department_id", "user_id"], how = "left")
train = pd.merge(train, user_aisle_features, on = ["aisle_id", "user_id"], how = "left")
train = pd.merge(train, user_features, on = "user_id")

print "start training ..."
# training settings
for col in ["eval_set", "product_id", "add_to_cart_order", "product_name", "department", "aisle"]:
    del train[col]

features = train.columns.difference(["reordered", "user_id", "order_id"])

#order_features = ["order_number", "days_since_prior_order", "order_dow", "order_hour_of_day"]
#user_product_features = ["user_product_last_orders", "user_product_order_number_avg", "user_product_order_number_skew", "user_product_orders_since_last_orders", "order_streak",
 #                        "user_product_first_orders", "user_product_purchase_avg", "user_product_occur_times", "user_product_reordered_rate", "user_product_order_rate", "user_product_order_rate", "user_product_orders_since_last_order", "user_product_rate_since_first_order",
#                         "is_top_k", "in_last_5_orders", "in_last_2_orders", "in_last_order"]
#user_features = ["user_order_avg", "user_purchase_frequency", "user_reordered_rate", "user_order_purchase_max", "user_reordered_times", "user_occur_times"]
#product_features = ["product_reordered_rate", "product_order_interval", "product_number_of_fans", "product_urgent_avg",
#                   "product_reordered_times", "product_hot_dow", "product_hot_hour", "product_occur_times", "product_order_number_occur_times", "product_first_occur"]
#special_features = ["aisle_id", "department_id"]
#features = order_features + user_product_features  + product_features + special_features + user_features

# logloss records:
# original features, 0.315469
# user_product_features_l1, 0.2537
# user_product_features_l2, 0.253393
# all user_features are useless 
# product_features, 0.249096
# special features, 0.248356
# all features, 0.247385
# new user_product_features, 0.252365
# new all features, 0.246619
# para tuning, 0.246251

print "training data size: ", train.shape
print "feature selection: "
features = ["order_number", "days_since_prior_order", "order_dow", "order_hour_of_day"]
features += ["product_occur_times", "product_reordered_rate",
             "product_order_interval", "product_interval_varities",
             "product_number_of_fans", "last_reordered_rate", "product_urgent_avg"]
features += ["user_occur_times", "user_reordered_times", "user_reordered_rate", "user_purchase_frequency", "user_purchase_varities", "user_product_fans"]
features += ["aisle_id", "department_id"]
features += ["is_top_k", "in_last_order"]
features += ["user_product_last_orders", "user_product_order_number_avg", "user_product_order_number_skew",
             "user_product_first_orders", "user_product_purchase_avg", "user_product_occur_times"]
features += ["user_product_orders_since_last_orders", "order_streak", "user_product_reordered_rate",
             "user_product_order_rate", "user_product_orders_since_last_order", "user_product_rate_since_first_order"]
features += ["user_product_purchase_varities", "user_product_purchase_frequency"]
features += ["user_aisle_reordered_rate", "aisle_reordered_rate", "user_department_reordered_rate", "department_hour_reordered_rate"]
features += ["last", "prev1", "prev2", "median", "mean"]
print "number of features: ", len(features)

print "split train - test sets."
X_train = train[train.order_id <= train.order_id.max() * 0.9][features]
Y_train = train[train.order_id <= train.order_id.max() * 0.9].reordered
X_valid = train[train.order_id > train.order_id.max() * 0.9][features]
Y_valid = train[train.order_id > train.order_id.max() * 0.9].reordered

print "training models ..."
#model = lgb.LGBMClassifier(objective='binary', num_leaves = 64, min_child_samples=3,
#        min_child_weight=2, subsample = 1, colsample_bytree = 0.9, learning_rate=0.1, n_estimators=400)
model = lgb.LGBMClassifier(objective='binary', num_leaves = 256, min_child_samples=3, max_depth = 12,
        min_child_weight=2, subsample = 1, colsample_bytree = 0.6, learning_rate=0.05, n_estimators=400)
model.fit(X_train, Y_train, eval_set = [(X_train, Y_train), (X_valid, Y_valid)], eval_metric = "logloss", verbose = 10, early_stopping_rounds = 20)

print "show feature importances: "
feature_importances = pd.DataFrame(zip(features, model.feature_importances_))
feature_importances.columns = ["features", "importances"]
feature_importances = feature_importances.sort_values(by = "importances")
print feature_importances

'''
cormat = np.corrcoef(train[features].sample(10000).T)
sns.heatmap(cormat, cbar = True, robust = True, cmap = "RdGy")
plt.xticks(rotation = 45)
plt.yticks(rotation = 45)
plt.title("Heatmap Plot of the Correlation Matrix")
'''
