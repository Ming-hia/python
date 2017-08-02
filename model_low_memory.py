import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings("ignore")

print "read data & features ..."
train = pd.read_csv("./downloads/order_products__train.csv")
orders = pd.read_csv("./downloads/orders.csv")
products = pd.read_csv("./downloads/products.csv")
departments = pd.read_csv("./downloads/departments.csv")
aisles = pd.read_csv("./downloads/aisles.csv")

product_features = pd.read_csv("./features/product_features.csv")
user_features = pd.read_csv("./features/user_features.csv")
user_product_features = pd.read_csv("./features/user_product_features.csv")

print "merge features ..."
products = pd.merge(products, departments, on = "department_id", how = "left")
products = pd.merge(products, aisles, on = "aisle_id", how = "left")
products = pd.merge(products, product_features, on = "product_id", how = "inner")

train_orders = orders[orders.eval_set == "train"]
del orders

train_orders = pd.merge(train_orders, user_product_features, on = "user_id", how = "left")
del user_product_features

train = pd.merge(train_orders, train, how = "left")
del train_orders
train.reordered = train.reordered.fillna(0)

train = pd.merge(train, products, on = "product_id")
del products

train = pd.merge(train, user_features, on = "user_id")
del user_features

print "start training ..."
# training settings
for col in ["order_id", "eval_set", "product_id", "add_to_cart_order", "product_name", "department", "aisle"]:
    del train[col]
    
features = train.columns.difference(["reordered", "user_id"])
print "training data size: ", train.shape
print "number of features: ", len(features)

scores = []

gkf = GroupKFold(n_splits = 10)
i = 1
print "training models ..."

for t, v in gkf.split(train[features], train.reordered, groups = train.user_id):
    print "cross - validation splits ..."
    X_train = train[features].iloc[t]
    Y_train = train.reordered.iloc[t]
    X_valid = train[features].iloc[v]
    Y_valid = train.reordered.iloc[v]
    model = lgb.LGBMClassifier(objective='binary',
                               max_depth = 6,
                               learning_rate=0.1,
                               n_estimators=1500)
    model.fit(X_train, Y_train, 
              eval_set = [(X_train, Y_train), (X_valid, Y_valid)], 
              eval_metric = "logloss", verbose = 10, early_stopping_rounds = 10)
    valid_pred = pd.DataFrame(model.predict_proba(X_valid))[1]
    score = f1_score(Y_valid, np.where(valid_pred > 0.21, 1, 0))
    print "fold {} local score:".format(i), score, "\n"
    scores.append(score)
    i += 1
    if i > 1:
        break

orders = pd.read_csv("./downloads/orders.csv")
products = pd.read_csv("./downloads/products.csv")
departments = pd.read_csv("./downloads/departments.csv")
aisles = pd.read_csv("./downloads/aisles.csv")

product_features = pd.read_csv("./features/product_features.csv")
user_features = pd.read_csv("./features/user_features.csv")
user_product_features = pd.read_csv("./features/user_product_features.csv")


products = pd.merge(products, departments, on = "department_id", how = "left")
products = pd.merge(products, aisles, on = "aisle_id", how = "left")
products = pd.merge(products, product_features, on = "product_id", how = "inner")

test_orders = orders[orders.eval_set == "test"]
del orders

test = pd.merge(test_orders, user_product_features, on = "user_id", how = "left")
del user_product_features
del test_orders

test = pd.merge(test, products, on = "product_id")
del products
test = pd.merge(test, user_features, on = "user_id")
del user_features

X_test = test[features]
test["pred"] = 0

pred = pd.DataFrame(model.predict_proba(test[features]))[1]
test["pred"] = np.where(pred > 0.2, 1, 0)
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
output.to_csv("./outputs/lgb.csv", index = False)
print "complete."