import pandas as pd
import numpy as np
import lightgbm as lgb
from random import sample
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

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
test_orders = orders[orders.eval_set == "test"]

train_orders = pd.merge(train_orders, user_product_features, on = "user_id", how = "left")
test = pd.merge(test_orders, user_product_features, on = "user_id", how = "left")

train = pd.merge(train_orders, train, how = "left")
train.reordered = train.reordered.fillna(0)

train = pd.merge(train, products, on = "product_id")
test = pd.merge(test, products, on = "product_id")

train = pd.merge(train, user_features, on = "user_id")
test = pd.merge(test, user_features, on = "user_id")

print "start training ..."
# training settings
n_fold = 10
features = train.columns.difference(["order_id", "user_id", "eval_set", "product_id", "add_to_cart_order", "reordered", "product_name", "department", "aisle"])

# feature - f1_score timeline: (max_depth = 4, learning_rate = 0.05, n_estimators = 100)
# ["days_since_prior_order", "order_dow", "order_hour_of_day", "department_id", "aisle_id", "reorder_rate", "orders", "reorders"] - 0.242
# ["purchase_order", "purchases", "product_importances_rate"] - 0.243 (+ 0.001)
# ["is_top_k"] - 0.307 (+ 0.064)
# ["days_since_prior_order_avg", "number_of_order_per_user", "total_items", "total_distinct_items", "average_basket"] - 0.320 (+ 0.013)
# ["in_last_order"] - 0.319 (- 0.001)
# set n_estimators <- 500 - 0.328 (+ 0.009)
# ["ordered_times", "reordered_times", "reordered_rate_exact", "days_since_prior_order_avg_exact", "purchase_order_avg", "purchase_order_std", "purchase_order_max", "purchase_importance_rate", "order_hour_of_day_avg", "order_hour_of_day_std"] - 0.340 (+ 0.012)
# set threshold <- 0.22 - 0.400 (+ 0.600)

print "cross validation split ..."
user_splits = k_fold(train.user_id.unique(), n_fold)

X_test = test[features]
test["pred"] = 0
#avg_threshold = 0
scores = []

def search_threshold(Y_valid, valid_pred):
    thres_list = []
    scores_list = []
    for thres in (0.18,0.19,0.20,0.21,0.22,0.23):
        score = f1_score(Y_valid, np.where(valid_pred > thres, 1, 0))
        thres_list.append(thres)
        scores_list.append(score)
    return pd.DataFrame({"threshold": thres_list,"f1_score": scores_list}).sort_values(by = "f1_score", ascending=False)

print "training models ..."
for i in range(n_fold):
    X_train = train[train.user_id.isin(user_splits[i]["train"])][features]
    Y_train = train[train.user_id.isin(user_splits[i]["train"])]["reordered"]
    X_valid = train[train.user_id.isin(user_splits[i]["valid"])][features]
    Y_valid = train[train.user_id.isin(user_splits[i]["valid"])]["reordered"]
    model = lgb.LGBMClassifier(objective='binary',
                               max_depth = 6,
                               learning_rate=0.05,
                               n_estimators=100)
    model.fit(X_train, Y_train, verbose = 10)
    valid_pred = pd.DataFrame(model.predict_proba(X_valid))[1]
    #threshold = float(sum(Y_valid)) / len(Y_valid)
    #avg_threshold += threshold
    #valid_pred = np.where(valid_pred > 0.22, 1, 0)
    #score = f1_score(Y_valid, valid_pred)
    score = search_threshold(Y_valid, valid_pred)
    print "fold {} local score:".format(i), score, "\n"
    #threshold = score.threshold[0]
    #avg_threshold += threshold
    scores.append(score)
    test["pred"] += pd.DataFrame(model.predict_proba(X_test))[1]

scores_df = pd.concat(scores)
#avg_threshold /= n_fold
test["pred"] /= n_fold

print "average score: ", scores_df.groupby("threshold").f1_score.mean().reset_index()
#print "average score: ", np.average(scores)
#print "average threshold: ", avg_threshold

test["pred"] = np.where(test.pred > 0.18, 1, 0)
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
