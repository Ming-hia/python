
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import gc

INPUT_FILE = "tables/train_merged.csv"
TEST_FILE="tables/merged_test.csv"
OUTPUT_FILE="results/xgb1.csv"

def loaddata():

    type = {"channel": np.int16, "price": np.float64, "demand_week_1": np.int16, "demand_week_2": np.int16,"demand_week_3": np.int16,
            "demand_week_4": np.int16, "demand_week_5": np.int16, "demand_week_6": np.int16, "demand_week_7": np.int16,
            "weight": np.float64, "inch": np.float64, "piece": np.float64, "brand": np.int8, "is_drink": np.int8,
            "pct": np.float64, "has_choc": np.int8, "has_vanilla": np.int8, "has_multigrain": np.int8,"is_bread": np.int8,
            "is_lata": np.int8, "hot_dog": np.int8, "sandwich": np.int8, "State": np.int8, "popularity": np.int8,
            "NaN": np.int8, "NickName": np.int8, "NonName": np.int8, "Group": np.int8, "Grocery": np.float64,
            "SuperChain": np.int8, "Pharmacy": np.int8, "Education": np.int8, "Cafe": np.int8, "Restuarant": np.int8}

    train=pd.read_csv(INPUT_FILE,header=0,dtype=type)
    train_demand6=train[train["demand_week_6"]>0]
    del train_demand6["demand_week_7"]
    train_demand6["is_nextweek"]=0
    train_demand6["demand"]=train_demand6["demand_week_6"]
    del train_demand6["demand_week_6"]

    train_demand7 = train[train["demand_week_7"] > 0]
    del train_demand7["demand_week_6"]
    train_demand7["is_nextweek"] = 1
    train_demand7["demand"] = train_demand7["demand_week_7"]
    del train_demand7["demand_week_7"]
    del train
    gc.collect()
    train=pd.concat([train_demand6,train_demand7],axis=0)

    del train_demand6
    del train_demand7
    gc.collect()

    train.index=range(train.shape[0])
    #train2 = pd.read_csv(INPUT_FILE, header=0)
    featurelist=list(train.columns)
    Y_train=train["demand"]
    #Y_train[Y_train<0]=0
    featurelist.remove("demand")
    featurelist.remove("price")
    X_train=train[featurelist]
    for fea in ["weight","inch","piece","pct"]:
        nulllist=X_train[fea].isnull()
        X_train[fea][nulllist]=-1
        is_na=pd.DataFrame(list(nulllist.apply(int)),dtype=np.int8,columns=[fea+"_isna"])
        X_train=X_train.join(is_na)
    return X_train,Y_train,featurelist

def rmsle(preds, dtrain):
    labels=dtrain.get_label()
    preds[preds < 1]=1
    #preds=np.round(preds)
    score=np.sqrt(1/float(len(labels))*(sum((np.log(preds+1)-np.log(labels+1))**2)))
    return 'RMSLE',score


def xgbmodel(X_train,Y_train,iflog=0,ifcross=1,k_folder=5,sample_seed=111):
    global num_round,max_depth,subsample,colsample_bytree,bst,auc,num_round,auclist_train,auclist_val
    param = {'max_depth':max_depth, 'eta':0.02, 'silent':1, 'min_child_weight':3, 'subsample' : subsample,'seed':111,
    "objective": "reg:linear",'colsample_bytree':colsample_bytree,'nthread':nthread,'tree_method':'exact'}
    x_model,x_val,y_model,y_val = train_test_split(X_train, Y_train, test_size = 0.3,random_state=sample_seed)
    dmodel=xgb.DMatrix(x_model,label=y_model)
    dval=xgb.DMatrix(x_val,label=y_val)
    watchlist  = [(dmodel,'train'),(dval,'validation')]

    bst = xgb.train(param, dmodel, num_round, watchlist,early_stopping_rounds=50,feval=rmsle)

    #bst = xgb.train(param, dmodel, num_round)
    #rmsle(bst.predict(dmodel),dmodel)

    #preds=bst.predict(dmodel)
    #labels=dmodel.get_label()
    #plt.hist(preds)
    #train_score=rmsle(bst.predict(dmodel),dmodel)
    #val_score=rmsle(bst.predict(dval),dval)

def score(bst):
    test = pd.read_csv(TEST_FILE, header=0)
    del test["price"]
    id=test["id"]
    del test["id"]
    for fea in ["weight", "inch", "piece", "pct"]:
        nulllist = test[fea].isnull()
        test[fea][nulllist] = -1
        is_na = pd.DataFrame(list(nulllist.apply(int)), dtype=np.int8, columns=[fea + "_isna"])
        test = test.join(is_na)
    dtest = xgb.DMatrix(test)
    scores=bst.predict(dtest)
    scores2=np.round(scores)
    scores2[scores2<1]=1
    scores2=pd.DataFrame(scores2,columns=["Demanda_uni_equil"],dtype=np.int64)
    id=pd.DataFrame(id,columns=["id"],dtype=np.int64)
    output=id.join(scores2)
    output.to_csv(OUTPUT_FILE,index=False)



def logfile(logstr):
    f=open("log.txt","a")
    f.write(logstr)
    f.write("\n")
    f.close()

X_train,Y_train,featurelist=loaddata()

max_depth=5
subsample=0.7
colsample_bytree=0.7
nthread=8
num_round=2000
sample_seed=111
xgbmodel(X_train,Y_train,iflog=0,ifcross=1,k_folder=5,sample_seed=111)

score(bst)

