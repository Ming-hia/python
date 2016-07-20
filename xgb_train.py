
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

INPUT_FILE = "tables/train_merged.csv"

def loaddata():
    type = {"channel": np.int8, "price": np.float16, "demand_week_1": np.int8, "demand_week_2": np.int8, "demand_week_3": np.int8,
            "demand_week_4": np.int8,"demand_week_5": np.int8, "demand_week_6": np.int8, "demand_week_7": np.int8,
            "weight": np.float16, "inch": np.float16, "piece": np.float16, "brand": np.int8, "is_drink": np.int8,
            "pct": np.float16, "has_choc": np.int8, "has_vanilla": np.int8, "has_multigrain": np.int8,"is_bread": np.int8,
            "is_lata": np.int8, "hot_dog": np.int8, "sandwich": np.int8, "State": np.int8,"popularity": np.int8,
            "NaN": np.int8, "NickName": np.int8, "NonName": np.int8, "Group": np.int8,"Grocery": np.float16,
            "SuperChain": np.int8, "Pharmacy": np.int8, "Education": np.int8, "Cafe": np.int8, "Restuarant": np.int8}
    train=pd.read_csv(INPUT_FILE,header=0,dtype=type)
    featurelist=list(train.columns)
    Y_train=train["demand_week_7"].values
    Y_train[Y_train<0]=0
    featurelist.remove("demand_week_7")
    X_train=train[featurelist]
    for fea in ["weight","inch","piece","pct"]:
        nulllist=X_train[fea].isnull()
        X_train[fea][nulllist]=-1
        is_na=pd.DataFrame(list(nulllist.apply(int)),dtype=np.int8,columns=[fea+"_isna"])
        X_train=X_train.join(is_na)

    return X_train,Y_train,featurelist

preds=np.array([1,2,3])
labels=np.array([1,2,4])

def rmsle(preds, dtrain):
    labels=dtrain.get_label()
    labels[labels<0]=0
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
    train_score=rmsle(bst.predict(dmodel),dmodel)
    val_score=rmsle(bst.predict(dval),dval)



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
num_round=500

xgbmodel(X_train,Y_train,iflog=0,ifcross=1,k_folder=5,sample_seed=111)



