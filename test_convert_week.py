#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

TEST_FILENAME = "./input/test.csv"
TRAIN_FILENAME = "./tables/train.csv"
INPUT_AGENT_FILE = "tables/agent.csv"
INPUT_CLIENT_FILE = "tables/client.csv"
INPUT_PRODUCT_FILE = "tables/product.csv"

OUTPUT_FILE="tables/merged_test.csv"



def loaddata():
    train = pd.read_csv(TRAIN_FILENAME, header=0)
    del train["agent"]
    del train["channel"]
    del train["route"]
    test=pd.read_csv(TEST_FILENAME,header=0)
    test.rename(columns={"Producto_ID": "product"}, inplace=True)
    test.rename(columns={"Cliente_ID": "client"}, inplace=True)
    test.rename(columns={"Agencia_ID": "agent"}, inplace=True)
    test.rename(columns={"Canal_ID": "channel"}, inplace=True)
    test.rename(columns={"Ruta_SAK": "route"}, inplace=True)
    return train,test

def merge(train,test):
    agent = pd.read_csv(INPUT_AGENT_FILE, header=0)
    client = pd.read_csv(INPUT_CLIENT_FILE, header=0)
    product = pd.read_csv(INPUT_PRODUCT_FILE, header=0)
    product.rename(columns={"Producto_ID": "product"}, inplace=True)
    agent.rename(columns={"Agencia_ID": "agent", "Value": "popularity", "NaN": "loc_not_available"}, inplace=True)
    client.rename(columns={"Cliente_ID": "client"}, inplace=True)
    test = pd.merge(test, train,how="left" ,on=["client","product"])
    test["is_nextweek"]=int(test["Semana"]==11)
    del test["Semana"]
    del test["demand_week_1"]
    del test["demand_week_2"]
    del test["price"]
    for i in range(2,8):
        fea="demand_week_"+str(i)
        fea_transfer="demand_week_"+str(i-2)
        test[fea][test[fea].isnull()]=0
        test.rename(columns={fea:fea_transfer}, inplace=True)
    test = pd.merge(test, product, on="product")
    test = pd.merge(test, agent, on="agent")
    test = pd.merge(test, client, on="client")
    del test["product"]
    del test["agent"]
    del test["client"]
    del test["route"];
    del test["TownName"];
    del test["TownID"]
    categorical_cols = ["brand", "State"]
    for key in categorical_cols:
        print "column " + key + ", " + str(len(test[key].unique())) + " values"
        test[key] = pd.Categorical.from_array(test[key]).codes
    return test


if __name__ == '__main__':
    train, test=loaddata()
    test=merge(train,test)

    test.to_csv(OUTPUT_FILE, index=False)
