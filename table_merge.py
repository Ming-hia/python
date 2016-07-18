# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

INPUT_AGENT_FILE = "tables/agent.csv"
INPUT_CLIENT_FILE = "tables/client.csv"
INPUT_PRODUCT_FILE = "tables/product.csv"
INPUT_TRAIN_TABLE_FILE = "tables/train.csv"

OUTPUT_FILE = "tables/train_merged.csv"

print ""
print "load csv files to pandas dataframes"
print ""

agent = pd.read_csv(INPUT_AGENT_FILE, header = 0)
client = pd.read_csv(INPUT_CLIENT_FILE, header = 0)
product = pd.read_csv(INPUT_PRODUCT_FILE, header = 0)
train = pd.read_csv(INPUT_TRAIN_TABLE_FILE, header = 0)

product.rename(columns={"Producto_ID":"product"}, inplace = True)
agent.rename(columns={"Agencia_ID":"agent", "Value":"popularity", "NaN":"loc_not_available"}, inplace = True)
client.rename(columns={"Cliente_ID":"client"}, inplace = True)


print "merge with product info."
train = pd.merge(train, product, on = "product")
del train["product"]

print "merge with agent info."
train = pd.merge(train, agent, on = "agent")
del train["agent"]

print "merge with client info."
train = pd.merge(train, client, on = "client")
del train["client"]


print "remove ID columns"
del train["route"]; del train["TownName"]; del train["TownID"]


print "encoder categorical columns"

categorical_cols = ["brand","State"]
for key in categorical_cols:
    print "column " + key + ", " + str(len(train[key].unique())) + " values"
    train[key] = pd.Categorical.from_array(train[key]).codes

print ""
print "write dataframe to local csv file"
print ""

train.to_csv(OUTPUT_FILE, index = False)

print "complete."
