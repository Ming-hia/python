#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

TEST_FILENAME = "test.csv"

TRAIN_TABLE_FILENAME = "tables/train.csv"
TEST_TABLE_FILENAME="tables/test.csv"

def rename(dataframe):
    dataframe.rename(columns={"Producto_ID": "product"}, inplace=True)
    dataframe.rename(columns={"Cliente_ID": "client"}, inplace=True)
    dataframe.rename(columns={"Agencia_ID": "agent"}, inplace=True)
    dataframe.rename(columns={"Canal_ID": "channel"}, inplace=True)
    dataframe.rename(columns={"Ruta_SAK": "route"}, inplace=True)
    return dataframe

if __name__ == "__main__":
    test = pd.read_csv(TEST_FILENAME, header=0)
    test = rename(test)

    train = pd.read_csv(TRAIN_TABLE_FILENAME, header = 0)
    test = pd.merge(test, train, how="left", on = ["client","product","agent","channel","route"])
    test.to_csv(TEST_TABLE_FILENAME,index=False)
