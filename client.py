# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

def space_reg(name):
    name = name.upper()
    name = name.replace("  "," ")
    name = name.replace("   "," ")
    return name

def find_words(name, wordict):
    for i in range(len(name)):
        if name[i] in wordict:
            return True
    return False


df = pd.read_csv("cliente_tabla.csv", header = 0)
df["NombreCliente"] = df["NombreCliente"].apply(space_reg)

IsDuplicated = df.duplicated()
df = df[~IsDuplicated]

fields = df["NombreCliente"].apply(lambda x: x.strip().split(" "))

df["NaN"] = df["NombreCliente"] == "NO IDENTIFICADO"
df["NickName"] = fields.apply(lambda x: len(x) == 1)
df["NonName"] = fields.apply(lambda x: find_words(x,["LA","EL","LOS","LAS"]))
df["Group"] = fields.apply(lambda x: find_words(x, ["GRUPO","EMPRESA","COMPAÑÍA","ORGANIZACIÓN"]))
df["Grocery"] = fields.apply(lambda x: find_words(x,["ABARROTES","MISCELANEA"]))
df["SuperChain"] = fields.apply(lambda x: find_words(x,["OXXO","CADENA","SUPERMERCADO","COMODIN"]))
df["Pharmacy"] = fields.apply(lambda x: find_words(x, ["FARMACIA","MEDICINA"]))
df["Education"] = fields.apply(lambda x: find_words(x, ["COLEGIO","UNIVERSIDAD"]))
df["Cafe"] = fields.apply(lambda x: find_words(x,["CAFETERIA","CAFÉ"]))
df["Restuarant"] = fields.apply(lambda x: find_words(x,["RESTAURANTE","HOTEL"]))

