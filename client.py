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

client_info={}
f=open("./input/cliente_tabla.csv","r")
var_name=f.readline().strip()
while 1:
    line = f.readline().strip()
    if line=="":
        break
    line=line.split(",")
    client=line[0]
    name=line[1]
    client_info[client]=name
f.close()

client_info=client_info.items()

df = pd.DataFrame(client_info, columns=var_name.split(","))
df["NombreCliente"] = df["NombreCliente"].apply(space_reg)

IsDuplicated = df.duplicated()
df = df[~IsDuplicated]

fields = df["NombreCliente"].apply(lambda x: x.strip().split(" "))

df["NaN"] = df["NombreCliente"].apply(lambda x:1 if x=="NO IDENTIFICADO" else 0)
df["NickName"] = fields.apply(lambda x: 1 if len(x) == 1 else 0)
df["NonName"] = fields.apply(lambda x: 1 if find_words(x,["LA","EL","LOS","LAS"]) else 0)
df["Group"] = fields.apply(lambda x: 1 if find_words(x, ["GRUPO","EMPRESA","COMPAÑÍA","ORGANIZACIÓN"]) else 0)
df["Grocery"] = fields.apply(lambda x: 1 if find_words(x,["ABARROTES","MISCELANEA"])  else 0)
df["SuperChain"] = fields.apply(lambda x: 1 if find_words(x,["OXXO","CADENA","SUPERMERCADO","COMODIN"])  else 0)
df["Pharmacy"] = fields.apply(lambda x: 1 if find_words(x, ["FARMACIA","MEDICINA"])  else 0)
df["Education"] = fields.apply(lambda x: 1 if find_words(x, ["COLEGIO","UNIVERSIDAD"])  else 0)
df["Cafe"] = fields.apply(lambda x: 1 if find_words(x,["CAFETERIA","CAFÉ"])  else 0)
df["Restuarant"] = fields.apply(lambda x: 1 if find_words(x,["RESTAURANTE","HOTEL"])  else 0)

# save for wide-table
del df["NombreCliente"]

df.to_csv("tables/client.csv", index = False)
