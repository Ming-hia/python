# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

df = pd.read_csv("town_state.csv", header = 0)

fields = df["Town"].apply(lambda x: x.strip().split(" "))
df["TownID"] = fields.apply(lambda x: x[0])
df["TownName"] = fields.apply(lambda x: " ".join(x[1:]))

popular = df.groupby("State")["Agencia_ID"].count()
popular = pd.DataFrame({"State":popular.index, "Value":popular})
df = pd.merge(df,popular, on="State", right_index = True)

# save for wide-table
del df["Town"]
df.to_csv("tables/agent.csv", index = False)
