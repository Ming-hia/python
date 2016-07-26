#!/usr/bin/env python
# -*- coding:utf-8 -*-

TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
CSV_FILENAME = "records.csv"

START_WEEKNUM = 3
END_WEEKNUM = 9
WEEKNUM_LENGTH = END_WEEKNUM - START_WEEKNUM + 1

class Record():
    def __init__(self, agent, channel, route, price):
        self.demand_list = [0] * WEEKNUM_LENGTH
        self.agent = agent
        self.channel = channel
        self.route = route
        self.price = price

    def update(self, week, demand):
        idx = (week - START_WEEKNUM)
        self.demand_list[idx] = demand


def get_var_index():
    var_index_dict = {"train":{}, "test":{}}
    
    with open(TRAIN_FILENAME, "r") as f:
        header_train = f.readline().strip().split(",")

    for idx,var in enumerate(header_train):
        var_index_dict["train"][var] = idx
    
    with open(TEST_FILENAME,"r") as f:
        header_test = f.readline().strip().split(",")

    for idx,var in enumerate(header_test):
        var_index_dict["test"][var] = idx

    return var_index_dict  

def read_data_streaming(var_index_dict):
    f = open(TRAIN_FILENAME, "r")
    f.readline()

    RecordSet = {}
    Total = 0
    
    while 1:
        line = f.readline().strip()
        if line == "":
            break
        line = line.split(",")
        train_header_dict = var_index_dict["train"]
        week = int(line[train_header_dict["Semana"]])
        client = int(line[train_header_dict["Cliente_ID"]])
        product = int(line[train_header_dict["Producto_ID"]])
        demand = int(line[train_header_dict["Demanda_uni_equil"]])
        agent = int(line[train_header_dict["Agencia_ID"]])
        channel = int(line[train_header_dict["Canal_ID"]])
        route = int(line[train_header_dict["Ruta_SAK"]])
        unit = int(line[train_header_dict["Venta_uni_hoy"]])
        spent = float(line[train_header_dict["Venta_hoy"]])
        price = ((spent / unit) if unit >0 else 0.0)
        
        if RecordSet.has_key((client, product)):
            RecordSet[(client, product)].update(week, demand)
        else:
            RecordSet[(client, product)] = Record(agent, channel, route, price)
            RecordSet[(client, product)].update(week, demand)

        Total += 1
        if (Total % 1000000 == 0):
            print "complete " + str(Total) + " rows"

    f.close()
    return RecordSet


def write_to_csv(RecordDict):
    f = open(CSV_FILENAME, "w")
    
    header_list = ["client","product","agent","channel","route","price"]
    for i in range(WEEKNUM_LENGTH):
        header_list.append("demand_week_" + str(i+1))

    f.write((",").join(header_list))
    f.write("\n")

    for (client,product),record in RecordDict.iteritems():
        f.write("%d,%d,%d,%d,%d,%.2f" % (client, product, record.agent, record.channel, record.route, record.price))
        for i in range(WEEKNUM_LENGTH):
            f.write(",%s" % str(record.demand_list[i]))
        f.write("\n")
    
    f.close()


if __name__ == '__main__':
    var_index_dict = get_var_index()
    RecordSet = read_data_streaming(var_index_dict)
    write_to_csv(RecordSet)
