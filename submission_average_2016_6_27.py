import pandas as pd
import numpy as np

File = 'train.csv'
IDX_SEMANA = 0
IDX_AGENCY = 1
IDX_CLIENT_ID = 4
IDX_PRODUCT_ID = 5
IDX_DEMAND_ID = 10

latest_demand_clpro = dict()
global_median = list()

total = 0

with open(File) as f:
    while True:
        line = f.readline().strip()
        total += 1
        if total % 5000000 == 0:
            print('Read {} lines...'.format(total))

        if line == '':
            break
        
        fields = line.split(',')
        try:
            semana = int(fields[IDX_SEMANA])
            agency = int(fields[IDX_AGENCY])
            client = int(fields[IDX_CLIENT_ID])
            product = int(fields[IDX_PRODUCT_ID])
            demand = int(fields[IDX_DEMAND_ID])
        except ValueError:
            continue

        if client != '' and product != '':
            hsh = (agency, client, product)
            if hsh in latest_demand_clpro:
                latest_demand_clpro[hsh] = ((.5 * latest_demand_clpro[hsh]) + (.5 * demand))
            else:
                latest_demand_clpro[hsh] = demand

        list.append(global_median, demand)

print('')
File = 'test.csv'
median_demanda = np.median(global_median)
IDX_SEMANA = 1
IDX_AGENCY = 2
IDX_CLIENT_ID = 5
IDX_PRODUCT_ID = 6
output = open('submission', 'w')
output.write('id' + ',' + 'Demanda_uni_equil' + '\n')

total = 0
with open(File) as f:
    while True:
        line = f.readline().strip()
        total += 1
        if total % 1000000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break

        fields = line.split(',')
        try:
            id = int(fields[0])
            semana = int(fields[IDX_SEMANA])
            agency = int(fields[IDX_AGENCY])
            client = int(fields[IDX_CLIENT_ID])
            product = int(fields[IDX_PRODUCT_ID])
        except ValueError:
            continue

        output.write(str(id) + ',')

        hsh = (agency, client, product)
        if hsh in latest_demand_clpro:
            demand = latest_demand_clpro[hsh]
            output.write(str(demand))
        else:
            output.write(str(round(median_demanda)))

        output.write("\n")

output.close()
print('')
print('Completed!')


