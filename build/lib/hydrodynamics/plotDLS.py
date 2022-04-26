#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys

string = sys.argv[1]
data = pd.read_csv(string,na_values='--')

#keys = {'A12','A13','A14','A15','A16','A17','A18','A19','A20'}
keys_bsa = ['A18','A19','A20']
keys_pan = ['A12','A13','A14']

figure = plt.figure(figsize=(7,7))
ax1 = figure.add_subplot(211)
ax2 = figure.add_subplot(212)
ax1.set_xlabel('Temperature $^{\circ}$C')
ax1.set_ylabel('Hydrodynamic radius nm')
ax2.set_xlabel('Temperature $^{\circ}$C')
ax2.set_ylabel('Hydrodynamic radius nm')

for kp,kb in zip(keys_pan, keys_bsa):
    index_b = data['Item'].map(lambda x: x.startswith(kb))
    index_p = data['Item'].map(lambda x: x.startswith(kp))
    ax1.plot(data[index_b]['DLS Temp (C)'], data[index_b]['Range1 Radius (N) (0.1-10nm)'],'o',label=kb)
    ax2.plot(data[index_p]['DLS Temp (C)'], data[index_p]['Range1 Radius (N) (0.1-10nm)'],'o',label=kp)

ax1.legend(title="BSA",loc='upper right')
ax2.legend(title="PAN",loc='upper right')
plt.savefig(os.path.splitext(string)[0] + '.svg')
plt.show()
