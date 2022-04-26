#!/usr/bin/env python3

import sys
import pandas as pd
import yaml

params = yaml.safe_load(open(sys.argv[1],'r'))
data = pd.read_csv(params['Fitted_data'])

print(params)

# Write list of chemical shifts as ChemEx input based on peaks with good dispersions
c13_shifts = open('c13_cs.txt','w')
h1_shifts = open('h1_cs.txt','w')
for peak in params['Peaks_ChemEx']:
    c13_string = peak.split('_')[2] + 'CD1 '
    c13_cs = data[data['assignment'] == peak].center_y_ppm.iloc[0]
    c13_ass_shift = c13_string + str(c13_cs) + "\n"
    c13_shifts.write(c13_ass_shift)

    h1_string = peak.split('_')[2] + 'QD1 '
    h1_cs = data[data['assignment'] == peak].center_x_ppm.iloc[0]
    h1_ass_shift = h1_string + str(h1_cs) + "\n"
    h1_shifts.write(h1_ass_shift)

c13_shifts.close()
h1_shifts.close()
    

