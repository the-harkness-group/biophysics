#!/usr/bin/env python3

import sys
import yaml

# Read yaml file to get parameters
parameters = yaml.load(open(sys.argv[1],'r'))
out_name = sys.argv[2]

csv = open(out_name,'w')
column_headers = 'OligoName,Sequence\n'
csv.write(column_headers)
for k in parameters.keys():
    row = 'DegP_CO_{}_F'.format(k) + ',' + '{}'.format(parameters[k]['primer_F']) + '\n'
    csv.write(row)
    row = 'DegP_CO_{}_R'.format(k) + ',' + '{}'.format(parameters[k]['primer_R']) + '\n'
    csv.write(row)

