#!/usr/bin/python
# Extract volumes from nlin fit based on assigned Sparky peak list

from os import sys

fin_sparky = open(sys.argv[1],"r")
fin_sparky_lines = fin_sparky.readlines()
fin_sparky.close()

fin_tab = open(sys.argv[2],"r")
fin_tab_lines = fin_tab.readlines()
fin_tab.close()

for m in fin_sparky_lines:
    sparky_list = m.split()

    if( len(sparky_list) >= 3 ):

        for n in fin_tab_lines:
            tab_list = n.split()

            if( len(tab_list) >= 20):

                if ( (sparky_list[1] == tab_list[6]) and (sparky_list[2] == tab_list[5]) ):
                    print(sparky_list[0],tab_list[19])
