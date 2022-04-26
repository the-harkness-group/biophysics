#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 10:51:55 2020

@author: robertharkness
"""

import sys
import os
import shutil
import tarfile
import yaml

# Read parameters and folders to be arranged
def main():
    
    params = yaml.safe_load(open(sys.argv[1],'r'))
    NMR_Expts = params['NMR_Experiments']
    RH_Folders = params['RH_Folders']
    
    arrangeFolders(NMR_Expts, RH_Folders)

# Move NMR experiment files into RH folders, make temp tar file for scp
def arrangeFolders(NMR_Expts, RH_Folders):
    
    tar = tarfile.open("temp.tgz","w:gz")
    for experiment,folder in zip(NMR_Expts, RH_Folders):
        
        os.makedirs(folder)
        shutil.move(f"{experiment}",f"./{folder}")
        tar.add(folder)
        
    tar.close()

main()
        