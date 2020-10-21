# -*- coding: utf-8 -*-
"""
Created on Tue May 26 20:45:27 2020

@author: Simon Zhang

Contains some general tool functions I might use.

Catalogue:
    - device, function to determine which machine I'm on and output prefix
    - get_fnames, function to obtain filenames given a path and filename patterns
    - order_list, function to arrange elements of list 1 according to contents of list 2

Changelog:
    - 26/05/2020 Creation. 
        - Moved device, get_fnames and order_list from uPL module.
        - Added gen_labels function
"""

#%% Loading
import os
import fnmatch
import sys
from pathlib import Path

#%%
"""
Function to determine from which of 3 devices I may use for work I am running this code.
Outputs a prefix to which I can attach a path starting in the OneDrive - UNSW folder.
    - Can add more devices
"""
def device():
    try: 
        os.listdir('C:/Users/z5059826/OneDrive - UNSW/')
        device = 'C:/Users/z5059826/OneDrive - UNSW/'
#        print('You are working on the uni desktop.')
    except:
        try:
            os.listdir('C:/Users/Simon/OneDrive - UNSW/')
            device = 'C:/Users/Simon/OneDrive - UNSW/'
#            print('You are working on the work laptop.')
        except:
            os.listdir('D:/OneDrive - UNSW/')
            device = 'D:/OneDrive - UNSW/'
#            print('You are working on the home desktop.')
    return device

#%%
"""
Function that takes in a directory and spits out all files in that directory that a) is of a certain type and
b) matches a certain pattern.
    - The 'path' input is a path that starts after the OneDrive - UNSW level
    - Can also just do the current directory
"""
def get_fnames(ftype, path='current', a='', b='',c=''):
    if path != 'current':
        path = device() + path
    else:
        path = Path().absolute()
    
    path_ori=Path().absolute()
    try: # the big try-except block is to ensure restoring the original working path regardless of exceptions in the body of the function. EAFP
        os.chdir(path)
        
        fnames=[]
        for file_name in os.listdir(path):
            if fnmatch.fnmatch(file_name, a+'*'+b+'*'+c+ftype):
#                print(file_name)
                fnames.append(file_name)
        os.chdir(path_ori)
    except:
        os.chdir(path_ori)
        sys.exit('Exception caught. Previous working directory restored')
    return fnames

#%%
"""
Function that reorders a list. Can be combined with get_fnames to output the correct order of filenames
"""
def order_list(mylist,myorder):
    if len(mylist) != len(myorder):
        sys.exit('The list and orders must be the same length')
    
    myholder = [None]*len(mylist)

    for i in range(len(myorder)):
        myholder[myorder[i]] = mylist[i]
        
    mylist = myholder
    return mylist

#%%
"""
Generate labels from parts of fnames
Default: gets rid of 3-letter extension names
"""
def gen_labels(fnames, start=0, end=-4):
    labels = []
    if type(fnames) != list:
        raise Exception('Please put fnames in list format')
    else:
        for f in fnames:
            labels.append(f[start:end])
    return labels