# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:43:21 2020

@author: z5059826

Version: 2020 04 19

uPL Toolbox

Use this module to  store all functions that will be used across multiple analyses
of uPL measurements conducted in TETB 147.
Updated as needed.

Catalogue:
    - load_correct_0d, function to load and correct spectral PL data given filenames
    - mplot, function to plot multiple PL spectra given filename patterns. Can also function as a pure multiple data importer
    
    - imap, imports data from given uPL maps
    - divPL, divides PL data into PLint_BB and PLint_defect
    - rcPL, divides PL data into PL(total), PLint_BB and PLint_defect for rows and columns, such as fingers, busbars and normal areas
    - cmapPL, draws a PL colourmap
    - plt_idx, plots PL spectra of points on a colourmap given the index
    - pt_idx, marks points on the colourmaps (commonly drawn by cmpPL) according to idx
    
    - lookup, uRob's function to look something up in an np array
    
    - temperature fitting functions:PL_BB_func, fit_temp1, bfit_temp, fit_temp2
        * bfit_temp uses old code of fit_temp1 so I have rewritten fit_temp2 to use as a single file temperature fitting
    
    - load_corr_ANU, ANU version of load_correct_0d, also imports given path
    - mplot_ANU, ANU version of mplot. Can also function as a pure multiple data importer
    
    * (19/04/20) mplot and get_fnames are dependent on the device function. Which means this module as is will only work for the 3 machines that I use.
    * (26/05/20) Moved device, get_fnames, and order_list to genSZ module
    * (xx/06/20) Added temp fitting functions
    * (19/08/20) Added ANU_load_corr
"""
#%% Loading
import numpy as np
import pandas as pd
import conSZ
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.optimize import curve_fit
from scipy.integrate import trapz
from pathlib import Path

import genSZ



#%% 
"""
Function to load spectral data collected at a single point, ie not mapping
Outputs everything needed for spectral plot (ie PL, wvl, energy) for a single file
Outputs other useful info: PL0 spectrum, integration time, number of averages
    - times: PL=PL-times*PL0
    
    - Cutoff: can cut from either direction. Cannot yet cut from both directions.
    - Si QE: the Si QE calibration uses the calibration from the Avantes lamp, with the bottom turret of the microscope on setting 6, which I suspect is empty.
"""
def load_correct_0d(filename, detector, times, xvar='eV', delimiter=',', cutoff=0):
    
        #load data. unpack transposes the ndarray output
        PL0, PL = np.loadtxt(filename, skiprows=80, usecols=(4,6), delimiter=delimiter, unpack=True)
        info = pd.read_csv(filename, delimiter=delimiter, usecols=(0,1), index_col='Name', 
                          header=0, names=['Name','Value'])
        t_int  = np.float(info.loc['intigration times(ms)'])
        n_avg  = np.float(info.loc['average number'])
       # Spectrum corrections setup
        path = genSZ.device()+'Modules/' # make sure that it's using the QE files in the Modules folder
        if detector == "InGaAs":
            wvl, QE = np.loadtxt(path+'QE_InGaAs.csv', unpack=True,
                             delimiter = ',', skiprows=1, usecols = (0, 1))
        elif detector == "Si":
#            wvl, QE = np.loadtxt(path+'QE_Si.csv', unpack=True,
#                             delimiter = ',', skiprows=1, usecols = (0, 1))
            wvl, QE = np.loadtxt(path+'QE_Si_Avantes.csv', unpack=True, # the calibration using the lamp
                             delimiter = ',', skiprows=1, usecols = (0, 3))
            PL = PL[138:2041]
            PL0 = PL0[138:2041]
        else:
            raise NameError('Detector must be string named InGaAs or Si')

        dw = np.gradient(wvl) # for elements in a 1D array it will output half the difference between the elements proceeding and following that element
        dw = dw/dw.max()
        eV = conSZ.nm2eV/wvl

        if xvar == 'eV':
            PL = (PL-times*PL0)/(QE*eV**2*dw)
        elif xvar == 'wvl':
            PL = (PL-times*PL0)/(QE*dw)
        else:
            raise Exception('The x-axis must be in wvl or eV')
            
        # Cut-off data where needed
        if cutoff < 0:
            PL = PL[:cutoff]
            PL0 = PL0[:cutoff]
            eV = eV[:cutoff]
            wvl = wvl[:cutoff]
        elif cutoff >= 0:
            PL = PL[cutoff:]
            PL0 = PL0[cutoff:]
            eV = eV[cutoff:]
            wvl = wvl[cutoff:]            
        return PL, wvl, eV, PL0, t_int, n_avg

#%% 
"""
Function to takes in multiple sets of data and plot multiple spectra.
Currently only used for uPL spectra, but with some adaptation should be able to be used for x-y graph.
    - Plots data for fnames under current path, or under specified path
        * can obtain the file names using the get_fnames function
        * path specified should start after the OneDrive level
    - Makes extensive use of the load_correct_0d function
    - Can switch between linear and semilogy graphs
    - There is an automatic pool of colors that this function uses. colors can also be input from the outside
    - Should be able to be integrated seemlessly to produce things like subplots
    - cutoff: can cut from either direction. Cannot yet cut from both directions.
    - ratio: to use in making sure of the same integration time and multipliers across different measurements
    - plot: setting this varible to 'n' will suppress the plotting function of this function
    
    - Must use the same type of delimiter (default comma)
    - Must use the same type of detector (default Si)
    - 13/08/20: (not tested) added offset function for both logy and linear graphs
"""       
def mplot(filenames, times, path='', labels='', detector='Si', plot=True, xvar='eV', norm=False, logy=True, colors='', delimiter=',',cutoff=0, ratio=0, offset=False):
    # prepare the ground
    if path != 'current':
        path = genSZ.device() + path
    elif path == 'current':
        path = Path().absolute()
    path_ori = Path().absolute()
    os.chdir(path)
    
    if detector=='Si': # which detector
#        length=2048-np.abs(cutoff) # this is for the default Si calibration with blackbody at 400C, which uRob rarely uses and is therefore not optimal
        length = 1903 - np.abs(cutoff) # this is measured with a lamp, the bottom turrent position #6 which I think is probably empty
    elif detector=="InGaAs":
        length=512-np.abs(cutoff)
    else:
        raise NameError('The detector must be Si or InGaAs')
       
    num_files = len(filenames) #the number of files this function will need to process

    if labels == '': # make labels the filenames, if no labels
        labels=[]
        for f in filenames:
            labels.append(os.path.splitext(f)[0])
            
    if colors == '' or colors==1: # the colours feature
        colorRev = False
        colors = [None]*num_files
    elif colors == -1:
        colorRev = True
        colors = [None]*num_files
    elif type(colors)==list:
        colorRev = False
    else:
        raise TypeError('The colors have to be in list format. The number of colors specificed have to equal the number of PL spectra')
#    try:
    if colorRev==True:
        mpl.rcParams['axes.prop_cycle'] = plt.cycler(color = plt.cm.rainbow(np.linspace(1,0.1,num_files)))
    else:
        mpl.rcParams['axes.prop_cycle'] = plt.cycler(color = plt.cm.rainbow(np.linspace(0.1,1,num_files)))
#    except:
#        pass

    if type(times)==float or type(times)==int: # allowing customisation of the times variable
        mul = np.empty((num_files,length))
        for i in range(len(mul)):
            mul[i]=times
    elif len(times)!=num_files:
        raise Exception('The times variable must be a float, or an array with the same length as the filenames array')
         
    #load the data
    PL = np.empty((num_files,length))
    wvl = np.empty((num_files,length))
    eV = np.empty((num_files,length))
    PL0 = np.empty((num_files,length))
    t_int = np.empty((num_files,1))
    n_avg = np.empty((num_files,1))
     
    for f in filenames:
        i = filenames.index(f)
        PL[i], wvl[i], eV[i], PL0[i], t_int[i], n_avg[i] = load_correct_0d(filename=f, times=times, detector=detector, xvar=xvar, delimiter=delimiter, cutoff=cutoff)
    
    # return to original directory
    os.chdir(path_ori)
    
    # account for possible different integration times or multipliers
    if ratio==0:
        pass
    elif type(ratio)==int or type(ratio)==float:
        for i in range(num_files):
            PL[i]=PL[i]/ratio
    elif len(ratio)==num_files:
        for i in range(num_files):
            PL[i]=PL[i]/ratio[i]
    else:
        raise Exception('The ratio variable must be zero, or have length 1, or an array with the same length as the filenames array')
    
    # Do we plot?
    if plot==False:
           return PL, wvl, eV, PL0, t_int, n_avg
    elif plot != True:
           raise NameError('The variable plot must be a boolean')
    # prep the data for plotting
    PL_plot = np.empty((num_files,length))
    xax = np.empty((num_files,length))
    
    if norm == True: #possible normalisation
        for i in range(num_files):
            PL_plot[i] = PL[i]/PL[i].max()
    elif norm == False:
        PL_plot = PL
    else:
        raise NameError('You must specify whether the graph should be normalised')
            
    if xvar == 'wvl': #x-axis wavelength or energy?
        xax = wvl
        xlabel = r'$ \rm Wavelength~[nm] $'
    elif xvar == 'eV':
        xax = eV
        xlabel = r'$ \rm Emission~Energy~[eV] $'
    else:
        raise Exception('The x-axis must be in wvl or eV')
   
    # plot, both the semilogy and the linear plot
    if logy == True:
        eps=0.3
        PL_plot=np.flip(PL_plot,axis=0)
        PL_plot=np.where(PL_plot>eps,PL_plot,eps)
        if type(offset) == int or type(offset) == float:
            for i in range(num_files):
                PL_plot[i]=PL_plot[i]*offset**i
        for i in range(num_files):
            plt.semilogy(xax[i],PL_plot[i],label=labels[i],color=colors[i],lw=1)
    elif logy == False:
        if type(offset) == int or type(offset) == float:
            for i in range(num_files):
                PL_plot[i]=PL_plot[i]+offset*i        
        for i in range(num_files):
            plt.plot(xax[i],PL_plot[i],label=labels[i],color=colors[i],lw=1) 
    else:
        raise Exception('The plot must be linear or semilogy, the variable must be a Boolean')
    
    plt.xlabel(xlabel)
    plt.legend()
    plt.ylabel(r'$ \rm PL~Intensity~[AU] $')
    plt.tight_layout()
        
    return PL, wvl, eV, PL0, t_int, n_avg    
#%%
"""
Function to import data from uPL maps. The map must be pre-processed in the standard way.
    - I'm choosing to make Nx Ny dx dy paramters because they're useful anyway as inputs, and knowing these parameters are very helpful in analysis
    - The data from here is already cut. Logic is consistent with other functions here: negative means cutting from the end (low eV, high wvl), positive vice versa
    - 26/04/20: it seems that the output PL matrix at the end (and the colourmap thereof) has the same orientation as the actual sample
"""    
def imap(filename, path, times, Nx, Ny, dx, dy, dark='PL0.csv', 
         detector='InGaAs', cutoff=0):
        
    # go to required directory
    if path != 'current':
        path = genSZ.device() + path
    else:
        path = Path().absolute()

    path_ori = Path().absolute()
    os.chdir(path)
    
    # set up map
    x = np.linspace(0, Nx*dx, Nx)
    y = np.linspace(0, Ny*dy, Ny)
    extent = (x.min(), x.max(), y.min(), y.max())
    
    # Import map and PL0
    PL0, noise = np.loadtxt(dark, skiprows=80, usecols=(6, 7), delimiter=',',
                        unpack=True)
    PL = np.loadtxt(filename)
    PL = PL[:, 2:]
    
    # Spectrum corrections setup: QE, wvl to eV, averaging wvl
    path = genSZ.device()+'Modules/' # make sure that it's using the QE files in the Modules folder
    if detector == "InGaAs":
        wvl, QE = np.loadtxt(path+'QE_InGaAs.csv', unpack=True,
                         delimiter = ',', skiprows=1, usecols = (0, 1))
    elif detector == "Si":
#            wvl, QE = np.loadtxt(path+'QE_Si.csv', unpack=True,
#                             delimiter = ',', skiprows=1, usecols = (0, 1))
        wvl, QE = np.loadtxt(path+'QE_Si_Avantes.csv', unpack=True, # the calibration using the lamp
                         delimiter = ',', skiprows=1, usecols = (0, 3))
        PL = PL[:,138:2041]
        PL0 = PL0[138:2041]
        noise = noise[138:2041] # end of Avantes calibration
    else:
        raise NameError('Detector must be string named InGaAs or Si')
    
    Nw = len(wvl)
    dw = np.gradient(wvl) # for elements in a 1D array it will output half the difference between the elements proceeding and following that element
    dw = dw/dw.max()
    eV = conSZ.nm2eV/wvl

    # spectral corrections
    PL_eV = (PL-times*PL0)/(QE*eV**2*dw)
    PL_wvl = (PL-times*PL0)/(QE*dw)
        
    # reshape the data: 2D -> 3D
    PL_eV = np.reshape(PL_eV, (Nx, Ny, Nw))
    PL_eV[1::2, :, :] = PL_eV[1::2, ::-1, :]
    PL_eV = np.flipud(PL_eV)
    PL_eV = np.rot90(PL_eV, 3)
    PL_wvl = np.reshape(PL_wvl, (Nx, Ny, Nw))
    PL_wvl[1::2, :, :] = PL_wvl[1::2, ::-1, :]
    PL_wvl = np.flipud(PL_wvl)
    PL_wvl = np.rot90(PL_wvl, 3)
    
    # Cut-off data where needed
    if cutoff<0:
        PL_eV = PL_eV[:,:,:cutoff]
        PL_wvl = PL_wvl[:,:,:cutoff]
        PL0 = PL0[:cutoff]
        eV = eV[:cutoff]
        wvl = wvl[:cutoff]
    elif cutoff >= 0:
        PL_eV = PL_eV[:,:,cutoff:]
        PL_wvl = PL_wvl[:,:,cutoff:]
        PL0 = PL0[cutoff:]
        eV = eV[cutoff:]
        wvl = wvl[cutoff:]
    # go back to previous directory
    os.chdir(path_ori)
    return PL_wvl, PL_eV, wvl, eV, PL0, noise, extent
    
#%%
"""
Function to divide PL data into band-to-band and defect PLs.
    - outputs 2 things: PLint_BB, PLint_defect
    - can divide on eV or wvl basis
    - should take in the PL that already has a the meaningless part cut off

    * uRob's code has 'Defect' capitalised, I output PL_defect not capitalised
"""    
def divPL(PL, div, xval, xvar='eV'):
    idx_cut = lookup(xval, div)
    if xvar == 'eV':
        PLint_BB = -trapz(PL[:, :, :idx_cut], xval[:idx_cut], axis=2)
        PLint_defect = -trapz(PL[:, :, idx_cut:], xval[idx_cut:], axis=2)
    elif xvar == 'wvl':
        PLint_BB = -trapz(PL[:, :, :idx_cut], xval[:idx_cut], axis=2)
        PLint_defect = -trapz(PL[:, :, idx_cut:], xval[idx_cut:], axis=2)
    else:
        raise NameError('Xvar must be eV or wvl, and xval must be the corresponding vector')

    PLint_BB = np.abs(PLint_BB)
    PLint_defect = np.abs(PLint_defect)

    return PLint_BB, PLint_defect

#%%
"""
Function to create horizontal and vertical strips of PL data
    - main function: to create fingers, busbar, and normal PLints
    - takes the FULL PL data. Also calls the divPL function.
"""
def rcPL(PL, div, xval, priority, cols=False, rows=False, xvar='eV'):
    
    PLint_BB, PLint_defect = divPL(PL=PL, div=div, xvar=xvar, xval=xval)
    PLint_BB1 = PLint_BB/np.max(PLint_BB)
    PLint_defect1 = PLint_defect/np.max(PLint_defect)
    
    # create masks
    mask_rows = np.ones(np.shape(PL)[0],dtype=bool)
    mask_cols = np.ones(np.shape(PL)[1],dtype=bool)
    
    try:
        cols = np.array(cols)
        rows = np.array(rows)
    except:
        raise Exception('Cannot convert cols and/or rows into numpy arrays')
    
    if np.shape(cols)!= (1,1):
        mask_cols[cols] = False
    elif cols != False:
        mask_cols[cols] = False
        
    if np.shape(rows)!= (1,1):
        mask_rows[rows] = False
    elif rows != False:
        mask_rows[rows] = False
        
    # apply masks to PL
    PL_norm = PL[mask_rows,:,:][:,mask_cols,:]
    
    if priority == 'cols':
        PL_cols = PL[:,~mask_cols,:]
        PL_rows = PL[:,mask_cols,:][~mask_rows,:,:]
    elif priority == 'rows':
        PL_rows = PL[~mask_rows,:,:]
        PL_cols = PL[mask_rows,:,:][:,~mask_cols,:]
    else:
        raise NameError('The priority variable must be string cols or rows. If only one is rows or cols present, then either one will do')

    # apply masks to PLint
    # for conciseness, PLint_BBnn is integrated normalised BB PL for normal areas
    PLint_BBn = PLint_BB[mask_rows,:][:,mask_cols]
    PLint_defectn = PLint_defect[mask_rows,:][:,mask_cols]
    
    if priority == 'cols':
        PLint_BBc = PLint_BB[:,~mask_cols]
        PLint_defectc = PLint_defect[:,~mask_cols]
        PLint_BBr = PLint_BB[:,mask_cols][~mask_rows,:]
        PLint_defectr = PLint_defect[:,mask_cols][~mask_rows,:]
    elif priority == 'rows':
        PLint_BBr = PLint_BB[~mask_rows,:]
        PLint_defectr = PLint_defect[~mask_rows,:]
        PLint_BBc = PLint_BB[mask_rows,:][:,~mask_cols]
        PLint_defectc = PLint_defect[mask_rows,:][:,~mask_cols]
    else:
        raise NameError('The priority variable must be string cols or rows')

    # normalise integrated BB/defect PL for each area
    PLint_BBnn = PLint_BBn/np.max(PLint_BBn)
    PLint_defectnn = PLint_defectn/np.max(PLint_defectn)
    try:
        PLint_BB1c = PLint_BBc/np.max(PLint_BBc)
        PLint_defect1c = PLint_defectc/np.max(PLint_defectc)
    except:
        PLint_BB1c = None
        PLint_defect1c = None
    try:
        PLint_BB1r = PLint_BBr/np.max(PLint_BBr)
        PLint_defect1r = PLint_defectr/np.max(PLint_defectr)
    except:
        PLint_BB1r = None
        PLint_defect1r = None
        
    return {'PL_norm':PL_norm, 'PL_rows':PL_rows, 'PL_cols':PL_cols, 'PLint_BB':PLint_BB, 'PLint_defect':PLint_defect, 'PLint_BB1':PLint_BB1, 'PLint_defect1':PLint_defect1,'PLint_BBn':PLint_BBn, 'PLint_defectn':PLint_defectn, 'PLint_BBc':PLint_BBc, 'PLint_defectc':PLint_defectc, 'PLint_BBr':PLint_BBr, 'PLint_defectr':PLint_defectr, 'PLint_BBnn':PLint_BBnn, 'PLint_defectnn':PLint_defectnn, 'PLint_BB1c':PLint_BB1c, 'PLint_defect1c':PLint_defect1c, 'PLint_BB1r':PLint_BB1r, 'PLint_defect1r':PLint_defect1r}

#%% 
"""
Function to draw a PL colourmap
"""

def cmapPL(ax, PLint, extent, section, aspect=10, LogNorm = False):
    if section == 'BB':
        cmap = plt.cm.inferno
    elif section == 'defect':
        cmap = plt.cm.viridis
    else:
        raise NameError('Section variable must be string of value BB or defect')
    
    # plot, with choice of log colours
    if LogNorm == False:
        im1 = ax.imshow(PLint, extent=extent, cmap = cmap)
        c1 = plt.colorbar(im1, aspect=aspect, ax=ax)
    elif LogNorm == True:
        im1 = ax.imshow(PLint, extent=extent, norm = mpl.colors.LogNorm(),cmap = cmap)
        c1 = plt.colorbar(im1, norm = mpl.colors.LogNorm(), aspect=aspect, ax=ax)
    else:
        raise NameError('LogNorm variable must be string of value n or y')
    # cosmetics
    ax.set_xlabel(r'x [mm]')
    ax.set_ylabel(r'y [mm]')
    
    if section == 'BB':
        c1.set_label(r'BB PL [AU]')
    elif section == 'defect':
        c1.set_label(r'Defect PL [AU]')
        
    return im1, c1
#%%
"""
Function to plot PL spectra of points on a colourmap given the index
    - Can choose between plotting all ponts indicated by the idx parameter; or an average of the points indicated by the idx paraemter
        * if the latter, then have to assign colour and label manually
"""
def plt_idx(ax, PL, xval, xvar, idx, x, y, logy='y', norm=False, avg='n', label=None, lw=2, capsize=3, errorevery=1, color=None):
    
    idx = np.array(idx)
    shape = np.shape(idx)
    Nidx = shape[1]
    
    # draw the average, or all lines?
    if avg == 'n':
        for k in range(Nidx):
            idx_x = idx[0, k]
            idx_y = idx[1, k]
            spectra = PL[idx_x, idx_y, :]
            if norm==False:
                m=1
            elif norm ==True:
                m= np.max(spectra)
            if logy == 'y':
                ax.semilogy(xval, spectra/m, lw=lw, color=color, label=(idx_y,idx_x))
            elif logy == 'n':
                ax.plot(xval, spectra/m, lw=lw, color=color, label=(idx_y,idx_x))       
    
    elif avg == 'y':
        spectra = np.empty_like(PL[1,1,:])
        
        for k in range(Nidx):
            idx_x = idx[0, k]
            idx_y = idx[1, k]
            spectra = np.vstack((spectra,PL[idx_x, idx_y, :]))
        spec_plot = np.average(spectra[1:,:], axis=0)
        std = np.std(spectra[1:,:],axis=0)

        if norm==False:
            m=1
        elif norm ==True:
            m= np.max(spec_plot)
            
        ax.errorbar(x=xval, y=spec_plot/m, yerr=std/m, lw=lw, label=label, color=color, ecolor=color, capsize=capsize, capthick=1, errorevery=errorevery)            
#        ax.semilogy(xval, spec_plot/m, lw=lw, label=label, color=color)
        if logy == 'y':
            ax.set_yscale('log')

    
    ax.set_ylabel(r'PL Intensity [AU]')
    ax.legend()
    
    if xvar == 'eV':
        ax.set_xlabel(r'Emission Energy [eV]')
    elif xvar == 'wvl':
        ax.set_xlabel(r'Wavelength [eV]')
    else:
        raise NameError('The xvar variable must have string value eV or wvl')
    
    plt.tight_layout()
    return

#%%
"""
Function to plot points onto the PL colourmap given the indices
"""
def pt_idx(ax, idx, x, y, color, ms=8, marker='D', mew=1):
    for i in range(len(idx)): # assign a colour for each part of idx
        idx_i=idx[i]
        color_i = color[i]
        
        idx_i = np.array(idx_i)
        shape = np.shape(idx_i)
        Nidx_i = shape[1]
        
        for k in range(Nidx_i):    
    #        color = plt.cm.Dark2(1.*k/Nidx_i)
            idx_i_x = idx_i[0, k]
            idx_i_y = idx_i[1, k]        
            # corresponding locations in mappings
            x_loc = y[idx_i_y]
            y_loc = x.max()-x[idx_i_x]        
            # mark locations in mappings
            ax.plot(x_loc, y_loc, marker='D', color=color_i, mew=1, ms=ms, mec='k')
    return

#%%
"""
Set of functions to fit for Si sample temperature based on the shape near the top of BB PL
- Only usable for low temp 
- Only for the bit next to the top of BB PL
- Xi is initial guess for parameter X.
    * T is temperature
    * PL0 is max intensity for a certain peak. For temp fitting this is the highest peak
    * E0 is the energy at which PL for this peak drops to zero
"""
def PL_BB_func(eV, *params):
    T = params[0]
    PL0 = params[1]
    E0 = params[2]
    
    Eth = const.k/const.e*T
    Edash = eV - E0
    PL = PL0*Edash**2*np.exp(-Edash/Eth)
    idx = np.where(eV < E0)
    PL[idx] = 0.0
    
    return PL

def fit_temp1(eV, PL, idx, Ti, PL0i, E0i):
       p0 = (Ti, PL0i, E0i)
       popt, pcov = curve_fit(PL_BB_func, eV[idx], PL[idx], p0)
       perr = np.sqrt(np.diag(pcov))
       
       PL_fit = PL_BB_func(eV, *popt)
       
       T, PL0, E0 = popt
       T_err, PL0_err, E0_err = perr
       
       return T, PL0, E0, T_err, PL_fit

def bfit_temp(PL, eV, eV_range, Ti):
    
    shapey = np.shape(PL)
    Ntemp = shapey[0]
    T = np.zeros(Ntemp)
    T_err = np.zeros(Ntemp)
    
    plt.figure()
    for k in range(Ntemp):
           eVk = eV[k, :]
           PLk = PL[k, :]
           
           idx = np.where(np.logical_and(eVk > eV_range[0], eVk < eV_range[1]))
           PL0i = PLk[idx].max()
           E0i = eVk[idx].min()
           Ti = Ti
           
           T[k], PL0, E0, T_err[k], PLk_fit = fit_temp1(eVk, PLk, idx, Ti, PL0i, E0i)
           
           plt.semilogy(eVk, PLk)
           plt.plot(eVk[idx], PLk_fit[idx], color='k')
           
#           print(T[k])
#           print(T_err[k])
           
    return T, T_err

def fit_temp2(PL, eV, eV_range, Ti):
    
    idx = np.where(np.logical_and(eV > eV_range[0], eV < eV_range[1]))

    PL0i = PL[idx].max()
    E0i = eV[idx].min()
    p0 = (Ti, PL0i, E0i)
    
    popt, pcov = curve_fit(PL_BB_func, eV[idx], PL[idx], p0)
    perr = np.sqrt(np.diag(pcov))
    
    PL_fit = PL_BB_func(eV, *popt)
    
    T, PL0, E0 = popt
    T_err, PL0_err, E0_err = perr
    
    param= PL0, PL0_err, E0, E0_err
    plt.semilogy(eV, PL)
    plt.plot(eV[idx], PL_fit[idx], color='k')
    
    return T, T_err, param

#%%
"""
Function to load uPL data measured at ANU, and correct for QE Of the detectors. 
- It is likely that the corrections has already in been applied in the measurement process
- Outputs both PL vs eV and PL_wvl vs wvl
- Also can shift up the data, something occassionaly needed for the ANU measurements for a better correction
- Has cutoff function
- Serves a similar role to the load_correct_0d function: although this one also goes to the file path to load.
- Also converts to eV if needed
- Outputs both original and normalised spectra
- Major improvement over original load_corr_0d (as of 21/08/20): can judge which detector was used, based on if it includes measurements taken at wvl > 1200 nm
"""

def load_corr_ANU(fname, path='', corr=False, shiftup=0, cutoffs=(0,-1)):
    # Go to the target directory
    if path != 'current':
        path = genSZ.device() + path
    else:
        path = Path().absolute()

    path_ori = Path().absolute()
    os.chdir(path)
    
    # load the correction coefficients 
#    np.seterr(invalid='ignore') # invalid (probably NaN) values exist in the below operation. use this to override the warning. Corresponding expression below sets it back
    corr_InGaAs = np.loadtxt(genSZ.device()+'Modules/Correction base files/ANU QE/no laser corrected.txt')[:,1]/np.loadtxt(genSZ.device()+'Modules/Correction base files/ANU QE/no laser.txt')[:,1]
    corr_Si = np.loadtxt(genSZ.device()+'Modules/Correction base files/ANU QE/Si corrected.txt')[:,1]/np.loadtxt(genSZ.device()+'Modules/Correction base files/ANU QE/Si uncorrected.txt')[:,1]
#    np.seterr(invalid='warn')
    
    # load and return to the original path
    wvl, PL = np.loadtxt(fname, unpack=True)
    os.chdir(path_ori)
    
    if np.any(wvl)>1200:
        detector = 'InGaAs'
    else:
        detector = 'Si'
    
    PL = PL+shiftup
    if corr == True:
        if detector == 'InGaAs':
            PL=PL*corr_InGaAs
        elif detector == 'Si':
            PL=PL*corr_Si
    elif corr != False:
        raise NameError('The correction variable must be True or False')

    PL_wvl = PL
    
    dw=np.gradient(wvl)
    dw=dw/dw.max()
    eV=conSZ.nm2eV/wvl
    
    PL = PL_wvl/(eV**2*dw)
     
    PL = PL[cutoffs[0]:cutoffs[1]]
    PL_wvl = PL_wvl[cutoffs[0]:cutoffs[1]]
    eV = eV[cutoffs[0]:cutoffs[1]]
    wvl = wvl[cutoffs[0]:cutoffs[1]]        
    
    PLn = PL/PL.max()
    PLn_wvl = PL_wvl/PL_wvl.max()

    return (eV, PL, PLn, wvl, PL_wvl, PLn_wvl)
#%%
"""
mplot function with ANU characteristics
- Major improvement over original load_corr_0d (as of 21/08/20): can judge which detector was used, based on if it includes measurements taken at wvl > 1200 nm
- Has eps feature, which is used to draw values below eps at eps, cleaning up the plot
- (21/08/20) some features in the original mplot not implemented yet
"""

def mplot_ANU(ax, fnames, path='', labels='', colors='', corr=False, shiftup=0, plot=True, xvar='eV', norm=False, logy=True, eps=False,cutoffs=(0,-1)):
    # prepare the ground
    if path != 'current':
        path_full = genSZ.device() + path
    elif path == 'current':
        path = Path().absolute()
    path_ori = Path().absolute()
    os.chdir(path_full)

    num_files = len(fnames)
    
    if labels == '': # make labels the filenames, if no labels
        labels=[]
        for f in fnames:
            labels.append(os.path.splitext(f)[0]) 
            
    if colors == '' or colors==1: # the colours feature
        colorRev = False
        colors = [None]*num_files # so that the plotting section doesn't throw an error
    elif colors == -1:
        colorRev = True
        colors = [None]*num_files
    elif type(colors) == list:
        colorRev = False
    else:
        raise TypeError('The colors have to be in list format. The number of colors specificed have to equal the number of PL spectra')
                        
    # if colors are specified, they will override this                   
    if colorRev == True:
        mpl.rcParams['axes.prop_cycle'] = plt.cycler(color = plt.cm.rainbow(np.linspace(1,0.1,num_files)))
    else:
        mpl.rcParams['axes.prop_cycle'] = plt.cycler(color = plt.cm.rainbow(np.linspace(0.1,1,num_files)))

    # Adjust for correction

    if corr == True or corr == False:
        corr = [corr]*num_files
    elif type(corr) != list or len(corr) != num_files:
        raise NameError('Please ensure variable corr is of type bool or list, and has the same number of elements as files to be processed. Lists use sq brackets []')
    
    # Adjust for shiftup
    if type(shiftup) == int or type(shiftup) == float:
        shiftup = np.full(num_files,shiftup)
    elif shiftup != list or len(shiftup) != num_files:
        raise NameError('Please ensure variable corr is of type int, float, or list, and has the same number of elements as files to be processed. Lists use sq brackets []')
        
    # load data
#    data = np.empty((0,0))
#    for f in fnames:
#        i = fnames.index(f)
#        data = np.concatenate((data,load_corr_ANU(f,path=path,corr=corr[i],shiftup=shiftup[i])))
#        print(data)
    data = []
    for f in fnames:
        i = fnames.index(f)
        data.append(load_corr_ANU(f,path=path,corr=corr[i],shiftup=shiftup[i],cutoffs=cutoffs))
    data= list(zip(*data)) # transpose the list of lists, and make it a list of lists instead of a list of tuples so I can call elements

    eV = data[0]
    PL = data[1]
    PLn = data[2]
    wvl = data[3]
    PL_wvl = data[4]
    PLn_wvl = data[5]

    # return to original directory
    os.chdir(path_ori)
    
    #x-axis wvl or eV?
    if xvar == 'wvl':
        xax = wvl
        xlabel = r'$ \rm Wavelength~[nm] $'
        PL_plot = PL_wvl
        if norm == True:
            PL_plot = PLn_wvl
    elif xvar == 'eV':
        xax = eV
        xlabel = r'$ \rm Emission~Energy~[eV] $'
        PL_plot = PL
        if norm == True:
            PL_plot = PLn
    else:
        raise Exception('The x-axis must be in wvl or eV')
    # Do we plot?
    if plot==False:
           return (eV, PL, PLn, wvl, PL_wvl, PLn_wvl)
    elif plot != True:
           raise NameError('The variable plot must be a boolean')

    # Why, I shall plot, of course
    if logy == False:
        for i in range(num_files):
            ax.plot(xax[i],PL_plot[i],label=labels[i],color=colors[i],lw=1)
    elif logy == True:
        for i in range(num_files):
#            print(np.shape(PL_plot[1]))
            if type(eps)==float or type(eps)==int: # apply epsilon cutting
                y = np.where(PL_plot[i]>eps,PL_plot[i],eps) # account for possible length differences between lines in PL_plot
                ax.semilogy(xax[i],y,label=labels[i],color=colors[i],lw=1)
            else:
                ax.semilogy(xax[i],PL_plot[i],label=labels[i],color=colors[i],lw=1)
    else:
        raise Exception('The plot must be linear or semilogy')
    ax.set_xlabel(xlabel)
    ax.legend()
    ax.set_ylabel(r'$ \rm PL~Intensity~[AU] $')
    plt.tight_layout()
   
    return (eV, PL, PLn, wvl, PL_wvl, PLn_wvl)
#%% 
"""
uRob's function to look something up, either to find the index of an element in an array,
or to find the element witht the same index in another array (if X~=[])
for instance: idx_cut = lookup(eV, 1.03) finds where 1.03 eV is in the array eV, or lookup((1,3,5),3,(2,4,6)) outputs 4.
"""

def lookup(Y, yval, X=[]):
    Y = np.array(Y)
    index = np.argmin(abs(Y - yval))

    if X == []:
        return index
    else:
        return X[index]