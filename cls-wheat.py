#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:19:58 2021

@author: pikl.m
"""

import os, glob, argparse
import numpy as np
from sann_all_n_rex_5_nm_SR_895_6 import * 
from sann_all_n_rex_5_nm_SR_895_6 import main as tellClass
from spectral import envi
from multiprocessing import Pool


#%%
def processImg(inFile):
    '''
    Classify image to green/dry leaves based on the sann_all_n_rex_5_nm_SR_895_6 model.
    
    Parameters
    ----------
    
    inFile : string
            ENVI hdr of image prepared for classification <-- "_avr_n895.hdr"
            
    Returns : ENVI classification
            Creates ENVI hdr image with extension "_cls.hdr"
            
    
    '''
    
    
    img = envi.open(inFile)
    ihdr = img.metadata
    ishape = img.shape
    
    data = np.reshape(img[:,:,:], (ishape[0]*ishape[1], ishape[2]))
    
    odata = np.zeros(data.shape[0])
    for i, v in enumerate(data):
        odata[i] = tellClass(v)
    
    
    
    # create output image
    ohdr = {}
    ohdr['description'] = 'Karel NN clasification'
    ohdr['lines'] = ishape[0]
    ohdr['samples'] = ishape[1]
    ohdr['bands'] = 1
    ohdr['data type'] = 1
    ohdr['interleave'] = 'bsq'
    ohdr['header offset'] = 0
    ohdr['byte order'] = 0
    ohdr['file type'] = 'ENVI Classificatioin'
    
    ohdr['classes'] = 6
    ohdr['band names'] = 'karelNN classification'
    ohdr['class names'] = ['unclassified',
                            'blue',
                            'dry',
                            'green',
                            'soil',
                            'tech']
    ohdr['class lookup'] = [0,0,0,0,102,255,255,255,0,51,204,51,204,153,0,179,179,179]
    
    oname = os.path.splitext(inFile)[0] + '_cls.hdr'
    
    oimg = envi.create_image(oname, ohdr, ext='.dat', force=True)
    mm = oimg.open_memmap(writable=True)
    
    mm[:,:,0] = np.reshape(odata, img.shape[:2])
    
    del(oimg)
    del(mm)
    del(img)
    
    
       

#%%
def p_processImg(inVar, n=-1):
    '''
    Paralel classification of PSI images to green/dry leaves.
    
    inVar : string
            file or directory with images prepared for classfication <-- _avr_b7_n895.hdr
            
    n : integer
            Number of processes, Default -1 --> all.
    '''

    if n == -1:
       n = None
    
    if os.path.isfile(inVar):
        if inVar.endswith('_avr_b7_n895.hdr'):
            flist = [inVar,]
        else:
            print('Unexpected input file')
            return(1)
        
    elif os.path.isdir(inVar):
        if inVar.endswith('/') == False:
            inVar +='/'
        flist = glob.glob(inVar +'*_avr_b7_n895.hdr')
        
    else:
        os.path.exists(inVar)
        

    pool = Pool(processes=n)
    pool.map(processImg, flist)
    
    pool.close()
    pool.join()
    
    
#%%  
def main():
    '''
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('inVar', help='file or directory with images prepared for classfication --> _avr_b7_n895.hdr', type=str)
    parser.add_argument('--n', type=int, default=-1, help='Number of processes, Default -1 --> all.')
    args = parser.parse_args()
    
    p_processImg(args.inVar, n=args.n)
    
if __name__=="__main__":
    main() 
    
    
    