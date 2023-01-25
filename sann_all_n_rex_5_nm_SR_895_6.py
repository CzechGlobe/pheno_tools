#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:10:31 2021

@author: pikl.m
"""
import numpy as np



#%% define variables

input_hidden_weights = np.array([
    [-4.92437000405234e+001, -2.16564197705067e+001, 5.73557509303117e+000, 2.70519982108059e+001, 1.67977789552704e+001, -2.35658822997960e+001, -2.42614734927442e+001 ],
    [-6.35487353202633e+002, -5.74817750789048e+002, 5.02033589758545e+002, -1.81535133356187e+002, -6.08940408158185e+001, 9.24432713783686e+000, 5.21208561824204e+001 ],
    [-2.85086455301930e+002, -3.55012005026532e+002, 8.25020848238386e+002, -4.14460163944917e+002, -4.46086607988267e+002, 1.29976930228247e+002, 1.41069811359791e+002 ],
    [-5.97969786563415e+001, -4.99360997104501e+001, 1.91458506070287e+001, -3.27548397604018e+001, -4.96792108573192e+002, 5.05300143476789e+001, 8.20416105153649e+001 ],
    [-4.25028164396528e+002, 1.87635408419282e+002, 1.21428093854227e+001, -2.52124130889062e+002, 2.78851160853751e+002, 8.99761538955703e+001, 6.41806077541556e+001 ]],
    dtype=np.float64)

hidden_bias = np.array([-1.33491144351568e+001, 1.95853204348680e+002, 5.82191938158675e+001, 7.38315401292369e+001, -1.11298448872931e+000], dtype=np.float64)

hidden_output_wts = np.array([
    [-1.23915784865406e+001, -1.81197145749327e+001, 9.51309428957673e+000, 1.52828801045438e+001, -1.48556864123520e+001 ],
    [-6.68238399640504e+000, 9.84589142295635e+000, 2.08203110961694e+000, -4.38134592491371e+000, -7.02065184554276e-001 ],
    [8.72211913762229e+001, 5.12710603155218e+000, -2.25886740044200e+001, 8.11004399451935e+000, 9.29913051161976e+001 ],
    [1.36270117543003e+001, 1.90405010889908e+001, -3.43717105327367e+001, 8.32750999564016e+000, -1.25584312179481e+001 ],
    [-8.19026966663177e+001, -1.57965430422876e+001, 4.52661302516350e+001, -2.74476292506706e+001, -6.48879189562169e+001 ]],
    dtype=np.float64)

output_bias = np.array([1.21205668413024e+001, -6.31719301091136e-001, -8.64657159742921e+001, 1.59911834951682e+000, 7.32422637806015e+001], dtype=np.float64)

max_input = np.array([1.84903114186851e+002, 5.89238754325260e+001, 4.05536332179931e+001, 3.30934256055363e+001, 4.70276816608997e+001, 2.40726643598616e+001, 1.92553041018388e+001], dtype=np.float64)

min_input = np.array([-4.89515151515152e+001, -1.75490693739425e+001, -1.09805414551607e+001, -1.34754653130288e+001, -1.15871404399323e+001, -1.17554991539763e+001, -1.78380952380952e+001], dtype=np.float64)

mean_inputs = np.array([8.12501582703937e-001, 6.87893510060953e-001, 6.70136201072581e-001, 6.98499525991564e-001, 7.35987309696838e-001, 9.03018920022443e-001, 1.02482042196341e+000], dtype=np.float64)

#%% define variables to be calculated
inputvct = np.empty(7, dtype=np.float64)
hidden = np.empty(5, dtype=np.float64)
output = np.empty(5, dtype=np.float64)

#%%  define functions

def logistic2(vect):
    '''
    '''
    
    pos100 = vect > 100
    posm100 = vect < -100
    posor = pos100 + posm100
    
    ovect = np.array(vect)
    ovect[pos100] = 1.0
    ovect[posm100] = 0.0
    ovect[~posor] = 1.0 /(1.0 +np.exp(-vect[~posor]))
    
    return(ovect)
    
    

def ScaleInputs(inputvect, minimum, maximum):
    '''
    Scale input spectrum to range defined by minimum and maximum
    
    Parameters
    ----------
    
    inputvect : 1D numpy array
            Input spectrum
            
    minimum : numeric
            
    maximum : numeric
    
    Returns : 1D numpy array
    
            
    '''
    
    delta = (maximum - minimum) / (max_input - min_input)   
    return(minimum - delta * min_input + delta * inputvect)
    
    
def ComputeFeedForwardSignals(invect, weights, bias, layer):
    '''
    This correspond to one layer of NN.
    
    Parameters
    ----------
    
    invect : 1D numpy array
            input spectrum, expected len = 7
            
    weights : 2D numpy array
            weights of neural layer
            
    bias : 1D numpy array
            bias of output layer
            
    layer : integer
            number of hidder layer. Start 0
            
    Returns : 1D numpy array
            weighted layer
    
    '''
    
    out = (invect * weights).sum(axis=1) +bias
    if layer == 0:
#        out1 = np.apply_along_axis(logistic, 0, out)
        out = logistic2(out)
    
    return(out)
    
    
def softmax(vec):
    '''
    Adjust values of last NN layer before class decision.
    
    Parameters
    ----------
    
    vec : 1D numpy array
    
    Returns
    -------
    1D numpy array
    '''
    
    if len(vec) == 0:
        print('softmax empty input')
    
    softsum = 0.0
    pos200 = vec > 200
    
    if pos200.any():
        ovec = np.array(vec == vec.max(), dtype=vec.dtype)
        
        return(ovec)
    
    else:
        ovec = np.exp(vec)
    
    softsum = ovec.sum()
    
    
    if softsum == 0.0:
        ovec = np.zeros(len(vec))
        #ovec = np.full(len(vec), 1/len(vec), dtype=vec.dtype) 
    
    else:
        ovec = ovec / softsum
    
    return(ovec)
    
#%%    
def main(spcVector, nodata=-9999):
    '''
    Classify spectrum.
    
    Parameters
    _________
    
    spcVector : numpy.array
            Reflectance spectrum with 7 bands. After normalyzing to XX wavelength.
            
    nodata : numeric
            Possible nodata value. Can accept 'data ignore value' from ENVI HDR.
            
    Returns 
    ________
    
    output : byte int
            Classification to one of following classes.\n
    
    possible output classes
    0 ... unclassified
    1 ... blue
    2 ... dry
    3 ... green
    4 ... soil
    5 ... tech
    '''
    
    # check for nodata values
    nodatapos = spcVector == nodata
    if nodatapos.any():
        spcVector[nodatapos] = mean_inputs[nodatapos]
    
    # scale input data
    inputvct = ScaleInputs(spcVector, 0, 1)
    
    # run NN
    hidden = ComputeFeedForwardSignals(inputvct, input_hidden_weights, hidden_bias, 0)
    output = ComputeFeedForwardSignals(hidden, hidden_output_wts, output_bias, 1)

    # define output class    
    ovec = softmax(output)
    
    #if ~ovec.all():
    if ovec.sum() == 0:
        c = 0
    else:
        pos = np.where(ovec == ovec.max())[0][0]
        c = pos+1
    
    return(np.byte(c))
    
    
    
    
    
    
