import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import UpSampling1D, concatenate, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam

import healpy as hp
from pygsp.graphs import SphereHealpix
from pygsp import filters

from deepsphere import healpy_layers as hp_layer
from deepsphere import gnn_layers as gnn
from deepsphere import utils
from deepsphere import plot

def rotate(sky,z,y,x,nside,p=3,pixel=True,forward=True,nest2ring=True):
    '''
    Up-samples the data, rotates map, then pools it to original nside. Map has to be in "NEST" ordering.
    
    Input:
    sky        map (In NEST ordering if nest2ring=True)
    z          longitude
    y          latitude
    x          about axis that goes through center of map (rotation of object centered in center)
    nside  
    p          up-samples data by 2**p 
    pixel      if True rotation happens in pixel space. Otherwise it happens in spherical harmonics space.
    forward    if True, +10degree rotation does +10degree rotation. Otherwise it does a -10 degree rotation
    nest2ring  if True converts NEST ordering to RING ordering before rotating, and RING to NEST after rotation.
               (rotation only works with RING ordering)
    
    Output:
    Rotated map
    
    '''
    #the point provided in rot will be the center of the map
    rot_custom = hp.Rotator(rot=[z,y,x],inv=forward)#deg=True
    
    if nest2ring == True:
        sky = hp.reorder(sky,n2r=True)
    
    up = hp.ud_grade(sky,nside*2**p)#up-sample
    if pixel == True:
        m_smoothed_rotated_pixel = rot_custom.rotate_map_pixel(up)
    else:
        m_smoothed_rotated_pixel = rot_custom.rotate_map_alms(up)#uses spherical harmonics instad
    down = hp.ud_grade(m_smoothed_rotated_pixel,nside)#down-sample
    
    if nest2ring == True:
        down = hp.reorder(down,r2n=True)
    
    return down

class HealpyUNet:
    """
    A graph UNet convolutional network.
    """
    
    def __init__(self, nside, indices, learning_rate, mask, mean, n_neighbors=20):
        """
        self, nside, indices, n_neighbors=20
        
        Initiates the UNet.
        Input:
        Output:
        """        
        #Create the instance attributes from inputs
        self.current_nside = nside
        self.current_indices = indices
        self.n_neighbors = n_neighbors
        self.learning_rate = learning_rate
        self.mask = mask
        self.mean = mean
        
    def model(self):
        """self, inputs"""
        
        inputs = Input(shape = (12*self.current_nside**2,1))
        x1 = self.HealpyChebyshev(self.current_nside, self.current_indices, K=10, Fout=16,
                                  initializer='he_normal', activation='relu', use_bias=True, use_bn=True)(inputs)
        
        x2 = self.HealpyPool(self.current_nside, self.current_indices, p=1, pool="AVG")(x1)
        x2 = self.HealpyChebyshev(self.current_nside, self.current_indices, K=10, Fout=32,
                                  initializer='he_normal', activation='relu', use_bias=True, use_bn=True)(x2)

        x3 = self.HealpyPool(self.current_nside, self.current_indices, p=1, pool="AVG")(x2)
        x3 = self.HealpyChebyshev(self.current_nside,self.current_indices,K=10,Fout=32,
                                  initializer='he_normal', activation='relu', use_bias=True, use_bn=True)(x3)
        
        xdeep = self.HealpyPool(self.current_nside, self.current_indices, p=1, pool="AVG")(x3)        
        xdeep = self.HealpyChebyshev(self.current_nside, self.current_indices, K=10, Fout=32,                 
                                  initializer='he_normal', activation='relu', use_bias=True, use_bn=True)(xdeep)
                
        x4 = self.HealpyPseudoConv_Transpose(self.current_nside,self.current_indices,p=1,Fout=32,
                                             kernel_initializer='he_normal')(xdeep)       
        #x4 = self.HealpyUpSample(self.current_nside, self.current_indices, p=1)(xdeep)
        x4 = concatenate([x4, x3], axis=2)
        x4 = BatchNormalization()(x4)
        #add convolution
        
        x5 = self.HealpyPseudoConv_Transpose(self.current_nside, self.current_indices, p=1, Fout=32, 
                                             kernel_initializer='he_normal')(x4)       
        #x5 = self.HealpyUpSample(self.current_nside, self.current_indices, p=1)(x4)        
        x5 = concatenate([x5, x2], axis=2)
        x5 = BatchNormalization()(x5)
        #add convolution
        
        x6 = self.HealpyPseudoConv_Transpose(self.current_nside, self.current_indices, p=1, Fout=32, 
                                             kernel_initializer='he_normal')(x5)       
        #x6 = self.HealpyUpSample(self.current_nside, self.current_indices, p=1)(x5)        
        x6 = concatenate([x6, x1], axis=2)
        x6 = BatchNormalization()(x6)
        #add convolution
        
        x7 = self.HealpyChebyshev(self.current_nside, self.current_indices, K=10, Fout=16, 
                                  initializer='he_normal', activation='relu', use_bias=True, use_bn=False)(x6)               
        
        
        output = self.HealpyChebyshev(self.current_nside, self.current_indices, K=10, Fout=1, 
                                      initializer='he_normal', activation='sigmoid', use_bias=True, use_bn=False)(x7)
            
        unet = Model(inputs,output)
        print(unet.summary(),flush=True)
        
        if self.learning_rate == None:
            unet.compile(optimizer='adam', loss='mse')
        else:
            unet.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
              
        return unet
            
    def HealpyUpSample(self, current_nside, current_indices, p):
        """
        :param p: Boost factor >=1 of the nside -> number of nodes increases by 4^p, note that the layer only checks if the dimensionality of the input is evenly divisible by 4^p and not if the ordering is correct (should be nested ordering)
        """
        layer = UpSampling1D(4**p)
        new_nside = int(current_nside * 2 ** p)
        self.current_indices = self._transform_indices(nside_in=current_nside, nside_out=new_nside, indices=current_indices)
        self.current_nside = new_nside
        return layer
        
    def HealpyChebyshev(self, current_nside, current_indices, K, Fout, initializer, activation, use_bias, use_bn):
        """
        Input:
        current_nside
        current_indices
        +
        :param K: Order of the polynomial to use
        :param Fout: Number of features (channels) of the output, default to number of input channels
        :param initializer: initializer to use for weight initialisation
        :param activation: the activation function to use after the layer, defaults to linear
        :param use_bias: Use learnable bias weights
        :param use_bn: Apply batch norm before adding the bias
        """
        sphere = SphereHealpix(subdivisions=current_nside, indexes=current_indices, nest=True, 
                               k=self.n_neighbors, lap_type='normalized')
        current_L = sphere.L
        
        layer = hp_layer.HealpyChebyshev(K, Fout, initializer, activation, use_bias, use_bn)
        
        actual_layer = layer._get_layer(current_L)
        
        return actual_layer
     
    def HealpyPseudoConv_Transpose(self, current_nside, current_indices, p, Fout, kernel_initializer):
        """
        :param p: Boost factor >=1 of the nside -> number of nodes increases by 4^p, note that the layer only checks if the dimensionality of the input is evenly divisible by 4^p and not if the ordering is correct (should be nested ordering)
        :param Fout: number of output channels
        :param kernel_initializer: initializer for kernel init
        """
        layer = hp_layer.HealpyPseudoConv_Transpose(p, Fout, kernel_initializer)
        new_nside = int(current_nside * 2 ** layer.p)
        self.current_indices = self._transform_indices(nside_in=current_nside, nside_out=new_nside, indices=current_indices)
        self.current_nside = new_nside
        return layer
           
    def HealpyPool(self, current_nside, current_indices, p, pool):
        """
        Input:
        current_nside
        current_indices
        +
        p         reduction factor >=1 of the nside -> number of nodes reduces by 4^p, note that the layer only checks if the dimensionality of the input is evenly divisible by 4^p and not if the ordering is correct (should be nested ordering).
        pool      type of pooling, can be "MAX" or  "AVG"
        
        Layer Parameters: (That have been set)
        None        
        """
        layer = hp_layer.HealpyPool(p, pool)
        new_nside = int(current_nside//2**layer.p)
        self.current_indices = self._transform_indices(nside_in=current_nside, nside_out=new_nside, indices=current_indices)
        self.current_nside = new_nside
        return layer
 
    def _transform_indices(self, nside_in, nside_out, indices):
        """
        Transforms a set of indices to an array of indices with a new nside. If the resulting map is smaller, it
        assumes that the reduction is sensible, i.e. all no new indices will be used during the reduction.
        :param nside_in: nside of the input indices
        :param nside_out: nside of the output indices
        :param indices: indices (pixel ids)
        :return: a new set of indices with nside_out
        """
        # simple case
        if nside_in == nside_out:
            return indices

        # down sample a binary mask
        mask_in = np.zeros(hp.nside2npix(nside_in))
        mask_in[indices] = 1.0
        mask_out = hp.ud_grade(map_in=mask_in, nside_out=nside_out, order_in="NEST", order_out="NEST")
        transformed_indices = np.arange(hp.nside2npix(nside_out))[mask_out > 1e-12]

        return transformed_indices