#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:49:24 2021

@author: henning
"""

from flax import linen as nn
from flax import optim

import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jax

#import nvif
from nvif import NVIF

import numpy as np
import scipy



import jax.profiler
server = jax.profiler.start_server(9999)


def generate_waveforms(x_dim, N=15):
    '''
    I assume that current waveforms of individual loads can have a phase shift
    and are non-linear. Non-linearity is modeled by a simple exponent.
    
    The phase shift is sampled uniformly between [-0.2*pi, 0.2*pi] and the 
    exponent from a geometric distribution with p = 1/4. (With odd exponent)
    
    Amplitude is assumed to vary between 0.3 and 1.3.
    
    '''
    
    np.random.seed(42069) #for reproducibility
    
    phase_shift = np.random.uniform(-0.3*np.pi, 0.3*np.pi, (N,))
    exponent = np.random.geometric(0.075, (N,))*2-1
    A = np.random.uniform(0.15, 2.0, (N,1))
    
    W = np.array([np.sin(np.linspace(0, 2*np.pi, x_dim) - phase_shift[i])**exponent[i] for i in range(N)])
    return A*W


def generate_state_seq(T, N=15):
    '''
    I assume that the number of appliances that change states from t to t+1
    follows a geometric distribution.
    
    Each appliance has an equal probability of switching.
    
    The initial state is sampled from a multi-variate Bernoulli with p=0.2
    
    '''
    np.random.seed(69) #for reproducibility
    
    state_seq = [np.random.binomial(1, 0.2, (N,))]
    for _ in range(T-1):
        num_switches = np.random.geometric(0.75)
        switches = np.random.choice(N, size=(num_switches,), replace=False)
        
        z_in = state_seq[-1].copy()
        for sw in switches:
            z_in[sw] = 1 - z_in[sw]
            
        state_seq.append(z_in)
        
    return np.array(state_seq)


def generate_synthetic(T, x_dim, N=15):
    '''
    I assume Gaussian measurement noise with sigma = 0.015
    '''
    W = generate_waveforms(x_dim, N)
    z = generate_state_seq(T, N)
    x = np.dot(z, W)
    x += np.random.normal(0, 0.015, x.shape) #additive noise
    
    return x, W, z



class p_zz(nn.Module):

    @nn.compact
    def __call__(self, zt, ztm1):
        #this is a trick to get a parameter for p
        p_ = nn.Dense(features=1, use_bias=False)(jnp.ones((1,)))
        p = -nn.softplus(p_)
        pm1 = p + p_
        
        #Geometric distribution: (1-p)**k*p -> k*log(1-p) + log(p)
        k = jnp.sum(jnp.abs(ztm1[None] - zt[:,None]), -1)
        
        return (k*pm1 + p)
    
    
class p_zz_fixed(nn.Module):
    
    def setup(self):
        
        z_dim = 10
        p = 0.75
        
        geo = lambda k: k*np.log(1-p) + np.log(p)            
        probs = geo(np.arange(z_dim))
        Zs = np.log(scipy.special.comb(z_dim, np.arange(z_dim)))
            
        Zs = jnp.array(Zs) + probs
        probs = jnp.array(probs)
        probs = probs-logsumexp(Zs)
        
        self.probs = probs


    def __call__(self, zt, ztm1):
        
        k = jnp.sum(jnp.abs(ztm1[None] - zt[:,None]), -1).astype(jnp.int32)
        
        return self.probs[k]
    
    
class p_xz(nn.Module):
    
    x_dim: int = 128
    
    
    @nn.compact
    def __call__(self, x, z):
        
        xhat = nn.Dense(features=self.x_dim, use_bias=False)(z)
        #this is a trick to get a strictly positive parameter for sigma
        #sgm = 0.1224745
        #log_sgm = -2.1
        #sgm = nn.softplus(nn.Dense(features=1, use_bias=False)(jnp.ones((1,))))
        sgm = 0.12247

        if len(xhat.shape) > 2:
            x = x[:,None]
            
        return -jnp.mean((x - xhat)**2 / (2 * sgm**2) + jnp.log(sgm), axis=-1)
        #return -jnp.mean((x - xhat)**2 / (2 * sgm**2) + log_sgm, axis=-1)



key = jax.random.PRNGKey(1)
h_dim = 96
num_samples = 256
z_dim = 10

x, W, z = generate_synthetic(10000, 156, N=10)
x = jnp.array(x)


N = NVIF(hidden_dim = h_dim, p_zz=p_zz_fixed, p_xz=p_xz, num_steps=128,
         num_samples=num_samples, z_dim=z_dim, x_dim=156)
print('starting train')
#optim.sgd.GradientDescent(learning_rate=0.003)
#optim.Adam(3E-4)
#a = N.train(x, optimizer = optim.sgd.GradientDescent(learning_rate=0.003), num_epochs=15)
a = N.train(x, optimizer = optim.Adam(3E-3), num_epochs=20)
