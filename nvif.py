#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 13:12:52 2020

@author: henning
"""

import functools
from dataclasses import field

from flax import linen as nn
from flax import optim

import jax.numpy as jnp
import jax
from jax.scipy.special import logsumexp

import numpy as np
import time


#numerically stable way to compute: log(1-exp(x)), i.e. inverte log probs
log1mexp = lambda x: jnp.where(-x > jnp.log(2), jnp.log1p(-jnp.exp(x)), jnp.log(-jnp.expm1(x)))
sg = lambda x: jax.lax.stop_gradient(x)



def compute_pis(p, n, EPS):
    
    N = p.shape[0]
    
    def compute_pi(q, n):
        
        q = q - logsumexp(q) + jnp.log(n)
        init = (q, 0, n)
        condfun = lambda op: jnp.max(op[0]) > EPS
        def bodyfun(carry):
            (q, iters, n) = carry
            q = jnp.clip(q, a_min=-jnp.inf, a_max=0)
            q = q - logsumexp(q) + jnp.log(n)
            return (q, iters+1, n)
        
        (q, iters, n) = jax.lax.while_loop(condfun, bodyfun, init)
        
        return q, iters
    
    init = compute_pi(p, n)
    xs = jnp.arange(N-n)+n
    
    def f(carry, n):
        pi_nm1, iters = carry
        pi, itr  = compute_pi(pi_nm1, n)
        return (pi, iters+itr), pi
    
    (_, iters), pis = jax.lax.scan(f, init, xs)
    
    return jnp.clip(jnp.flipud(pis), a_min=-jnp.inf, a_max=0), iters
        


def tilles_step(key, p, n, EPS=1E-4):

    N = p.shape[0]

    selected = jnp.ones((N,))
    pi_ip1 = jnp.zeros((N,)) #pi(k,i+1)
    
    pis, iters = compute_pis(p, n, EPS)
    state = (key, selected, pi_ip1)
    
    def loop_body(i, state):
        key, selected, pi_ip1 = state
        pi = pis[i]
        
        r = log1mexp(pi-pi_ip1)
        r = r+jnp.log(selected)
        key, subkey = jax.random.split(key)
        
        elim = jax.random.categorical(key, r)
        selected = selected.at[elim].set(0)
        
        return (key, selected, pi)
    
    key, selected, pi = jax.lax.fori_loop(0, N-n, loop_body, state)
    
    return key, selected.astype(jnp.bool_), pi
    


def tilles_sampler(n, EPS):
    
    def sample(key, p, n, EPS):
        
        tilles_ = lambda x,y: tilles_step(x,y,n, EPS)
        keys = jax.random.split(key, p.shape[0])
        
        keys, samples, pi  = jax.vmap(tilles_)(keys, p)
        return keys[-1], samples, pi
    
    f = lambda key, p: sample(key, p, n, EPS)
    return jax.jit(f)



def select_half_by_mask(x, mask):
    '''
    JAX does not support polymorphic boolean indexing, this is a hack/trick.
    This function selects half of the entries in x based on mask.
    Half of the entries in mask should be 1.

    '''
    N = mask.shape[-1]//2
    
    mask = jnp.where(mask, jnp.arange(N*2), jnp.log(jnp.zeros(N*2)))
    mask = jnp.sort(mask, 1)[:,N:].astype(jnp.int32)
    
    if len(x.shape) > 2:
        index_fun = lambda m,x: x[m,:]
    else:
        index_fun = lambda m,x: x[m]
    
    return jax.vmap(index_fun)(mask, x)


class head(nn.Module):
    
  @nn.compact
  def __call__(self, x):
            
      #x = nn.tanh(nn.Dense(features=128)(x))
      x = nn.tanh(nn.Dense(features=64)(x))
      x = nn.Dense(features=1)(x)
      sp = -nn.softplus(x)
      
      return jnp.concatenate([sp, sp + x], -1) #p(z|x), 1-p(z|x)



class LSTM(nn.Module):
    @functools.partial(
        nn.transforms.scan,
        variable_broadcast='params',
        split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, x):
        return nn.LSTMCell()(carry, x)



class fnet_sampler(nn.Module):
    
    z_dim: int
    num_samples: int
    
    head_module: nn.Module = head
    temp_module: nn.Module = LSTM
    EPS: float = 1E-3
    
    def setup(self):
        self.sampler = tilles_sampler(self.num_samples, self.EPS)
    
    
    @nn.compact        
    def __call__(self, sample_state, x):
        
        N = self.num_samples
        
        key, state = sample_state
        state, h_temp = self.temp_module()(state, x)
        
        T = x.shape[0]
        
        z = jnp.zeros((T, 2*self.num_samples, self.z_dim))
        q = jnp.zeros((T, 2*self.num_samples)) #q(z|x)
        
        q_z = head()(h_temp)
        
        z = jax.ops.index_update(z, jax.ops.index[:, 0, 0], 1)
        q = jax.ops.index_add(q, jax.ops.index[:, 0], q_z[:,0])
        q = jax.ops.index_add(q, jax.ops.index[:, 1], q_z[:,1])
        
        expand_agenda = lambda x, l: \
            jax.ops.index_update(x, jax.ops.index[:, l:2*l], x[:,:l])
            
        
        for i in range(1,int(np.log2(N))+1): 
            h_temp_bc = jnp.broadcast_to(h_temp[:,None], (T, 2**i, h_temp.shape[-1]))
            conditions = jnp.concatenate([h_temp_bc,z[:,:2**i,:i]],-1)
            q_z = head()(conditions)
        
            z = expand_agenda(z, 2**i)
            q = expand_agenda(q, 2**i)
            
            z = jax.ops.index_update(z, jax.ops.index[:, :2**i, i], 1)
            q = jax.ops.index_add(q, jax.ops.index[:, :2**i], q_z[:,:,0])
            q = jax.ops.index_add(q, jax.ops.index[:, 2**i:2**(i+1)], q_z[:,:,1])

        #at this point, the agenda is full
        sample_probs = q 
        r = jnp.zeros_like(q) #actual inclusion probabilities
        
        def fill_agenda(x, x_samples):
            x = jax.ops.index_update(x, jax.ops.index[:, :N], x_samples)
            x = jax.ops.index_update(x, jax.ops.index[:, N:], x_samples)
            return x
        
        for i in range(int(np.log2(N))+1, self.z_dim):
            #draw num_sample samples without replacement from agenda
            key, samples, inclu_prob = self.sampler(key, sample_probs)
            
            r += inclu_prob
            
            q = fill_agenda(q, select_half_by_mask(q, samples))
            z = fill_agenda(z, select_half_by_mask(z, samples))
            r = fill_agenda(r, select_half_by_mask(r, samples))
                
            h_temp_bc = jnp.broadcast_to(h_temp[:,None], (T, N, h_temp.shape[-1]))
            conditions = jnp.concatenate([h_temp_bc,z[:,:N,:i]],-1)
            
            q_z = head()(conditions)
            
            sample_probs = jnp.concatenate([q_z[...,0],q_z[...,1]],-1)
            
            z = jax.ops.index_update(z, jax.ops.index[:, :N, i], 1)
            q = jax.ops.index_add(q, jax.ops.index[:, :N], q_z[:,:,0])
            q = jax.ops.index_add(q, jax.ops.index[:, N:], q_z[:,:,1])
            
            
        #last sampling step
        key, samples, inclu_prob = self.sampler(key, sample_probs)
        r += inclu_prob
        
        q = select_half_by_mask(q, samples)
        z = select_half_by_mask(z, samples)
        r = select_half_by_mask(r, samples)
        
        return (key, state), q, r, z
    

class scan_wrapper(nn.Module):
    
    pzz_fun: nn.Module
    
    @functools.partial(
        nn.transforms.scan,
        variable_broadcast='params',
        split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, x):
        
        (wtm1, ztm1), (pxzt, zt, qt) = carry, x
        pzz = self.pzz_fun(zt, ztm1)
        
        wtm1, ztm1 = sg(wtm1), sg(ztm1)
        
        p_joint = pzz + wtm1
        viterbi_prev = jnp.argmax(p_joint, -1)
        p_joint = logsumexp(pzz + wtm1, -1) + pxzt
        
        pxx = logsumexp(p_joint-qt)
        w = p_joint - qt - sg(pxx)
        
        return (w,zt), (w, p_joint, viterbi_prev)

 
    
    
class nvif(nn.Module):
    
    p_zz: nn.Module
    p_xz: nn.Module
    x_dim: int
    z_dim: int
    
    sampler: nn.Module = fnet_sampler
    num_samples: int = 512
    sampler_cfg: dict = field(default_factory=dict)
    
    def setup(self):
        self.sampler_cfg.update({'num_samples': self.num_samples,
                                 'z_dim':self.z_dim})
        
        self.sample_fun = self.sampler(**self.sampler_cfg)
        
        self.pxz_fun = self.p_xz(x_dim=self.x_dim)
        self.pzz_fun = self.p_zz()
        self.scan_ = scan_wrapper(self.pzz_fun)
 
    
    def __call__(self, s_state, ztm1, wtm1, x):
        
        s_state, q, r, z = self.sample_fun(s_state, x)
        pxz = self.pxz_fun(x,z)
        
        init = (wtm1, ztm1)
        xs = (pxz, sg(z), sg(q))
                
        (wtm1,ztm1), (w, p_joint, viterbi_prev) = self.scan_(init, xs)
        r -= jnp.log(self.num_samples)
        
        return s_state, ztm1, wtm1, (z, q, r, w, p_joint, viterbi_prev)
    
    


class NVIF:
    
    def __init__(self, **kwargs):
        
        nvif_kwargs = {}
        other_kwargs = {}
        for kw in kwargs:
            if kw in nvif.__annotations__:
                nvif_kwargs[kw] = kwargs[kw]
            else:
                other_kwargs[kw] = kwargs[kw]
                
        if not 'hidden_dim' in other_kwargs:
            other_kwargs['hidden_dim'] = 128
        if not 'num_steps' in other_kwargs:
            other_kwargs['num_steps'] = 256
        
        self.kwargs = other_kwargs
        self.nvif = nvif(**nvif_kwargs)
        
        
    def train(self, x, seed = 420, optimizer = optim.Adam(3E-4), num_epochs=100, **kwargs):
        nv = self.nvif
        nv_apply = jax.jit(nv.apply)
        
        key = jax.random.PRNGKey(seed)
        num_samples, z_dim = nv.num_samples, nv.z_dim
        h_dim, num_steps = self.kwargs['hidden_dim'], self.kwargs['num_steps']
        
        if 'z0' in kwargs:
            z0 = kwargs['z0']
        else:
            z0 = jax.random.bernoulli(key, p = 0.2, shape=(num_samples, z_dim))*1.0
        
        init_input = (key, (jnp.zeros((h_dim,)), jnp.zeros((h_dim,)))), z0, jnp.zeros((num_samples,))-jnp.log(num_samples), jax.random.uniform(key, (num_steps, nv.x_dim))
        params = nv.init(key, *init_input)
        
        w0 = init_input[2]
        
        @jax.jit
        def train_step(optimizer, s_state, ztm1, wtm1, x):
            """Train one step."""
        
            def loss_fn(params):
                new_state, new_ztm1, new_wtm1, other = nv_apply(params, s_state, ztm1, wtm1, x)
                q,r,pj = other[1], other[2], other[4]
                elbo = jnp.exp(q-sg(r))*(pj-sg(q))
                loss = -jnp.mean(elbo)
                return loss, (new_state, new_ztm1, new_wtm1)
            
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, (s_state, ztm1, wtm1)), grad = grad_fn(optimizer.target)
            optimizer = optimizer.apply_gradient(grad)
            
            return optimizer, s_state, ztm1, wtm1, loss
        
        print('Done jitting')
        
        def train_epoch(optimizer, x, z0, w0, s0):
            """ Train one epoch."""
            opt = optimizer
            ztm1, wtm1, s_state = z0, w0, s0
            losses = []
            
            s = time.time()
            for t in range(int(np.ceil(x.shape[0]//num_steps))):
                x_slice = x[t*num_steps:(t+1)*num_steps]
                opt, s_state, ztm1, wtm1, loss = train_step(opt, sg(s_state), sg(ztm1), sg(wtm1), x_slice)
                losses.append(loss)
                print(t, loss, time.time()-s)
                
            return opt, np.mean(losses)
                
        if not hasattr(self, 'optimizer'):
            optimizer = optimizer.create(params)
            self.losses = []
        else:
            print('Resuming')
            optimizer = self.optimizer
        
        s0 = init_input[0]
        for i in range(num_epochs):
            s = time.time()
            s0 = (jax.random.split(s0[0], 2)[0], *s0[1:])
            optimizer, loss = train_epoch(optimizer, x, z0, w0, s0)
            print(f'{i+1}/{num_epochs} Epoch loss: {loss:.3f}, took seconds:{(time.time()-s):.3f}')
            self.optimizer = optimizer
            self.losses.append(loss)
            
        self.z0, self.w0, self.s0 = z0, w0, s0
        
        return optimizer.target, params, self.losses
    
    
    def inference(self, x):
        
        s0, z0, w0 = self.s0, self.z0, self.w0
        (z, q, r, w, p_joint, vit_prev) = self.nvif.apply(self.optimizer.target, s0, z0, w0, x)[-1]
        
        prev = vit_prev[-1][jnp.argmax(p_joint[-1])]
        states = [z[-1][jnp.argmax(p_joint[-1])]]
        
        for i in range(vit_prev.shape[0]-2,-1,-1):
            states.append(z[i,prev])
            prev = vit_prev[i][prev]
            
        states.append(z[0,prev])
        states.reverse()
            
        return jnp.array(states)
    
        

        
        
    
        
        
        

        
    