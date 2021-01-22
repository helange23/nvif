# Neural Variational Filtering and Inference
This repository contains an efficient approximate algorithm for inference and learning for temporal graphical models with binary latent variables.
This code is accompanied by the following [paper](https://ieeexplore.ieee.org/abstract/document/8683552). A video explaining the algorithm and code will follow soon.

# Usage
In order to use the algorithm, probability densities for p(z_t | z_{t-1}) and p(x_t | z_t) need to be defined. The algorithm has two control knobs that can trade off computational burden and accurary. The first control know is the number of samples used to approximate the data likelihood (num_samples) whereas the second control knob controls the accuracy of the underlying sampler (EPS).

The input to p(x_t | z_t) is x_t (num_steps x x_dim) and z_t (num_steps x num_samples x z_dim) and the output is the log probability (num_steps x num_samples). Because jax.lax.scan is employed, input for p(z_t | z_{t-1}) does not have the num_steps dimension, i.e. it is z_t (num_samples x z_dim) and z_{t-1} (num_samples x z_dim) and the output is again the log probability for each combination of states, i.e. (num_samples x num_samples). The algorithm there scales quadratic in num_samples.

Once these two potentially parameterized functions are defined, the model can be fit and inference can be performed by:

```python
from nvif import NVIF

N = NVIF(p_zz=p_zz_fixed, p_xz=p_xz, num_steps=128,
         num_samples=512, z_dim=15, x_dim=156)
N.train(x[:5000], optimizer = optim.Adam(3E-3), num_epochs=20)

z_hat = N.inference(x[5000:])
```

See nilm_example.py for an example of how to use the algorithm in the context of a synthetic problem inspired by Non-Intrusive Load Monitoring. This readme file will be expanded upon soon.
