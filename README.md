# Neural Variational Filtering and Inference
This repository contains an efficient approximate algorithm for inference and learning for temporal graphical models with binary latent variables.
This code is accompanied by the following [paper](https://ieeexplore.ieee.org/abstract/document/8683552). A video explaining the algorithm and code will follow soon.

# Usage
In order to use the algorithm, probability densities for p(z_t | z_{t-1}) and p(x_t | z_t) need to be defined. The algorithm has two control knobs that can trade off computational burden and accurary. The first control know is the number of samples used to approximate the data likelihood (num_samples) whereas the second control knob controls the accuracy of the underlying sampler (EPS).

The input to p(x_t | z_t) is `x_t: (num_steps, x_dim)` and `z_t: (num_steps, num_samples, z_dim)` and the output is the log probability for each state `(num_steps, num_samples)`. Because `jax.lax.scan` is employed, input for p(z_t | z_{t-1}) does not have the num_steps dimension, i.e. it is `z_t: (num_samples, z_dim)` and `z_{t-1}: (num_samples, z_dim)` and the output is again the log probability for each combination of states, i.e. `(num_samples, num_samples)`. The algorithm therefore scales quadratically in num_samples.

Once these two potentially parameterized functions are defined, the model can be fit and inference can be performed by:

```python
from nvif import NVIF

N = NVIF(p_zz=p_zz_fixed, p_xz=p_xz, num_steps=128,
         num_samples=512, z_dim=15, x_dim=156)
N.train(x[:5000], optimizer = optim.Adam(3E-3), num_epochs=20)

z_hat = N.inference(x[5000:])
```

See `nilm_example.ipynb` for an example of how to use the algorithm in the context of a synthetic problem inspired by Non-Intrusive Load Monitoring. This readme file will be expanded upon soon.

## Frequently Asked Questions
* Why does the time required per epoch vary over time?
  * Most of the time is spent sampling _without replacement_ from the auxiliary distribution $Q$. Sampling without replacement according to pre-defined inclusion probabilities is difficult and in some cases even impossible. The difficulty of sampling without replacement increases when the distribution to sample from has lower entropy. During training, the entropy of $Q$ decreases (in the beginning most states have the same probability) making the sampling step more time consuming.
* Is the model that is being performed learning and inference on a Factorial Hidden Markov Model (FMM)?
  * __No!__ FMMs make the assumption that the individual latent chains are marginally indepedent. NVIF does not require this assumption making the class of models that NVIF can perform inference and learning on a lot richer. Speficially for NILM, the 'independence between chains'-assumption is oftentimes not great because, from experience, you want to constrain the number of latent states that switch states.
* What are potential avenues for future work?
  * The underlying sampler (Yves Tilles elimination sampler) is slow but accurate. There are faster but less accurate alternatives such as, e.g. the Pareto sampler. Studying the effects of swapping out the sampler might considerably speed up inference.
  * The main contribution of NVIF is an approximate algorithm for inference and learning in temporal models with binary latent states. However, little research has gone into the best instantiations (`p_zz` and `p_xz`) to solve e.g. Non-Intrusive Load Monitoring or Energy Disaggregation. The performance of NVIF can most likely improved substantially by finding better choices for `p_zz` and `p_xz`.
  * So far, NVIF has only been evaluated in the context of NILM, applying the algorithm to other problems that require inference of binary latent states might be interesting.
