
import tensorflow_probability.substrates.jax as tfp
from jax import random, numpy as jnp, vmap

from jaxns import resample

tfpd = tfp.distributions


# Generate data

num_samples = 10

event_times = jnp.concatenate([jnp.linspace(0., 1., num_samples) ** 2,
                               2. + jnp.linspace(0., 1., num_samples) ** 2])


from jaxns import Prior, Model

# Build model
B = 2.0


def lam_t(mu, alpha, beta, t):
    dt = t[:, None] - event_times[None, :]
    # a-a, a-b
    phi = alpha * jnp.exp(-beta * jnp.where(dt > 0, dt, 0)) * jnp.where(dt > 0, 1, 0)
    lam = mu + jnp.sum(phi, axis=1)
    return lam


def log_likelihood(mu, alpha, beta):
    """
    Poisson likelihood.
    """
    lam = lam_t(mu, alpha, beta, event_times)
    Lam = mu + B * event_times.size

    print("priors input to log likelihood = ")
    print(mu)
    lout = jnp.sum(jnp.log(lam)) - Lam
    print("lout = ", lout)
    return lout


def prior_model():
    mu = yield Prior(tfpd.HalfCauchy(loc=0, scale=1.), name='mu')
    beta = yield Prior(tfpd.HalfNormal(num_samples), name='beta')
    alpha = yield Prior(tfpd.Uniform(0., B * beta), name='alpha')

    print("Priors look like this:")
    print(mu)
    return mu, alpha, beta


model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

model.sanity_check(random.PRNGKey(0), S=100)