import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from functools import partial

from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.linear_gaussian_ssm import lgssm_smoother as serial_lgssm_smoother
from dynamax.linear_gaussian_ssm import parallel_lgssm_smoother
from dynamax.linear_gaussian_ssm import lgssm_posterior_sample as serial_lgssm_posterior_sample
from dynamax.linear_gaussian_ssm import parallel_lgssm_posterior_sample

def allclose(x, y, atol=1e-2):
    m = jnp.max(jnp.abs(x-y))
    if m > atol:
        print(f'\nmax diff: {m} > {atol}')
        return False
    else:
        return True

def make_static_lgssm_params(key):
    latent_dim = 4
    observation_dim = 2

    dt = 0.1
    F = jnp.eye(latent_dim) + dt * jnp.eye(latent_dim, k=2)
    Q = 1. * jnp.kron(jnp.array([[dt**3/3, dt**2/2],
                                 [dt**2/2, dt]]),
                      jnp.eye(latent_dim // 2))
    H = jnp.eye(observation_dim, latent_dim)
    R = 0.5 ** 2 * jnp.eye(observation_dim)
    μ0 = jnp.array([0.,0.,1.,-1.])
    Σ0 = jnp.eye(latent_dim)

    lgssm = LinearGaussianSSM(latent_dim, observation_dim)
    params, _ = lgssm.initialize(key,
                                 initial_mean=μ0,
                                 initial_covariance=Σ0,
                                 dynamics_weights=F,
                                 dynamics_covariance=Q,
                                 emission_weights=H,
                                 emission_covariance=R)
    return lgssm, params

class TestParallelLGSSMSmoother():
    """ Compare parallel and serial lgssm smoothing implementations."""

    seed = 0 
    key = jr.PRNGKey(seed)
    key1, key2 = jr.split(key)

    lgssm, model_params = make_static_lgssm_params(key1)
    
    inf_params = model_params

    num_timesteps = 50
    _, emissions = lgssm.sample(model_params, key2, num_timesteps)

    serial_posterior = serial_lgssm_smoother(inf_params, emissions)
    parallel_posterior = parallel_lgssm_smoother(inf_params, emissions)

    def test_filtered_means(self):
        assert allclose(self.serial_posterior.filtered_means, self.parallel_posterior.filtered_means)

    def test_filtered_covariances(self):
        assert allclose(self.serial_posterior.filtered_covariances, self.parallel_posterior.filtered_covariances)

    def test_smoothed_means(self):
        assert allclose(self.serial_posterior.smoothed_means, self.parallel_posterior.smoothed_means)

    def test_smoothed_covariances(self):
        assert allclose(self.serial_posterior.smoothed_covariances, self.parallel_posterior.smoothed_covariances)

    def test_marginal_loglik(self):
        assert jnp.allclose(self.serial_posterior.marginal_loglik, self.parallel_posterior.marginal_loglik)


class TestParallelLGSSMSampler():
    """Compare parallel and serial lgssm posterior sampling implementations in expectation."""
    
    seed = 0 
    key = jr.PRNGKey(seed)
    key1, key2, key3, key4 = jr.split(key, 4)

    lgssm, model_params = make_static_lgssm_params(key1)
    
    inf_params = model_params

    num_timesteps = 50
    _, emissions = lgssm.sample(model_params, key2, num_timesteps)

    num_samples = 1000
    serial_keys = jr.split(key3, num_samples)
    parallel_keys = jr.split(key4, num_samples)
    serial_samples = vmap(serial_lgssm_posterior_sample, in_axes=(0,None,None))(
                          serial_keys, inf_params, emissions)
    parallel_samples = vmap(parallel_lgssm_posterior_sample, in_axes=(0, None, None))(
                            parallel_keys, inf_params, emissions)

    def test_sampled_means(self):
        serial_mean = self.serial_samples.mean(axis=0)
        parallel_mean = self.parallel_samples.mean(axis=0)
        assert allclose(serial_mean, parallel_mean, atol=1e-1)

    def test_sampled_covariances(self):
        # samples have shape (N, T, D): vmap over the T axis, calculate cov over N axis
        serial_cov = vmap(partial(jnp.cov, rowvar=False), in_axes=1)(self.serial_samples)
        parallel_cov = vmap(partial(jnp.cov, rowvar=False), in_axes=1)(self.parallel_samples)
        assert allclose(serial_cov, parallel_cov, atol=1e-1)