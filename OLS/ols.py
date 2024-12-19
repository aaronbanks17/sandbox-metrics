import pandas as pd 
import statsmodels.formula.api as smf
import jax.numpy as jnp 
import chex 
import jax

"""
This file implements OLS in the following ways:
    1. using the statsmodels module
    2. using the matrix formulation 
    3. FOC formulation with JAX 
"""

@chex.dataclass 
class Setting:
    """
    Research question: What is the effect of an agent's
    level of education on their wages? 
    """
    # parameter
    beta: jnp.ndarray           # true parameter
    # outcome
    wages: jnp.ndarray          # outcome of interest
    # covariates
    education: jnp.ndarray      # num years of education since high school
    experience: jnp.ndarray     # num years working
    gender: jnp.ndarray         # indicator of gender
    bads: jnp.ndarray           # general detractors from wage improvments (e.g. prison time)
    ability: jnp.ndarray        # measure of skill falling between 0 and 1 (1 is most)
    location: jnp.ndarray       # location variable denoted by index


def DGP(num_obs:int=100, seed:int=17)->Setting:
    """
    Data generating process that returns a Setting in
    accordance with the Gauss-Markov assumptions
    """
    # PRNG keys
    key = jax.random.key(seed)
    subkey1, subkey2, subkey3, subkey4, subkey5, subkey6 = jax.random.split(key, 6)
    
    _cities = jnp.array([
        0,  # Chicago
        1,  # Boston
        2,  # New York
        3,  # San Francisco
        4,  # Seattle
        5   # Austin 
        ])

    _beta = [50, 0.5, 0.6, 0.1, -0.3, 0.4, 0.2]

    # generate covariates
    education = jax.random.randint(subkey1, shape=(num_obs,), minval=1, maxval=12)
    experience = jax.random.randint(subkey2, shape=(num_obs,), minval=1, maxval=45)
    gender = jax.random.bernoulli(subkey3, p=0.5, shape=(num_obs,)) 
    bads = jax.random.normal(subkey4, shape=(num_obs,))
    ability = jax.random.uniform(subkey5, shape=(num_obs,), minval=0, maxval=1)
    location = jax.random.choice(subkey6, _cities, shape=(num_obs,), p=None) 
    # generate structural error
    epsilon = jax.random.normal(subkey3, shape=(num_obs,))
    # generate outcomes
    wages = (
        _beta[0]
        + _beta[1] * education
        + _beta[2] * experience
        + _beta[3] * gender
        + _beta[4] * bads
        + _beta[5] * ability
        + _beta[6] * location
        + epsilon
    )
    _settings_dict = dict(
        beta=_beta,wages=wages, education=education, experience=experience, gender=gender,
        bads=bads, ability=ability, location=location)
    
    return Setting(**_settings_dict) 

controls = DGP(10)
