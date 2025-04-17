import pandas as pd 
import numpy as np
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
    num_obs: int                # number of observations
    num_loc: int                # number of cities in the data
    loc_ind: jnp.ndarray        # unique locational values
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

def DGP(num_obs:int=100, misspefication:bool=False, seed:int=17)->Setting:
    """
    Data generating process that returns a Setting in
    accordance with the Gauss-Markov assumptions

    Args:
        num_obs (int, optional): number of observations
        misspecification (bool, optional): set true model to linear or non-linear, 
                                           defaults to linear
        seed (int, optonal): seed for PRNG

    Returns:
        Setting: realization of the data generating process 
    """

    # PRNG keys
    key = jax.random.key(seed)
    subkey1, subkey2, subkey3, subkey4, subkey5, subkey6 = jax.random.split(key, 6)
    
    _beta = [50, 0.5, 0.6, 0.1, -0.3, 0.4, 0.2]
    _cities = jnp.array([
        0,  # Chicago
        1,  # Boston
        2,  # New York
        3,  # San Francisco
        4,  # Seattle
        5   # Austin 
        ])

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
        num_obs=num_obs, num_loc=_cities.size, loc_ind=_cities, beta=_beta,wages=wages, education=education,
        experience=experience, gender=gender, bads=bads, ability=ability, location=location)
    
    return Setting(**_settings_dict) 

def module_OLS(data:Setting)->jnp.ndarray:
    """Estimate OLS with statmodels module"""

    # convert jax arrays to pandas object for compatibility
    # use np.array() to explicity detach jax object from jax coputational graph
    df = pd.DataFrame({
            'wages': np.array(data.wages),
            'education': np.array(data.education),
            'experience': np.array(data.experience),
            'gender': np.array(data.gender),
            'bads': np.array(data.bads),
            'ability': np.array(data.ability),
            'location': np.array(data.location)
        })

    # Specify the regression formula
    formula = "wages ~ education + experience + gender + bads + ability + C(location)"

    # Fit the model using statsmodels
    model = smf.ols(formula=formula, data=df).fit()
    model.summary()

    return

def matrix_OLS(data:Setting)->jnp.ndarray:
    """Estimate OLS via matrix formulation"""

    # transforms locations into fixed effects
    onehot_loc = (data.location[:, None] == data.loc_ind).astype(int)
    onehot_loc = onehot_loc[:, 1:] 

    # form data matrix
    constant = jnp.ones(shape=(data.num_obs, 1))
    X = jnp.column_stack((
        constant, 
        data.education, 
        data.experience, 
        data.gender, 
        data.bads, 
        data.ability,
        onehot_loc
    ))
    y = data.wages.reshape(-1, 1)

    #FIXME multicolinearity somewhere -- X is not inveritble
    _beta_hat = jnp.linalg.inv(X.T @ X) @ X.T @ y

    return 

def foc_OLS(data:Setting)->jnp.ndarray:
    """Estimate OLS as the BLUE"""
    return

def robust_SE()->jnp.ndarray:
    """Eicker-Huber-White Standard Errors
    
    Start here:
        https://en.wikipedia.org/wiki/Heteroskedasticity-consistent_standard_errors
    """
    return  


# --- TESTS --- #
data = DGP(10)
matrix_OLS(data)
