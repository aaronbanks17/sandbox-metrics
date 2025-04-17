import os 
from methods.settings import settings 
import jax
import jax.numpy as jnp 
from jaxopt import LBFGS
from functools import partial
import chex 
import dataclasses
from typing import Tuple
import time
from tqdm import tqdm
from itertools import product
from jaxopt import ScipyMinimize

@chex.dataclass
class FirmDefinition: 
    mu    : float        # age cost-multiplier
    R     : float        # replacement cost
    beta  : float        # discount factor
    gamma : float        # euler constant
    ages  : jnp.ndarray  # machine age
    V     : jnp.ndarray  # value function

@chex.dataclass(frozen=True)
class FirmData:
    actions : jnp.ndarray # observed sequence of actions
    states  : jnp.ndarray # observed sequence of ages
    periods : int         # time horizon of data series

def _update(params:FirmDefinition, updates:dict) -> FirmDefinition:
    """
    Update FirmDefinition instance as we update parameters
    """
    # ensure updates are all valid
    valid_updates = {k: v for k, v in updates.items() if hasattr(params, k)}
    # return new FirmDefinition instance
    return dataclasses.replace(params, **valid_updates)

def DGP(T:int=20000, seed:int=17) -> Tuple[FirmDefinition, FirmData]:
    """Generate data (DGP) to simulate Rust (1987)/Hotz-Miller (1993)"""

    # initialize FirmDefinition with true parameters
    V_init = jnp.zeros(5)
    ages   = jnp.arange(1, 6)
    params = dict(
        mu = -1.0, R = -3.0, beta = 0.9,
        gamma = 0.5775, ages = ages, V = V_init
        )
    params = FirmDefinition(**params)
    # compute value function at true parameters
    params = vfi(params)
    # set PRNG keys
    key = jax.random.PRNGKey(seed)
    subkey1, subkey2 = jax.random.split(key)
    # compute conditional value functions
    V_0, V_1 = _choice_specific_V(params)
    # simulate data based on (state, action) each period given 
    def _sim(state, _):
        # get state variables resulting from last period
        age, key = state
        # split keys
        key, subkey1, subkey2 = jax.random.split(key, 3)
        # sample choice-specific cost shocks
        eps0 = jax.random.gumbel(subkey1)
        eps1 = jax.random.gumbel(subkey2)
        # compute ex-post choice-specific utilities
        u0 = V_0[age - 1] + eps0
        u1 = V_1 + eps1
        # set optimal action 
        a = jnp.argmax(jnp.array([u0, u1]))
        # update age based on optimal action this period
        next_age = jnp.where(a == 0, jnp.minimum(age + 1, 5), 1)
        return (next_age, key), (age, a)
    
    # draw an initial starting age and run simulation
    init_state = jax.random.randint(subkey1, shape=(), minval=1, maxval=6)
    _, (states, acts) = jax.lax.scan(_sim, (init_state, subkey2), jnp.arange(T))
    _data = FirmData(actions=acts, states=states, periods=T)
    # report data attributes
    print(
        f"[DGP] Generated series of {T} observations,", 
        f"with initial age {init_state}",
        f"and {acts.sum()} observed replacement decisions")
    # set FirmDefinition to mimick econmetrician's information set
    params = dict(
        mu = jnp.nan, R = jnp.nan, beta = 0.9,
        gamma = 0.5775, ages = ages, V = V_init
        )
    _definition = FirmDefinition(**params)

    # return data for estimation
    return _definition, _data 

def _choice_specific_V(params:FirmDefinition) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """compute choice-specific conditional value functions"""
    next_st = jnp.minimum(params.ages + 1, 5)
    V_0 = params.mu * params.ages + params.beta * params.V[next_st - 1]
    V_1 = params.R + params.beta * params.V[0]
    return V_0, V_1

def _bellman_update(V:jnp.ndarray, params:FirmDefinition) -> jnp.ndarray:
    """Given a current value of V, compute a FP Bellman iteration"""
    # determine the next state based on evolution of a_t
    next_state = jnp.minimum(params.ages + 1, 5)
    state_ix   = next_state - 1
    # compute choice-specific cond. value fiuntions
    V_0, V_1 = _choice_specific_V(params)
    # calculate log-sum 
    exp_sum = jnp.exp(V_0) + jnp.exp(V_1)
    V_update = params.gamma + jnp.log(exp_sum)
    return V_update

@partial(jax.jit, static_argnums=(1, 2))
def vfi(params:FirmDefinition, maxiter:int = 500, tol:float=1e-8) -> FirmDefinition:
    """Compute value function at given parameters"""
    # define loop body of value function iteraction
    def _body(V, _):
        V_new = _bellman_update(V, params)
        return V_new, None
    # solve for fixed point
    init = jnp.zeros_like(params.V)
    V_fp, _ = jax.lax.scan(_body, init, None, length=maxiter)
    return _update(params, {"V": V_fp})

def log_likelihood(theta:jnp.ndarray, params:FirmDefinition, data:FirmData,
                   vfi_kwargs:dict=dict(maxiter=500, tol=1e-8)) -> float:
    """Evaluate (negative) log-likelihood at a candidate theta"""

    # upack theta
    _mu, _R = theta
    # update parameters with current guess of theta
    params = _update(params, {"mu": _mu, "R": _R})
    # compute value function at current theta
    params = vfi(params, **vfi_kwargs)
    # compute choice-specific value functions
    V_0, V_1 = _choice_specific_V(params)
    # compute choice probabilities at each state
    cont_values = V_0[data.states - 1] 
    rep_values  = V_1 + jnp.zeros_like(data.states)
    utilities   = jnp.stack([cont_values, rep_values], axis=1)
    logit_probs = jax.nn.softmax(utilities, axis=1)  
    # fetch the probabilities of observed actions
    ix = jnp.arange(data.periods)
    action_probs = logit_probs[ix, data.actions]
    # compute negative log-likelihood
    ll = -jnp.sum(jnp.log(action_probs + 1e-12))

    return ll 

def replacement_prob(data:FirmData) -> jnp.ndarray:
    """Estimate the replacement probability used in HM (1996) approach"""
    # get frequency of each state
    state_counts = jnp.bincount(data.states, minlength=6)[1:]
    # get frequency of replacement in each state
    repl_counts = jnp.bincount(data.states, weights=data.actions, minlength=6)[1:]
    repl_probs  = repl_counts / state_counts
    return repl_probs

def nfp_Rust(params:FirmDefinition, data:FirmData,
             solver_kwargs: dict = dict(maxiter=1000, tol=1e-8),
             vfi_kwargs: dict = dict(maxiter=500, tol=1e-8)
             ) -> FirmDefinition:
    """Estimate utility function via Rust (1987) NFP algorithm"""
        
    # set objetive and solver object
    _obj = lambda theta: log_likelihood(theta, params, data, vfi_kwargs)
    solver = LBFGS(fun=_obj, **solver_kwargs)
    # define parameter grid
    grid = [-4.0, -3.0, -2.0, -1.0]
    lattice = list(product(grid, grid))
    print(f"[RUST NFP] Evaluating {len(lattice)} starting points...")
    results = []
    for theta0 in tqdm(lattice, desc="[RUST NFP]"):
        theta0 = jnp.array(theta0)
        try:
            start_time = time.time()
            res = solver.run(theta0)
            elapsed = time.time() - start_time
            # extract params
            _mle = res.state.value
            mu_hat, R_hat = res.params
            results.append({
                "MLE": _mle, "params": (mu_hat, R_hat),
                "init": tuple(theta0.tolist()),"time": elapsed
            })

        except Exception as e:
            results.append({
                "MLE": jnp.inf, "params": (jnp.nan, jnp.nan),
                "init": tuple(theta0.tolist()), "time": None,
                "error": str(e)
            })
    # find the best result by lowest loss
    best = min(results, key=lambda r: r["MLE"])
    mu_best, R_best = best["params"]
    # report NFP estimates
    print(f"[RUST NFP] Best result:\n",
          f"\t\t init = {best['init']}\n",
          f"\t\t mu   = {mu_best:.2f}\n", 
          f"\t\t R    = {R_best:.2f}\n",
          f"\t\t MLE  = {best['MLE']:.2f}")

    return _update(params, {"mu": mu_best, "R": R_best})

def analytic_HM(params:FirmDefinition, data:FirmData) -> FirmDefinition:
    """
    Estimate utility function via the Hotz-Miller CCP
    approach using the Arcidiacono-Miller formula inversion
    """

    # generate observed data and fixed model structure
    params, data = DGP()
    # estimate replacement probabilities nonparametrically
    p_hat = replacement_prob(data)
    # compute log-odds for each state
    log_odds = jnp.log(p_hat / (1 - p_hat))
    # compute adjustment term: beta * (log P(a_t+1) - log P(1))
    log_p1 = jnp.log(p_hat[0])   
    log_p2 = jnp.log(p_hat[1])   
    # use analytic formulas to solve for mu
    mu_hat = (log_odds[0] - log_odds[4] - params.beta * (log_p2 - log_p1)) / 4.0
    # use analytic fomrula to solve for R
    R_hat = log_odds[4] + 5 * mu_hat
    # report analytic estimates
    print(f"[HM ANALYTIC] Estimated parameters using Arcidiacono-Miller Inversion:")
    print(f"\t\t mu = {mu_hat:.2f}")
    print(f"\t\t R  = {R_hat:.2f}")
    
    return _update(params, {"mu": mu_hat, "R": R_hat})

def main(all_seeds) -> None: 
    # generate data
    params, data = DGP()
    # estimate model using Rust NFP algorithm
    nfp_Rust(params, data)
    # solve model analytically
    analytic_HM(params, data)
    # FIXME if there is time, implement Forward simulation
    return None 
   
if __name__=="__main__":

    # create output directories
    out_dir = os.path.join(settings['output_directory'], 'homework_3')
    _sub =['figures', 'tables', 'data']
    for fol in _sub:
        url = os.path.join(out_dir, fol)
        os.makedirs(url, exist_ok=True)

    # run estimation steps
    main()

