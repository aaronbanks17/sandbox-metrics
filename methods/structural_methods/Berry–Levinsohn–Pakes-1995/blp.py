import pandas as pd 
import os 
import dataclasses
import jax.numpy as jnp 
import chex 
import jax
import matplotlib.pyplot as plt
from functools import partial
import seaborn as sns 
from methods.settings import settings 
from typing import Tuple, Callable
from jax import lax 
import copy
import jax.scipy.optimize as jsp
import numpy as np

@chex.dataclass
class MarketData:
    """
    Discrete Choice model of a market with differentiated products.
    """
    # exogenous variables
    X         : jnp.ndarray  # product characteristics
    w         : jnp.ndarray  # product-level cost shifter
    z         : jnp.ndarray  # product-market-level cost shifter 
    mc        : jnp.ndarray  # marginal cost 
    Z_blp     : jnp.ndarray  # BLP demand side instruments
    # parameters
    beta      : jnp.ndarray  #  marginal utility of characteristics 
    gamma     : jnp.ndarray  # additional marinal cost of prod/mkt shifters
    alpha     : jnp.ndarray  # mean price sensitivity
    sig_alpha : jnp.ndarray  # price sensitivity dispersion 
    xi        : jnp.ndarray  # product-market FE
    eta       : jnp.ndarray  # supply side shock
    v         : jnp.ndarray  # consumer-level price sensitivity
    # endogenous variables
    price     : jnp.ndarray  # price vector
    share     : jnp.ndarray  # equilibrium shares
    delta     : jnp.ndarray  # delta from BLP share inversion
    # IDs
    prod_ids  : jnp.ndarray  # product (firm) IDs
    mkt_ids   : jnp.ndarray  # market IDs

@chex.dataclass(frozen=True)
class MarketDefinition:
    """
    Constants characterizing market setting
    """
    num_firms : int
    num_consumers : int
    num_markets : int 

def _update(controls: MarketData, updates: dict) -> MarketData:
    """
    Update MarketData instance

    ARGS:
        instance (MarketData): The existing MarketData instance.
        updates (dict): A dictionary where keys are attribute names and values are new values.

    RETURNS:
        MarketData: A _new_ MarketData instance with updated values.
    """
    # ensure updates are all valid
    valid_updates = {k: v for k, v in updates.items() if hasattr(controls, k)}
    return dataclasses.replace(controls, **valid_updates)

def blp_instruments(X:jnp.ndarray, mkt_ids:jnp.ndarray, markets:int=100) -> jnp.ndarray:
    r"""
    Construct BLP Demand-side instruments 
        Z_{j,k} = \sum_{h \in m} X_{h,k} - X_{j,k}
    """

    # ensure IDs start at zero
    adj_ids = mkt_ids - 1 
    # fetch characteristics shape
    X = X[:, 1:] # only use non-constant characteristics
    N, K = X.shape
    # first, sum across characteristics k in each market
    market_sum = []
    for k in range(K):
        col_sum = jnp.bincount(adj_ids, weights=X[:, k], length=markets)
        market_sum.append(col_sum)
    # stack to get shape markets x products
    market_sum = jnp.stack(market_sum, axis=1)
    # subtract out own characteristic value 
    Z = market_sum[adj_ids, :] - X
    return Z

def DGP(consumers:int=1000, markets:int=100, products:int=3, seed:int=17) -> Tuple[MarketData, MarketDefinition]:
    r"""
    Simulate market data for the setting

    ARGS:
        consumers (int, optional): number of consumers per market
        markets (int, optional): number of markets
        products (int, optional): number of products per market (does not include outside option)
        seed (int, optional): seed for PRNG

    RETURNS:
        Setting (dataclass): simulated data, setting eqm objects to nan

    MODEL: 
        U_ijm = X_jm * \beta - \alpha_i * p_jm + \x_jm + \epsilon_ijm
            - \alpha_i = \alpha + \sigma_\alpha * v_ip
        \alpha_i = \alpha + \sigma_alpha * v_i

        MC_jm = \gamma_1 + \gamma_2 * W_j + \gamma_3 * Z_jm + \eta_jm 
    """

    # PRNG keys
    key = jax.random.key(seed)
    subkey1, subkey2, subkey3, subkey4, subkey5, subkey6, subkey7 = jax.random.split(key, 7)

    # draw product characteristics and demand shocks
    X_1 = jnp.ones(shape=(markets * products, ))
    X_2 = jax.random.uniform(subkey1, shape=(markets * products, ))
    X_3 = jax.random.normal(subkey2, shape=(markets * products, ))
    X   = jnp.array([X_1, X_2, X_3]).T
    xi  = jax.random.normal(subkey3, shape=(products * markets))
    # draw cost shifters
    z   = jax.random.lognormal(subkey5, sigma=1, shape=(markets * products))
    w   = jax.random.lognormal(subkey4, sigma=1, shape=(products))
    w   = jnp.tile(w, (markets,)) # product shocks are common across all markets
    # draw consumer-level preferences and cost shock
    v   = jax.random.lognormal(subkey6, sigma=1, shape=(markets * consumers))
    eta = jax.random.lognormal(subkey7, sigma=1, shape=(markets * products))
    # true parameters
    beta      = jnp.array([5.0, 1.0, 1.0])
    alpha     = jnp.array(1.0)
    sig_alpha = jnp.array(1.0)
    gamma     = jnp.array([2.0, 1.0, 1.0])
    theta     = jnp.concatenate([beta, jnp.array([alpha]), jnp.array([sig_alpha])])
    # compute marginal costs
    shifters  = jnp.column_stack([jnp.ones_like(w), w, z])
    mc        = (shifters @ gamma + eta).ravel()
    # endogeous/estimated vectors
    price     = jnp.empty(shape=(products * markets, ))
    share     = jnp.empty(shape=(products * markets, ))
    delta     = jnp.empty(shape=(products * markets, ))
    # create ids 
    prod_ids  = jnp.arange(0, products)
    prod_ids  = jnp.tile(prod_ids, markets)
    mkt_ids   = jnp.arange(1, markets + 1)
    mkt_ids   = jnp.tile(mkt_ids, products)
    # demand side instruments
    Z_blp = blp_instruments(X, mkt_ids, markets)

    data_dict = dict(X=X, w=w,z=z,mc=mc, beta=beta, gamma=gamma, alpha=alpha,
        sig_alpha=sig_alpha, xi=xi, eta=eta,v=v, price=price, share=share,
        delta=delta, prod_ids=prod_ids, mkt_ids=mkt_ids, Z_blp=Z_blp
    )
    const_dict = dict(num_firms=products, num_consumers=consumers, num_markets=markets)

    return MarketData(**data_dict), MarketDefinition(**const_dict)

def compute_delta(p:jnp.ndarray, controls:MarketData) -> jnp.ndarray:
    """
    Given a price, compute the product-market contribution to consumer utility
    """
    with jax.ensure_compile_time_eval():
        X = controls.X 

    # compute delta with currrent controls
    return X @ controls.beta - controls.alpha * p + controls.xi 

@partial(jax.jit, static_argnums=2)
def consumer_utility(p:jnp.ndarray, controls:MarketData, constants:MarketDefinition):
    """
    Compute consumer's indirect utilities (inside goods)

    ARGS:

    RETURNS:
    """
    with jax.ensure_compile_time_eval():
        v = controls.v
        M = constants.num_markets
        N = constants.num_consumers
        J = constants.num_firms 

    # construct product-market heterogeneity (delta_jm)
    delta = controls.delta.reshape(M, J)[:, :, None]
    # contruct indiviual level heterogeneity (mu_ijm)
    mu = -1.0 * (controls.sig_alpha * v).reshape(M, 1, N) * p.reshape(M, J, 1)
    # compute utility 
    utility = delta + mu
    return utility

@partial(jax.jit, static_argnums=2)
def choice_probability(p:jnp.ndarray, controls:MarketData, constants:MarketDefinition) -> jnp.ndarray:
    """
    Compute consumer's choice probabilities (inside goods)

    ARGS:

    RETURNS:
    """
    utility  = consumer_utility(p, controls, constants)
    exp_util = jnp.exp(utility)
    exp_sum  = 1 + jnp.sum(exp_util, axis=1, keepdims=True) # since outside option excluded
    return exp_util / exp_sum 

@partial(jax.jit, static_argnums=2)
def demand(p:jnp.ndarray, controls:MarketData, constants:MarketDefinition) -> jnp.ndarray:
    """
    Compute product-market shares

    ARGS:

    RETURNS:
    """
    choice_prob = choice_probability(p, controls, constants)
    return jnp.mean(choice_prob, axis=2).ravel() # average over consumers

@partial(jax.jit, static_argnums=2)
def blp_derivative(p:jnp.ndarray, controls:MarketData, constants:MarketDefinition) -> jnp.ndarray:
    """
    Compute simulated form of choice probability ijm w.r.t price jm
    given in Berry, Levinsohn, and Pakes (1995) Eq. (6.9a)

    ARGS:

    RETURNS:
    """
    with jax.ensure_compile_time_eval():
        M = constants.num_markets
        N = constants.num_consumers
        v = controls.v

    # compute choice probabilities
    s_ijm = choice_probability(p, controls, constants)
    # compute dmu/dp_jm (just  alpha_i - alpha)
    dmu_dp = -1.0 * (controls.sig_alpha * v).reshape(M, -1, N)
    # compute integral argument
    val = s_ijm * (1.0 - s_ijm) * dmu_dp
    # average over consumers 
    return val.mean(axis=2).ravel()

@partial(jax.jit, static_argnums=2)
def lerner_fp(p:jnp.ndarray, controls:MarketData, constants:MarketDefinition) -> jnp.ndarray:
    """
    Compute current value of Lerner equilibrium fixed point 
    condition for a given price vector

    ARGS:

    RETURNS:
    """

    # compute product-market shares
    s_j   = demand(p, controls, constants)
    # compute share derivatives
    ds_dp = blp_derivative(p, controls, constants)
    # compute fixed-point condition from FOC
    return controls.mc - (s_j / ds_dp)

def solve_prices_fp(controls:MarketData, constants:MarketDefinition,
                    tol:float=1e-7, maxiter=10000, damp:float=1.0,
                    verbose:bool=True) -> MarketData:
    """
    Solve for prices using fixed point iteration
    """

    # fix constants args of funcs
    _fp  = partial(lerner_fp, constants=constants)
    _der = partial(blp_derivative, constants=constants)

    # creat logs
    if verbose:
        header = f"{'ITERATION':>10} {'MEAN PRICE':>15} {'ERROR':>15} {'MEAN DERIVATIVE':>25} {'POS. DERIVATIVE':>15}"
        print('-' * len(header)) 
        print(header)
        print('-' * len(header)) 

    # initialize prices and controls
    _p        = jnp.ones_like(controls.price)
    _controls = copy.deepcopy(controls)
    for _ in range(maxiter):
        # Compute price update using Lerner fixed point
        p_out = _fp(_p, _controls)
        p_out = damp * p_out + (1-damp)* _p
        der   = _der(p_out, _controls)
        # Compute metrics
        err = jnp.max(jnp.abs(p_out - _p))
        mean_p = jnp.mean(p_out)
        mean_d = jnp.mean(der)
        posi_d = (der > 0).any()
        # log
        if verbose:
            print(
                f"iter: {_:4d} | "
                f"p: {mean_p:12.6f} | "
                f"err: {err:12.6e} | "
                f"d: {mean_d:15.6f} | "
                f"pos: {str(posi_d):>5}"
            )
        # update delta/controls with estimated price
        new_delta = compute_delta(p_out, _controls)
        _controls  = _update(_controls, {'delta': new_delta})
        # asses update
        _p = p_out
        if err < tol:
            print(f"[   PRICE SOLVER    ] Convgered with error {err:12.6e}")
            break

    # update market data
    new_price   = _p
    new_share   = demand(new_price, _controls, constants)
    update_dict = dict(price=new_price, share=new_share)

    return  _update(controls, update_dict)

@partial(jax.jit, static_argnums=1)
def delta_contraction(controls:MarketData, constants:MarketDefinition) -> jnp.ndarray:
    """
    Invert Market shares to recover value of product-market
    contribution to utility (delta_jm)

    ARGS:

    RETURNS:
    """
    with jax.ensure_compile_time_eval():
        obs_share = controls.share
        obs_price = controls.price

    share_hat     = demand(obs_price, controls, constants)
    log_share_hat = jnp.log(share_hat)
    return controls.delta + jnp.log(obs_share) - log_share_hat

@jax.jit
def construct_instruments(controls:MarketData) -> jnp.ndarray:
    """
    Construct matrix of "BLP" instruments and cost shifters
    """
    with jax.ensure_compile_time_eval():
        X = controls.X 
        W = controls.w
        z = controls.z
        Z_blp = controls.Z_blp
    # instruments
    return jnp.hstack([X, Z_blp, W.reshape(-1,1), z.reshape(-1, 1)])

@jax.jit
def iv_reg(controls:MarketData) -> MarketData:
    """
    Estimate theta_2 = (beta, alpha) via 2SLS
    """
    with jax.ensure_compile_time_eval():
        X = controls.X 

    # collect instruments
    inst = construct_instruments(controls)
    # collect characteristics
    x = jnp.hstack([X, controls.price.reshape(-1,1)])
    # compute projection matrix
    inv_inst  = jnp.linalg.inv(inst.T @ inst)
    proj_inst = inst @ inv_inst @ inst.T
    # compute sandwhich formula
    theta = jnp.linalg.inv(x.T @ proj_inst @ x) @ (x.T @ proj_inst @ controls.delta)
    # update controls
    beta, alpha = theta[:3], theta[3]
    return _update(controls, {'beta': beta, 'alpha': alpha})

@partial(jax.jit, static_argnums=(1, 2, 3))
def inner_loop(controls:MarketData, constants:MarketDefinition, 
               maxiter:int=1000, tol:float=10e-15
               ) -> Tuple[MarketData, jnp.ndarray, int]:
    """
    Solve for delta in the nested fixed-point 
    procedure via contraction mapping.
    """

    with jax.ensure_compile_time_eval():
        obs_price = controls.price
        X = controls.X

    def _body_fun(state, _):
        # extract items from state
        _controls, _old, _conv = state
        # compute new delta
        _new = delta_contraction(_controls, constants)
        # compute error
        out_err = jnp.max(jnp.abs(_new - _old))
        # update the convergence flag (JAX boolean object)
        out_conv = jnp.logical_or(_conv, out_err < tol)
        # if converged, keep state; otherwise, update via jax.lax.cond
        out_controls = jax.lax.cond(
            out_conv,
            lambda _: _controls,
            lambda _: _update(_controls, {"delta": _new}),
            operand=None,
        )
        out_delta = jax.lax.cond(
            out_conv,
            lambda _: _old,
            lambda _: _new,
            operand=None,
        )
        return (out_controls, out_delta, out_conv), out_err

    # inner loop initialization info
    init_delta = compute_delta(obs_price, controls)
    init_state = (controls, init_delta, jnp.array(False))
    iterations = jnp.arange(maxiter)
    # run contraction map iteration
    _state, _err = lax.scan(_body_fun, init_state, iterations)
    _controls, _delta, _conv = _state
    _err = _err[-1]

    # 2SLS estimation of (beta, alpha)
    _controls = iv_reg(_controls)
    resid = _controls.delta - X @ _controls.beta - _controls.alpha * obs_price
    # update the structural error
    _controls = _update(_controls, {'xi': resid})

    return _controls, _err, _conv

@partial(jax.jit, static_argnums=(4,5))
def _objective_gmm(sig_alpha:float, inst:jnp.ndarray, weights:jnp.ndarray,
             controls:MarketData, constants:MarketDefinition,
             inner_kwargs:Tuple=(('maxiter', 1000), ('tol', 10e-15))
             ) -> float:
    """
    Compute the GMM objective Q(sigma_alpha) = xi'(Z W Z')xi
    """
    kwargs = dict(inner_kwargs)
    # update controls with value of \sigma_alpha
    _controls = _update(controls, {"sig_alpha": sig_alpha})
    # solve inner loop and extract controls from output
    output = inner_loop(_controls, constants, maxiter=kwargs['maxiter'], tol=kwargs['tol'])   
    _controls = output[0]     
    # compute moment  \xi' Z W Z' \xiw
    g = inst.T @ _controls.xi
    # compute objective
    obj_val = g.T @ weights @ g 
    return obj_val

partial(jax.jit, static_argnums=(1,2,3))
def multi_start(obj_fun:Callable[[jnp.ndarray], jnp.ndarray],
                lower:float=0.1, upper:float=10.0,
                step:float=0.1) -> float:
    """
    Perform a grid search over [lower, upper] in increments of `step`
    to find the best (lowest) value of the objective function obj_fun(sig_alpha).
    """
    grid = jnp.arange(lower, upper + step, step)
    obj_vals = jax.vmap(obj_fun)(grid)
    best_idx = jnp.argmin(obj_vals)
    best_sa  = grid[best_idx]
    return jnp.array([best_sa])

def outer_loop(controls:MarketData, constants:MarketDefinition, 
               start_kwargs:Tuple=(('lower', 0.1), ('upper',10.0), ('step', 0.1)),
               inner_kwargs:Tuple=(('maxiter', 1000), ('tol', 10e-14)),
               ridge:float=1e-6, multi:bool=True) -> MarketData:
    """
    Two-step GMM estimation.
    """

    # unpack kwargs and set mult-start search func
    kwargs   = dict(inner_kwargs)
    s_kwargs = dict(start_kwargs)
    search   = partial(multi_start, **s_kwargs)

    # -- STEP 1 GMM -- #

    # compute initial weighting matrix
    inst     = construct_instruments(controls)
    weights1 = jnp.linalg.inv(inst.T @ inst)  # FIXME why are we taking an inverse here?
    # set step 1 objective as func Q(\sigma_alpha)
    _Q1 = lambda sig_alpha: _objective_gmm(sig_alpha, inst, weights1, controls, constants, inner_kwargs=inner_kwargs)
    # first step minimization
    # sa_init = jnp.array([controls.sig_alpha])
    sa_init = search(_Q1) if multi else jnp.array([controls.sig_alpha])
    print(f"[OUTER LOOP - STEP 1] start: {sa_init}, multi-start: {multi}")
    res1 = jsp.minimize(fun=_Q1, x0=sa_init, method="BFGS")
    print(f"[OUTER LOOP - STEP 1] sigma_alpha = {res1.x[0]}")
    # update controls after step 1 minimization
    _controls = _update(controls, {"sig_alpha": res1.x[0]})
    _controls, _err, _conv = inner_loop(_controls, constants, kwargs['maxiter'], tol=kwargs['tol'])
    print(f"[INNER LOOP - STEP 1] converged with error {_err.item():.6e}")
    
    # -- STEP 2 GMM -- #

    # compute optimal weighting matrix given \hat(\sigma_alpha)
    xi     = _controls.xi.reshape(-1, 1)
    moment = (inst.T @ (xi @ xi.T) @ inst) / xi.shape[0]
    moment += ridge * jnp.eye(moment.shape[0]) # for stability in the inversion
    weights_opt = jnp.linalg.inv(moment) 
    # set step 2 objective as func Q(\sigma_alpha)
    _Q2 = partial(_objective_gmm, inst=inst, weights=weights_opt, 
                    controls=_controls, constants=constants,
                    inner_kwargs=inner_kwargs)
    # Second-step optimization
    sa_init = search(_Q2) if multi else jnp.array([res1.x[0]])
    print(f"[OUTER LOOP - STEP 2] start: {sa_init}")
    res2    = jsp.minimize(fun=_Q2, x0=sa_init, method="BFGS")
    print(f"[OUTER LOOP - STEP 2] sigma_alpha = {res2.x[0]:.4f}")
    # step 2 minimization
    _controls = _update(_controls, {"sig_alpha": res2.x[0]})
    _controls, _err, _conv = inner_loop(_controls, constants, kwargs['maxiter'], tol=kwargs['tol'])
    start_val = _Q2(sa_init)
    end_val = _Q2(jnp.array([_controls.sig_alpha]))
    if not _conv:
        raise ValueError(f"[INNER LOOP - STEP 2] Did not converge; achieved error {_err.item():.4e}")
    else:
        print(f"[INNER LOOP - STEP 2] converged with error {_err.item():.6e}")
        print(f"[   GMM OBJECTIVE   ] Routine finished with start, end values: ({start_val:.4e}, {end_val:.4e})\n")

    return _controls, start_val, end_val

def compute_mc(p:jnp.ndarray, controls:MarketData, constants:MarketDefinition) -> MarketData:
    """compute marginal costs given parameters"""

    # compute product-market shares
    s_j   = demand(p, controls, constants)
    # compute share derivatives
    ds_dp = blp_derivative(p, controls, constants)
    # return mc
    mc = p + s_j /  ds_dp
    return _update(controls, {'mc': mc})

def compute_avg_sub(controls:MarketData, constants:MarketDefinition) -> jnp.ndarray:
    """
    Compute the average substitution patterns across all markets

    ARGS:
        controls (MarketData): Market data containing product characteristics, prices, etc.
        constants (MarketDefinition): Market structure constants.

    RETURNS:
        jnp.ndarray: Average substitution matrix (J x J), where J is the number of products per market.
    """
    with jax.ensure_compile_time_eval():
        M, J = constants.num_markets, constants.num_firms

    # compute jacobian
    ds_dp = jax.jacfwd(demand, argnums=0)(controls.price, controls, constants)
    # reshape into (M, J, M, J) to segment markets correctly
    ds_dp = ds_dp.reshape(M, J, M, J)
    # extract only the within-market blocks (ignore cross-market derivatives)
    ds_dp_marketwise = jnp.einsum('mjmk->mjk', ds_dp) 
    # compute MRS for each market
    mrs_matrix = -ds_dp_marketwise / jnp.expand_dims(jnp.diagonal(ds_dp_marketwise, axis1=1, axis2=2), axis=2)
    # Compute the average across all markets
    avg_mrs = jnp.mean(mrs_matrix, axis=0)
    return avg_mrs

def gmm_distribution(lower:int, upper:int, true_vals:bool=False, scale_noise:float=0.1,
        dgp_kwargs:dict={'consumers':1000, 'markets':100, 'products':3},
        price_solver_kwargs:dict={'maxiter':1000, 'tol':10e-15, 'damp':0.7, 'verbose':False},
        outer_loop_kwargs:dict={'inner_kwargs':(('maxiter', 1000), ('tol', 10e-14)), 'ridge':1e-6, 'multi':False},
        suffix:str='') -> None:
    """
    Estimate BLP over a grid of PRNG seeds and return a DataFrame with 
    the estimated sig_alpha and beta coefficients (b1, b2, b3) for each run.
    """
    results = []
    
    for seed in range(lower, upper):
        print(f"---------------- SEED {seed} ----------------")
        try:
            # simulate data for given seed
            controls, constants = DGP(seed=seed, **dgp_kwargs)
            # determine equilibrium ("observed") prices and shares 
            controls = solve_prices_fp(controls, constants, **price_solver_kwargs)
            if not true_vals:
                key      = jax.random.key(seed + 42)
                noise    = jax.random.normal(key, shape=controls.beta.shape) * scale_noise
                new_b    = controls.beta + noise
                new_a    =  controls.alpha + scale_noise  
                new_sa   = controls.sig_alpha + scale_noise
                new_xi   = jnp.zeros_like(controls.xi)
                controls = _update(
                    controls, {'beta': new_b, 'alpha': new_a, 'sig_alpha': new_sa, 'xi': new_xi}
                    )

            sig_alpha_start = controls.sig_alpha
            alpha_start = controls.alpha 

            # estimate (beta, alpha, sigma_alpha)
            controls, start, end = outer_loop(controls, constants, **outer_loop_kwargs)
            
            # extract estimated params
            sig_alpha_end = float(controls.sig_alpha)
            alpha_end     = float(controls.alpha)
            beta_vals     = [float(b) for b in controls.beta]
            mean_v        = float(jnp.mean(controls.v))
            
            results.append({
                'seed': seed,
                'sig_alpha_start':sig_alpha_start,
                'sig_alpha_end': sig_alpha_end,
                'b1': beta_vals[0],
                'b2': beta_vals[1],
                'b3': beta_vals[2],
                'alpha_start': alpha_start,
                'alpha_end': alpha_end,
                'mean_v': mean_v,
                'obj_start' : start,
                'obj_end': end
            })
            
        except Exception as e:
            print(f"Error encountered for seed {seed}: {e}")
            
    # collect results into dataframe and save
    df = pd.DataFrame(results)
    out_url = os.path.join(settings['output_directory'], f'homework_1/data/gmm_simulation_{suffix}.pkl')
    df.to_pickle(out_url)
    return None

def _trim_percentiles(series, lower=3, upper=95):
    low, high = np.percentile(series, [lower, upper])
    return series[(series >= low) & (series <= high)]

def plot_gmm_results(suffix:str="") -> None:
    """
    Plot resulting distribution of parameters
    from gmm_distribution
    """
    # Load the results dataframe from pickle.
    url = os.path.join(settings['output_directory'], f"homework_1/data/gmm_simulation_{suffix}.pkl")
    df = pd.read_pickle(url)

    df['alpha_end'] *= -1.0

    # Ensure the output directory exists.
    out_dir = os.path.join(settings['output_directory'], 'homework_1/figures')

    def _bin_width(data):
        # Freedman - Diaconis Rule
        q75, q25 = np.percentile(data.dropna(), [75, 25])  
        iqr = q75 - q25
        bin_width = 2 * iqr / np.cbrt(len(data))  
        if bin_width==0:
            return 10
        bins = int((data.max() - data.min()) / bin_width)
        max_bins = 100  
        bins = min(bins, max_bins)
        return bins if bins > 1 else 10  

    def _plot_histogram(data, beta_label, filename):
        trimmed_data = _trim_percentiles(data)
        num_bins = _bin_width(trimmed_data)  # Calculate optimal bins

        plt.figure(figsize=figsize_latex)
        plt.hist(trimmed_data, bins=num_bins, alpha=1, color=colors[0], label=beta_label, density=True)
        plt.axvline(trimmed_data.mean(), color='black', linestyle='dashed', linewidth=0.8)
        plt.xlabel('Estimated Value', fontsize=label_fontsize)
        plt.ylabel('Density', fontsize=label_fontsize)
        plt.title(fr'Distribution of Estimated {beta_label}', fontsize=title_fontsize)
        plt.tight_layout()
        sns.despine()
        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)

        out_url = os.path.join(out_dir, filename)
        plt.savefig(out_url, dpi=300, bbox_inches='tight')
        plt.close()

    # Define figure size and font sizes
    figsize_latex = (2.4, 2.2)
    label_fontsize = 7
    title_fontsize = 8
    ticks_fontsize = 6

    # Set color palette
    colors = sns.color_palette("Set2", 3)

    # --- Histograms of beta, alpha coeffs --- #
    _plot_histogram(df['b1'], r'$\beta_1$', f"gmm_simulation_beta1_{suffix}.pdf")
    _plot_histogram(df['b2'], r'$\beta_2$', f"gmm_simulation_beta2_{suffix}.pdf")
    _plot_histogram(df['b3'], r'$\beta_3$', f"gmm_simulation_beta3_{suffix}.pdf")

    _plot_histogram(df['alpha_end'], r'$\alpha$', f"gmm_simulation_alpha_{suffix}.pdf")
    _plot_histogram(df['sig_alpha_end'], r'$\sigma_{\alpha}$', f"gmm_simulation_sigma_alpha_{suffix}.pdf")

    df['random_coeff'] = -1 * (df['alpha_end'] + df['sig_alpha_end'] * df['mean_v'])
    _plot_histogram(df['random_coeff'], r'$\alpha_i$', f"gmm_simulation_alpha_rand_{suffix}.pdf")

    # --- Check Objective Convergence --- #
    
    # inidicate improper convergence
    df['invalid'] = (df['obj_end'] > df['obj_start']).astype(int)
    counts = df['invalid'].value_counts().sort_index()

    plt.figure()
    plt.bar(['True', 'False'], counts, alpha=1, color=colors[0]) 
    plt.xlabel(r'Final $Q(\theta)$ $<$ Initial $Q(\theta)$')
    plt.ylabel('Frequency')
    plt.title(r'GMM Routine Finished Properly')
    plt.tight_layout()
    sns.despine()

    out_url = os.path.join(out_dir, f"gmm_simulation_conv_{suffix}.pdf")
    plt.savefig(out_url)
    plt.close()
    
    return None


def main(all_seeds:bool=False):
    """Entry point for executing Homework 1"""
    # Simulate market
    controls, constants = DGP(consumers=1000, markets=100, seed=0)

    # Solve for equilibrium prices and shares
    controls = solve_prices_fp(controls, constants, maxiter=10000, tol=10e-15, damp=0.7)

    # Run GMM estimation
    kwargs = (('maxiter', 1000), ('tol', 10e-14))
    s_kwargs = (('lower', 0.0), ('upper',5.0), ('step', 0.1))
    controls, _, _ = outer_loop(controls, constants,
                                inner_kwargs=kwargs, start_kwargs=s_kwargs,
                                ridge=0.0, multi=False)
    print(f"Beta estimates: {controls.beta}")
    print(f"alpha estimate: {-1 * controls.alpha}")
    print(f"sigma_alpha estimate: {controls.sig_alpha}")
    if all_seeds:
        gmm_distribution(lower=0, upper=101)

if __name__ == "__main__":

    # create output directories
    out_dir = os.path.join(settings['output_directory'], 'BLP')
    _sub =['figures', 'tables', 'data']
    for fol in _sub:
        url = os.path.join(out_dir, fol)
        os.makedirs(url, exist_ok=True)

    # set float type 
    jax.config.update("jax_enable_x64", True)
    main()
