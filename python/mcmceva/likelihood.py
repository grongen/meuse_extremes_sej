import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.special import gamma
from scipy.stats import gumbel_r, lognorm
from scipy.special import gamma

def _beta_pdf_vec(x, left=-0.5, right=0.5, alpha=6, beta=9):
    out_of_bounds = (x < left) | (x > right)
    in_bounds = ~out_of_bounds

    pdf = np.zeros(len(x))
    
    z = x[in_bounds] - left / (right - left)
    B = gamma(alpha) * gamma(beta) / gamma(alpha + beta)
    pdf[in_bounds] = z**(alpha - 1) * (1 - z) ** (beta - 1) / B

    return pdf
    

def beta_pdf(x, left=-0.5, right=0.5, alpha=6, beta=9):
    if np.size(x) > 1:
        return _beta_pdf_vec(x, left, right, alpha, beta)    
    
    if x < left or x > right:
        return 0.0
    
    z = x - left / (right - left)
    B = gamma(alpha) * gamma(beta) / gamma(alpha + beta)
    pdf = z**(alpha - 1) * (1 - z) ** (beta - 1) / B
    
    return pdf


class GEV:
    def __init__(self) -> None:
        self.params = ['loc', 'scale', 'c']
        
    @staticmethod
    def uninformed_prior(theta):
        
        loc, scale, c = theta
        if (c - 10 < c < 10) and (scale > 0):
            return 0.0
        else:
            return np.NINF

    @staticmethod
    def _informed_prior_vec(theta):

        lbound = 1e-10
        loc, scale, c = theta[:, 0], theta[:, 1], theta[:, 2]
        idx = ((-0.5 < c) & (c < 0.5)) & (scale > lbound) & (loc > 0)
        
        logp = np.full(c.shape, np.NINF)
        logp[idx] = np.log(beta_pdf(c[idx]))

        return logp
    
    @staticmethod
    def informed_prior(theta):
        if np.ndim(theta) > 1:
            return GEV._informed_prior_vec(theta)
        
        loc, scale, c = theta

        lbound = 1e-10
        
        if not ((-0.5 < c < 0.5) and (scale > lbound) and (loc > 0)):
            return np.NINF
        else:
            loc, scale, c = theta
            
            return np.log(beta_pdf(c))
    
    def jeffreys_scale_prior(theta):
        loc, scale, c = theta
        # Scale bounds
        lbound, ubound = 1e-3, 1e6
        N = np.log(ubound) - np.log(lbound)
        return np.log(1/(scale * N))


    @staticmethod
    def _logp_c0(z):
        return np.exp(-np.exp(-z)) * np.exp(-z)

    @staticmethod
    def _logp_cnon0(c, z):
        part = 1 - c * z
        return np.exp(-(part ** (1 / c))) * part ** (1 / c - 1)


    @classmethod
    def _log_likelihood_obs_vec(cls, theta, obs):
    
        loc, scale, c = theta[:, 0], theta[:, 1], theta[:, 2]
        # Standardize
        z = (obs[None, :] - loc[:, None]) / scale[:, None]
        # Calculate probability density
        logp = np.zeros_like(c)
        idxgumbel = c == 0.0
        if idxgumbel.any():
            logp[idxgumbel] = np.log(1.0 / scale[idxgumbel] * cls._logp_c0(z[idxgumbel])).sum(axis=1)

        if (~idxgumbel).any():
            # If z is sorted, only check the first and last value
            nongumbel = np.where(~idxgumbel)[0]
            invalid = (c[nongumbel, None] * z[nongumbel]).max(1) > 1
            logp[nongumbel[invalid]] = np.NINF

            # This will cause an underflow
            # The last value in part is the smallest (of z is sorted)
            valid = nongumbel[~invalid]
            p = 1.0 / scale[valid, None] * cls._logp_cnon0(c[valid, None], z[valid])
            invalid2 = p.min(1) <= 0.0
            logp[valid[invalid2]] = np.NINF
            
            valid2 = valid[~invalid2]
            logp[valid2] = np.log(p[~invalid2]).sum(1)

            # Convert potential nans to np.NINF
            logp[~np.isfinite(logp)] = np.NINF

        return logp

    @classmethod
    def log_likelihood_obs(cls, theta, obs):
        if np.ndim(theta) > 1:
            return cls._log_likelihood_obs_vec(theta, obs)
    
        loc, scale, c = theta

        # Standardize
        z = (obs - loc) / scale

        # Calculate probability density
        if c == 0.0:
            logp = np.log(1.0 / scale * cls._logp_c0(z)).sum()

        else:
            # If z is sorted, only check the first and last value
            if ((c * z) > 1).any():
                return np.NINF
            # This will cause an underflow
            # The last value in part is the smallest (of z is sorted)
            else:
                p = 1.0 / scale * cls._logp_cnon0(c, z)
                if (p <= 0.0).any():
                    return np.NINF
                
                logp = np.log(p).sum()

        return logp if np.isfinite(logp) else np.NINF
    
    @classmethod
    def _log_likelihood_ej_vec(cls, theta, ej):

        loc, scale, c = theta[:, 0], theta[:, 1], theta[:, 2]
        logp_ej_tot = np.zeros_like(c)

        for p, dist in ej.items():
            x = cls.ppf(theta, p)
            logp_ej_tot += dist.logpdf(x)

            vals = dist.logpdf(x)

            if np.isnan(vals).any():
                raise ValueError()

        return logp_ej_tot

    @classmethod
    def log_likelihood_ej(cls, theta, ej):
        if np.ndim(theta) > 1:
            return cls._log_likelihood_ej_vec(theta, ej)
    
        logp_ej_tot = 0
        for p, dist in ej.items():
            x = cls.ppf(theta, p)
            logp_ej_tot += dist.logpdf(x)

        return logp_ej_tot

    @classmethod
    def log_posterior_obs_only(cls, theta, obs):
        return cls.informed_prior(theta) + cls.log_likelihood(theta, obs)
    
    @classmethod
    def log_posterior_ej_only(cls, theta, ej):
        logp_prior, logp_ej = cls.informed_prior(theta), cls.log_likelihood_ej(theta, ej)
        return logp_prior + logp_ej
    


    @classmethod
    def log_posterior_both(cls, theta, obs, ej, fobs):
            
        Nobs = len(obs)

        # if fobs is not None:
        fobs = fobs / Nobs
        # else:
            # fobs = 1.0

        # if fobs > 1:
        #     fej = 1/ fobs
        #     fobs = 1
        # else:
        #     fej = 1

        return cls.informed_prior(theta) + fobs * cls.log_likelihood(theta, obs) + cls.log_likelihood_ej(theta, ej)
    
 
    @classmethod
    def _log_posterior_infer_vec(cls, theta, obs=None, ej=None, fej=None):
            
        logl = cls._informed_prior_vec(theta)
        idx = np.isfinite(logl)
        if not idx.any():
            return logl

        if obs is not None:
            logl[idx] += cls._log_likelihood_obs_vec(theta[idx], obs)
        
        if ej is not None:
            # Give ej a weight equal to 10 per cent of the observations
            logl[idx] += cls._log_likelihood_ej_vec(theta[idx], ej) 

        return logl

    @classmethod
    def log_posterior_infer(cls, theta, obs=None, ej=None, fej=None):
        if np.ndim(theta) > 1:
            return GEV._log_posterior_infer_vec(theta, obs, ej)
            
        logl = cls.informed_prior(theta)

        if np.isinf(logl):
            return logl
        
        if obs is not None:
            logl += cls.log_likelihood_obs(theta, obs)
        
        if ej is not None:
            # Give ej a weight equal to 10 per cent of the observations
            if fej is None: fej = 1
            logl += fej * cls.log_likelihood_ej(theta, ej) 

        return logl
    
    @classmethod
    def log_posterior_infer_vec(cls, theta, obs=None, ej=None, fej=None):
            
        logl = cls.informed_prior_vec(theta)
        
        if obs is not None:
            logl += cls.log_likelihood_obs_vec(theta, obs)
        
        if ej is not None:
            # Give ej a weight equal to 10 per cent of the observations
            if fej is None: fej = 1
            logl += fej * cls.log_likelihood_ej(theta, ej) 

        return logl


    @staticmethod
    def get_x0_bounds(obs):
        return dict(x0=(obs.mean(), obs.std(), 0.01), bounds=((1e-6, np.inf), (1e-6, np.inf), (-10, 10)))
    
    @classmethod
    def get_mcmc_x0(cls, *args, **kwargs):
        return _mcmc_x0(cls, *args, **kwargs)
    
    @classmethod
    def get_mcmc_x0_ej(cls, obs):
        return _mcmc_x0(cls, obs)
    
    @staticmethod    
    def _ppf_vec(theta, p):
        loc, scale, c = theta[:, 0], theta[:, 1], theta[:, 2]

        gumbel_idx = c == 0.0
        x = np.zeros_like(c)
        if gumbel_idx.any():
            x[gumbel_idx] = loc[gumbel_idx] - scale[gumbel_idx] * np.log(-np.log(1 - p)),
        if (~gumbel_idx).any():
            x[~gumbel_idx] = loc[~gumbel_idx] + (scale[~gumbel_idx] / -c[~gumbel_idx]) * ((-np.log(1 - p)) ** c[~gumbel_idx] - 1)
        return x        

    @staticmethod    
    def ppf(theta, p):
        if np.ndim(theta) > 1:
            return GEV._ppf_vec(theta, p)
    
        loc, scale, c = theta
        if c == 0.0:
            x = loc - scale * np.log(-np.log(1 - p))
        else:
            x = loc + (scale / -c) * ((-np.log(1 - p)) ** c - 1)
        return x
    


def _get_nnl(func):
    return lambda *args: -np.nan_to_num(func(*args), neginf=-1e12)
            

class GEVObsEJSampler:
    """This class helps to find valid starting point for the MCMC sampling. In most
    cases using the mean and standard deviations of the observations as location and
    scale parameter works fine (with a close to zero tail shape 'c'). However, with
    expert judgments that greatly deviate from the observations, such an initial guess
    might give an -inf likelihood, which obstructs finding a maximum likelihood estimate
    to start from. In this case, random variations are made within a suitable range
    to still find a good starting point."""
    def __init__(self, dist, obs: np.ndarray, ej: dict, fej: float, log_prob_fn: callable, param_variation_scale: np.ndarray):
        
        self.obs = obs
        self.fej = fej
        self.ej = ej
        self.xscale = param_variation_scale
        self.log_prob_fn = log_prob_fn
        self.dist = dist
        
        # Create a tuple with arguments for the log_posterior_infer function
        self.args = (obs, ej, fej)
        
    def _get_theta_obs(self):
        _loc0 = self.obs.mean()
        _scale0 = self.obs.std()
        return _loc0, _scale0, 0.0
    
    def _interp_estimate(self, p, quantile):
        return np.interp(quantile, list(self.ej[p].estimates.keys()), list(self.ej[p].estimates.values()))

    def _get_theta_ej(self):
        """Function to get a valid starting point for inference for expert judgments only.
        This is a gumbel dist fitted to the median estimates"""
        ydata = list(self.ej.keys())
        xdata = [self._interp_estimate(p=p, quantile=0.5) for p in ydata]
        def func(xdata, loc, scale):
            return 1 - gumbel_r(loc, scale).cdf(xdata)            
        _loc0, _scale0 = curve_fit(func, xdata, ydata, p0=(xdata[0], 10))[0]
        return _loc0, _scale0, 0.0
    
    def _get_theta_both(self):
        """Function to get a valid starting point for inference for both observations and expert judgments.
        This is a gumbel dist fitted to the min observation and the 1000 ARI 95th estimate"""
        ydata = [0.63, 0.001]
        xdata = [np.min(self.obs), self._interp_estimate(p=0.001, quantile=0.95)]
        def func(xdata, loc, scale):
            return 1 - gumbel_r(loc, scale).cdf(xdata)            
        _loc0, _scale0 = curve_fit(func, xdata, ydata, p0=(xdata[0], 10))[0]
        return _loc0, _scale0, 0.0

    def _get_ml_valid(self):
        
        # Get an initial guess based on the observations
        if self.obs is not None:
            try:
                # x0 = self._try_random_variations(self._get_theta_obs(), 1, minp=0.0001).squeeze()
                x0 = self._get_theta_obs()
            except:
                # x0 = self._try_random_variations(self._get_theta_both(), 1, minp=0.0001).squeeze()            
                x0 = self._get_theta_both()

        elif self.obs is None:
            # x0 = self._try_random_variations(self._get_theta_ej(), 1, minp=0.0001).squeeze()
            x0 = self._get_theta_ej()

        # Find the maximum likelihood estimates
        nll = _get_nnl(self.log_prob_fn)    
        optimum = minimize(nll, args=self.args, x0=x0)

        return optimum.x

    def _try_random_variations(self, x0, N, minp=0.1):
        """Generates N random variations from a starting point x0.
        infeasible combinations are ignored. minp states what part
        needs to be accepted (at least). If not, the starting point
        is probably bad.
        """
        pos = []
        k = 0
        while len(pos) < N:
            k += 1
            rand = np.random.randn(len(x0))
            potential = x0 + np.array(self.xscale)[:len(x0)] * rand
            if not np.isinf(self.log_prob_fn(potential, obs=self.obs, ej=self.ej, fej=self.fej)):
                pos.append(potential)

            if k > N / minp:
                raise ValueError(f"Less than {minp * 100:.2f}% percent is effective. Check scale.")
            
        return np.array(pos)

    def initialize(self, nwalkers):

        # Get a maximum likelihood estimate
        opt = self._get_ml_valid()
        # Get a number of valid starting points for the walkers
        pos = self._try_random_variations(opt, nwalkers, minp=0.01)

        return pos
        

