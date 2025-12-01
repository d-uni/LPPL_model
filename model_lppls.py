import numpy as np
import pandas as pd
from scipy.optimize import minimize


class ModelLPPLS:
    """
    Log-Periodic Power Law Singularity (LPPLS) model for bubble detection.

    This class fits the LPPLS function to a price time series using
    non-linear least squares optimization.

    Parameters
    ----------
    t : array-like
        Time points (e.g., in years).
    p : array-like
        Observed price time series (must be positive).

    Attributes
    ----------
    t : np.ndarray
        Time grid.
    p : np.ndarray
        Observed prices.
    logp : np.ndarray
        Log of observed prices.
    params : dict or None
        Dictionary of fitted parameters {A, B, C1, C2, m, omega, tc}.
    fitted : bool
        True if the model was successfully fitted.
    result : OptimizeResult
        Full optimization output (from scipy.optimize.minimize).
    """

    # ---------- Initialization ---------- #
    def __init__(self, t: np.ndarray, p: np.ndarray):
        self.t = np.asarray(t)
        self.p = np.asarray(p)
        self.logp = np.log(self.p)
        self.params = None
        self.fitted = False
        self.result = None
        self.calibration_date = None

    def set_calibration_date(self, date):
        self.calibration_date = pd.to_datetime(date)

    # ---------- Internal Utilities ---------- #
    def _design_matrix(self, tc: float, m: float, omega: float) -> np.ndarray:
        """Design matrix for linear regression part (internal)."""
        dt = np.maximum(tc - self.t, 1e-9)
        f = dt**m
        g = f * np.cos(omega * np.log(dt))
        h = f * np.sin(omega * np.log(dt))
        return np.column_stack([np.ones_like(self.t), f, g, h])

    def _solve_linear_params(self, tc: float, m: float, omega: float):
        """Analytical OLS solution for A, B, C1, C2 (internal)."""
        X = self._design_matrix(tc, m, omega)
        beta, *_ = np.linalg.lstsq(X, self.logp, rcond=None)
        return beta  # A, B, C1, C2

    def _check_bounds(self, tc: float, m: float, omega: float) -> bool:
        """Stylized LPPL parameter constraints (Filimonov–Sornette)."""
        return (
            self.t[-1]-60/365 < tc < self.t[-1] + 300 / 365
            and 0.1 <= m <= 0.9
            and 6 <= omega <= 13
        )

    def _sse(self, params):
        """Sum of squared errors (objective function)."""
        tc, m, omega = params
        if not self._check_bounds(tc, m, omega):
            return np.inf
        A, B, C1, C2 = self._solve_linear_params(tc, m, omega)
        y_pred = self.lppls(self.t, A, B, C1, C2, tc, m, omega)
        return np.sum((self.logp - y_pred) ** 2)

    # ---------- Public Methods ---------- #
    def lppls(self, t, A, B, C1, C2, tc, m, omega):
        """
        Compute the expected log-price for given parameters.
        """
        dt = np.maximum(tc - t, 1e-9)
        f = dt**m
        return A + B * f + C1 * f * np.cos(omega * np.log(dt)) + C2 * f * np.sin(
            omega * np.log(dt)
        )
    def _check_qualified_fit(self, tc: float, m: float, omega: float) -> bool:
        """
        Filimonov–Sornette (2013) qualified fit constraints.

        tc must lie within (-60, 365) days relative to the last observation.
        m must lie in (0, 1)
        omega must lie in [2, 15]
        """
        tc_lower = - 60 / 365.25
        tc_upper = 365 / 365.25
        tc_rel = tc - self.t[-1]

        return (
            tc_lower < tc_rel < tc_upper   # tc in allowed range
            and 0+1e-4 < m < 1-1e-4                  # stricter than optimization bounds
            and 2+1e-4 <= omega <= 15-1e-4           # qualified range
        )
    
    def fit(self, initial_guess, method: str = "Nelder-Mead", options=None):
        """
        Fit the LPPLS model by minimizing the sum of squared errors.

        Parameters
        ----------
        initial_guess : list or array-like
            Initial guess for [tc, m, omega].
        method : str, optional
            Optimization method for scipy.optimize.minimize.
        options : dict, optional
            Additional options for the optimizer.

        Returns
        -------
        self : ModelLPPLS
            The fitted model (allows method chaining).

        Raises
        ------
        ValueError
            If optimization fails or model fitting is unsuccessful.
        """
        result = minimize(self._sse, initial_guess, method=method, options=options)
        self.result = result

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        tc, m, omega = result.x
        A, B, C1, C2 = self._solve_linear_params(tc, m, omega)
        self.params = dict(tc=float(tc), m=float(m), omega=float(omega), A=float(A), B=float(B), C1=float(C1), C2=float(C2))
        # Check qualified-fit constraints
        if self._check_qualified_fit(tc, m, omega):
            self.fitted = True
        else:
            self.fitted = False

        return self
    
    def fit_multistart(self, n_runs = 10, tol = 0.01):
        """
        Robust multistart fitting: run several fits with randomized initial guesses
        and keep the best run (lowest RMSE) among those that pass the qualified-fit check.

        - n_runs : number of random starting points (default 10)
        - tol : early-stop RMSE threshold (default 0.01)
            If any qualified fit reaches RMSE < tol, stop early.

        Random draws:
         m ~ Uniform(0,1)
         omega ~ Uniform(1,50)
         tc0 = t_last + Uniform(0.01, 0.5)

        Calls self.fit(initial_guess) (no method/options passed).
        Allows negative B (no bubble direction restriction).
        """
        np.random.seed(42)
        best = None
        best_rmse = np.inf
        
        for _ in range(int(n_runs)):
         # Randomized initial guess
         tc0 = self.t[-1] + float(np.random.uniform(0.01, 0.5))
         m0 = float(np.random.uniform(0.0, 1.0))
         omega0 = float(np.random.uniform(1.0, 50.0))

         try:
            candidate = ModelLPPLS(self.t, self.p)  # fresh instance
            candidate.fit([tc0, m0, omega0])        # DO NOT pass method/options
         except Exception:
            continue

         # must pass qualified-fit
         if not candidate.fitted:
            continue

         # compute RMSE
         pars = candidate.params
         y_pred = candidate.lppls(
            self.t, pars["A"], pars["B"], pars["C1"], pars["C2"],
            pars["tc"], pars["m"], pars["omega"]
         )
         rmse = np.sqrt(np.mean((self.logp - y_pred) ** 2))

         # early stop if RMSE good enough
         if rmse < tol:
            self.params = candidate.params
            self.result = candidate.result
            self.fitted = True
            return self

         # compute R^2 (same as before)
         ss_res = np.sum((self.logp - y_pred) ** 2)
         ss_tot = np.sum((self.logp - np.mean(self.logp)) ** 2)
         r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

         # keep best RMSE so far
         if np.isfinite(rme := rmse) and rmse < best_rmse:
            best_rmse = rmse
            best = {
                "model": candidate,
                "rmse": float(rmse),
                "r2": float(r2),
            }

        if best is None:
         self.fitted = False
         raise RuntimeError("fit_multistart: no qualified fit found in any random start.")

        # adopt best candidate
        chosen = best["model"]
        self.params = chosen.params
        self.result = chosen.result
        self.fitted = True
        return self

 
    def summary(self, calibration_date=None):
        """
        Return fitted parameters and key derived metrics as a one-row DataFrame.

        Parameters
        ----------
        calibration_date : float or datetime-like, optional
            If provided, it will appear in the 'calibration_date' column
            (useful when summarizing multiple fits).

        Returns
        -------
        pd.DataFrame
            One-row DataFrame containing:
            calibration_date, tc, A, B, C1, C2, m, omega, r2, rmse, kappa, sign
        """
        if not self.fitted:
            raise RuntimeError("Fit the model first using .fit(initial_guess).")

         # --- Extract fitted parameters ---
        A, B, C1, C2, tc, m, omega = (
            self.params["A"], self.params["B"], self.params["C1"], self.params["C2"],
            self.params["tc"], self.params["m"], self.params["omega"]
         )

         # --- Derived parameters ---
        kappa = -B
        sign = 1 if kappa > 0 else -1

         # --- Build DataFrame row ---
        row = {
            "calibration_date": self.calibration_date,
            "tc": tc,
            "A": A,
            "B": B,
            "C1": C1,
            "C2": C2,
            "m": m,
            "omega": omega,
            "kappa": kappa,
            "sign": sign,
        }

        return pd.DataFrame([row])
