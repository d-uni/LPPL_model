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
            self.t[-1] < tc < self.t[-1] + 300 / 365
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

        tc must lie within (-60, 252) days relative to the last observation.
        m must lie in (0, 1)
        omega must lie in [2, 15]
        """
        tc_lower = - 60 / 365.25
        tc_upper = 252 / 365.25
        tc_rel = tc - self.t[-1]

        return (
            tc_lower < tc_rel < tc_upper   # tc in allowed range
            and 0 < m < 1                  # stricter than optimization bounds
            and 2 <= omega <= 15           # qualified range
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
