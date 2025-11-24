import numpy as np
import pandas as pd
from model_lppls import ModelLPPLS

class RollingLPPLCalibrator:
    """
    Rolling LPPLS model calibration over a time series.

    This class fits the LPPLS model repeatedly over a sliding (rolling)
    time window, allowing analysis of parameter stability and predicted
    critical times (t_c) over time.
    """

    # ---------- Initialization ---------- #
    def __init__(self, t, p):
        """
        Parameters
        ----------
        t : array-like
            Time points in years (in increasing order).
        p : array-like
            Positive price values corresponding to t.
        """
        self.t = np.asarray(t).flatten()
        self.p = np.asarray(p).flatten()
        self.results = pd.DataFrame()

        self.model_params = None
        self.acceptance_criteria = None

    def set_acceptance_thresholds(self, r2_min, rmse_max, kappa_min, tc_horizon_years):
        """
        Define criteria to accept or reject fitted LPPLS models.

        Parameters
        ----------
        r2_min : float
            Minimum acceptable R² value.
        rmse_max : float
            Maximum acceptable RMSE value.
        kappa_min : float
            Minimum absolute B value (|B| = κ).
        tc_horizon_years : float
            Maximum time horizon (in years) for predicted critical time beyond window end.
        """
        self.acceptance_criteria = {
            "r2_min": r2_min,
            "rmse_max": rmse_max,
            "kappa_min": kappa_min,
            "tc_horizon_years": tc_horizon_years,
        }

    # ---------- Helpers ---------- #
    def r2_score(self, y_true, y_pred):
        """Compute coefficient of determination (R²)."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    def _fit_window(self, t, p, n_runs: int = 10, tol: float = 0.01):
        """Fit LPPLS model on a single rolling window using multistart."""
        model = ModelLPPLS(t, p)
        try:
            model.fit_multistart(n_runs=n_runs, tol=tol)
        except Exception:
            return None
        return model if model.fitted else None

    # ---------- Main rolling calibration ---------- #
    def run(self, start_roll, end_roll, window_years, step_years):
        """
        Perform rolling LPPLS calibration across a given time range.

        Parameters
        ----------
        start_roll : float
            Start time (in years) for the first calibration window end.
        end_roll : float
            End time (in years) for the last calibration window end.
        window_years : float
            Length of each calibration window in years.
        step_years : float
            Step size (in years) between consecutive window ends.

        Returns
        -------
        pd.DataFrame
            DataFrame containing parameters and metrics for each credible fit.
        """
        rows = []
        current_end = start_roll

        while current_end <= end_roll:
            current_start = current_end - window_years
            mask = (self.t > current_start) & (self.t <= current_end)
            t_window, p_window = self.t[mask], self.p[mask]

            if len(t_window) < 20:  # skip too-short windows
                current_end += step_years
                continue

            model = self._fit_window(t_window, p_window)
            if model is None:
                current_end += step_years
                continue

            pars = model.params
            A, B, C1, C2, tc, m, omega = (
                pars["A"], pars["B"], pars["C1"], pars["C2"],
                pars["tc"], pars["m"], pars["omega"]
            )

            # Compute metrics
            logp = np.log(p_window)
            y_pred = model.lppls(t_window, A, B, C1, C2, tc, m, omega)
            r2 = self.r2_score(logp, y_pred)
            rmse = np.sqrt(np.mean((logp - y_pred) ** 2))
            kappa = -B
            sign = 1 if kappa > 0 else -1

            # Apply acceptance criteria
            crit = self.acceptance_criteria
            if (
                tc > t_window[-1] + crit["tc_horizon_years"]
                or r2 < crit["r2_min"]
                or rmse > crit["rmse_max"]
                or abs(kappa) < crit["kappa_min"]
            ):
                current_end += step_years
                continue

            rows.append({
                "window_start": current_start,
                "window_end": current_end,
                "window_size": window_years,
                "tc": tc,
                "A": A, "B": B, "C1": C1, "C2": C2,
                "m": m, "omega": omega,
                "r2": r2, "rmse": rmse,
                "kappa": kappa, "sign": sign,
            })

            current_end += step_years

        self.results = pd.DataFrame(rows)
        return self.results

    # ---------- Summary ---------- #
    def summary(self, n=5):
        """Print a summary of rolling calibration results."""
        return self.results
