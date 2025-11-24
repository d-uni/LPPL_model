import numpy as np
import pandas as pd
from model_lppls import ModelLPPLS

class DistributionLPPLCalibrator_for_different_Windows:
    """
    LPPLS distribution analysis by varying the calibration window size.

    For a fixed target date, this class repeatedly fits the LPPLS model
    over multiple window sizes (e.g., from 3 months to 1 year),
    building a distribution of predicted critical times (t_c)
    and fitted model parameters.
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

    # ---------- Configuration ---------- #
    def set_model_params_initial_guess(self, tc_offset, m0, omega0):
        """
        Set the initial guess configuration for LPPLS fitting.

        Parameters
        ----------
        tc_offset : float
            Offset (in years) added to the last window point for initial t_c guess.
        m0 : float
            Initial guess for exponent m (typically between 0.1 and 0.9).
        omega0 : float
            Initial guess for angular frequency (typically between 6 and 13).
        """
        self.model_params = {
            "tc_offset": tc_offset,
            "m0": m0,
            "omega0": omega0
        }

    def set_acceptance_thresholds(self, r2_min, rmse_max, kappa_min, tc_horizon_years):
        """
        Define filtering thresholds to accept only credible LPPLS fits.

        Parameters
        ----------
        r2_min : float
            Minimum acceptable R² value.
        rmse_max : float
            Maximum acceptable RMSE value.
        kappa_min : float
            Minimum absolute B value (|B| = κ).
        tc_horizon_years : float
            Maximum time horizon (in years) allowed for predicted t_c beyond window end.
        """
        self.acceptance_criteria = {
            "r2_min": r2_min,
            "rmse_max": rmse_max,
            "kappa_min": kappa_min,
            "tc_horizon_years": tc_horizon_years
        }

    # ---------- Utility ---------- #
    def r2_score(self, y_true, y_pred):
        """Compute coefficient of determination (R²)."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    def _fit_window(self, t, p, n_runs: int=10, tol: float=0.01):
        """Fit a single LPPLS model using multistart."""
        model = ModelLPPLS(t, p)
        try:
            model.fit_multistar(n_runs=n_runs, tol=tol)
        except Exception:
            return None
        return model if model.fitted else None

    # ---------- Distribution over window sizes ---------- #
    def tc_distribution_over_window(
        self,
        target_date: float,
        min_window_years: float,
        max_window_years: float,
        step_window_years: float,
        acceptance_thresholds: bool = False
    ):
        """
        Estimate the distribution of predicted critical times (t_c)
        for a fixed target date, varying the calibration window length.

        Parameters
        ----------
        target_date : float
            The end time (in years) of each calibration window.
        min_window_years : float
            Minimum window size (in years), e.g., 1/12 for 1 month.
        max_window_years : float
            Maximum window size (in years), e.g., 1.0 for 1 year.
        step_window_years : float
            Step increment (in years) between consecutive window sizes.
        acceptance_thresholds : bool, optional
            If True, only retain fits satisfying acceptance criteria.

        Returns
        -------
        pd.DataFrame
            DataFrame containing fitted parameters and metrics
            for each window size.
        """
        results = []
        window_sizes = np.arange(
            min_window_years, max_window_years + step_window_years, step_window_years
        )

        for w in window_sizes:
            start = target_date - w
            mask = (self.t > start) & (self.t <= target_date)
            t_window, p_window = self.t[mask], self.p[mask]

            if len(t_window) < 20:
                continue

            model = self._fit_window(t_window, p_window)
            if model is None:
                continue

            pars = model.params
            A, B, C1, C2, tc, m, omega = (
                pars["A"], pars["B"], pars["C1"], pars["C2"],
                pars["tc"], pars["m"], pars["omega"]
            )

            logp = np.log(p_window)
            y_pred = model.lppls(t_window, A, B, C1, C2, tc, m, omega)
            r2 = self.r2_score(logp, y_pred)
            rmse = np.sqrt(np.mean((logp - y_pred) ** 2))
            kappa = -B
            sign = 1 if kappa > 0 else -1

            if acceptance_thresholds:
                crit = self.acceptance_criteria
                if (
                    tc > t_window[-1] + crit["tc_horizon_years"]
                    or r2 < crit["r2_min"]
                    or rmse > crit["rmse_max"]
                    or abs(kappa) < crit["kappa_min"]
                ):
                    continue

            results.append({
                "window_years": w,
                "tc": tc,
                "A": A, "B": B, "C1": C1, "C2": C2,
                "m": m, "omega": omega,
                "r2": r2, "rmse": rmse,
                "kappa": kappa, "sign": sign
            })

        self.results = pd.DataFrame(results)
        return self.results
    
class DistributionLPPLCalibrator_for_different_Dates:
    """
    LPPLS distribution analysis by varying the calibration end date.

    For a fixed window size, this class repeatedly fits the LPPLS model
    over multiple calibration dates preceding a target date.
    The resulting distribution of predicted critical times (t_c)
    helps assess the model’s stability through time.
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

    # ---------- Configuration ---------- #
    def set_model_params_initial_guess(self, tc_offset, m0, omega0):
        """Configure initial parameter guesses for LPPLS fitting."""
        self.model_params = {
            "tc_offset": tc_offset,
            "m0": m0,
            "omega0": omega0
        }

    def set_acceptance_thresholds(self, r2_min, rmse_max, kappa_min, tc_horizon_years):
        """Define filtering thresholds for credible LPPLS fits."""
        self.acceptance_criteria = {
            "r2_min": r2_min,
            "rmse_max": rmse_max,
            "kappa_min": kappa_min,
            "tc_horizon_years": tc_horizon_years
        }

    # ---------- Helpers ---------- #
    def r2_score(self, y_true, y_pred):
        """Compute coefficient of determination (R²)."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
       
    def _fit_window(self, t, p, n_runs: int = 10, tol: float = 0.01):
        """Fit a single LPPLS model to a specific calibration window using multistart."""
        model = ModelLPPLS(t, p)
        try:
            model.fit_multistart(n_runs=n_runs, tol=tol)
        except Exception:
            return None
        return model if model.fitted else None

    # ---------- Distribution across calibration dates ---------- #
    def tc_distribution_over_dates(
        self,
        target_date: float,
        window_years: float,
        distribution_window_years: float,
        step_years: float,
        acceptance_thresholds: bool = False
    ):
        """
        Estimate distribution of predicted t_c by shifting calibration end date.

        Parameters
        ----------
        target_date : float
            Reference date (in years) relative to data start.
        window_years : float
            Fixed calibration window size (in years).
        distribution_window_years : float
            Time range (before target_date) over which to vary the calibration end date.
        step_years : float
            Step between consecutive calibration dates (in years).
        acceptance_thresholds : bool, optional
            If True, apply acceptance filters to retain only credible fits.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per calibration date and its fitted parameters.
        """
        results = []
        end_dates = np.arange(
            target_date - distribution_window_years,
            target_date + step_years,
            step_years
        )

        for d in end_dates:
            start = d - window_years
            mask = (self.t > start) & (self.t <= d)
            t_window, p_window = self.t[mask], self.p[mask]

            if len(t_window) < 20:
                continue

            model = self._fit_window(t_window, p_window)
            if model is None:
                continue

            pars = model.params
            A, B, C1, C2, tc, m, omega = (
                pars["A"], pars["B"], pars["C1"], pars["C2"],
                pars["tc"], pars["m"], pars["omega"]
            )

            logp = np.log(p_window)
            y_pred = model.lppls(t_window, A, B, C1, C2, tc, m, omega)
            r2 = self.r2_score(logp, y_pred)
            rmse = np.sqrt(np.mean((logp - y_pred) ** 2))
            kappa = -B
            sign = 1 if kappa > 0 else -1

            if acceptance_thresholds:
                crit = self.acceptance_criteria
                if (
                    tc > t_window[-1] + crit["tc_horizon_years"]
                    or r2 < crit["r2_min"]
                    or rmse > crit["rmse_max"]
                    or abs(kappa) < crit["kappa_min"]
                ):
                    continue

            results.append({
                "calibration_date": d,
                "tc": tc,
                "A": A, "B": B, "C1": C1, "C2": C2,
                "m": m, "omega": omega,
                "r2": r2, "rmse": rmse,
                "kappa": kappa, "sign": sign
            })

        self.results = pd.DataFrame(results)
        return self.results
