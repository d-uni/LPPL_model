import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from tqdm import tqdm
from distribution_calibrators import (
    DistributionLPPLCalibrator_for_different_Windows,
    DistributionLPPLCalibrator_for_different_Dates,
)

class LPPLIndicatorGenerator:

    def __init__(
        self,
        price_series: pd.Series,
        qualified_fit_r2_min: float = 0.9,
        qualified_fit_rmse_max: float = 0.06,
        qualified_fit_kappa_min: float = 0.05,
        qualified_fit_tc_horizon_years: float = 0.5,

    ):
        """
        Parameters
        ----------
        price_series : pd.Series
            Close prices with DateTimeIndex.
        r2_min, rmse_max, kappa_min, tc_horizon_years :
            Acceptance thresholds for qualified LPPL fits.
        """
        self.price_series = price_series.dropna().copy()
        self.dates = self.price_series.index

        # Time in years since start
        self.t_series = (self.dates - self.dates[0]).days / 365.25
        self.p_series = self.price_series.values.squeeze()


        # --- instantiate calibrators ---
        self.cal_window = DistributionLPPLCalibrator_for_different_Windows(
            self.t_series, self.p_series
        )
        self.cal_window.set_acceptance_thresholds(
            r2_min=qualified_fit_r2_min,
            rmse_max=qualified_fit_rmse_max,
            kappa_min=qualified_fit_kappa_min,
            tc_horizon_years=qualified_fit_tc_horizon_years,
        )
        self.cal_window.set_model_params_initial_guess(tc_offset=0.12, m0=0.55, omega0=9.0)

        self.cal_date = DistributionLPPLCalibrator_for_different_Dates(
            self.t_series, self.p_series
        )
        self.cal_date.set_acceptance_thresholds(
            r2_min=qualified_fit_r2_min,
            rmse_max=qualified_fit_rmse_max,
            kappa_min=qualified_fit_kappa_min,
            tc_horizon_years=qualified_fit_tc_horizon_years,
        )
        self.cal_date.set_model_params_initial_guess(tc_offset=0.12, m0=0.55, omega0=9.0)

    def build_indicator_table(
        self,
        years_for_calibration: float = 1.5,
        diff_windows_indicator_min_window_years: float = 1 / 12,
        diff_windows_indicator_max_window_years: float = 1.0,
        diff_windows_indicator_step_window_years: float = 10 / 365,
        diff_dates_indicator_neighborhood_years: float = 1 / 12,
        diff_dates_indicator_step_years_dates: float = 5 / 365.25,
        crash_HORIZON_years_for_indicator: float = 0.25,
    ) -> pd.DataFrame:

        rows = []

        t_values = self.t_series         # skip first day (no return)
        dates = self.dates

        for t_current, date_current in tqdm(
            zip(t_values, dates),
            total=len(t_values),
            desc="Computing LPPL indicators",
        ):
            # Default row with NaNs (for days with no valid indicator)
            row = {
                "date": date_current,
                "n_fits_diff_windows_percentage (%)": 0,
                "n_fits_diff_dates_percentage (%)": 0,
                "Proba_crash_in_HORIZON_time_diff_windows": np.nan,
                "Proba_crash_in_HORIZON_time_diff_dates": np.nan,
                "trustworthiness_percentage": np.nan,
                "bubble_sign": np.nan,
            }

            # require enough history before we even try
            if t_current < years_for_calibration:
                rows.append(row)
                continue

            # ---------- 1) distribution over window sizes ----------
            dist_w = self.cal_window.tc_distribution_over_window(
                target_date=t_current,
                min_window_years=diff_windows_indicator_min_window_years,
                max_window_years=diff_windows_indicator_max_window_years,
                step_window_years=diff_windows_indicator_step_window_years,
                acceptance_thresholds=True,
            )

            if dist_w.empty or "tc" not in dist_w.columns or len(dist_w) < 2:
                rows.append(row)
                continue
            n_windows_total = len(
                  np.arange(diff_windows_indicator_min_window_years, diff_windows_indicator_max_window_years + diff_windows_indicator_step_window_years, diff_windows_indicator_step_window_years)
                  )
            row["n_fits_diff_windows_percentage (%)"] = int(len(dist_w)/n_windows_total * 100)

            # KDE over tc (relative to current time)
            tc_w_all = dist_w["tc"].values
            tc_w_rel = tc_w_all - t_current

            # need at least 2 points for KDE
            if len(tc_w_rel) < 2:
                rows.append(row)
                continue

            kde_tc = gaussian_kde(tc_w_rel)
            density = kde_tc(tc_w_rel)
            dist_w = dist_w.copy()
            dist_w["density"] = density

            # best window = one in highest-density region of tc
            best_row = dist_w.loc[dist_w["density"].idxmax()]
            best_window = float(best_row["window_years"])

            # crash probability using window-variation only
            kde_w = gaussian_kde(tc_w_rel)
            P_crash_window = kde_w.integrate_box_1d(0.0, crash_HORIZON_years_for_indicator)
            row["Proba_crash_in_HORIZON_time_diff_windows"] = float(P_crash_window)

            # ---------- 2) distribution over calibration dates ----------
            dist_d = self.cal_date.tc_distribution_over_dates(
                target_date=t_current,
                window_years=best_window,
                distribution_window_years=diff_dates_indicator_neighborhood_years,
                step_years=diff_dates_indicator_step_years_dates,
                acceptance_thresholds=True,
            )

            if dist_d.empty or len(dist_d) < 2:
                rows.append(row)
                continue

            n_dates_total = len(np.arange(
                                t_current - diff_dates_indicator_neighborhood_years,
                                t_current + diff_dates_indicator_step_years_dates,
                                diff_dates_indicator_step_years_dates
            )
            )

            row["n_fits_diff_dates_percentage (%)"] = int(len(dist_d) / n_dates_total * 100)

            tc_d_all = dist_d["tc"].values
            tc_d_rel = tc_d_all - t_current

            if len(tc_d_rel) < 2:
                rows.append(row)
                continue

            kde_d = gaussian_kde(tc_d_rel)
            P_crash_dates = kde_d.integrate_box_1d(0.0, crash_HORIZON_years_for_indicator)
            row["Proba_crash_in_HORIZON_time_diff_dates"] = float(P_crash_dates)

            # ---------- 3) combined signal: mean sign + joint crash probability ----------
            sign_mean = np.mean(
                np.concatenate([dist_w["sign"].values, dist_d["sign"].values])
            )
            row["bubble_sign"] = float(sign_mean)


            P_conf = min(P_crash_window, P_crash_dates)
            overlap = self.kde_overlap_on_interval(kde_w, kde_d, 0.0, crash_HORIZON_years_for_indicator)
            row["trustworthiness_percentage"] = P_conf * overlap

            rows.append(row)

        indicators = pd.DataFrame(rows).set_index("date")
        self.indicators_ = indicators
        return indicators

    def kde_overlap_on_interval(self, kde1, kde2, a, b, n_grid=400):
        """
        Compute overlap integral int_a^b min(f1, f2) dt for two KDEs.
        Returns a scalar in [0, 1] (on full support) or [0, <=1] on [a,b].
        """
        grid = np.linspace(a, b, n_grid)
        f1 = kde1(grid)
        f2 = kde2(grid)
        f_min = np.minimum(f1, f2)
        overlap = np.trapz(f_min, grid)
        return float(overlap)

    def to_csv(self, path: str) -> None:
        """Save last computed indicator table to CSV (skipping initial empty rows)."""
        if not hasattr(self, "indicators_"):
          raise RuntimeError("Run build_indicator_table() before saving to CSV.")

        df = self.indicators_.copy()

        # Remove rows with no valid indicators
        df = df.dropna(
            subset=[
              "Proba_crash_in_HORIZON_time_diff_windows",
              "Proba_crash_in_HORIZON_time_diff_dates",
              "trustworthiness_percentage",
              "bubble_sign"
            ],
            how="all"
        )
        df.to_csv(path)
        return df