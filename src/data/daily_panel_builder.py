from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.data.csv_panel_loader import curve_sort_key
from src.data.time_utils import default_alignment_tolerance, merge_on_timestamp, to_timestamp_series


def format_grid_column(value: float) -> str:
    token = str(value).replace("-", "m").replace(".", "p")
    return f"iv_mny_{token}"


def _curve_builder_config(config: dict | None) -> dict:
    return (config or {}).get("curve_builder", {})


def _interp_with_linear_extrapolation(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    interpolated = np.interp(grid, x, y)
    if len(x) < 2:
        return interpolated

    left_denom = x[1] - x[0]
    right_denom = x[-1] - x[-2]
    left_slope = (y[1] - y[0]) / left_denom if abs(left_denom) > 1e-12 else 0.0
    right_slope = (y[-1] - y[-2]) / right_denom if abs(right_denom) > 1e-12 else 0.0

    left_mask = grid < x[0]
    right_mask = grid > x[-1]
    if left_mask.any():
        interpolated[left_mask] = y[0] + left_slope * (grid[left_mask] - x[0])
    if right_mask.any():
        interpolated[right_mask] = y[-1] + right_slope * (grid[right_mask] - x[-1])
    return interpolated


def _interp_with_flat_edges(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    return np.interp(grid, x, y, left=y[0], right=y[-1])


def _fit_ridge_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    max_degree: int,
    ridge_alpha: float,
) -> tuple[np.ndarray, int]:
    degree = max(1, min(int(max_degree), len(x) - 1))
    design = np.vander(x, N=degree + 1, increasing=True)
    penalty = np.eye(degree + 1, dtype=float) * float(ridge_alpha)
    penalty[0, 0] = 0.0
    coeffs = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    grid_design = np.vander(grid, N=degree + 1, increasing=True)
    return grid_design @ coeffs, degree


def _clip_curve(curve: np.ndarray, y: np.ndarray, clip_multiplier: float) -> np.ndarray:
    observed_min = float(np.min(y))
    observed_max = float(np.max(y))
    lower = max(observed_min * 0.5, 1e-4)
    upper = max(observed_max * float(clip_multiplier), lower + 1e-4)
    return np.clip(curve, lower, upper)


def _simple_to_log_moneyness(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -0.95, None)
    return np.log1p(clipped)


def _svi_total_variance(k: np.ndarray, params: np.ndarray) -> np.ndarray:
    a, b, rho, m, sigma = params
    shifted = k - m
    return a + b * (rho * shifted + np.sqrt(shifted**2 + sigma**2))


def _fit_svi_curve(
    x: np.ndarray,
    y: np.ndarray,
    dte_years: np.ndarray,
    grid: np.ndarray,
    regularization: float,
    max_nfev: int,
    m_bounds: tuple[float, float],
) -> tuple[np.ndarray, dict[str, float | int]]:
    if len(x) < 3:
        raise ValueError("SVI fit requires at least three distinct points.")

    k_obs = _simple_to_log_moneyness(x)
    grid_k = _simple_to_log_moneyness(grid)
    t_obs = np.clip(dte_years.astype(float), 1e-6, None)
    t_ref = float(np.median(t_obs))
    w_obs = np.clip((y.astype(float) ** 2) * t_obs, 1e-8, None)

    atm_idx = int(np.argmin(np.abs(k_obs)))
    atm_w = float(w_obs[atm_idx])
    span = float(max(np.ptp(k_obs), 1e-3))
    curvature_scale = float(max(np.std(k_obs), 0.05))
    a0 = max(atm_w * 0.50, 1e-6)
    b0 = max((float(np.max(w_obs)) - float(np.min(w_obs))) / span, 1e-3)
    m0 = float(k_obs[atm_idx])
    sigma0 = curvature_scale
    initial = np.array([a0, b0, 0.0, m0, sigma0], dtype=float)

    lower = np.array([1e-8, 1e-5, -0.999, float(m_bounds[0]), 1e-4], dtype=float)
    upper = np.array([5.0, 10.0, 0.999, float(m_bounds[1]), 2.0], dtype=float)

    scales = np.array(
        [
            max(atm_w, 1e-4),
            max(b0, 1e-3),
            1.0,
            max(abs(m_bounds[1] - m_bounds[0]), 1e-2),
            max(sigma0, 1e-2),
        ],
        dtype=float,
    )
    reg_strength = max(float(regularization), 0.0)

    def residuals(params: np.ndarray) -> np.ndarray:
        model = _svi_total_variance(k_obs, params)
        data_resid = (model - w_obs) / np.maximum(w_obs, 1e-4)
        if reg_strength <= 0.0:
            return data_resid
        reg_resid = np.sqrt(reg_strength) * ((params - initial) / scales)
        return np.concatenate([data_resid, reg_resid])

    result = least_squares(
        residuals,
        x0=np.clip(initial, lower, upper),
        bounds=(lower, upper),
        method="trf",
        loss="soft_l1",
        f_scale=0.1,
        max_nfev=int(max_nfev),
    )
    params = result.x
    fitted_w = np.clip(_svi_total_variance(grid_k, params), 1e-8, None)
    fitted_iv = np.sqrt(fitted_w / max(t_ref, 1e-6))
    diagnostics = {
        "svi_success": int(result.success),
        "svi_nfev": int(result.nfev),
        "svi_cost": float(result.cost),
    }
    return fitted_iv, diagnostics


def _fit_curve_for_method(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    dte_years: np.ndarray,
    curve_cfg: dict,
) -> tuple[np.ndarray, str, dict[str, float | int]]:
    method = str(method).lower()
    max_degree = int(curve_cfg.get("max_polynomial_degree", 3))
    ridge_alpha = float(curve_cfg.get("ridge_alpha", 1e-4))
    diagnostics: dict[str, float | int] = {}

    if method == "poly_ridge":
        interpolated, degree = _fit_ridge_polynomial(
            x=x,
            y=y,
            grid=grid,
            max_degree=max_degree,
            ridge_alpha=ridge_alpha,
        )
        return interpolated, f"poly_ridge_deg{degree}", diagnostics
    if method == "legacy_interp":
        return _interp_with_flat_edges(x=x, y=y, grid=grid), "legacy_interp", diagnostics
    if method == "svi":
        interpolated, diagnostics = _fit_svi_curve(
            x=x,
            y=y,
            dte_years=dte_years,
            grid=grid,
            regularization=float(curve_cfg.get("svi_regularization", 0.05)),
            max_nfev=int(curve_cfg.get("svi_max_nfev", 1000)),
            m_bounds=tuple(curve_cfg.get("svi_m_bounds", [-0.35, 0.35])),
        )
        return interpolated, "svi_raw", diagnostics
    return _interp_with_linear_extrapolation(x=x, y=y, grid=grid), "linear_extrap", diagnostics


@dataclass
class DailyIVPanelBuilder:
    moneyness_grid: list[float]
    target_dte_days: int
    dte_tolerance_days: int = 7

    def build(
        self,
        option_rows: pd.DataFrame,
        underlying_df: pd.DataFrame,
        symbol: str,
        config: dict | None = None,
    ) -> pd.DataFrame:
        required = {"date", "expiration_date", "strike", "implied_volatility"}
        missing = required.difference(option_rows.columns)
        if missing:
            raise ValueError(f"Option rows are missing required columns: {sorted(missing)}")

        frame = option_rows.copy()
        frame["date"] = to_timestamp_series(frame["date"])
        frame["expiration_date"] = to_timestamp_series(frame["expiration_date"])
        if "spot" in frame.columns:
            merged = frame.copy()
        else:
            spot = underlying_df.copy()
            spot["date"] = to_timestamp_series(spot["date"])
            tolerance = default_alignment_tolerance(config or {})
            merged = merge_on_timestamp(
                frame,
                spot[["date", "close"]],
                on="date",
                tolerance=tolerance,
            ).rename(columns={"close": "spot"})
            merged = merged.dropna(subset=["spot"])
        if merged.empty:
            raise ValueError("No overlapping option and underlying rows were found.")

        merged["dte"] = (
            (merged["expiration_date"] - merged["date"]).dt.total_seconds() / 86400.0
        )
        merged = merged.loc[
            (merged["dte"] >= self.target_dte_days - self.dte_tolerance_days)
            & (merged["dte"] <= self.target_dte_days + self.dte_tolerance_days)
        ].copy()
        if merged.empty:
            raise ValueError(
                "No option rows fell inside the requested maturity bucket. "
                "Use CSV/parquet fallback if Alpaca historical coverage is insufficient."
            )
        merged["moneyness"] = merged["strike"] / merged["spot"] - 1.0
        merged = merged.loc[np.isfinite(merged["moneyness"]) & np.isfinite(merged["implied_volatility"])]
        merged = merged.loc[merged["implied_volatility"] > 0]
        if merged.empty:
            raise ValueError("No valid implied volatilities were available after filtering.")

        rows: list[dict[str, float | str | int]] = []
        grid = np.asarray(self.moneyness_grid, dtype=float)
        curve_cfg = _curve_builder_config(config)
        min_points = int(curve_cfg.get("min_unique_moneyness_points", 3))
        require_both_sides = bool(curve_cfg.get("require_both_sides_of_atm", True))
        fit_method = str(curve_cfg.get("fit_method", "poly_ridge")).lower()
        fallback_fit_method = str(curve_cfg.get("fallback_fit_method", "legacy_interp")).lower()
        clip_multiplier = float(curve_cfg.get("iv_clip_multiplier", 1.5))
        allow_low_quality_fallback = bool(curve_cfg.get("allow_low_quality_fallback", False))
        allow_single_point_flat_fill = bool(curve_cfg.get("allow_single_point_flat_fill", False))

        for trade_date, group in merged.groupby("date"):
            grouped = (
                group.groupby("moneyness", as_index=False)["implied_volatility"]
                .mean()
                .sort_values("moneyness")
            )
            x = grouped["moneyness"].to_numpy(dtype=float)
            y = grouped["implied_volatility"].to_numpy(dtype=float)
            dte_years = (group.groupby("moneyness", as_index=False)["dte"].mean().sort_values("moneyness")["dte"].to_numpy(dtype=float) / 365.0)
            num_points = int(len(x))
            has_left = bool(np.any(x < 0.0))
            has_right = bool(np.any(x > 0.0))
            quality_ok = num_points >= min_points and (
                not require_both_sides or (has_left and has_right)
            )
            if not quality_ok and not allow_low_quality_fallback:
                continue
            diagnostics: dict[str, float | int] = {}
            if num_points < 2:
                if allow_low_quality_fallback and allow_single_point_flat_fill and num_points == 1:
                    interpolated = np.repeat(y[0], len(grid))
                    method_used = "single_point_flat_fill"
                else:
                    continue
            else:
                try:
                    interpolated, method_used, diagnostics = _fit_curve_for_method(
                        method=fit_method,
                        x=x,
                        y=y,
                        grid=grid,
                        dte_years=dte_years,
                        curve_cfg=curve_cfg,
                    )
                except Exception:
                    if not allow_low_quality_fallback or fallback_fit_method == fit_method:
                        continue
                    interpolated, fallback_method_used, diagnostics = _fit_curve_for_method(
                        method=fallback_fit_method,
                        x=x,
                        y=y,
                        grid=grid,
                        dte_years=dte_years,
                        curve_cfg=curve_cfg,
                    )
                    method_used = f"{fit_method}_fallback_{fallback_method_used}"

            if num_points >= 2:
                interpolated = _clip_curve(interpolated, y=y, clip_multiplier=clip_multiplier)

            row: dict[str, float | str | int] = {
                "date": trade_date,
                "underlying": symbol,
                "dte_bucket": self.target_dte_days,
                "curve_num_points": num_points,
                "curve_has_left": int(has_left),
                "curve_has_right": int(has_right),
                "curve_x_min": float(np.min(x)),
                "curve_x_max": float(np.max(x)),
                "curve_quality_ok": int(quality_ok),
                "curve_fit_method": method_used,
            }
            if "stale_bars" in group.columns:
                point_staleness = (
                    group.groupby("moneyness", as_index=False)["stale_bars"]
                    .min()
                    .sort_values("moneyness")
                )
                row["curve_num_carried_points"] = int((point_staleness["stale_bars"] > 0).sum())
                row["curve_num_fresh_points"] = int((point_staleness["stale_bars"] == 0).sum())
                row["curve_avg_stale_bars"] = float(point_staleness["stale_bars"].mean())
                row["curve_max_stale_bars"] = int(point_staleness["stale_bars"].max())
            row.update(diagnostics)
            for grid_value, iv in zip(self.moneyness_grid, interpolated):
                row[format_grid_column(grid_value)] = float(iv)
            rows.append(row)

        panel = pd.DataFrame(rows)
        if panel.empty:
            raise ValueError(
                "Curve-quality filtering removed every timestamp. "
                "Relax curve_builder settings or improve the underlying option-chain coverage."
            )
        panel = panel.sort_values("date").reset_index(drop=True)
        curve_columns = sorted(
            [column for column in panel.columns if column.startswith("iv_mny_")],
            key=curve_sort_key,
        )
        meta_columns = [
            column
            for column in [
                "curve_num_points",
                "curve_has_left",
                "curve_has_right",
                "curve_x_min",
                "curve_x_max",
                "curve_quality_ok",
                "curve_fit_method",
                "curve_num_carried_points",
                "curve_num_fresh_points",
                "curve_avg_stale_bars",
                "curve_max_stale_bars",
                "svi_success",
                "svi_nfev",
                "svi_cost",
            ]
            if column in panel.columns
        ]
        return panel[["date", "underlying", "dte_bucket", *meta_columns, *curve_columns]]
