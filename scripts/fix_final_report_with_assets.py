#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


LISTINGS_PACKAGE = "\\usepackage{listings}\n"
LISTINGS_STYLE = r"""
\lstdefinestyle{csvstyle}{
basicstyle=\ttfamily\tiny,
breaklines=true,
breakatwhitespace=false,
columns=fullflexible,
keepspaces=true,
showstringspaces=false,
frame=single
}
""".strip(
    "\n"
)

OLD_APPENDIX_MACRO = r"""
\newcommand{\insertappendixcsv}[3]{%
\begin{landscape}
\scriptsize
\begin{center}
\captionof{table}{#1}
\label{#2}
\IfFileExists{\assetroot/tables/#3}{\csvautolongtable{\assetroot/tables/#3}}{\fbox{\parbox[c][1.8in][c]{0.8\linewidth}{\centering Missing table\\[0.4em]\texttt{\assetroot/tables/#3}}}}
\end{center}
\end{landscape}}
""".strip(
    "\n"
)

INTERMEDIATE_APPENDIX_MACRO = r"""
\newcommand{\insertappendixcsv}[3]{%
\begin{landscape}
\begin{center}
\captionof{table}{#1}
\label{#2}
\IfFileExists{\assetroot/tables/#3}{\lstinputlisting[style=csvstyle]{\assetroot/tables/#3}}{\fbox{\parbox[c][1.8in][c]{0.8\linewidth}{\centering Missing table\\[0.4em]\texttt{\assetroot/tables/#3}}}}
\end{center}
\end{landscape}}
""".strip(
    "\n"
)

NEW_APPENDIX_MACRO = r"""
\newcommand{\insertappendixcsv}[3]{%
\begin{landscape}
\begin{center}
\captionof{table}{#1}
\label{#2}
\lstinputlisting[style=csvstyle]{\assetroot/tables/#3}
\end{center}
\end{landscape}}
""".strip(
    "\n"
)

DISABLED_APPENDIX_MACRO = r"""
\newcommand{\insertappendixcsv}[3]{}
""".strip(
    "\n"
)

FEATURE_TABLE_CALL = r"\insertappendixcsv{Feature definitions used across the study.}{tab:feature_definitions_main}{tab_feature_definitions.csv}"

FEATURE_TABLE_BLOCK = r"""
\begin{table}[H]
\centering
\small
\caption{Feature groups used across the early and final phases of the study.}
\label{tab:feature_definitions_main}
\begin{tabular}{p{3.2cm}p{8.0cm}cc}
\toprule
Feature group & Main variables & Early daily/hourly & Final 5-minute \\
\midrule
IV curve state & Seven fixed-grid IV values across moneyness & Yes & Yes \\
ATM level features & ATM IV and ATM IV change & ATM IV only & Yes \\
Underlying move features & Return and absolute return & Return only & Yes \\
Realized-volatility features & Short-window and long-window realized volatility & Short-window only & Yes \\
Liquidity and range features & Range percentage, log volume, volume z-score & No & Yes \\
Curve-shape features & Curve slope and curve curvature & No & Yes \\
Calendar feature & Scaled DTE bucket & Yes & Yes \\
\bottomrule
\end{tabular}
\end{table}
""".strip(
    "\n"
)

OLD_FEATURE_SECTION_BLOCK = r"""
\section{Feature engineering evolution}
The initial feature set was intentionally compact. It contained lagged IV-curve values, ATM IV, underlying return, realized-volatility proxies, and DTE bucket information, giving 11 features per timestamp in the early studies. The early datasets contained 260 daily samples, 543 hourly H1 samples, 2045 year-long hourly samples, and 4030 two-year hourly walk-forward samples.

The final 5-minute study expanded the feature set to 19 features because higher-frequency forecasting benefits more from local state variables. The enriched feature set included IV grid values, ATM IV, ATM IV change, underlying return, absolute return, realized volatility, long-window realized volatility, range percentage, log volume, volume z-score, curve slope, curve curvature, and a scaled DTE bucket. The base 5-minute walk-forward dataset contained 35,260 supervised samples, 19 features, and 7 curve outputs. This richer state representation was one of the main reasons the later 5-minute results became materially stronger. To make the feature design explicit, Table~\ref{tab:feature_definitions_main} lists the final feature definitions and whether each feature group was used in the early daily/hourly experiments or only in the final 5-minute study.

\begin{table}[H]
\centering
\small
\caption{Feature groups used across the early and final phases of the study.}
\label{tab:feature_definitions_main}
\begin{tabular}{p{3.2cm}p{8.0cm}cc}
\toprule
Feature group & Main variables & Early daily/hourly & Final 5-minute \\
\midrule
IV curve state & Seven fixed-grid IV values across moneyness & Yes & Yes \\
ATM level features & ATM IV and ATM IV change & ATM IV only & Yes \\
Underlying move features & Return and absolute return & Return only & Yes \\
Realized-volatility features & Short-window and long-window realized volatility & Short-window only & Yes \\
Liquidity and range features & Range percentage, log volume, volume z-score & No & Yes \\
Curve-shape features & Curve slope and curve curvature & No & Yes \\
Calendar feature & Scaled DTE bucket & Yes & Yes \\
\bottomrule
\end{tabular}
\end{table}
""".strip(
    "\n"
)

RICH_FEATURE_SECTION_BLOCK = r"""
\section{Feature engineering evolution}
The feature design changed materially over the life of the project, and that evolution was one of the main reasons the final intraday results became credible. The earliest daily and hourly pipelines were intentionally conservative. They used a compact 11-feature state consisting of the lagged seven-node IV curve, ATM IV, an underlying return feature, a short-window realized-volatility proxy, and a scaled DTE-bucket indicator. That design was appropriate for a minimum viable research system because it kept the state small, easy to debug, and easy to align across providers. It also made the early negative result interpretable: when those compact low-frequency states failed to let the LSTM beat the stronger classical baselines consistently, the issue was not hidden feature complexity.

The final 5-minute study expanded that state to 19 features per timestamp because intraday IV dynamics depend on much more than the last quoted curve level. At that frequency, the model benefits from information about recent underlying movement, short-run volatility clustering, trading activity, and simple smile geometry. The final 5-minute walk-forward dataset therefore used the seven IV-grid values plus a set of low-cost but informative engineered features: ATM IV, ATM IV change, underlying return, absolute return, short-window realized volatility, long-window realized volatility, range percentage, log volume, volume z-score, curve slope, curve curvature, and a scaled DTE bucket. The base 5-minute dataset contained 35,260 supervised samples, 19 input features, and 7 output curve nodes. This richer state representation is directly tied to the later experimental result that the LSTM only became convincingly superior once the study moved to denser intraday data.

Table~\ref{tab:feature_definitions_main} hard-codes the final feature families used in the thesis version of the methodology. The point of the table is not merely to list variables, but to show why each family was retained and when it entered the research program.

\begin{table}[H]
\centering
\footnotesize
\caption{Feature families used across the study and their methodological role.}
\label{tab:feature_definitions_main}
\begin{tabular}{p{2.8cm}p{3.5cm}p{5.6cm}cc}
\toprule
Feature family & Main variables & Why included & Early daily/hourly & Final 5-minute \\
\midrule
IV curve state & Seven fixed-grid IV values across moneyness & Core state of the problem. These nodes represent the current smile and allow every model to learn local persistence, wing asymmetry, and cross-moneyness dependence. & Yes & Yes \\
ATM level and change & ATM IV, ATM IV change & Captures the central level of the curve and short-run re-pricing pressure around the most liquid part of the smile. ATM IV change was added later because intraday dynamics are often driven by local level shifts before they propagate to the full curve. & ATM IV only & Yes \\
Underlying move features & Return, absolute return & Links smile dynamics to directional moves and the size of the latest underlying shock. Absolute return was added later because the magnitude of the move matters even when its sign is less informative. & Return only & Yes \\
Realized-volatility features & Short-window RV, long-window RV & Gives the model a fast and slow proxy for recent volatility regime. This helps distinguish transient noise from broader variance-state changes. & Short-window only & Yes \\
Liquidity and activity features & Range percentage, log volume, volume z-score & Introduced only in the 5-minute setting to proxy intraday market activity, local information arrival, and abnormal trading intensity. These were unnecessary in the earliest MVP but became useful once the data frequency increased. & No & Yes \\
Curve-shape features & Slope, curvature & Adds a low-dimensional geometric summary of the smile. These features help the network reason about whether the curve is steepening, flattening, or kinking near ATM even before it processes all seven nodes jointly. & No & Yes \\
Calendar feature & Scaled DTE bucket & Keeps the single-maturity-bucket framing explicit and preserves limited term-structure information within the chosen DTE tolerance window. & Yes & Yes \\
\bottomrule
\end{tabular}
\end{table}

The feature set can also be understood as three layers. First, the IV-grid values and ATM features define the market object being forecast. Second, the underlying-return and realized-volatility features provide a compact description of the short-run state of the underlying. Third, the liquidity, activity, and curve-shape features provide contextual information that becomes materially more useful at 5-minute frequency than at hourly or daily frequency. This layered structure is important because it explains why the final model improvement should not be attributed to architecture alone. The LSTM only became convincing once it was fed a state representation rich enough to match the time scale of the forecasting task.

This is also where the thesis can explain why some candidate feature families were not taken further. No option-level order-book variables, quote-level imbalance measures, or full arbitrage-constrained smile parameterizations were required for the final result. Those additions may improve a production system, but they were deliberately excluded from the core thesis pipeline to keep the experimental story interpretable. The final feature design therefore balances practicality and expressiveness: it is much richer than the early MVP, but still simple enough that each family has a clear economic and modeling rationale.
""".strip(
    "\n"
)

APPENDIX_TABLES_SECTION_TITLE = r"\section{CSV tables exported from the asset pack}"

APPENDIX_TABLES_SECTION_REPLACEMENT = r"""
\section{CSV tables exported from the asset pack}

The full CSV exports referenced throughout the report are included in the accompanying \texttt{thesis\_report\_assets/tables/} folder and are intended to be consulted externally rather than embedded in the compiled PDF. They were removed from the appendix body because the full raw CSV listings are large enough to create avoidable compilation and memory issues in Overleaf, while adding little interpretive value beyond the summarized tables and figures already discussed in the main text.

The most important external tables are:
\begin{itemize}
\item \texttt{tab\_standardized\_overall\_summary.csv} for the final standardized LSTM versus matched-MLP comparison;
\item \texttt{tab\_best\_threshold\_by\_finalist.csv} for the threshold-tuning results;
\item \texttt{tab\_final\_robustness\_addendum.csv} for the final significance and robustness checks;
\item \texttt{tab\_5min\_lstm\_grid\_summary.csv} and \texttt{tab\_5min\_baseline\_grid\_summary.csv} for the full 5-minute model grids.
\end{itemize}

For provenance and file-level mapping, see \texttt{thesis\_report\_assets/summaries/README\_asset\_manifest.md} and \texttt{thesis\_report\_assets/summaries/report\_asset\_map.csv}.
""".strip(
    "\n"
)


FOLLOW_UPS = {
    r"\insertassetfigure{Hourly capacity sweep across depth and hidden-size configurations.}{fig:hourly_capacity_sweep}{fig_hourly_capacity_sweep.png}": (
        r"Figure~\ref{fig:hourly_capacity_sweep} makes the hourly capacity result easy to interpret. "
        r"The main gain came from widening the recurrent state, not from adding another layer, which is why the later intraday experiments kept depth modest and increased width only where the data volume justified it."
    ),
    r"\insertassetfigure{Hourly stitched walk-forward ablation summary, showing how sequence length, forecast horizon, and regularization choices altered the LSTM's statistical and economic performance.}{fig:hourly_walkforward_ablation}{fig_hourly_walkforward_ablation_summary.png}": (
        r"Figure~\ref{fig:hourly_walkforward_ablation} also explains why the hourly evidence remained mixed. "
        r"The sequence-length and horizon panels show that the problem became materially harder as the target moved further away, while the regularization panel shows that structural choices affected economic stability more than they changed headline RMSE."
    ),
    r"\insertassetfigure{5-minute LSTM net PnL grid across sequence lengths and horizons. This figure complements the RMSE grid by showing that the best trading configuration differs from the best pure forecasting configuration.}{fig:five_min_pnl_grid}{fig_5min_pnl_grid.png}": (
        r"Taken together, Figures~\ref{fig:five_min_rmse_grid} and~\ref{fig:five_min_pnl_grid} are central to the thesis narrative. "
        r"Figure~\ref{fig:five_min_rmse_grid} shows that the LSTM established a broad statistical edge once the study moved to 5-minute data, while Figure~\ref{fig:five_min_pnl_grid} shows that the economically best model is not mechanically the one with the very lowest RMSE. "
        r"This is exactly why the later stages of the report separate forecasting quality from trading quality instead of treating them as interchangeable."
    ),
    r"\insertassetfigure{Standardized common-window comparison between final LSTM finalists and matched MLP baselines. These are the cleanest cross-horizon comparisons because all models are clipped to the same prediction windows.}{fig:standardized_comparison}{fig_standardized_common_window_comparison.png}": (
        r"Figure~\ref{fig:threshold_sweep} and Figure~\ref{fig:standardized_comparison} should be read as the transition from broad model search to defensible final comparison. "
        r"Figure~\ref{fig:threshold_sweep} shows that longer horizons needed stricter trade activation, while Figure~\ref{fig:standardized_comparison} removes the remaining fairness problem by forcing the finalists and their matched baselines onto identical evaluation windows. "
        r"Only after this standardization does it become reasonable to make cross-model claims in the final discussion."
    ),
    r"\insertassetfigure{Summary of the final robustness addendum. The figure condenses the standardized-comparison follow-up into significance, moneyness-region, and regime-level evidence.}{fig:robustness_addendum_main}{fig_final_robustness_addendum_summary.png}": (
        r"Figure~\ref{fig:robustness_addendum_main} is therefore doing more than summarizing extra appendix material. "
        r"It visually links the three most important robustness questions: whether the LSTM wins are statistically significant, whether they are concentrated only in one part of the smile, and whether they survive different volatility regimes. "
        r"The final claim would be much weaker without this figure because the standardized comparison alone would still leave those questions open."
    ),
    r"\insertassetfigure{Execution-sensitivity summary under wider spreads and latency-proxy stress. Absolute profits compress, but the ranking of the strongest finalists remains intact.}{fig:execution_sensitivity_main}{fig_execution_sensitivity.png}": (
        r"Figure~\ref{fig:execution_sensitivity_main} should be interpreted alongside the preceding table rather than as a decorative repetition of it. "
        r"The point of the figure is that all three horizons suffer under harsher execution assumptions, but the compression is gradual rather than catastrophic and the ranking of the strongest LSTM candidates is preserved. "
        r"That is what allows the economic conclusion to remain credible even after costs are made more conservative."
    ),
    r"\insertassetfigure{Placebo target-shuffle comparison for one LSTM and one MLP. Both families deteriorate under the placebo condition, which supports the claim that the real models are learning structured signal rather than benefiting from accidental leakage.}{fig:placebo_comparison_main}{fig_placebo_comparison.png}": (
        r"Figure~\ref{fig:placebo_comparison_main} is especially important for the methodological argument of the report. "
        r"Because both the LSTM and the MLP deteriorate sharply under the shuffled-target placebo, the figure provides a simple visual rebuttal to the concern that the final results are being driven by hidden leakage or a trivial artifact of the pipeline."
    ),
    r"\insertassetfigure{Vega-weighted-loss ablation for the two final H1 LSTM candidates. The figure makes clear that vega-weighting did not improve the tested finalists either statistically or economically.}{fig:vega_weighted_ablation_main}{fig_vega_weighted_ablation.png}": (
        r"Figure~\ref{fig:vega_weighted_ablation_main} turns what might otherwise read like a minor negative result into a useful design conclusion. "
        r"The visual comparison shows that once the model class and data frequency were already strong enough, reweighting the loss by vega did not add incremental value, so the final pipeline remained with the simpler unweighted objective."
    ),
    r"\insertassetfigure{Shape-diagnostics comparison across the final H1 regularization variants. The unregularized and partially regularized variants stay close on RMSE but produce noticeably worse curve geometry.}{fig:shape_diagnostics_main}{fig_shape_diagnostics_comparison.png}": (
        r"Figure~\ref{fig:shape_diagnostics_main} is also where the report most clearly separates pure prediction error from structural forecast quality. "
        r"The figure shows that it is possible to stay near the same RMSE while still generating much less stable curve geometry, which is why the final model selection did not rely on RMSE alone."
    ),
    r"\insertassetfigure{Representative current, forecast, and realized IV curves for the final H1 candidate. This gives a qualitative view of how the model adjusts the smile shape rather than only reporting aggregate error metrics.}{fig:repr_curve_h1_main}{fig_representative_forecast_vs_realized_curve_h1.png}": (
        r"Figure~\ref{fig:repr_curve_h1_main} is included precisely because the aggregate metrics cannot show how the model moves the curve. "
        r"In this representative non-degenerate example, the forecast does not merely shift the curve up or down uniformly; it adjusts the left wing, the ATM region, and the right side by different amounts. "
        r"That qualitative behavior is consistent with the claim that the final LSTM is learning local smile dynamics rather than producing a trivial level forecast."
    ),
    r"\insertassetfigure{Execution-aware equity curve for the final selected model on the saved backtest output. This figure is placed in the main discussion because the main claim is not purely statistical; it also depends on the stability of the economic layer.}{fig:equity_curve_main}{fig_execution_equity_curve_final_model.png}": (
        r"Figure~\ref{fig:equity_curve_main} therefore acts as the economic counterpart to the standardized RMSE evidence. "
        r"The important point is not that the curve rises monotonically, but that the cumulative path reflects repeated incremental gains rather than one isolated burst. "
        r"That makes the final economic conclusion more defensible than a report that cited only one summary PnL number."
    ),
    r"\insertassetfigure{Representative forecast versus realized curve: H12.}{fig:app_repr_h12}{fig_representative_forecast_vs_realized_curve_h12.png}": (
        r"Figure~\ref{fig:app_repr_h12} extends the qualitative curve check to the medium-horizon winner and shows that the h12 model still produces structured smile adjustments rather than a uniform level shift."
    ),
    r"\insertassetfigure{Representative forecast versus realized curve: H24.}{fig:app_repr_h24}{fig_representative_forecast_vs_realized_curve_h24.png}": (
        r"Figure~\ref{fig:app_repr_h24} provides the same qualitative check for the longest retained horizon, where the task is hardest and the smile geometry is correspondingly more difficult to preserve."
    ),
}

APPENDIX_TABLE_CALLS = [
    ("Baseline walk-forward summary.", "tab:app_baseline_walkforward", "tab_baseline_walkforward_all.csv"),
    ("LSTM ablation summary.", "tab:app_lstm_ablation", "tab_lstm_ablation_summary.csv"),
    ("5-minute LSTM grid summary.", "tab:app_lstm_grid", "tab_5min_lstm_grid_summary.csv"),
    ("5-minute baseline grid summary.", "tab:app_baseline_grid", "tab_5min_baseline_grid_summary.csv"),
    ("Best threshold by finalist.", "tab:app_best_threshold", "tab_best_threshold_by_finalist.csv"),
    ("Standardized overall summary.", "tab:app_standardized_summary", "tab_standardized_overall_summary.csv"),
    ("Final robustness addendum.", "tab:app_robustness_table", "tab_final_robustness_addendum.csv"),
    ("Daily phase results.", "tab:app_daily_phase", "tab_daily_phase_results.csv"),
    ("Early hourly results.", "tab:app_early_hourly", "tab_early_hourly_results.csv"),
    ("Hourly capacity sweep.", "tab:app_hourly_capacity", "tab_hourly_capacity_sweep.csv"),
    ("Hourly sequence-length ablation.", "tab:app_hourly_seq", "tab_hourly_seq_length_ablation.csv"),
    ("Hourly horizon ablation.", "tab:app_hourly_horizon", "tab_hourly_horizon_ablation.csv"),
    ("Hourly regularization ablation.", "tab:app_hourly_reg", "tab_hourly_regularization_ablation.csv"),
    ("Frozen finalist summary.", "tab:app_finalists", "tab_finalists_summary.csv"),
    ("Feature definitions.", "tab:app_feature_defs", "tab_feature_definitions.csv"),
]


def explicit_appendix_table_block(caption: str, label: str, filename: str) -> str:
    return (
        "\\begin{landscape}\n"
        "\\begin{center}\n"
        f"\\captionof{{table}}{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\lstinputlisting[style=csvstyle]{{\\assetroot/tables/{filename}}}\n"
        "\\end{center}\n"
        "\\end{landscape}"
    )


def apply_once(text: str, target: str, replacement: str) -> str:
    if target not in text:
        raise ValueError(f"Expected target text not found: {target[:80]!r}")
    return text.replace(target, replacement, 1)


def patch_report(text: str) -> str:
    if LISTINGS_PACKAGE not in text:
        text = apply_once(text, "\\usepackage{url}\n", "\\usepackage{url}\n" + LISTINGS_PACKAGE)
    if LISTINGS_STYLE not in text:
        marker = "\\newcommand{\\assetroot}{thesis_report_assets}\n"
        text = apply_once(text, marker, marker + "\n" + LISTINGS_STYLE + "\n")
    if OLD_APPENDIX_MACRO in text:
        text = apply_once(text, OLD_APPENDIX_MACRO, DISABLED_APPENDIX_MACRO)
    if INTERMEDIATE_APPENDIX_MACRO in text:
        text = apply_once(text, INTERMEDIATE_APPENDIX_MACRO, DISABLED_APPENDIX_MACRO)
    if NEW_APPENDIX_MACRO in text:
        text = apply_once(text, NEW_APPENDIX_MACRO, DISABLED_APPENDIX_MACRO)
    text = text.replace("this \\.tex file", "this .tex file")
    if FEATURE_TABLE_CALL in text:
        text = apply_once(text, FEATURE_TABLE_CALL, FEATURE_TABLE_BLOCK)
    if OLD_FEATURE_SECTION_BLOCK in text:
        text = apply_once(text, OLD_FEATURE_SECTION_BLOCK, RICH_FEATURE_SECTION_BLOCK)

    appendix_start = text.find(APPENDIX_TABLES_SECTION_TITLE)
    bibliography_marker = "\\begin{thebibliography}{9}"
    if appendix_start != -1:
        appendix_end = text.find(bibliography_marker, appendix_start)
        if appendix_end == -1:
            raise ValueError("Appendix CSV section found, but bibliography marker was not found after it.")
        text = text[:appendix_start] + APPENDIX_TABLES_SECTION_REPLACEMENT + "\n\n" + text[appendix_end:]

    for caption, label, filename in APPENDIX_TABLE_CALLS:
        macro_call = f"\\insertappendixcsv{{{caption}}}{{{label}}}{{{filename}}}"
        block = explicit_appendix_table_block(caption, label, filename)
        if macro_call in text:
            text = apply_once(text, macro_call, block)

    for figure_call, follow_up in FOLLOW_UPS.items():
        if follow_up not in text:
            text = apply_once(text, figure_call, figure_call + "\n\n" + follow_up)

    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch the final thesis report to use asset-safe tables and richer figure discussion.")
    parser.add_argument("tex_path", help="Path to the .tex report file to patch in place.")
    parser.add_argument("--backup-suffix", default=".bak")
    args = parser.parse_args()

    tex_path = Path(args.tex_path)
    original = tex_path.read_text(encoding="utf-8")
    patched = patch_report(original)
    backup_path = tex_path.with_name(tex_path.name + args.backup_suffix)
    if not backup_path.exists():
        backup_path.write_text(original, encoding="utf-8")
    tex_path.write_text(patched, encoding="utf-8")
    print(f"Patched report: {tex_path}")
    print(f"Backup: {backup_path}")


if __name__ == "__main__":
    main()
