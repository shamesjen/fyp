# Thesis Report Asset Manifest

This folder collects the final thesis-ready asset pack for Overleaf import.

## What Was Generated

- Figures: 23
- Tables: 15
- Summary files: 2

## Source Policy

- Preferred existing saved artifacts over rerunning experiments.
- Reused final standardized 5-minute outputs whenever common-window comparisons were required.
- Reused final robustness addendum CSVs for DM tests, bucket results, regime analysis, execution sensitivity, placebo, vega-weighted loss, and shape diagnostics.
- Reused saved hourly and early-phase summary CSVs for historical-phase tables.

## Copied vs Transformed vs Newly Plotted

- Direct CSV copies: 5
- Derived / transformed CSV tables: 10
- Newly plotted PNG figures: 23

## Placeholders

- Placeholder assets: 0

No placeholder assets were required.

## Notes And Assumptions

- `fig_execution_equity_curve_final_model.png` uses the strongest neural model from the final overlap-safe benchmark, `xLSTM`, averaged across seeds.
- `fig_carry_retuned_tradeoff.png` is the main trade-off figure for the thesis and compares standardized-path RMSE with tuned-threshold net PnL on the retuned carry benchmark, including the plain LSTM control, the attention-pooled LSTM control, the transformer control, the xLSTM winner, and the strongest baselines.
- `fig_xlstm_thresholded_equity_curve.png` is copied from the best-threshold xLSTM carry benchmark (`b2/e256`, threshold `0.0050`) and is the main equity-curve figure referenced by `thesis_rewritten.tex`.
- `fig_svi_carry_fit_mix.png` summarizes the fit-method mix in the carry-plus-SVI panel and is used in the rewritten SVI sensitivity section.
- `fig_bootstrap_net_pnl_ci.png` plots 95% circular block-bootstrap confidence intervals for tuned net PnL on the final retuned-carry economic frontier, including the transformer control alongside the strongest baseline cluster, and is used in `thesis.tex`.
- `fig_bootstrap_winner_delta_ci.png` plots the xLSTM winner's tuned net-PnL lead over the strongest challengers, including the transformer control, with 95% circular block-bootstrap confidence intervals; the vertical axis lists challenger models and positive values indicate an xLSTM lead.
- `fig_neural_ablation_retuned_carry.png` summarizes the nine recurrent variants on the final retuned carry benchmark, including the attention-pooled LSTM intermediary, and is used in the appendix ablation section of `thesis_final.tex`.
- Representative forecast-vs-realized curve figures use the observation whose total curve MAE is closest to the sample median, to avoid cherry-picking best or worst examples.
- Hourly regularization exports include the saved `base`, `shape off`, and `smoothness off` variants. A saved hourly `both off` run was not found.
- Full source-to-output mapping is in `report_asset_map.csv`.
