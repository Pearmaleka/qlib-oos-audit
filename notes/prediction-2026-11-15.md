# Prediction recorded before the ALSTM and LSTM results

Written on 2026-11-15 at about 05:05 (Hong Kong time), after the tree models,
the MLP, Linear and GRU had been scored and after one seed of a LightGBM run
with a rank label, and before any ALSTM or LSTM prediction existed.

Observation so far: the official configs that train on a cross-sectionally
*ranked* label (Linear, GRU) lost less Rank IC after July 2020 than the configs
that train on a *z-scored* label (LightGBM, XGBoost, CatBoost, DoubleEnsemble,
MLP). One seed of LightGBM with the rank label kept a post-2020 Rank IC of
about 0.029, against 0.015 for the official LightGBM.

Hypothesis: the label normalisation, not the model family, explains most of
the difference in robustness after 2020.

Predictions, to be checked against runs that had not finished when this was written:

1. ALSTM and LSTM (official configs, ranked label) keep at least 40% of their
   published-window Rank IC after July 2020, more than every z-scored-label model
   (which kept 21-34%).
2. The other four seeds of LightGBM with a ranked label keep a post-2020 Rank IC
   above 0.02.
3. XGBoost, CatBoost and the MLP with a ranked label have a higher post-2020
   Rank IC than with their official z-scored label.
4. Linear with a z-scored label has a lower post-2020 Rank IC than the official
   Linear.
