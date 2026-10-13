# Football Experiments (2021)

Two self-study experiments in predicting football match outcomes with neural networks,
written in August 2021 while I was teaching myself deep learning from the Manning books.

This repository is kept as a **historical record**, not as a showcase of how I write code today.
Its value is in showing where I started; the "What I would change today" section shows where I went from there.

## The experiments

Both experiments share the same data pipeline, built on match statistics exported from Wyscout
(about 100 per-team metrics per match: shots, passes, duels, recoveries, pressing zones, set pieces, ...).

| Folder | Question | Model |
|---|---|---|
| [`categorical/`](categorical/) | Given a match's statistics, was it a home win, a draw or an away win? | Dense network with hyperparameter search via KerasTuner, comparing Random Search, Bayesian Optimization and Hyperband |
| [`recurrent/`](recurrent/) | Given a team's last three matches, what will the fourth look like? | Bidirectional LSTM followed by stacked LSTM layers, regressing the next match's statistics |

### Pipeline

1. **Parser** (`*Parser.py`): cleans each team's Excel export, renames the multi-level Wyscout headers,
   extracts the formation and joins home and away rows of the same match by date.
2. **Merger** (`*Merger.py`): concatenates all teams, encodes teams and formations as categories,
   computes the score differential and the HOME / DRAW / AWAY label, drops the goal columns
   and adds engineered features such as
   *defensive duels lost* and *shots conceded per defensive duel lost*.
3. **Model** (`categorical.py`, `recurrent.py`): scaling with `QuantileTransformer`, training,
   TensorBoard logging, evaluation and a prediction helper. Intermediate results could also be stored in PostgreSQL.

## History

The commits keep their original 2021 dates.

- `first commit` (13 Aug 2021): the complete, coherent version of each experiment.
- `major modifications` (15–16 Aug 2021): I started a refactor. Among other fixes it introduced a separate
  scaler for the targets in `recurrent/` and stopped re-fitting the scaler on prediction data in `categorical/`.
  The refactor moved the pipeline into shared `Parser`, `Merger` and `PredictionDataBuilder` modules.
- Many later commits were reset and are gone, together with the corrections they contained.
  What survives is only this early stage of the work, so several of the issues listed below
  had already been addressed in versions that no longer exist.
  For the same reason the latest version of the two scripts imports modules that are not in the repository.
- In 2026 I merged the two original repositories into this one and restored the pipeline files from the first commits.
  The 2021 code has not been edited.

The Wyscout data is licensed and is not included, so the code cannot be run as-is.

## Design choices

- **Score-Differential as a feature (deliberate).** In `categorical/` the goal difference is kept among the inputs on purpose:
  the network learns how match statistics, including goal difference, map to the outcome, and at prediction time
  `PredictionDataBuilder` was meant to supply those features from each team's recent matches, where goal difference
  acts as a form indicator. The trade-off: during training the label is exactly the sign of that column, so the reported
  accuracy measures how well the network learned the rule, not how well it forecasts. Today I would also train a variant
  without it, to see how much the other statistics carry on their own.

## What I would change today

**Evaluation**
- Split the data by time (train on earlier matches, test on later ones) instead of shuffling,
  so the score reflects real forecasting rather than interpolation between neighbouring matches.
- Fit the scalers on the training set only. They are currently fitted on the whole dataset before the split.
- Split `X` and `y` in the same `train_test_split` call in `recurrent/`. They stay aligned only because both calls use the same seed.
- Compare against simple baselines (always predict a home win, bookmaker odds) before reading anything into the accuracy.

**Modelling**
- Sample the learning rate, regularization and decay rate on a log scale. With `step=10` on a linear scale,
  KerasTuner never moves them off their minimum value.
- Encode teams and formations with embeddings or one-hot vectors rather than integer codes.
- Use a fresh `Dropout` layer per block instead of reusing one instance.
- Guard the engineered ratios against division by zero.

**Engineering**
- Extract one shared pipeline package instead of two near-identical copies of the parser and merger.
- Do not delete source data files during parsing (`RecurrentParser.adjust_rows`), and fix its
  `len(dataframe.index - 1)` typo.
- Generate unique output file names deterministically rather than with a random suffix.
- Fix the column-renaming typo that labels the back-pass columns as forward passes.
- Turn the scripts into a CLI with configuration, remove module-level side effects, pin dependencies and add tests:
  unit tests for the parser on a small fixture, plus a check that no label information reaches the features.

## Stack

Python, TensorFlow 2 / Keras, KerasTuner, scikit-learn, pandas, PostgreSQL (SQLAlchemy), TensorBoard, matplotlib.
