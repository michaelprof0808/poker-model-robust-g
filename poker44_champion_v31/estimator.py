"""Compact, leakage-safe supervised estimator for Poker44 bot detection.

The estimator is a soft-voting ensemble of a regularized categorical linear
model (logistic regression over the fixed one-hot fraction features) and a
complementary nonlinear model (histogram gradient boosting). Probabilities are
calibrated out-of-fold with ``CalibratedClassifierCV`` so no calibration data
leaks from training. Every component uses a fixed seed for determinism, and the
estimator degrades gracefully on tiny or single-class data.
"""

from __future__ import annotations

import numpy as np

SEED = 44


class ChampionEstimator:
    """Fit/predict wrapper returning P(bot) as a finite value in [0, 1]."""

    def __init__(self, seed: int = SEED, calibrate: bool = True):
        self.seed = seed
        self.calibrate = calibrate
        self._ensemble = None
        self._calibrator = None
        self._constant: float | None = None
        self.n_features_: int | None = None
        # Populated at fit time; surfaced for the artifact manifest.
        self.calibration_: str = "none"

    @property
    def hyperparameters(self) -> dict:
        return {
            "seed": self.seed,
            "calibrate": self.calibrate,
            "linear": {"C": 0.5, "class_weight": "balanced", "max_iter": 2000, "solver": "lbfgs"},
            "gb": {
                "learning_rate": 0.08,
                "max_depth": 3,
                "max_iter": 200,
                "l2_regularization": 1.0,
                "min_samples_leaf": 5,
            },
            "voting": "soft",
            "weights": [1.0, 1.0],
        }

    def _build_ensemble(self):
        from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        linear = Pipeline(
            steps=[
                ("scale", StandardScaler(with_mean=True, with_std=True)),
                (
                    "logreg",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=self.seed,
                    ),
                ),
            ]
        )
        nonlinear = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_depth=3,
            max_iter=200,
            l2_regularization=1.0,
            min_samples_leaf=5,
            random_state=self.seed,
        )
        return VotingClassifier(
            estimators=[("linear", linear), ("gb", nonlinear)],
            voting="soft",
            weights=[1.0, 1.0],
        )

    def fit(self, X, y, groups=None) -> ChampionEstimator:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        self.n_features_ = X.shape[1]
        self._calibrator = None
        self.calibration_ = "none"

        classes = np.unique(y)
        if classes.size < 2:
            # Single-class training set: predict a constant near the sole class,
            # kept strictly inside (0, 1) so downstream metrics stay finite.
            sole = int(classes[0]) if classes.size else 0
            self._constant = 0.95 if sole == 1 else 0.05
            self._ensemble = None
            return self
        self._constant = None

        ensemble = self._build_ensemble()
        min_class = int(np.min(np.bincount(y)))
        if self.calibrate and min_class >= 2:
            self._fit_calibrator(ensemble, X, y, groups, min_class)
        # The final ensemble is always fit on ALL data; calibration is a
        # monotone post-map learned out-of-fold so it never leaks.
        ensemble.fit(X, y)
        self._ensemble = ensemble
        return self

    def _fit_calibrator(self, ensemble, X, y, groups, min_class) -> None:
        """Learn a monotone sigmoid calibrator from out-of-fold probabilities.

        When >= 2 distinct release groups are available the folds are
        release-grouped (StratifiedGroupKFold) so no release straddles a
        calibration train/validation split. With a single group we fall back to
        an honest, documented item-level stratified OOF (a single release cannot
        support release-grouped calibration); if even that is infeasible, no
        calibrator is fit.
        """
        from sklearn.base import clone
        from sklearn.model_selection import (
            StratifiedGroupKFold,
            StratifiedKFold,
            cross_val_predict,
        )

        n_groups = len(np.unique(groups)) if groups is not None else 0
        splitter = None
        split_groups = None
        mode = "none"
        if groups is not None and n_groups >= 2:
            splits = min(3, n_groups)
            splitter = StratifiedGroupKFold(n_splits=splits)
            split_groups = np.asarray(groups)
            mode = "grouped_oof"
        elif min_class >= 2:
            splits = max(2, min(3, min_class))
            splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=self.seed)
            mode = "stratified_oof"

        if splitter is None:
            return
        try:
            oof = cross_val_predict(
                clone(ensemble), X, y, cv=splitter, groups=split_groups,
                method="predict_proba",
            )
            col = self._bot_column(np.unique(y))
            oof_bot = np.clip(oof[:, col].astype(np.float64), 1e-6, 1 - 1e-6)
            calibrator = self._fit_sigmoid(oof_bot, y)
            # Calibration may improve probability scale, but it must never
            # reverse P(bot) ordering. Tiny OOF samples can fit a negative
            # sigmoid slope even when the final all-data ensemble is correctly
            # oriented; reject that calibrator and keep raw probabilities.
            slope = float(calibrator.coef_[0, 0])
            if not np.isfinite(slope) or slope <= 0.0:
                return
            self._calibrator = calibrator
            self.calibration_ = mode
        except Exception:  # noqa: BLE001 -- calibration is best-effort; fall back
            self._calibrator = None
            self.calibration_ = "none"

    def _fit_sigmoid(self, scores: np.ndarray, y: np.ndarray):
        from sklearn.linear_model import LogisticRegression

        logit = np.log(scores / (1.0 - scores)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(logit, y)
        return lr

    @staticmethod
    def _bot_column(classes) -> int:
        classes = list(classes)
        return classes.index(1) if 1 in classes else len(classes) - 1

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self._constant is not None or self._ensemble is None:
            value = 0.5 if self._constant is None else self._constant
            return np.full(X.shape[0], float(value), dtype=np.float64)
        proba = self._ensemble.predict_proba(X)
        col = self._bot_column(self._ensemble.classes_)
        out = np.clip(proba[:, col].astype(np.float64), 1e-6, 1 - 1e-6)
        if self._calibrator is not None:
            logit = np.log(out / (1.0 - out)).reshape(-1, 1)
            out = self._calibrator.predict_proba(logit)[:, 1].astype(np.float64)
        return np.clip(out, 0.0, 1.0)
