"""Train-only preprocessing for anonymous SECOM sensor features."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class LeakageSafePreprocessor:
    max_features: int = 128
    variance_epsilon: float = 1e-12
    medians_: np.ndarray | None = None
    numeric_indices_: np.ndarray | None = None
    indicator_indices_: np.ndarray | None = None
    candidate_centers_: np.ndarray | None = None
    candidate_scales_: np.ndarray | None = None
    selected_candidate_indices_: np.ndarray | None = None
    selected_scores_: np.ndarray | None = None
    candidate_source_indices_: np.ndarray | None = None
    candidate_kinds_: np.ndarray | None = None
    train_rows_seen_: int = 0

    def fit(self, features: np.ndarray, labels: np.ndarray) -> LeakageSafePreprocessor:
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        if features.ndim != 2 or features.shape[0] != labels.size:
            raise ValueError("invalid training feature/label shapes")
        if np.unique(labels).size != 2:
            raise ValueError("training data must contain both classes")
        if self.max_features < 1:
            raise ValueError("max_features must be positive")

        observed = np.sum(~np.isnan(features), axis=0)
        medians = np.zeros(features.shape[1], dtype=np.float64)
        usable = observed > 0
        medians[usable] = np.nanmedian(features[:, usable], axis=0)
        imputed = np.where(np.isnan(features), medians, features)

        numeric_std = np.std(imputed, axis=0)
        numeric_indices = np.flatnonzero(
            usable & np.isfinite(numeric_std) & (numeric_std > self.variance_epsilon)
        )
        missing = np.isnan(features).astype(np.float64)
        missing_std = np.std(missing, axis=0)
        indicator_indices = np.flatnonzero(
            (observed > 0)
            & (observed < features.shape[0])
            & (missing_std > self.variance_epsilon)
        )
        if numeric_indices.size == 0 and indicator_indices.size == 0:
            raise ValueError("no usable training features")

        numeric_candidates = imputed[:, numeric_indices]
        indicator_candidates = missing[:, indicator_indices]
        candidates = np.concatenate((numeric_candidates, indicator_candidates), axis=1)
        centers = np.mean(candidates, axis=0)
        scales = np.std(candidates, axis=0)
        scales = np.where(scales > self.variance_epsilon, scales, 1.0)
        standardized = (candidates - centers) / scales

        positive = standardized[labels == 1]
        negative = standardized[labels == 0]
        numerator = np.square(np.mean(positive, axis=0) - np.mean(negative, axis=0))
        denominator = (
            np.var(positive, axis=0)
            + np.var(negative, axis=0)
            + self.variance_epsilon
        )
        scores = np.nan_to_num(
            numerator / denominator,
            nan=0.0,
            posinf=np.finfo(np.float64).max,
            neginf=0.0,
        )
        candidate_source_indices = np.concatenate(
            (numeric_indices, indicator_indices)
        ).astype(np.int64)
        candidate_kinds = np.asarray(
            ["value"] * numeric_indices.size + ["missing_indicator"] * indicator_indices.size
        )
        # Stable tie-breaking by candidate order makes selection reproducible.
        ranking = np.argsort(-scores, kind="stable")
        selected = ranking[: min(self.max_features, ranking.size)]

        self.medians_ = medians
        self.numeric_indices_ = numeric_indices.astype(np.int64)
        self.indicator_indices_ = indicator_indices.astype(np.int64)
        self.candidate_centers_ = centers
        self.candidate_scales_ = scales
        self.selected_candidate_indices_ = selected.astype(np.int64)
        self.selected_scores_ = scores[selected]
        self.candidate_source_indices_ = candidate_source_indices
        self.candidate_kinds_ = candidate_kinds
        self.train_rows_seen_ = int(features.shape[0])
        return self

    def _check_fitted(self) -> None:
        required = (
            self.medians_,
            self.numeric_indices_,
            self.indicator_indices_,
            self.candidate_centers_,
            self.candidate_scales_,
            self.selected_candidate_indices_,
            self.selected_scores_,
            self.candidate_source_indices_,
            self.candidate_kinds_,
        )
        if any(value is None for value in required):
            raise RuntimeError("preprocessor is not fitted")

    def transform(self, features: np.ndarray) -> np.ndarray:
        self._check_fitted()
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.medians_.size:
            raise ValueError(
                f"expected (*, {self.medians_.size}) features, got {features.shape}"
            )
        imputed = np.where(np.isnan(features), self.medians_, features)
        numeric = imputed[:, self.numeric_indices_]
        indicators = np.isnan(features[:, self.indicator_indices_]).astype(np.float64)
        candidates = np.concatenate((numeric, indicators), axis=1)
        standardized = (
            candidates - self.candidate_centers_
        ) / self.candidate_scales_
        selected = standardized[:, self.selected_candidate_indices_]
        if not np.isfinite(selected).all():
            raise ValueError("preprocessing produced non-finite values")
        return np.asarray(selected, dtype=np.float32)

    @property
    def output_features(self) -> int:
        self._check_fitted()
        return int(self.selected_candidate_indices_.size)

    def manifest(self) -> dict[str, Any]:
        self._check_fitted()
        selected_sources = self.candidate_source_indices_[
            self.selected_candidate_indices_
        ]
        selected_kinds = self.candidate_kinds_[self.selected_candidate_indices_]
        selected = [
            {
                "output_index": output_index,
                "source_feature_index": int(source_index),
                "kind": str(kind),
                "fisher_score_train_only": float(score),
            }
            for output_index, (source_index, kind, score) in enumerate(
                zip(
                    selected_sources,
                    selected_kinds,
                    self.selected_scores_,
                    strict=True,
                )
            )
        ]
        return {
            "schema": "fabyield_preprocessor.v1",
            "fit_scope": "train_partition_only",
            "train_rows_seen": self.train_rows_seen_,
            "input_features": int(self.medians_.size),
            "numeric_candidates": int(self.numeric_indices_.size),
            "missing_indicator_candidates": int(self.indicator_indices_.size),
            "output_features": self.output_features,
            "steps": [
                "training_median_imputation",
                "training_only_zero_variance_filter",
                "training_only_standardization",
                "training_label_only_fisher_feature_selection",
                "missingness_indicators_when_variable_in_training",
            ],
            "selected_features": selected,
        }

    def save_npz(self, path: Path) -> None:
        self._check_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                medians=self.medians_,
                numeric_indices=self.numeric_indices_,
                indicator_indices=self.indicator_indices_,
                candidate_centers=self.candidate_centers_,
                candidate_scales=self.candidate_scales_,
                selected_candidate_indices=self.selected_candidate_indices_,
                selected_scores=self.selected_scores_,
                candidate_source_indices=self.candidate_source_indices_,
                candidate_kinds=self.candidate_kinds_,
            )
        temporary.replace(path)

    def state_sha256(self) -> str:
        self._check_fitted()
        digest = hashlib.sha256()
        for array in (
            self.medians_,
            self.numeric_indices_,
            self.indicator_indices_,
            self.candidate_centers_,
            self.candidate_scales_,
            self.selected_candidate_indices_,
            self.selected_scores_,
            self.candidate_source_indices_,
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
            digest.update(contiguous.tobytes())
        digest.update("\0".join(str(value) for value in self.candidate_kinds_).encode())
        return digest.hexdigest()
