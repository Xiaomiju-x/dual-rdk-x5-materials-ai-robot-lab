"""Immutable source, split, architecture, and evaluation contracts."""
from __future__ import annotations

SOURCE_ID = "nist_chips_sem_metrology_v1_0"
SOURCE_TITLE = "Detection Limits for SEM Image Segmentation"
SOURCE_DOI = "10.18434/mds2-3838"
SOURCE_VERSION = "1.0"
SOURCE_LANDING_PAGE = "https://data.nist.gov/od/id/mds2-3838"
SOURCE_LICENSE = "NIST Open License"
SOURCE_LICENSE_URL = "https://www.nist.gov/open/license"

CLAIM_BOUNDARY = (
    "NIST created these images with simulation and custom software. This "
    "candidate concerns simulated SEM dimensional-metrology segmentation; it "
    "is not trained or validated on real wafer defects or production fab data."
)

TRAIN_SETS = (1, 2, 3, 4, 5)
SEALED_TEST_SET = 6
SOURCE_IMAGE_SIZE = 512
INPUT_SIZE = 128
EXPECTED_IMAGES_PER_SET = 567
EXPECTED_NOISE_LEVELS = 27
EXPECTED_CONTRAST_LEVELS = 21
SPLIT_SEED = "sem-metrology-v2-quality-cell-split-20260728"

ARCHIVES = {
    "intensity_sets.zip": {
        "official_url": "https://data.nist.gov/od/ds/mds2-3838/intensity_sets.zip",
        "bytes": 419_556_677,
        "sha256": "5cd9f4caff80e9afab83515032347a17e9974554ea148f01280090504807e078",
    },
    "mask_sets.zip": {
        "official_url": "https://data.nist.gov/od/ds/mds2-3838/mask_sets.zip",
        "bytes": 276_853,
        "sha256": "5925dc95478e2cfc3c9ec54bfef888c7596db35fdf41bb929d6b96b8562ab562",
    },
    "metrics_sets.zip": {
        "official_url": "https://data.nist.gov/od/ds/mds2-3838/metrics_sets.zip",
        "bytes": 787_689,
        "sha256": "a2b89522c6d8d3fae6afefa97dd99932b564d6455beb96ae0c69083622cffa36",
    },
}

METRICS_MEMBER = "Image_quality/data_quality_results_set{set_id}.csv"
INTENSITY_REFERENCE_MEMBER = "masks/set{set_id}_cex_noise_000_contrast_100.tiff"
MASK_MEMBER = "masks/mask_set{set_id}_cex_noise_000_contrast_100.tiff"

FROZEN_BASELINE = {
    "name": "SEM-Metrology-X5-Lite",
    "version": "official-subset-baseline-v1",
    "historical_set6_dice": 0.5393491013712962,
    "parameter_count": 20_385,
    "boundary": (
        "Historical set6 result from the frozen subset baseline. It is not a "
        "v2 validation target and cannot be used for v2 model selection."
    ),
}

MODEL_CONTRACT = {
    "name": "SEM-Metrology-X5-TinyUNet-Q",
    "version": "v2-candidate",
    "input_shape": [1, 1, INPUT_SIZE, INPUT_SIZE],
    "max_parameters": 250_000,
    "allowed_module_families": [
        "Conv2d",
        "ReLU",
        "MaxPool2d",
        "UpsampleNearest",
        "Concat",
        "AdaptiveAvgPool2d",
    ],
    "mapper_status": "NOT_RUN",
    "x5_status": "NOT_RUN",
}

# These thresholds are fixed before any v2 set6 access.
NON_TEST_GATE = {
    "schema": "icmat_sem_v2_non_test_gate_spec.v2",
    "data": {
        "require_all_train_sets": list(TRAIN_SETS),
        "images_per_set": EXPECTED_IMAGES_PER_SET,
        "require_official_archive_sha256": True,
        "require_binary_nonidentical_masks": True,
    },
    "architecture": {
        "max_parameters": MODEL_CONTRACT["max_parameters"],
        "static_input_shape": MODEL_CONTRACT["input_shape"],
        "forbidden_modules": ["ConvTranspose2d", "LSTM", "GRU", "MultiheadAttention"],
    },
    "performance": {
        "macro_dice_min": 0.80,
        "worst_quality_quartile_dice_min": 0.62,
        "boundary_f1_min": 0.70,
        "fnr_max": 0.25,
        "fpr_max": 0.15,
        "delta_vs_retrained_frozen_baseline_min": 0.08,
        "delta_vs_best_simple_threshold_min": 0.12,
    },
    "calibration": {
        "quality_ece_max": 0.10,
        "conformal_alpha": 0.10,
        "selective_coverage_min": 0.50,
        "accepted_macro_dice_min": 0.75,
    },
    "set6_policy": {
        "accesses_allowed_after_pass": 1,
        "used_for_model_selection": False,
        "mapper_allowed": False,
        "x5_allowed": False,
        "production_integration_allowed": False,
    },
}
