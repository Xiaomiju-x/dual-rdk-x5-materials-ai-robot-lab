"""Immutable contracts for the NIST CHIPS simulated-SEM candidate."""
from __future__ import annotations

SOURCE_ID = "nist_chips_sem_metrology_v1_0"
SOURCE_TITLE = "Detection Limits for SEM Image Segmentation"
SOURCE_DOI = "10.18434/mds2-3838"
SOURCE_VERSION = "1.0"
SOURCE_LICENSE = "NIST Open License"
SOURCE_LICENSE_URL = "https://www.nist.gov/open/license"
SOURCE_LANDING_PAGE = "https://data.nist.gov/od/id/mds2-3838"

CLAIM_BOUNDARY = (
    "This candidate uses simulated SEM images of quasi-circular structures. "
    "It is not trained or validated on real wafer defects or production fab data."
)

INPUT_SIZE = 128
SOURCE_IMAGE_SIZE = 512
TRAIN_SETS = (1, 2, 3, 4, 5)
LOCKED_TEST_SET = 6
VALID_SUBSET_TRAIN_SETS = (1, 2, 3, 5)
BOUNDARY_TOLERANCE_PX = 2
DEFAULT_THRESHOLD = 0.5

ARCHIVES = {
    "intensity_sets.zip": {
        "official_url": "https://data.nist.gov/od/ds/mds2-3838/intensity_sets.zip",
        "sha256": "5cd9f4caff80e9afab83515032347a17e9974554ea148f01280090504807e078",
        "required_for_full_corpus": True,
        "description": "Six sets of 567 simulated SEM images.",
    },
    "mask_sets.zip": {
        "official_url": "https://data.nist.gov/od/ds/mds2-3838/mask_sets.zip",
        "nist_cache_url": (
            "https://nist-oar-cache.s3.amazonaws.com/prd/fst4/"
            "mds2-3838/mask_sets.zip"
        ),
        "bytes": 276853,
        "sha256": "5925dc95478e2cfc3c9ec54bfef888c7596db35fdf41bb929d6b96b8562ab562",
        "required_for_full_corpus": True,
        "description": "Six masks and six maximum-contrast/minimum-noise intensity images.",
    },
    "metrics_sets.zip": {
        "official_url": "https://data.nist.gov/od/ds/mds2-3838/metrics_sets.zip",
        "nist_cache_url": (
            "https://nist-oar-cache.s3.amazonaws.com/prd/fst4/"
            "mds2-3838/metrics_sets.zip"
        ),
        "bytes": 787689,
        "sha256": "a2b89522c6d8d3fae6afefa97dd99932b564d6455beb96ae0c69083622cffa36",
        "required_for_full_corpus": True,
        "description": "Image-quality and official model evaluation metrics.",
    },
}

INTENSITY_MEMBER = "masks/set{set_id}_cex_noise_000_contrast_100.tiff"
MASK_MEMBER = "masks/mask_set{set_id}_cex_noise_000_contrast_100.tiff"
METRICS_MEMBER = "Image_quality/data_quality_results_set{set_id}.csv"

EXPECTED_METRICS_ROWS_PER_SET = 567
MODEL_CHANNELS = (8, 16, 24)
MODEL_NAME = "SEM-Metrology-X5-Lite"
MODEL_VERSION = "official-subset-baseline-v1"
