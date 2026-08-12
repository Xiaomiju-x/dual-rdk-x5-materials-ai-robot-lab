"""Immutable data, feature, target, and deployment contracts for PropNet."""
from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = "icmat_propnet.v2"
SPLIT_SEED = 20260728

SOURCE_ID = "nist_jarvis_dft"
SOURCE_VERSION = "v11 file 64391379, dated 2025-09-24"
SOURCE_DOI = "10.6084/m9.figshare.6815699.v11"
SOURCE_LICENSE = "CC BY 4.0"
SOURCE_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
SOURCE_EXPECTED_BYTES = 48_447_610
SOURCE_EXPECTED_MD5 = "c3179161ef0d029cf1fa70da463aac6e"
SOURCE_EXPECTED_SHA256 = (
    "f9e0a3309f0000d5de1ec9e49c93963109ea45e63f451103f8f6595d2eabf7f5"
)
SOURCE_EXPECTED_MEMBER = "jdft_3d-9-24-2025.json"
SOURCE_EXPECTED_ROWS = 93_902

CLAIM_BOUNDARY = (
    "Computed-property screening on a version-pinned public JARVIS-DFT dataset. "
    "This is not experimental ground truth, fab-line validation, a BPU binary, "
    "live-X5 evidence, or production integration."
)

# Atomic-number order. Fractions in this order form the first 118 inputs.
ELEMENTS = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)
ELEMENT_TO_Z = {symbol: index + 1 for index, symbol in enumerate(ELEMENTS)}

CRYSTAL_SYSTEMS = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
)

FEATURE_NAMES = (
    tuple(f"element_fraction_{symbol}" for symbol in ELEMENTS)
    + (
        "composition_n_elements_norm",
        "composition_entropy_norm",
        "atomic_number_mean_norm",
        "atomic_number_std_norm",
        "atomic_number_min_norm",
        "atomic_number_max_norm",
    )
    + tuple(f"stoich_fraction_rank_{index}" for index in range(1, 9))
    + (
        "nat_log1p",
        "density_log1p",
        "volume_per_atom_log1p",
        "lattice_ratio_sorted_1",
        "lattice_ratio_sorted_2",
        "lattice_ratio_sorted_3",
        "angle_cos_sorted_1",
        "angle_cos_sorted_2",
        "angle_cos_sorted_3",
        "spacegroup_norm",
    )
    + tuple(f"crystal_system_{name}" for name in CRYSTAL_SYSTEMS)
)


@dataclass(frozen=True)
class TargetSpec:
    name: str
    unit: str
    source_fields: tuple[str, ...]
    required: bool
    minimum: float
    maximum: float


TARGET_SPECS = (
    TargetSpec(
        name="formation_energy_peratom",
        unit="eV/atom",
        source_fields=("formation_energy_peratom",),
        required=True,
        minimum=-10.0,
        maximum=10.0,
    ),
    TargetSpec(
        name="optb88vdw_bandgap",
        unit="eV",
        source_fields=("optb88vdw_bandgap",),
        required=True,
        minimum=0.0,
        maximum=50.0,
    ),
    TargetSpec(
        name="ehull",
        unit="eV/atom",
        source_fields=("ehull",),
        required=False,
        minimum=0.0,
        maximum=20.0,
    ),
    TargetSpec(
        name="mbj_bandgap",
        unit="eV",
        source_fields=("mbj_bandgap",),
        required=False,
        minimum=0.0,
        maximum=50.0,
    ),
    TargetSpec(
        name="electronic_dielectric_mean",
        unit="relative_permittivity",
        source_fields=("epsx", "epsy", "epsz"),
        required=False,
        minimum=0.0,
        maximum=500.0,
    ),
)

PRIMARY_TARGETS = ("formation_energy_peratom", "optb88vdw_bandgap")
ACTIVE_SPLITS = ("train", "tune", "calibration", "test")
SPLIT_NAMES = (*ACTIVE_SPLITS, "quarantine")
SPLIT_TO_CODE = {name: index for index, name in enumerate(SPLIT_NAMES)}
CODE_TO_SPLIT = {index: name for name, index in SPLIT_TO_CODE.items()}

# Fine but tolerant bins for an approximate, chemistry-agnostic structure
# fingerprint. Cross-split collisions are quarantined rather than relabelled.
STRUCTURE_LATTICE_BIN = 0.02
STRUCTURE_ANGLE_BIN_DEG = 1.0
STRUCTURE_PAIR_DISTANCE_BIN = 0.02
STRUCTURE_PAIR_QUANTILES = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)

MODEL_HIDDEN_DIMS = (128, 64)
MODEL_INPUT_SHAPE = (1, 1, 1, len(FEATURE_NAMES))
MODEL_OUTPUT_SHAPE = (1, len(TARGET_SPECS))
ONNX_OPSET = 11
ALLOWED_ONNX_OPS = frozenset({"Constant", "Gemm", "Relu", "Reshape"})

if len(ELEMENTS) != 118:
    raise RuntimeError(f"periodic table contract must contain 118 elements, got {len(ELEMENTS)}")
if len(FEATURE_NAMES) != 149:
    raise RuntimeError(f"feature contract changed unexpectedly: {len(FEATURE_NAMES)}")
