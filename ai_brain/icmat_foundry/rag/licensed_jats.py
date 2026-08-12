"""Offline, fail-closed ingestion for the licensed Europe PMC JATS corpus."""
from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .contracts import (
    NAMESPACES,
    ChunkV1,
    ContractError,
    SourceAssetV1,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)

LICENSE_ID = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
LICENSE_URL_ALIASES = frozenset(
    {
        "http://creativecommons.org/licenses/by/4.0/",
        LICENSE_URL,
    }
)
JATS_CANDIDATE_RELATIVE_PATH = Path(
    "research/icmat_foundry/open_corpus_20260728/jats_candidates"
)
JATS_EXPANSION_CANDIDATE_RELATIVE_PATH = Path(
    "research/icmat_foundry/open_corpus_20260728/jats_expansion_candidates"
)
MIN_CHUNK_CHARS = 600
TARGET_CHUNK_CHARS = 900
MAX_CHUNK_CHARS = 1200
MAX_ATOMIC_UNIT_CHARS = 420
MAX_XML_BYTES = 8 * 1024 * 1024

_DOCTYPE_RE = re.compile(br"<!\s*DOCTYPE\b", re.IGNORECASE)
_ENTITY_RE = re.compile(br"<!\s*ENTITY\b", re.IGNORECASE)
_SAFE_NLM_JATS_PUBLIC_ID = (
    b"-//NLM//DTD JATS (Z39.96) Journal Archiving and Interchange DTD "
    b"with MathML3 v1.4 20241031//EN"
)
_SAFE_NLM_JATS_SYSTEM_ID = b"JATS-archivearticle1-4-mathml3.dtd"
_SAFE_NLM_JATS_DOCTYPE_RE = re.compile(
    br'\A\s*<!DOCTYPE\s+article\s+PUBLIC\s+"'
    + re.escape(_SAFE_NLM_JATS_PUBLIC_ID)
    + br'"\s+"'
    + re.escape(_SAFE_NLM_JATS_SYSTEM_ID)
    + br'"\s*>\s*',
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s+")
_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)
_EXCLUDED_NODE_NAMES = frozenset(
    {
        "ack",
        "award-group",
        "fig",
        "fig-group",
        "funding-group",
        "funding-source",
        "graphic",
        "inline-graphic",
        "media",
        "ref",
        "ref-list",
        "supplementary-material",
        "table",
        "table-wrap",
        "table-wrap-group",
    }
)
_BLOCKED_SECTION_TYPES = frozenset(
    {
        "ack",
        "acknowledgments",
        "funding",
        "ref-list",
        "references",
        "supplementary-material",
        "supplementary-materials",
    }
)
_BLOCKED_SECTION_TITLE_TERMS = (
    "acknowledg",
    "funding",
    "funders",
    "references",
    "supplementary material",
)


class JatsSecurityError(ContractError):
    """Raised when a JATS input violates the offline XML security policy."""


class JatsLicenseError(ContractError):
    """Raised when the article does not prove the fixed CC BY 4.0 license."""


@dataclass(frozen=True, slots=True)
class LicensedJatsSpec:
    source_id: str
    filename: str
    namespace: str
    pmcid: str
    doi: str
    title: str
    source_url: str
    xml_source_url: str
    xml_sha256: str
    coverage_tags: tuple[str, ...]
    required_phrases: tuple[str, ...]


LICENSED_JATS_SPECS: tuple[LicensedJatsSpec, ...] = (
    LicensedJatsSpec(
        source_id="eupmc_pmc7722901_crabnet",
        filename="PMC7722901.xml",
        namespace="electronic_materials_property",
        pmcid="PMC7722901",
        doi="10.1038/s41467-020-19964-7",
        title=(
            "Predicting materials properties without crystal structure: "
            "deep representation learning from stoichiometry"
        ),
        source_url="https://europepmc.org/articles/PMC7722901",
        xml_source_url=(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7722901/fullTextXML"
        ),
        xml_sha256="34c3e05d4edbbf5e86ac34cc87dbf78c26d3a3948123b70cd7574546a0ebd426",
        coverage_tags=("crabnet", "composition_property_prediction"),
        required_phrases=("predicting materials properties",),
    ),
    LicensedJatsSpec(
        source_id="eupmc_pmc10132970_human_machine_process",
        filename="PMC10132970.xml",
        namespace="fab_process_metrology_yield",
        pmcid="PMC10132970",
        doi="10.1038/s41586-023-05773-7",
        title="Human–machine collaboration for improving semiconductor process development",
        source_url="https://europepmc.org/articles/PMC10132970",
        xml_source_url=(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10132970/fullTextXML"
        ),
        xml_sha256="d7c3fb9c0369989fbefd3aef1904babbd3a81b9b77430bc058a3bdb3632f645f",
        coverage_tags=("human_machine_collaboration", "semiconductor_process_development"),
        required_phrases=("human-machine", "semiconductor process"),
    ),
    LicensedJatsSpec(
        source_id="eupmc_pmc10169811_crystal_growth",
        filename="PMC10169811.xml",
        namespace="fab_process_metrology_yield",
        pmcid="PMC10169811",
        doi="10.1038/s41598-023-34732-5",
        title=(
            "Data-driven automated control algorithm for floating-zone crystal "
            "growth derived by reinforcement learning"
        ),
        source_url="https://europepmc.org/articles/PMC10169811",
        xml_source_url=(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10169811/fullTextXML"
        ),
        xml_sha256="61a48c97c679301eade20133158f69779d25bd48db0cb9c571231827859cf80a",
        coverage_tags=("crystal_growth_control", "reinforcement_learning"),
        required_phrases=("floating-zone", "crystal growth"),
    ),
    LicensedJatsSpec(
        source_id="eupmc_pmc10611205_soft_sensing",
        filename="PMC10611205.xml",
        namespace="fab_process_metrology_yield",
        pmcid="PMC10611205",
        doi="10.3390/s23208363",
        title="Soft-Sensing Regression Model: From Sensor to Wafer Metrology Forecasting",
        source_url="https://europepmc.org/articles/PMC10611205",
        xml_source_url=(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10611205/fullTextXML"
        ),
        xml_sha256="23f6f63cfa24846a07113f2ae35b044668279c7bcbae42c928aa13349e6e1c10",
        coverage_tags=("soft_sensing", "wafer_metrology_forecasting"),
        required_phrases=("soft-sensing", "wafer metrology"),
    ),
    LicensedJatsSpec(
        source_id="eupmc_pmc12475497_ellipsometry",
        filename="PMC12475497.xml",
        namespace="fab_process_metrology_yield",
        pmcid="PMC12475497",
        doi="10.1038/s41467-025-63511-1",
        title=(
            "Ultra-wide-field imaging Mueller matrix spectroscopic ellipsometry "
            "for semiconductor metrology"
        ),
        source_url="https://europepmc.org/articles/PMC12475497",
        xml_source_url=(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12475497/fullTextXML"
        ),
        xml_sha256="3ee4759604a4aca1037431970a2592d8c753ff488255e0b61f468fb5f3a31320",
        coverage_tags=("ellipsometry_metrology", "wafer_spatial_metrology"),
        required_phrases=("ellipsometry", "semiconductor metrology"),
    ),
    LicensedJatsSpec(
        source_id="eupmc_pmc10302880_hbm_3d_metrology",
        filename="PMC10302880.xml",
        namespace="opto_packaging_reliability",
        pmcid="PMC10302880",
        doi="10.3390/s23125470",
        title=(
            "Robust Detection, Segmentation, and Metrology of High Bandwidth "
            "Memory 3D Scans Using an Improved Semi-Supervised Deep Learning Approach"
        ),
        source_url="https://europepmc.org/articles/PMC10302880",
        xml_source_url=(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10302880/fullTextXML"
        ),
        xml_sha256="4b97d9206a5a841d14f83bf6ba57380397e3763ccece851885e0cb45ccac7266",
        coverage_tags=("hbm_3d_metrology", "semi_supervised_segmentation"),
        required_phrases=("high bandwidth memory", "3d"),
    ),
    LicensedJatsSpec(
        source_id="eupmc_pmc10972495_packaging_identification",
        filename="PMC10972495.xml",
        namespace="opto_packaging_reliability",
        pmcid="PMC10972495",
        doi="10.3390/mi15030418",
        title=(
            "IC Packaging Material Identification via a Hybrid Deep Learning "
            "Framework with CNN–Transformer Bidirectional Interaction"
        ),
        source_url="https://europepmc.org/articles/PMC10972495",
        xml_source_url=(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10972495/fullTextXML"
        ),
        xml_sha256="8e8ac0362b3f99e06421bcf4fa82752cb9127c5470e459a0da9532dde9fc0587",
        coverage_tags=("packaging_material_identification", "cnn_transformer"),
        required_phrases=("packaging material", "transformer"),
    ),
    LicensedJatsSpec(
        source_id="eupmc_pmc8472661_advanced_packaging",
        filename="PMC8472661.xml",
        namespace="opto_packaging_reliability",
        pmcid="PMC8472661",
        doi="10.3390/ma14185342",
        title=(
            "An Overview of AI-Assisted Design-on-Simulation Technology for "
            "Reliability Life Prediction of Advanced Packaging"
        ),
        source_url="https://europepmc.org/articles/PMC8472661",
        xml_source_url=(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8472661/fullTextXML"
        ),
        xml_sha256="f7e0234079fc984483fa2021df62c2449e63f8dbcebbe4519344b947135bd644",
        coverage_tags=("advanced_packaging_reliability", "design_on_simulation"),
        required_phrases=("advanced packaging", "reliability"),
    ),
    LicensedJatsSpec(
        source_id="eupmc_pmc9035975_3d_ic",
        filename="PMC9035975.xml",
        namespace="opto_packaging_reliability",
        pmcid="PMC9035975",
        doi="10.1038/s41598-022-08179-z",
        title="Artificial intelligence deep learning for 3D IC reliability prediction",
        source_url="https://europepmc.org/articles/PMC9035975",
        xml_source_url=(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9035975/fullTextXML"
        ),
        xml_sha256="bae37e94ce73d5e351613fc696ffd91c718b5bd6f63a7007f5d6bc792253007a",
        coverage_tags=("3d_ic_reliability", "deep_learning"),
        required_phrases=("3d ic", "reliability"),
    ),
    LicensedJatsSpec(
        source_id="eupmc_pmc9182149_wlp",
        filename="PMC9182149.xml",
        namespace="opto_packaging_reliability",
        pmcid="PMC9182149",
        doi="10.3390/ma15113897",
        title=(
            "Predicting Wafer-Level Package Reliability Life Using Mixed "
            "Supervised and Unsupervised Machine Learning Algorithms"
        ),
        source_url="https://europepmc.org/articles/PMC9182149",
        xml_source_url=(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9182149/fullTextXML"
        ),
        xml_sha256="dc0fe71de993e260b8e48cbead5735849770c9457f4b686de360ee801b6b5e75",
        coverage_tags=("wlp_reliability", "mixed_machine_learning"),
        required_phrases=("wafer-level package", "reliability"),
    ),
)

LICENSED_JATS_EXPANSION_SPECS: tuple[LicensedJatsSpec, ...] = (
    LicensedJatsSpec(
        source_id="eupmc_pmc10935296_binary_semiconductor_bandgap",
        filename="PMC10935296.xml",
        namespace="electronic_materials_property",
        pmcid="PMC10935296",
        doi="10.3390/nano14050445",
        title=(
            "Feature-Assisted Machine Learning for Predicting Band Gaps of "
            "Binary Semiconductors"
        ),
        source_url="https://europepmc.org/articles/PMC10935296",
        xml_source_url=(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10935296/fullTextXML"
        ),
        xml_sha256="b774d734254d2e3fb3b6bef8966d1c1404bd683e93d3d8980cb36b1e98195f01",
        coverage_tags=("binary_semiconductor", "band_gap_prediction"),
        required_phrases=("binary semiconductors", "band gaps"),
    ),
    LicensedJatsSpec(
        source_id="eupmc_pmc6874674_materials_transfer_learning",
        filename="PMC6874674.xml",
        namespace="electronic_materials_property",
        pmcid="PMC6874674",
        doi="10.1038/s41467-019-13297-w",
        title=(
            "Enhancing materials property prediction by leveraging computational "
            "and experimental data using deep transfer learning"
        ),
        source_url="https://europepmc.org/articles/PMC6874674",
        xml_source_url=(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6874674/fullTextXML"
        ),
        xml_sha256="e1bcbc6e9766f11ab64b0158475facae279bdc0a65f3f95744d4a52fd9282d6d",
        coverage_tags=("deep_transfer_learning", "computational_experimental_bridge"),
        required_phrases=("deep transfer learning", "experimental"),
    ),
    LicensedJatsSpec(
        source_id="eupmc_pmc9279333_experimental_level_prediction",
        filename="PMC9279333.xml",
        namespace="electronic_materials_property",
        pmcid="PMC9279333",
        doi="10.1038/s41598-022-15816-0",
        title="Moving closer to experimental level materials property prediction using AI",
        source_url="https://europepmc.org/articles/PMC9279333",
        xml_source_url=(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9279333/fullTextXML"
        ),
        xml_sha256="6d08c4da428642a204e923d73e67da9ceaedecc99039d4f6dac2094c9b7f2d45",
        coverage_tags=("experimental_level_prediction", "materials_ai"),
        required_phrases=("experimental", "materials property prediction"),
    ),
    LicensedJatsSpec(
        source_id="eupmc_pmc10820808_hybrid_perovskite_bandgap",
        filename="PMC10820808.xml",
        namespace="electronic_materials_property",
        pmcid="PMC10820808",
        doi="10.3390/molecules29020499",
        title=(
            "Prediction of Organic\u2013Inorganic Hybrid Perovskite Band Gap by "
            "Multiple Machine Learning Algorithms"
        ),
        source_url="https://europepmc.org/articles/PMC10820808",
        xml_source_url=(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10820808/fullTextXML"
        ),
        xml_sha256="0d991dbc8db9fec9a87902642da4d1b0e5f8fbf55f75291334042885ab7cccdf",
        coverage_tags=("hybrid_perovskite", "band_gap_prediction"),
        required_phrases=("organic-inorganic", "band gap"),
    ),
)

EXPANSION_EXCLUSION_REASONS: Mapping[str, str] = {
    "PMC10241826.xml": "isolated_unlisted_strict_xml_policy",
    "PMC10601046.xml": "excluded_cc_by_nc_nd",
    "PMC10870658.xml": "isolated_unlisted_strict_xml_policy",
    "PMC10879764.xml": "excluded_cc_by_nc_nd",
    "PMC11538560.xml": "excluded_cc_by_nc_nd",
    "PMC11742729.xml": "excluded_cc_by_nc_nd",
    "PMC12947634.xml": "excluded_not_cc_by_4_0",
    "PMC8756567.xml": "excluded_cc_by_nc_nd",
    "PMC9122959.xml": "excluded_cc_by_nc_nd",
    "PMC9218748.xml": "isolated_unlisted_strict_xml_policy",
}


@dataclass(frozen=True, slots=True)
class ParsedLicensedArticle:
    spec: LicensedJatsSpec
    authors: tuple[str, ...]
    publication_year: str
    license_text: str
    license_node_sha256: str
    chunks: tuple[ChunkV1, ...]

    @property
    def source_asset(self) -> SourceAssetV1:
        return SourceAssetV1(
            source_id=self.spec.source_id,
            source_uri=self.spec.xml_source_url,
            sha256=self.spec.xml_sha256,
            access_mode="licensed_fulltext_readonly",
            license_id=LICENSE_ID,
        )

    def catalog_record(self) -> dict[str, Any]:
        return {
            "source_id": self.spec.source_id,
            "namespace": self.spec.namespace,
            "primary_namespace": self.spec.namespace,
            "paper_family_id": self.spec.pmcid,
            "pmcid": self.spec.pmcid,
            "doi": self.spec.doi,
            "title": self.spec.title,
            "authors": list(self.authors),
            "publication_year": self.publication_year,
            "license_id": LICENSE_ID,
            "license_url": LICENSE_URL,
            "license_verified_from": "JATS front/article-meta/permissions/license",
            "license_node_sha256": self.license_node_sha256,
            "source_url": self.spec.source_url,
            "xml_source_url": self.spec.xml_source_url,
            "xml_filename": self.spec.filename,
            "xml_sha256": self.spec.xml_sha256,
            "access_mode": "licensed_fulltext_readonly",
            "evidence_kind": "literature_knowledge",
            "coverage_tags": list(self.spec.coverage_tags),
            "chunk_count": len(self.chunks),
            "claim_boundary": (
                "Published CC BY 4.0 literature knowledge; not local laboratory "
                "measurement, fab production ground truth, or X5 runtime evidence."
            ),
        }


@dataclass(frozen=True, slots=True)
class LicensedJatsCorpus:
    articles: tuple[ParsedLicensedArticle, ...]

    @property
    def chunks(self) -> tuple[ChunkV1, ...]:
        return tuple(chunk for article in self.articles for chunk in article.chunks)

    def chunks_by_namespace(self) -> dict[str, tuple[ChunkV1, ...]]:
        return {
            namespace: tuple(
                chunk for chunk in self.chunks if chunk.namespace == namespace
            )
            for namespace in NAMESPACES
        }

    def assets_by_namespace(self) -> dict[str, tuple[SourceAssetV1, ...]]:
        return {
            namespace: tuple(
                article.source_asset
                for article in self.articles
                if article.spec.namespace == namespace
            )
            for namespace in NAMESPACES
        }

    def source_catalog(self, *, created_at: str) -> dict[str, Any]:
        records = [article.catalog_record() for article in self.articles]
        return {
            "schema": "icmat.rag.licensed_source_catalog.v2",
            "created_at": created_at,
            "status": "LICENSED_FULLTEXT_CANDIDATE_OFFLINE",
            "license_policy": {
                "required_license": LICENSE_ID,
                "required_license_url": LICENSE_URL,
                "verification_location": "JATS front/article-meta/permissions/license",
                "third_party_nodes_ingested": False,
            },
            "source_count": len(records),
            "chunk_count": sum(record["chunk_count"] for record in records),
            "namespace_counts": {
                namespace: {
                    "paper_count": sum(
                        article.spec.namespace == namespace for article in self.articles
                    ),
                    "chunk_count": sum(
                        len(article.chunks)
                        for article in self.articles
                        if article.spec.namespace == namespace
                    ),
                }
                for namespace in NAMESPACES
            },
            "records": records,
            "evidence_boundary": (
                "All records are attributed published literature_knowledge. None are "
                "real_measurement or local fab production ground truth."
            ),
        }


@dataclass(frozen=True, slots=True)
class _TextUnit:
    text: str
    paragraph_id: str


@dataclass(frozen=True, slots=True)
class _BodyUnit:
    text: str
    paragraph_id: str
    section_id: str
    title_path: tuple[str, ...]


def _local_name(tag: object) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def _search_normalize(value: str) -> str:
    return _normalize_text(value).translate(_DASH_TRANSLATION).lower()


def _direct_child(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if _local_name(child.tag) == name), None)


def _direct_children(element: ET.Element, name: str) -> tuple[ET.Element, ...]:
    return tuple(child for child in element if _local_name(child.tag) == name)


def _safe_element_text(element: ET.Element) -> str:
    parts: list[str] = []

    def visit(node: ET.Element) -> None:
        if node.text:
            parts.append(node.text)
        for child in node:
            if _local_name(child.tag) not in _EXCLUDED_NODE_NAMES:
                visit(child)
            if child.tail:
                parts.append(child.tail)

    visit(element)
    return _normalize_text(" ".join(parts))


def _required_child(element: ET.Element, name: str, label: str) -> ET.Element:
    child = _direct_child(element, name)
    if child is None:
        raise ContractError(f"JATS is missing {label}")
    return child


def _read_xml_bytes(path: Path, allowed_root: Path) -> bytes:
    resolved_root = allowed_root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    if resolved_path.parent != resolved_root:
        raise JatsSecurityError("JATS path escapes the fixed corpus directory")
    if path.is_symlink():
        raise JatsSecurityError("JATS symlinks are not accepted")
    payload = resolved_path.read_bytes()
    if not payload or len(payload) > MAX_XML_BYTES:
        raise JatsSecurityError("JATS input is empty or exceeds the fixed size limit")
    if _ENTITY_RE.search(payload):
        raise JatsSecurityError("ENTITY declarations are forbidden")
    return payload


def _parse_xml(payload: bytes) -> ET.Element:
    if _DOCTYPE_RE.search(payload):
        sanitized, replacements = _SAFE_NLM_JATS_DOCTYPE_RE.subn(b"", payload, count=1)
        if replacements != 1 or _DOCTYPE_RE.search(sanitized):
            raise JatsSecurityError("unapproved DOCTYPE declaration is forbidden")
        payload = sanitized
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ContractError(f"invalid JATS XML: {exc}") from exc
    for element in root.iter():
        if (
            isinstance(element.tag, str)
            and element.tag.startswith("{http://www.w3.org/2001/XInclude}")
        ):
            raise JatsSecurityError("XInclude is forbidden")
    return root


def _article_identity(article_meta: ET.Element) -> tuple[str, str, str]:
    title_group = _required_child(article_meta, "title-group", "article title-group")
    title = _safe_element_text(
        _required_child(title_group, "article-title", "article title")
    )
    identifiers: dict[str, str] = {}
    for element in _direct_children(article_meta, "article-id"):
        kind = element.attrib.get("pub-id-type", "")
        value = _safe_element_text(element)
        if kind and value and kind not in identifiers:
            identifiers[kind] = value
    return identifiers.get("pmcid", ""), identifiers.get("doi", ""), title


def _license_evidence(article_meta: ET.Element) -> tuple[str, str]:
    permissions = _required_child(article_meta, "permissions", "permissions")
    licenses = _direct_children(permissions, "license")
    if not licenses:
        raise JatsLicenseError("JATS permissions has no license node")
    for license_node in licenses:
        text = _safe_element_text(license_node)
        normalized = text.lower()
        hrefs = {
            value.strip()
            for element in license_node.iter()
            for name, value in element.attrib.items()
            if _local_name(name) == "href"
        }
        attribution_text = (
            "creative commons attribution" in normalized
            or "creative commons attribution (cc by)" in normalized
        )
        if hrefs.intersection(LICENSE_URL_ALIASES) and attribution_text and "4.0" in normalized:
            digest = sha256_bytes(canonical_json_bytes({"text": text, "hrefs": sorted(hrefs)}))
            return text, digest
    raise JatsLicenseError("JATS license does not prove CC BY 4.0")


def _authors_and_year(article_meta: ET.Element) -> tuple[tuple[str, ...], str]:
    authors: list[str] = []
    for group in article_meta.iter():
        if _local_name(group.tag) != "contrib-group":
            continue
        for contrib in _direct_children(group, "contrib"):
            if contrib.attrib.get("contrib-type", "author") != "author":
                continue
            name = _direct_child(contrib, "name")
            if name is None:
                continue
            surname_node = _direct_child(name, "surname")
            given_node = _direct_child(name, "given-names")
            surname = _safe_element_text(surname_node) if surname_node is not None else ""
            given = _safe_element_text(given_node) if given_node is not None else ""
            display = _normalize_text(f"{given} {surname}")
            if display and display not in authors:
                authors.append(display)
        if authors:
            break
    year = ""
    for pub_date in article_meta.iter():
        if _local_name(pub_date.tag) != "pub-date":
            continue
        year_node = _direct_child(pub_date, "year")
        if year_node is not None:
            year = _safe_element_text(year_node)
            if year:
                break
    if not authors or not year:
        raise ContractError("JATS attribution requires authors and publication year")
    return tuple(authors), year


def _section_is_blocked(section: ET.Element, title: str) -> bool:
    section_type = _search_normalize(section.attrib.get("sec-type", ""))
    if section_type in _BLOCKED_SECTION_TYPES:
        return True
    normalized_title = _search_normalize(title)
    return any(term in normalized_title for term in _BLOCKED_SECTION_TITLE_TERMS)


def _split_long_unit(text: str, max_chars: int) -> tuple[str, ...]:
    if len(text) <= max_chars:
        return (text,)
    words = text.split()
    if len(words) <= 1:
        return tuple(text[start : start + max_chars] for start in range(0, len(text), max_chars))
    parts: list[str] = []
    current: list[str] = []
    current_length = 0
    for word in words:
        added = len(word) + (1 if current else 0)
        if current and current_length + added > max_chars:
            parts.append(" ".join(current))
            current = [word]
            current_length = len(word)
        else:
            current.append(word)
            current_length += added
    if current:
        parts.append(" ".join(current))
    return tuple(parts)


def _paragraph_units(
    paragraphs: Sequence[tuple[str, str]],
    *,
    max_body_chars: int,
) -> tuple[_TextUnit, ...]:
    units: list[_TextUnit] = []
    for paragraph_id, paragraph in paragraphs:
        sentences = tuple(
            sentence
            for sentence in _SENTENCE_BOUNDARY_RE.split(paragraph)
            if sentence
        )
        for sentence in sentences or (paragraph,):
            for part in _split_long_unit(sentence, max_body_chars):
                units.append(_TextUnit(text=part, paragraph_id=paragraph_id))
    return tuple(units)


def _group_length(units: Sequence[_TextUnit]) -> int:
    return sum(len(unit.text) for unit in units) + max(0, len(units) - 1)


def _pack_units(
    units: Sequence[_TextUnit],
    *,
    min_body_chars: int,
    target_body_chars: int,
    max_body_chars: int,
) -> tuple[tuple[_TextUnit, ...], ...]:
    groups: list[list[_TextUnit]] = []
    current: list[_TextUnit] = []
    for unit in units:
        proposed = _group_length((*current, unit))
        if current and proposed > max_body_chars:
            groups.append(current)
            current = [unit]
        else:
            current.append(unit)
            if _group_length(current) >= target_body_chars:
                groups.append(current)
                current = []
    if current:
        groups.append(current)
    if len(groups) > 1 and _group_length(groups[-1]) < min_body_chars:
        previous = groups[-2]
        tail = groups[-1]
        while len(previous) > 1 and _group_length(tail) < min_body_chars:
            candidate = previous[-1]
            remaining = previous[:-1]
            moved = [candidate, *tail]
            if (
                _group_length(remaining) < min_body_chars
                or _group_length(moved) > max_body_chars
            ):
                break
            previous.pop()
            tail.insert(0, candidate)
        if _group_length((*previous, *tail)) <= max_body_chars:
            groups[-2] = [*previous, *tail]
            groups.pop()
    return tuple(tuple(group) for group in groups if group)


def _unique_in_order(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _section_chunks(
    *,
    spec: LicensedJatsSpec,
    authors: tuple[str, ...],
    publication_year: str,
    section: ET.Element,
    section_id: str,
    title_path: tuple[str, ...],
) -> tuple[ChunkV1, ...]:
    paragraphs: list[tuple[str, str]] = []
    paragraph_ordinal = 0
    for child in section:
        if _local_name(child.tag) != "p":
            continue
        paragraph_ordinal += 1
        text = _safe_element_text(child)
        if not text:
            continue
        paragraph_id = child.attrib.get("id") or f"p{paragraph_ordinal:03d}"
        paragraphs.append((paragraph_id, text))
    if not paragraphs:
        return ()
    section_title = " > ".join(title_path) or "Untitled body section"
    header = f"Section: {section_title}\n"
    max_body = MAX_CHUNK_CHARS - len(header)
    min_body = max(1, MIN_CHUNK_CHARS - len(header))
    target_body = max(min_body, TARGET_CHUNK_CHARS - len(header))
    units = _paragraph_units(paragraphs, max_body_chars=max_body)
    groups = _pack_units(
        units,
        min_body_chars=min_body,
        target_body_chars=target_body,
        max_body_chars=max_body,
    )
    attribution = (
        f"{spec.title}; {', '.join(authors)}; {publication_year}; "
        f"PMCID {spec.pmcid}; DOI {spec.doi}; {LICENSE_ID}."
    )
    chunks: list[ChunkV1] = []
    for chunk_ordinal, group in enumerate(groups, start=1):
        paragraph_ids = _unique_in_order(unit.paragraph_id for unit in group)
        paragraph_locator = (
            paragraph_ids[0]
            if len(paragraph_ids) == 1
            else f"{paragraph_ids[0]}..{paragraph_ids[-1]}"
        )
        text = header + " ".join(unit.text for unit in group)
        chunks.append(
            ChunkV1.create(
                namespace=spec.namespace,
                source_id=spec.source_id,
                source_title=spec.title,
                source_uri=spec.source_url,
                locator=(
                    f"jats:section={section_id};paragraph={paragraph_locator};"
                    f"chunk={chunk_ordinal:03d}"
                ),
                evidence_kind="literature_knowledge",
                text=text,
                license_id=LICENSE_ID,
                metadata={
                    "access_mode": "licensed_fulltext_readonly",
                    "attribution": attribution,
                    "body_only": True,
                    "coverage_tags": list(spec.coverage_tags),
                    "doi": spec.doi,
                    "excluded_nodes": sorted(_EXCLUDED_NODE_NAMES),
                    "license_url": LICENSE_URL,
                    "license_verified_from_jats": True,
                    "measurement_status": "published_literature_not_local_measurement",
                    "paragraph_ids": list(paragraph_ids),
                    "pmcid": spec.pmcid,
                    "section_id": section_id,
                    "section_title_path": list(title_path),
                    "xml_filename": spec.filename,
                    "xml_sha256": spec.xml_sha256,
                },
            )
        )
    return tuple(chunks)


def _render_body_units(units: Sequence[_BodyUnit]) -> str:
    sections: list[str] = []
    current_section = ""
    current_title_path: tuple[str, ...] = ()
    current_text: list[str] = []
    for unit in units:
        if unit.section_id != current_section:
            if current_text:
                title = " > ".join(current_title_path) or "Untitled body section"
                sections.append(f"Section: {title}\n{' '.join(current_text)}")
            current_section = unit.section_id
            current_title_path = unit.title_path
            current_text = [unit.text]
        else:
            current_text.append(unit.text)
    if current_text:
        title = " > ".join(current_title_path) or "Untitled body section"
        sections.append(f"Section: {title}\n{' '.join(current_text)}")
    return "\n".join(sections)


def _partition_body_units(
    units: Sequence[_BodyUnit],
) -> tuple[tuple[_BodyUnit, ...], ...]:
    """Find a deterministic complete partition with every rendered block in range."""
    frozen = tuple(units)
    best_cost: list[int | None] = [None] * (len(frozen) + 1)
    previous: list[int | None] = [None] * (len(frozen) + 1)
    best_cost[0] = 0

    for end in range(1, len(frozen) + 1):
        for start in range(end - 1, -1, -1):
            rendered_length = len(_render_body_units(frozen[start:end]))
            if rendered_length > MAX_CHUNK_CHARS:
                break
            if rendered_length < MIN_CHUNK_CHARS or best_cost[start] is None:
                continue
            candidate_cost = best_cost[start] + (
                rendered_length - TARGET_CHUNK_CHARS
            ) ** 2
            if best_cost[end] is None or candidate_cost < best_cost[end]:
                best_cost[end] = candidate_cost
                previous[end] = start

    if previous[-1] is None:
        raise ContractError(
            "eligible JATS body cannot be partitioned into complete 600-1200 "
            "character chunks"
        )

    groups: list[tuple[_BodyUnit, ...]] = []
    end = len(frozen)
    while end:
        start = previous[end]
        if start is None:
            raise ContractError("internal JATS partition reconstruction failure")
        groups.append(frozen[start:end])
        end = start
    groups.reverse()
    return tuple(groups)


def _body_units(
    *,
    body: ET.Element,
) -> tuple[_BodyUnit, ...]:
    units: list[_BodyUnit] = []

    def collect_section(
        section: ET.Element,
        titles: tuple[str, ...],
        ordinal_path: tuple[int, ...],
    ) -> None:
        title_node = _direct_child(section, "title")
        title = _safe_element_text(title_node) if title_node is not None else ""
        if _section_is_blocked(section, title):
            return
        current_titles = (*titles, title) if title else titles
        section_id = section.attrib.get("id") or (
            "sec-" + ".".join(str(value) for value in ordinal_path)
        )
        paragraph_ordinal = 0
        for child in section:
            if _local_name(child.tag) != "p":
                continue
            paragraph_ordinal += 1
            paragraph = _safe_element_text(child)
            if not paragraph:
                continue
            paragraph_id = child.attrib.get("id") or f"p{paragraph_ordinal:03d}"
            sentences = tuple(
                sentence
                for sentence in _SENTENCE_BOUNDARY_RE.split(paragraph)
                if sentence
            )
            for sentence in sentences or (paragraph,):
                for part in _split_long_unit(sentence, MAX_ATOMIC_UNIT_CHARS):
                    units.append(
                        _BodyUnit(
                            text=part,
                            paragraph_id=paragraph_id,
                            section_id=section_id,
                            title_path=current_titles,
                        )
                    )

        nested_ordinal = 0
        for child in section:
            if _local_name(child.tag) != "sec":
                continue
            nested_ordinal += 1
            collect_section(
                child,
                current_titles,
                (*ordinal_path, nested_ordinal),
            )

    top_level_ordinal = 0
    for child in body:
        if _local_name(child.tag) != "sec":
            continue
        top_level_ordinal += 1
        collect_section(child, (), (top_level_ordinal,))
    return tuple(units)


def _extract_body_chunks(
    *,
    root: ET.Element,
    spec: LicensedJatsSpec,
    authors: tuple[str, ...],
    publication_year: str,
) -> tuple[ChunkV1, ...]:
    body = _required_child(root, "body", "article body")
    units = _body_units(body=body)
    if not units:
        raise ContractError(f"JATS body produced no eligible chunks: {spec.pmcid}")
    groups = _partition_body_units(units)
    attribution = (
        f"{spec.title}; {', '.join(authors)}; {publication_year}; "
        f"PMCID {spec.pmcid}; DOI {spec.doi}; {LICENSE_ID}."
    )
    chunks: list[ChunkV1] = []
    for chunk_ordinal, group in enumerate(groups, start=1):
        paragraph_ids = _unique_in_order(unit.paragraph_id for unit in group)
        section_ids = _unique_in_order(unit.section_id for unit in group)
        title_paths = tuple(dict.fromkeys(unit.title_path for unit in group))
        text = _render_body_units(group)
        start_unit = group[0]
        end_unit = group[-1]
        chunks.append(
            ChunkV1.create(
                namespace=spec.namespace,
                source_id=spec.source_id,
                source_title=spec.title,
                source_uri=spec.source_url,
                locator=(
                    "jats:"
                    f"section={quote(start_unit.section_id, safe='._:-')}"
                    f"..{quote(end_unit.section_id, safe='._:-')};"
                    f"paragraph={quote(start_unit.paragraph_id, safe='._:-')}"
                    f"..{quote(end_unit.paragraph_id, safe='._:-')};"
                    f"chunk={chunk_ordinal:03d}"
                ),
                evidence_kind="literature_knowledge",
                text=text,
                license_id=LICENSE_ID,
                metadata={
                    "access_mode": "licensed_fulltext_readonly",
                    "attribution": attribution,
                    "body_only": True,
                    "coverage_tags": list(spec.coverage_tags),
                    "doi": spec.doi,
                    "excluded_nodes": sorted(_EXCLUDED_NODE_NAMES),
                    "license_url": LICENSE_URL,
                    "license_verified_from_jats": True,
                    "measurement_status": "published_literature_not_local_measurement",
                    "paragraph_ids": list(paragraph_ids),
                    "pmcid": spec.pmcid,
                    "section_ids": list(section_ids),
                    "section_title_paths": [list(path) for path in title_paths],
                    "xml_filename": spec.filename,
                    "xml_sha256": spec.xml_sha256,
                },
            )
        )
    return tuple(chunks)


def ingest_licensed_jats_article(
    path: Path,
    *,
    allowed_root: Path,
    spec: LicensedJatsSpec,
) -> ParsedLicensedArticle:
    if spec.namespace not in NAMESPACES[1:]:
        raise ContractError("licensed finals sources cannot enter the frozen phosphor namespace")
    if path.name != spec.filename:
        raise ContractError("JATS filename does not match the fixed source specification")
    payload = _read_xml_bytes(path, allowed_root)
    observed_sha256 = sha256_bytes(payload)
    if observed_sha256 != spec.xml_sha256:
        raise ContractError(f"JATS SHA-256 mismatch for {spec.pmcid}")
    root = _parse_xml(payload)
    if _local_name(root.tag) != "article":
        raise ContractError("JATS root must be article")
    front = _required_child(root, "front", "article front")
    article_meta = _required_child(front, "article-meta", "article-meta")
    pmcid, doi, title = _article_identity(article_meta)
    if (pmcid, doi, title) != (spec.pmcid, spec.doi, spec.title):
        raise ContractError(f"JATS identity mismatch for {spec.filename}")
    license_text, license_node_sha256 = _license_evidence(article_meta)
    authors, publication_year = _authors_and_year(article_meta)
    chunks = _extract_body_chunks(
        root=root,
        spec=spec,
        authors=authors,
        publication_year=publication_year,
    )
    searchable = _search_normalize(
        title + " " + " ".join(chunk.text for chunk in chunks)
    )
    missing = [
        phrase
        for phrase in spec.required_phrases
        if _search_normalize(phrase) not in searchable
    ]
    if missing:
        raise ContractError(
            f"JATS required coverage phrases missing for {spec.pmcid}: {missing}"
        )
    if any(chunk.evidence_kind != "literature_knowledge" for chunk in chunks):
        raise ContractError("licensed JATS chunks must remain literature_knowledge")
    if any(
        not MIN_CHUNK_CHARS <= len(chunk.text) <= MAX_CHUNK_CHARS
        for chunk in chunks
    ):
        raise ContractError("licensed JATS chunk is outside the fixed 600-1200 range")
    if sha256_file(path.resolve()) != observed_sha256:
        raise ContractError("JATS source changed during ingestion")
    return ParsedLicensedArticle(
        spec=spec,
        authors=authors,
        publication_year=publication_year,
        license_text=license_text,
        license_node_sha256=license_node_sha256,
        chunks=chunks,
    )


def ingest_licensed_jats_corpus(
    corpus_dir: Path,
    *,
    specs: Sequence[LicensedJatsSpec] = LICENSED_JATS_SPECS,
    require_exact_inventory: bool = True,
) -> LicensedJatsCorpus:
    root = corpus_dir.resolve(strict=True)
    expected_names = {spec.filename for spec in specs}
    observed_names = {path.name for path in root.glob("*.xml")}
    missing = sorted(expected_names - observed_names)
    unexpected = sorted(observed_names - expected_names)
    if missing or (require_exact_inventory and unexpected):
        raise ContractError(
            f"JATS corpus inventory mismatch; missing={missing}, unexpected={unexpected}"
        )
    for field_name in ("source_id", "filename", "pmcid", "doi"):
        values = [getattr(spec, field_name) for spec in specs]
        if len(values) != len(set(values)):
            raise ContractError(
                f"JATS source specification contains duplicate {field_name}"
            )
    articles = tuple(
        ingest_licensed_jats_article(
            root / spec.filename,
            allowed_root=root,
            spec=spec,
        )
        for spec in specs
    )
    chunk_ids = [chunk.chunk_id for article in articles for chunk in article.chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ContractError("licensed JATS corpus produced duplicate chunk IDs")
    return LicensedJatsCorpus(articles=articles)


def combine_licensed_jats_corpora(
    *corpora: LicensedJatsCorpus,
) -> LicensedJatsCorpus:
    articles = tuple(
        article
        for corpus in corpora
        for article in corpus.articles
    )
    for field_name in ("source_id", "pmcid", "doi", "filename"):
        values = [getattr(article.spec, field_name) for article in articles]
        if len(values) != len(set(values)):
            raise ContractError(
                f"combined licensed corpus contains duplicate {field_name}"
            )
    chunk_ids = [chunk.chunk_id for article in articles for chunk in article.chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ContractError("combined licensed corpus contains duplicate chunk IDs")
    return LicensedJatsCorpus(articles=articles)
