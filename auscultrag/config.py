"""Paths, label spaces and global constants. Paths come from the environment so nothing is hardcoded
in downstream stages; every stage resolves locations through this module and the manifest."""
from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from the project's git-ignored .env into os.environ (existing vars win)."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value.strip():
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(Path(__file__).resolve().parents[1] / ".env")

REPO_ROOT = Path(__file__).resolve().parents[1]
# Raw corpora and derived artifacts live outside git; by default in <repo>/raw_data and <repo>/artifacts
# (directories or symlinks), overridable with AUSCULTRAG_RAW / AUSCULTRAG_ARTIFACTS.
RAW_ROOT = Path(os.environ.get("AUSCULTRAG_RAW", REPO_ROOT / "raw_data"))
ARTIFACT_ROOT = Path(os.environ.get("AUSCULTRAG_ARTIFACTS", REPO_ROOT / "artifacts"))
# Small, irreplaceable outputs (manifest, splits, reports) are also committed under the repo.
COMMITTED_ROOT = REPO_ROOT / "artifacts_committed"

SEED = 20261009

FAMILIES = ("cough", "respiratory", "cardiac")
SPLITS = ("knowledge_base", "dev", "test")

# Every corpus keeps its own label space (R5). Keys are label_space_id.
LABEL_SPACES: dict[str, tuple[str, ...]] = {
    "coswara_covid8": (
        "healthy", "no_resp_illness_exposed", "positive_asymp", "positive_mild",
        "positive_moderate", "recovered_full", "resp_illness_not_identified", "under_validation",
    ),
    "tbscreen_tb2": ("NTB", "TB"),
    # ICBHI's native 8 diagnoses (Asthma: 1 patient, LRTI: 2) cannot support a patient-disjoint
    # evaluation, so the primary space is the chronic / non-chronic / healthy grouping used in the
    # ICBHI literature. The native diagnosis is kept in meta_diagnosis_native.
    "icbhi_chronic3": ("Chronic", "Healthy", "NonChronic"),
    "circor_outcome2": ("Abnormal", "Normal"),
    # Harmonised adventitious-sound space shared by three chest-auscultation corpora (one id per
    # corpus so rows stay separable, R5): CAS = continuous (wheeze, rhonchi, stridor),
    # DAS = discontinuous (crackles). Identical class names across corpora are deliberate: a shared
    # label space removes the "corpus identity shrinks the label space" shortcut.
    "icbhi_adv4": ("CAS", "CAS&DAS", "DAS", "Normal"),
    "sprsound_adv4": ("CAS", "CAS&DAS", "DAS", "Normal"),
    # KAUH has only 2 patients with both CAS and DAS, too few for a class; those recordings are excluded
    # and KAUH uses the 3-class subset (same class names as the other two corpora).
    "kauh_adv3": ("CAS", "DAS", "Normal"),
    # COUGHVID has no closed-set ground truth; rows carry this id with has_closed_set_label = False.
    "coughvid_none": (),
}
