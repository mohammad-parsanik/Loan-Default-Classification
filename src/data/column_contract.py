"""
The column contract: what the upstream feed promises, in machine-readable form.

`contract/columns.json` is the single source for the column SET, its ORDER,
and the per-column handling flags this project needs (which columns are
features, which are binary, which carry a sentinel that must not be clipped
or scaled). It is authored on the ETL side and vendored here; the copy in
this repo carries the code-relevant fields only — column *semantics* live in
the local-only `etl_integration/` folder and are deliberately not committed.

Everything downstream keys on NAMES from this file, never on the order a
`SELECT *` happened to return. `project_config.META_COLS` and
`BINARY_FEATURES` are derived here rather than hand-maintained, so the two
cannot drift.

Deliberately dependency-free (json + pathlib) so `project_config` can import
it without a cycle. A malformed contract fails at import, not at fit time.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONTRACT_PATH = BASE_DIR / "contract" / "columns.json"
# Vendored full copy from the ETL repo (gitignored, local-only). When present
# it is used to cross-check the tracked file — see _check_vendored_copy.
VENDORED_PATH = BASE_DIR / "etl_integration" / "columns.json"

_ROLES = {"key", "feature", "label", "meta"}
# Fields the tracked contract keeps; also the fields compared against the
# vendored copy. Anything outside this list is prose and stays out of git.
_CODE_FIELDS = ("ordinal", "name", "type", "role", "nullable",
                "binary", "sentinel", "clip", "scale")


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _validate(doc: dict, path: Path) -> list[dict]:
    """Ordinals 1..N with no gaps, unique names, exactly one known role each."""
    cols = doc.get("columns")
    if not cols:
        raise ValueError(f"{path}: no columns")

    ordinals = [c.get("ordinal") for c in cols]
    if ordinals != list(range(1, len(cols) + 1)):
        raise ValueError(
            f"{path}: ordinals must be 1..{len(cols)} in order, got "
            f"{ordinals[:5]}… — a gap or a reorder means the file was hand-edited."
        )

    names = [c.get("name") for c in cols]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"{path}: duplicate column name(s) {sorted(dupes)}")

    for c in cols:
        if c.get("role") not in _ROLES:
            raise ValueError(
                f"{path}: column {c.get('name')!r} has role {c.get('role')!r}, "
                f"expected one of {sorted(_ROLES)}"
            )
    return cols


def _check_vendored_copy(cols: list[dict], version: int) -> None:
    """
    Warn if the tracked contract and the local-only vendored ETL copy disagree
    on any field code reads. The tracked file drops the vendored copy's prose
    notes, so only _CODE_FIELDS are compared. No-op when the folder is absent
    (the normal case on a machine that never had the ETL repo).
    """
    if not VENDORED_PATH.exists():
        return
    try:
        vendored = _load(VENDORED_PATH)
        v_cols = vendored["columns"]
    except Exception as e:                                   # pragma: no cover
        logger.warning(f"Could not read {VENDORED_PATH}: {e}")
        return

    def projection(rows):
        return [tuple(c.get(f) for f in _CODE_FIELDS) for c in rows]

    if vendored.get("contract_version") != version or projection(v_cols) != projection(cols):
        logger.warning(
            f"{CONTRACT_PATH.name} disagrees with the vendored ETL copy at "
            f"{VENDORED_PATH}. The vendored copy is the newer of the two by "
            "convention — refresh the tracked file (code fields only, no notes) "
            "before training or scoring."
        )


_doc = _load(CONTRACT_PATH)
_cols = _validate(_doc, CONTRACT_PATH)

CONTRACT_VERSION: int = _doc["contract_version"]
TABLE: str = _doc["table"]

#: The 64 feature columns, in contract ordinal order. THIS is feature identity.
FEATURE_ORDER: list[str] = [c["name"] for c in _cols if c["role"] == "feature"]
#: Everything that is not a feature: keys, labels, identifiers, label horizon.
META_COLS: list[str] = [c["name"] for c in _cols if c["role"] != "feature"]
#: 0/1 features — exempt from clipping and scaling.
BINARY_FEATURES: list[str] = [c["name"] for c in _cols if c.get("binary")]
#: Features carrying a coded value that percentile-clipping would destroy.
NO_CLIP: set[str] = {c["name"] for c in _cols if c.get("clip") is False}
#: Features whose raw value must reach the model unscaled.
NO_SCALE: set[str] = {c["name"] for c in _cols if c.get("scale") is False}
#: name -> sentinel value, for the invariant checks and for documentation.
SENTINELS: dict[str, float] = {
    c["name"]: c["sentinel"] for c in _cols if "sentinel" in c
}

_check_vendored_copy(_cols, CONTRACT_VERSION)


def feature_ordinal(name: str) -> int:
    """Contract position of a feature; len(FEATURE_ORDER) for unknown names."""
    try:
        return FEATURE_ORDER.index(name)
    except ValueError:
        return len(FEATURE_ORDER)


if __name__ == "__main__":
    assert len(FEATURE_ORDER) == 64, len(FEATURE_ORDER)
    assert len(META_COLS) == 7, META_COLS
    assert len(FEATURE_ORDER) + len(META_COLS) == len(_cols)
    assert NO_CLIP == NO_SCALE, "clip/scale exemptions diverged — check the contract"
    assert set(SENTINELS) == NO_CLIP
    assert not (set(BINARY_FEATURES) & NO_CLIP)
    print(f"contract v{CONTRACT_VERSION} for {TABLE}: "
          f"{len(FEATURE_ORDER)} features, {len(META_COLS)} meta, "
          f"{len(BINARY_FEATURES)} binary, {len(NO_CLIP)} sentinel-bearing — OK")
