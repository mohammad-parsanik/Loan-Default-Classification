"""
Explicit parameter bundle for the manager-code scoring entry point
(src/inference/scoring.py::run_scoring).

Design: a manager codebase integrating scoring for several data sources may
need different business knobs per call (e.g. one source scores with a
certainty threshold, another doesn't). Reaching into project_config.py
globals doesn't support that. ScoringParams is the explicit alternative —
every field can be set per call; anything left unset is filled from
project_config.py, with a single consolidated warning naming exactly which
fields defaulted, so a caller silently forgetting a parameter is visible
in the logs rather than invisible.

`calibration_df` and `output_path` are NOT part of the warn-on-default set:
there is no project_config equivalent for either (a DataFrame can't live in
a config file; "don't write a CSV" is a legitimate default, not a mistake).
"""

import dataclasses
import logging
from typing import Optional

import pandas as pd

import project_config as config

logger = logging.getLogger(__name__)

# field name -> project_config attribute name
_CONFIG_FALLBACKS = {
    "certainty_act_threshold": "CERTAINTY_ACT_THRESHOLD",
    "carve_current_cat_ge":    "CARVE_CURRENT_CAT_GE",
    "calibration_min_stratum_n": "CALIBRATION_MIN_STRATUM_N",
    "api_data_ttl_days":       "API_DATA_TTL_DAYS",
    "pred_dedup_latest":       "PRED_DEDUP_LATEST",
    "cost_matrix":             "COST_MATRIX",
    "called_log_path":         "API_CALL_LOG",
}


@dataclasses.dataclass
class ScoringParams:
    # Required: which trained model to score with. No project-wide default
    # makes sense for "which model" — omitting this is always an error.
    bundle_path: str

    # Data inputs — no config fallback exists for these; a None here is a
    # normal, deliberate choice, not a missing parameter.
    calibration_df: Optional[pd.DataFrame] = None
    output_path: Optional[str] = None

    # Business knobs — fall back to project_config.py when None, with a
    # warning (see resolve()).
    called_log_path: Optional[str] = None
    certainty_act_threshold: Optional[float] = None
    carve_current_cat_ge: Optional[int] = None
    calibration_min_stratum_n: Optional[int] = None
    api_data_ttl_days: Optional[int] = None
    pred_dedup_latest: Optional[bool] = None
    cost_matrix: Optional[list] = None

    def resolve(self) -> "ScoringParams":
        """
        Return a copy with every None business-knob field filled from
        project_config, logging one warning naming whichever fields
        defaulted. Idempotent — resolving an already-resolved instance
        finds nothing left to fill and warns nothing.
        """
        if not self.bundle_path:
            raise ValueError(
                "ScoringParams.bundle_path is required — no project-wide "
                "default exists for 'which trained model to use'."
            )

        resolved = dataclasses.replace(self)
        defaulted = []
        for field, config_attr in _CONFIG_FALLBACKS.items():
            if getattr(resolved, field) is None:
                value = getattr(config, config_attr, None)
                setattr(resolved, field, value)
                defaulted.append(f"{field}={value!r}")

        if defaulted:
            logger.warning(
                "ScoringParams: %d parameter(s) not passed explicitly, "
                "using project_config defaults — %s",
                len(defaulted), ", ".join(defaulted),
            )
        return resolved


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.WARNING)

    # Self-check: omitted fields default from config; explicit values win;
    # missing bundle_path raises; resolve() is idempotent.
    p = ScoringParams(bundle_path="model_bundle.pkl")
    r = p.resolve()
    assert r.carve_current_cat_ge == config.CARVE_CURRENT_CAT_GE
    assert r.cost_matrix == config.COST_MATRIX
    assert r.calibration_df is None and r.output_path is None   # never defaulted

    p2 = ScoringParams(bundle_path="x.pkl", carve_current_cat_ge=99)
    r2 = p2.resolve()
    assert r2.carve_current_cat_ge == 99          # explicit value preserved
    assert r2.cost_matrix == config.COST_MATRIX    # other fields still default

    r3 = r2.resolve()
    assert r3.carve_current_cat_ge == 99           # idempotent

    try:
        ScoringParams(bundle_path="").resolve()
        raise AssertionError("expected ValueError for missing bundle_path")
    except ValueError:
        pass

    print("scoring_params.py self-check OK")
