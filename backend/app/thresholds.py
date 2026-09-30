# ==========================================
# VISIONINSPECT AI
# UNIFIED THRESHOLD DEFAULTS (single source of truth)
# Phase 1: Threshold & Settings Unification
# ==========================================
# Canonical defaults:
#   Blur: 20, Anomaly: 90,
#   Critical: 80, High: 60, Medium: 40,
#   Classification confidence: 0.50
# All routers / services must import from here instead of
# hardcoding their own copies.

UNIFIED_DEFAULTS = {
    "blur_threshold": 20.0,
    "anomaly_threshold": 90.0,
    "classification_confidence_threshold": 0.50,
    "critical_severity_cutoff": 80.0,
    "high_severity_cutoff": 60.0,
    "medium_severity_cutoff": 40.0,
}

# Ordering constraint for severity cutoffs.
SEVERITY_ORDER = (
    "critical_severity_cutoff",
    "high_severity_cutoff",
    "medium_severity_cutoff",
)


def validate_severity_ordering(vals: dict) -> dict:
    """Reset severity cutoffs to unified defaults if misordered."""
    try:
        if not (
            float(vals["critical_severity_cutoff"])
            > float(vals["high_severity_cutoff"])
            > float(vals["medium_severity_cutoff"])
        ):
            vals["critical_severity_cutoff"] = UNIFIED_DEFAULTS[
                "critical_severity_cutoff"
            ]
            vals["high_severity_cutoff"] = UNIFIED_DEFAULTS[
                "high_severity_cutoff"
            ]
            vals["medium_severity_cutoff"] = UNIFIED_DEFAULTS[
                "medium_severity_cutoff"
            ]
    except (KeyError, TypeError, ValueError):
        pass
    return vals
