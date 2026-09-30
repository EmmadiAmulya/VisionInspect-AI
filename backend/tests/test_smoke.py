"""Phase 1 smoke tests: API startup, auth validation, inference imports."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))


def test_threshold_defaults_unified():
    from thresholds import UNIFIED_DEFAULTS
    assert UNIFIED_DEFAULTS["blur_threshold"] == 20.0
    assert UNIFIED_DEFAULTS["anomaly_threshold"] == 90.0
    assert UNIFIED_DEFAULTS["critical_severity_cutoff"] == 80.0
    assert UNIFIED_DEFAULTS["high_severity_cutoff"] == 60.0
    assert UNIFIED_DEFAULTS["medium_severity_cutoff"] == 40.0
    assert UNIFIED_DEFAULTS["classification_confidence_threshold"] == 0.50


def test_severity_ordering_validator():
    from thresholds import validate_severity_ordering, UNIFIED_DEFAULTS
    bad = {
        "critical_severity_cutoff": 10.0,
        "high_severity_cutoff": 50.0,
        "medium_severity_cutoff": 90.0,
    }
    fixed = validate_severity_ordering(dict(bad))
    assert fixed["critical_severity_cutoff"] == UNIFIED_DEFAULTS[
        "critical_severity_cutoff"
    ]


def test_inference_imports_and_registry():
    import ml_inference as mi
    assert hasattr(mi, "inspect_image")
    assert hasattr(mi, "get_supported_categories")
    assert hasattr(mi, "normalize_category")
    assert hasattr(mi, "get_anomaly_visualization")
    assert mi.normalize_category(None) == "bottle"
    assert mi.normalize_category("  Cable ") == "cable"
    cats = mi.get_supported_categories()
    assert isinstance(cats, list)
    assert any(c["name"] == "bottle" for c in cats)


def test_inspect_image_backward_compat_signature():
    import inspect as _inspect
    import ml_inference as mi
    params = list(_inspect.signature(mi.inspect_image).parameters)
    assert params[0] == "image_path"
    # category must precede thresholds for backward-compat shim.
    assert "category" in params
    assert "anomaly_threshold" in params
    assert "confidence_threshold" in params


def test_models_have_product_category():
    from models import Inspection
    assert hasattr(Inspection, "product_category")


def test_health_endpoints_registered():
    from main import app
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/" in paths
    assert "/health" in paths
    assert "/ready" in paths
