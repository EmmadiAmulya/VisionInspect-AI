"""Phase 4 integration: category upload contract + settings + history filter."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))


def test_upload_accepts_category_param():
    from routers.inspection_router import upload_and_inspect
    import inspect as _inspect
    params = _inspect.signature(upload_and_inspect).parameters
    assert "category" in params
    assert "category_query" in params


def test_categories_endpoint_exists():
    from routers.inspection_router import list_product_categories
    assert callable(list_product_categories)


def test_visualize_endpoint_exists():
    from routers.inspection_router import visualize_anomaly
    assert callable(visualize_anomaly)


def test_history_supports_category_filter():
    from routers.inspection_router import inspection_history
    import inspect as _inspect
    assert "category" in _inspect.signature(inspection_history).parameters


def test_settings_schema_has_confidence():
    from routers.inspection_router import SettingsUpdateSchema, DEFAULT_SETTINGS
    assert "classification_confidence_threshold" in DEFAULT_SETTINGS
    assert "classification_confidence_threshold" in (
        SettingsUpdateSchema.model_fields
    )
