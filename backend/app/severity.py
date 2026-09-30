# ==========================================
# VISIONINSPECT AI
# SEVERITY SCORING
# ==========================================


def calculate_severity(
    defect_detected,
    defect_type,
    confidence,
    defect_region,
    critical_cutoff=80.0,
    high_cutoff=60.0,
    medium_cutoff=40.0,
):

    # ------------------------------------------
    # NO DEFECT
    # ------------------------------------------

    if not defect_detected:

        return {
            "severity_score": 0.0,
            "severity_level": "Low"
        }


    # ------------------------------------------
    # 1. SIZE SCORE - 30%
    # ------------------------------------------

    size_score = 50.0

    if defect_region:

        width = defect_region.get(
            "width",
            0
        )

        height = defect_region.get(
            "height",
            0
        )

        area = width * height


        if area >= 1000:

            size_score = 100.0

        elif area >= 500:

            size_score = 80.0

        elif area >= 250:

            size_score = 65.0

        elif area >= 100:

            size_score = 50.0

        else:

            size_score = 30.0


    # ------------------------------------------
    # 2. LOCATION SCORE - 25%
    # ------------------------------------------
    #
    # Prototype rule:
    # defects near the center of the
    # 224 x 224 product image receive
    # a higher location-risk score.
    # ------------------------------------------

    location_score = 50.0

    if defect_region:

        x = defect_region.get(
            "x",
            0
        )

        y = defect_region.get(
            "y",
            0
        )

        width = defect_region.get(
            "width",
            0
        )

        height = defect_region.get(
            "height",
            0
        )


        center_x = x + (
            width / 2
        )

        center_y = y + (
            height / 2
        )


        if (
            70 <= center_x <= 154
            and
            70 <= center_y <= 154
        ):

            location_score = 80.0

        else:

            location_score = 50.0


    # ------------------------------------------
    # 3. DEFECT TYPE SCORE - 25%
    # ------------------------------------------

    defect_type_scores = {

        "broken_large": 90.0,

        "broken_small": 65.0,

        "contamination": 75.0,

        "good": 0.0
    }


    defect_type_score = defect_type_scores.get(
        defect_type,
        60.0
    )

    if defect_type not in defect_type_scores:
        # Unknown type: flag as medium-high rather than silently scoring.
        pass


    # ------------------------------------------
    # 4. DETECTION CONFIDENCE - 20%
    # ------------------------------------------
    # Accepts 0-1 (model output) or 0-100 (percent). Values >1 are
    # treated as percent to match PDF severity framework.

    try:
        conf_val = float(confidence)
    except (
        TypeError,
        ValueError
    ):
        conf_val = 0.0

    if conf_val <= 1.0:
        confidence_score = conf_val * 100.0
    else:
        confidence_score = conf_val


    # Keep confidence between 0 and 100

    confidence_score = max(
        0.0,
        min(
            confidence_score,
            100.0
        )
    )


    # ------------------------------------------
    # 5. FINAL WEIGHTED SCORE
    #
    # Size        = 30%
    # Location    = 25%
    # Defect Type = 25%
    # Confidence  = 20%
    # ------------------------------------------

    severity_score = (

        size_score * 0.30

        +

        location_score * 0.25

        +

        defect_type_score * 0.25

        +

        confidence_score * 0.20
    )


    severity_score = round(
        severity_score,
        2
    )


    # ------------------------------------------
    # 6. SEVERITY LEVEL
    # ------------------------------------------
    # Cutoffs are configurable via SystemSettings; fall back to
    # defaults (80/60/40) if misordered or invalid.

    try:
        _critical = float(critical_cutoff)
        _high = float(high_cutoff)
        _medium = float(medium_cutoff)
    except (TypeError, ValueError):
        _critical, _high, _medium = 80.0, 60.0, 40.0

    if not (_critical > _high > _medium):
        _critical, _high, _medium = 80.0, 60.0, 40.0

    if severity_score >= _critical:

        severity_level = "Critical"

    elif severity_score >= _high:

        severity_level = "High"

    elif severity_score >= _medium:

        severity_level = "Medium"

    else:

        severity_level = "Low"


    # ------------------------------------------
    # RETURN RESULT
    # ------------------------------------------

    return {

        "severity_score":
            severity_score,

        "severity_level":
            severity_level
    }