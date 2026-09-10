# ==========================================
# VISIONINSPECT AI
# SEVERITY SCORING
# ==========================================


def calculate_severity(
    defect_detected,
    defect_region,
    reconstruction_error
):

    # ------------------------------------------
    # NO DEFECT
    # ------------------------------------------

    if not defect_detected:

        return {
            "severity_score": 0,
            "severity_level": "Low"
        }


    # ------------------------------------------
    # 1. SIZE SCORE - 30%
    # ------------------------------------------

    size_score = 50

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

            size_score = 100

        elif area >= 500:

            size_score = 80

        elif area >= 250:

            size_score = 60

        elif area >= 100:

            size_score = 40

        else:

            size_score = 20


    # ------------------------------------------
    # 2. LOCATION SCORE - 25%
    # ------------------------------------------
    #
    # For the current prototype:
    # central areas are treated as more important.
    # ------------------------------------------

    location_score = 50

    if defect_region:

        x = defect_region.get(
            "x",
            0
        )

        y = defect_region.get(
            "y",
            0
        )

        center_x = x + (
            defect_region.get(
                "width",
                0
            ) / 2
        )

        center_y = y + (
            defect_region.get(
                "height",
                0
            ) / 2
        )


        # 224 x 224 processed image

        if (
            70 <= center_x <= 154
            and
            70 <= center_y <= 154
        ):

            location_score = 80

        else:

            location_score = 50


    # ------------------------------------------
    # 3. DEFECT TYPE SCORE - 25%
    # ------------------------------------------

    defect_type_score = 60


    # Current model only returns
    # "Potential Defect".

    # This is a moderate risk score until
    # exact defect classification is added.


    # ------------------------------------------
    # 4. DETECTION CONFIDENCE - 20%
    # ------------------------------------------

    # Current model uses reconstruction error.
    #
    # Higher reconstruction error indicates
    # stronger anomaly.
    #
    # Current threshold is 0.002435...
    # ------------------------------------------

    confidence_score = 50


    if reconstruction_error >= 0.0030:

        confidence_score = 100

    elif reconstruction_error >= 0.0027:

        confidence_score = 85

    elif reconstruction_error >= 0.0025:

        confidence_score = 70

    elif reconstruction_error >= 0.0023:

        confidence_score = 55

    else:

        confidence_score = 40


    # ------------------------------------------
    # FINAL WEIGHTED SCORE
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


    # Round score

    severity_score = round(
        severity_score,
        2
    )


    # ------------------------------------------
    # SEVERITY LEVEL
    # ------------------------------------------

    if severity_score >= 80:

        severity_level = "Critical"

    elif severity_score >= 60:

        severity_level = "High"

    elif severity_score >= 40:

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