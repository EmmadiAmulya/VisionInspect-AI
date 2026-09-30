# ==========================================
# VISIONINSPECT AI
# QUALITY ASSESSMENT
# ==========================================


def assess_quality(
    status,
    defect_detected,
    defect_type,
    classification_confidence,
    severity_score,
    severity_level,
    image_quality=None
):

    # ------------------------------------------
    # DEFAULT VALUES
    # ------------------------------------------

    quality_decision = "ACCEPT"
    risk_level = "Low"
    recommendation = "Product meets the current inspection criteria."


    # ------------------------------------------
    # IMAGE QUALITY CHECK
    # ------------------------------------------

    if image_quality is not None:

        quality_ok = image_quality.get(
            "quality_ok",
            True
        )

        if not quality_ok:

            return {

                "quality_status": "REJECT",

                "quality_decision": "REJECT",

                "risk_level": "High",

                "recommendation":
                    "Image quality is insufficient. "
                    "Capture or upload a clearer product image."
            }


    # ------------------------------------------
    # NO DEFECT
    # ------------------------------------------

    if not defect_detected:

        return {

            "quality_status": "PASS",

            "quality_decision": "ACCEPT",

            "risk_level": "Low",

            "recommendation":
                "No defect detected. "
                "Product meets the current inspection criteria."
        }


    # ------------------------------------------
    # DEFECT DETECTED
    # ------------------------------------------

    quality_decision = "REJECT"


    # ------------------------------------------
    # DETERMINE RISK LEVEL
    # ------------------------------------------

    if severity_level == "Critical":

        risk_level = "Critical"

        recommendation = (
            "Critical defect detected. "
            "Product should be immediately removed "
            "from the production line and reviewed."
        )


    elif severity_level == "High":

        risk_level = "High"

        recommendation = (
            "High-risk defect detected. "
            "Product should be rejected and "
            "reviewed by the quality team."
        )


    elif severity_level == "Medium":

        risk_level = "Medium"

        recommendation = (
            "Medium-risk defect detected. "
            "Product should undergo quality review "
            "before acceptance."
        )


    elif severity_level == "Low":

        risk_level = "Low"

        recommendation = (
            "Minor defect detected. "
            "Product should be reviewed according "
            "to the current quality criteria."
        )

    else:

        # Unknown severity: fail safe to High risk.
        risk_level = "High"

        recommendation = (
            "Defect detected with unknown severity. "
            "Product should be rejected and "
            "reviewed by the quality team."
        )


    # ------------------------------------------
    # ADD DEFECT-SPECIFIC INFORMATION
    # ------------------------------------------
    # Prepend defect detail while preserving risk-level guidance.

    defect_detail = ""

    if defect_type == "broken_large":

        defect_detail = (
            "Large break detected. "
            "Reject the product and inspect the "
            "production process for possible damage. "
        )


    elif defect_type == "broken_small":

        defect_detail = (
            "Small break detected. "
            "Reject or review the product according "
            "to the manufacturing quality criteria. "
        )


    elif defect_type == "contamination":

        defect_detail = (
            "Contamination detected. "
            "Reject the product and inspect the "
            "production process for contamination sources. "
        )

    if defect_detail:
        recommendation = defect_detail + recommendation


    # ------------------------------------------
    # FINAL QUALITY STATUS
    # ------------------------------------------

    quality_status = "FAIL"


    # ------------------------------------------
    # RETURN ASSESSMENT
    # ------------------------------------------

    return {

        "quality_status":
            quality_status,

        "quality_decision":
            quality_decision,

        "risk_level":
            risk_level,

        "recommendation":
            recommendation
    }