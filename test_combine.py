from allelion.acmg.combine import class_2015, class_from_points, points_of, posterior


def crit(code, direction, strength, status="met"):
    return {"code": code, "direction": direction, "strength": strength, "status": status}


def test_point_thresholds():
    assert class_from_points(10, False) == "Pathogenic"
    assert class_from_points(9, False) == "Likely pathogenic"
    assert class_from_points(6, False) == "Likely pathogenic"
    assert class_from_points(5, False) == "Uncertain significance"
    assert class_from_points(0, False) == "Uncertain significance"
    assert class_from_points(-1, False) == "Likely benign"
    assert class_from_points(-7, False) == "Benign"
    assert class_from_points(12, True) == "Benign"


def test_points_and_posterior():
    c = [crit("PVS1", "pathogenic", "VeryStrong"), crit("PM2", "pathogenic", "Supporting"), crit("BP4", "benign", "Supporting"),
         crit("PS3", "pathogenic", "Strong", "not_evaluated")]
    assert points_of(c) == 8
    assert abs(posterior(8) - 350 * 0.1 / (349 * 0.1 + 1)) < 1e-9
    assert posterior(10) > 0.99 and 0.899 < posterior(6) < 0.99  # six points sits at the 0.90 boundary


def test_2015_rules():
    assert class_2015([crit("PVS1", "pathogenic", "VeryStrong"), crit("PS1", "pathogenic", "Strong")]) == "Pathogenic"
    assert class_2015([crit("PVS1", "pathogenic", "VeryStrong"), crit("PM2", "pathogenic", "Moderate")]) == "Likely pathogenic"
    assert class_2015([crit("BS1", "benign", "Strong"), crit("BP4", "benign", "Supporting")]) == "Likely benign"
    assert class_2015([crit("PS3", "pathogenic", "Strong"), crit("PM1", "pathogenic", "Moderate"), crit("BS1", "benign", "Strong"), crit("BP4", "benign", "Supporting")]) == "Uncertain significance"
