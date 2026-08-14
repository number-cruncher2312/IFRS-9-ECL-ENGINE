from scripts.jokd import assign_stage

def test_defaulted_loan_is_stage_3():
    assert assign_stage(0.01, 0.02, 1) == 3


def test_no_default_and_no_significant_deterioration_is_stage_1():
    assert assign_stage(0.01, 0.015, 0) == 1


def test_pd_doubled_and_increased_by_0_005_is_stage_2():
    assert assign_stage(0.01, 0.02, 0) == 2


def test_default_takes_precedence_over_stage_2():
    assert assign_stage(0.01, 0.02, 1) == 3