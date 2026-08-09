from pathlib import Path

from evaluation.build_hebbian_followup_decision import (
    _read_test_log,
    relative_change,
    select_branch,
)


def test_relative_change_uses_baseline_magnitude_and_handles_zero():
    assert relative_change(2.0, 3.0) == 0.5
    assert relative_change(-2.0, -3.0) == -0.5
    assert relative_change(0.0, 1.0) is None


def test_branch_a_has_priority_when_joint_gate_passes():
    assert (
        select_branch(
            performance_pass=True,
            health_pass=True,
            integrity_pass=True,
            health_improved=True,
            direction_improved_scale_abnormal=True,
            enc3_isolated_failure=True,
            update_noise_primary=True,
        )
        == "A"
    )


def test_failed_performance_health_and_direction_select_branch_d():
    assert (
        select_branch(
            performance_pass=False,
            health_pass=False,
            integrity_pass=True,
            health_improved=False,
            direction_improved_scale_abnormal=False,
            enc3_isolated_failure=False,
            update_noise_primary=False,
        )
        == "D"
    )


def test_test_log_reader_accepts_utf8_bom_and_utf16(tmp_path: Path):
    utf8_log = tmp_path / "utf8.log"
    utf16_log = tmp_path / "utf16.log"
    utf8_log.write_text("73 passed", encoding="utf-8-sig")
    utf16_log.write_text("70 passed", encoding="utf-16")

    assert _read_test_log(utf8_log) == "73 passed"
    assert _read_test_log(utf16_log) == "70 passed"
