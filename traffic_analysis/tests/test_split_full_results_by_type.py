import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.split_full_results_by_type import normalize_type, split_full_results_by_type  # noqa: E402


def test_normalize_type():
    assert normalize_type(" Fire  /Car ") == "fire__car"
    assert normalize_type(None) == "other"
    assert normalize_type("  ") == "other"


def test_split_full_results(tmp_path):
    input_dir = tmp_path / "full_results_2025"
    input_dir.mkdir()

    # incidents map
    incidents_path = tmp_path / "incidents.csv"
    pd.DataFrame(
        {
            "incident_id": ["1", "2", "3"],
            "Type": ["Fire ", "Hazard", "Other"],
        }
    ).to_csv(incidents_path, index=False)

    # A_final_output with Type
    pd.DataFrame(
        {
            "row_index": [0, 1, 2],
            "IncidentId": ["1", "2", "3"],
            "Type": ["Fire ", "Hazard", "Other"],
            "Data_Type": ["Speed", "Speed", "Speed"],
        }
    ).to_csv(input_dir / "A_final_output.csv", index=False)

    # common_table with Type（覆盖三类）
    pd.DataFrame(
        {
            "row_index": [0, 1, 2],
            "IncidentId": ["1", "2", "3"],
            "Type": ["Fire ", "Hazard", "Other"],
        }
    ).to_csv(input_dir / "A_final_common_table.csv", index=False)

    # score_table with Type
    pd.DataFrame(
        {
            "row_index": [0, 1, 2],
            "IncidentId": ["1", "2", "3"],
            "Type": ["Fire ", "Hazard", "Other"],
            "ImpactScore": [0.1, 0.2, 0.3],
        }
    ).to_csv(input_dir / "A_final_score_table.csv", index=False)

    # percentile_analysis with Type
    pd.DataFrame(
        {
            "IncidentId": ["1", "2", "3"],
            "Type": ["Fire ", "Hazard", "Other"],
            "Data_Type": ["Speed", "Speed", "Speed"],
        }
    ).to_csv(input_dir / "A_percentile_analysis.csv", index=False)

    # error table without Type
    pd.DataFrame(
        {
            "row_index": [5, 6],
            "incident_id": ["1", "2"],
            "error": ["e1", "e2"],
        }
    ).to_csv(input_dir / "A_error_traffic_table.csv", index=False)

    summary = split_full_results_by_type(
        input_dir=input_dir,
        output_dir=input_dir,
        incidents_path=incidents_path,
        chunk_size=1,
    )

    # 针对出现的类别检查文件存在性
    target_files = [
        "A_final_output.csv",
        "A_final_common_table.csv",
        "A_final_score_table.csv",
        "A_percentile_analysis.csv",
        "A_error_traffic_table.csv",
    ]

    for type_value in summary["A_final_output.csv"].keys():
        type_dir = input_dir / type_value
        assert type_dir.exists()
        for fname in target_files:
            # 只有当该文件在 summary 中出现该类型时才检查存在
            if type_value in summary.get(fname, {}):
                assert (type_dir / fname).exists()

    # Row counts should match summary
    assert summary["A_final_output.csv"]["fire"] == 1
    assert summary["A_final_output.csv"]["hazard"] == 1
    assert summary["A_final_output.csv"]["other"] == 1

    # Error table type from mapping
    fire_dir = input_dir / "fire"
    err_fire = pd.read_csv(fire_dir / "A_error_traffic_table.csv")
    assert len(err_fire) == 1
    assert str(err_fire.iloc[0]["incident_id"]) == "1"
