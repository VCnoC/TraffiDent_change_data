# -*- coding: utf-8 -*-
"""
评分计算模块单元测试

测试 processors/scorer.py 中的所有类和函数
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.scorer import (
    SEVERITY_LEVELS,
    TIME_TAG_MORNING_RUSH,
    TIME_TAG_NOON_RUSH,
    TIME_TAG_EVENING_RUSH,
    TIME_TAG_NON_RUSH,
    TIME_TAG_ERROR,
    compute_time_tag,
    SensorScore,
    IncidentScore,
    IncidentScorer,
    score_incident,
    create_score_table,
)
from processors.percentile_analyzer import (
    PercentileResult,
    SensorAnalysisResult,
    AnalysisResult,
)


class TestSeverityLevels:
    """测试严重程度等级定义"""

    def test_levels_exist(self):
        """测试等级定义存在"""
        assert 'NONE' in SEVERITY_LEVELS
        assert 'LOW' in SEVERITY_LEVELS
        assert 'MEDIUM' in SEVERITY_LEVELS
        assert 'HIGH' in SEVERITY_LEVELS
        assert 'SEVERE' in SEVERITY_LEVELS

    def test_level_ranges(self):
        """测试等级范围"""
        # NONE: 0.0-0.1
        assert SEVERITY_LEVELS['NONE'][0] == 0.0
        assert SEVERITY_LEVELS['NONE'][1] == 0.1

        # SEVERE: 0.7-1.0
        assert SEVERITY_LEVELS['SEVERE'][0] == 0.7
        assert SEVERITY_LEVELS['SEVERE'][1] == 1.0


class TestSensorScore:
    """测试 SensorScore 数据类"""

    def test_create_sensor_score(self):
        """测试创建传感器评分"""
        score = SensorScore(
            sensor_idx=0,
            occupancy_score=0.5,
            speed_score=0.4,
            volume_score=0.3,
            combined_score=0.42,
            time_series_pattern={'peak_position': 0.5, 'trend': 0.0}
        )

        assert score.sensor_idx == 0
        assert score.combined_score == 0.42


class TestIncidentScore:
    """测试 IncidentScore 数据类"""

    def test_create_incident_score(self):
        """测试创建事故评分"""
        sensor_score = SensorScore(
            sensor_idx=0,
            occupancy_score=0.5,
            speed_score=0.4,
            volume_score=0.3,
            combined_score=0.42
        )

        score = IncidentScore(
            incident_id="12345",
            row_index=0,
            sensor_scores={0: sensor_score},
            overall_score=0.42,
            severity_level='MEDIUM',
            severity_label='中等影响',
            peak_time_offset=2,
            duration_estimate=6
        )

        assert score.incident_id == "12345"
        assert score.overall_score == 0.42
        assert score.severity_level == 'MEDIUM'


class TestIncidentScorer:
    """测试 IncidentScorer 类"""

    @pytest.fixture
    def sample_analysis_result(self):
        """创建测试用分析结果"""
        occ_result = PercentileResult(
            data_type='occupancy',
            actual_values=np.array([0.5, 0.6, 0.7]),
            percentile_values={50: np.array([0.4, 0.5, 0.6])},
            deviations=np.array([0.25, 0.2, 0.17]),
            anomaly_scores=np.array([0.5, 0.5, 0.5])
        )

        spd_result = PercentileResult(
            data_type='speed',
            actual_values=np.array([35.0, 30.0, 25.0]),
            percentile_values={50: np.array([50.0, 55.0, 60.0])},
            deviations=np.array([-0.3, -0.45, -0.58]),
            anomaly_scores=np.array([-0.5, -1.0, -1.0])
        )

        vol_result = PercentileResult(
            data_type='volume',
            actual_values=np.array([80.0, 90.0, 100.0]),
            percentile_values={50: np.array([75.0, 85.0, 95.0])},
            deviations=np.array([0.07, 0.06, 0.05]),
            anomaly_scores=np.array([0.0, 0.0, 0.0])
        )

        sensor_result = SensorAnalysisResult(
            sensor_idx=0,
            occupancy=occ_result,
            speed=spd_result,
            volume=vol_result,
            overall_anomaly_score=0.5
        )

        return AnalysisResult(
            incident_id="12345",
            sensor_results={0: sensor_result},
            percentiles_used=[10, 25, 50, 75, 90],
            errors=[]
        )

    def test_init_default_weights(self):
        """测试默认权重"""
        scorer = IncidentScorer()

        assert scorer.weights['occupancy'] == 0.4
        assert scorer.weights['speed'] == 0.4
        assert scorer.weights['volume'] == 0.2

    def test_init_custom_weights(self):
        """测试自定义权重"""
        scorer = IncidentScorer(weights={
            'occupancy': 0.5,
            'speed': 0.3,
            'volume': 0.2
        })

        assert scorer.weights['occupancy'] == 0.5
        assert scorer.weights['speed'] == 0.3

    def test_score(self, sample_analysis_result):
        """测试评分计算"""
        scorer = IncidentScorer()
        result = scorer.score(
            incident_id="12345",
            row_index=0,
            analysis_result=sample_analysis_result
        )

        assert result.incident_id == "12345"
        assert 0 <= result.overall_score <= 1
        assert result.severity_level in SEVERITY_LEVELS

    def test_score_empty_analysis(self):
        """测试空分析结果"""
        scorer = IncidentScorer()

        empty_result = AnalysisResult(
            incident_id="12345",
            sensor_results={},
            percentiles_used=[],
            errors=[]
        )

        result = scorer.score(
            incident_id="12345",
            row_index=0,
            analysis_result=empty_result
        )

        assert result.overall_score == 0.0
        assert result.severity_level == 'NONE'

    def test_determine_severity_none(self):
        """测试无影响等级"""
        scorer = IncidentScorer()
        level, label = scorer._determine_severity(0.05)

        assert level == 'NONE'
        assert label == '无影响'

    def test_determine_severity_low(self):
        """测试轻微影响等级"""
        scorer = IncidentScorer()
        level, label = scorer._determine_severity(0.2)

        assert level == 'LOW'
        assert label == '轻微影响'

    def test_determine_severity_medium(self):
        """测试中等影响等级"""
        scorer = IncidentScorer()
        level, label = scorer._determine_severity(0.4)

        assert level == 'MEDIUM'
        assert label == '中等影响'

    def test_determine_severity_high(self):
        """测试较大影响等级"""
        scorer = IncidentScorer()
        level, label = scorer._determine_severity(0.6)

        assert level == 'HIGH'
        assert label == '较大影响'

    def test_determine_severity_severe(self):
        """测试严重影响等级"""
        scorer = IncidentScorer()
        level, label = scorer._determine_severity(0.85)

        assert level == 'SEVERE'
        assert label == '严重影响'

    def test_compute_type_score(self):
        """测试单类型评分"""
        scorer = IncidentScorer()

        result = PercentileResult(
            data_type='occupancy',
            actual_values=np.array([0.5]),
            percentile_values={},
            deviations=np.array([0.1]),
            anomaly_scores=np.array([0.5, -0.5, 1.0])
        )

        score = scorer._compute_type_score(result)

        # 平均绝对值：(0.5 + 0.5 + 1.0) / 3 = 0.67
        expected = np.mean(np.abs([0.5, -0.5, 1.0]))
        assert abs(score - expected) < 0.01

    def test_compute_type_score_empty(self):
        """测试空数据评分"""
        scorer = IncidentScorer()

        result = PercentileResult(
            data_type='occupancy',
            actual_values=np.array([]),
            percentile_values={},
            deviations=np.array([]),
            anomaly_scores=np.array([])
        )

        score = scorer._compute_type_score(result)

        assert score == 0.0

    def test_estimate_timing(self):
        """测试时间估计"""
        scorer = IncidentScorer(time_window=12)

        sensor_scores = {
            0: SensorScore(
                sensor_idx=0,
                occupancy_score=0.5,
                speed_score=0.5,
                volume_score=0.3,
                combined_score=0.45,
                time_series_pattern={'peak_position': 0.6}
            )
        }

        peak_offset, duration = scorer._estimate_timing(sensor_scores)

        # peak_position = 0.6, window = 25
        # offset = (0.6 - 0.5) * 25 = 2.5 -> 2
        assert isinstance(peak_offset, int)
        assert isinstance(duration, int)


class TestScoreIncident:
    """测试 score_incident 便捷函数"""

    def test_convenience_function(self):
        """测试便捷函数"""
        occ_result = PercentileResult(
            data_type='occupancy',
            actual_values=np.array([0.5]),
            percentile_values={},
            deviations=np.array([0.1]),
            anomaly_scores=np.array([0.5])
        )

        sensor_result = SensorAnalysisResult(
            sensor_idx=0,
            occupancy=occ_result,
            speed=occ_result,  # 简化
            volume=occ_result,
            overall_anomaly_score=0.5
        )

        analysis_result = AnalysisResult(
            incident_id="12345",
            sensor_results={0: sensor_result},
            errors=[]
        )

        result = score_incident(
            incident_id="12345",
            row_index=0,
            analysis_result=analysis_result
        )

        assert isinstance(result, IncidentScore)


class TestCreateScoreTable:
    """测试 create_score_table 函数"""

    def test_create_table(self):
        """测试创建表格"""
        sensor_score = SensorScore(
            sensor_idx=0,
            occupancy_score=0.5,
            speed_score=0.4,
            volume_score=0.3,
            combined_score=0.42
        )

        incident_score = IncidentScore(
            incident_id="12345",
            row_index=0,
            sensor_scores={0: sensor_score},
            overall_score=0.42,
            severity_level='MEDIUM',
            severity_label='中等影响'
        )

        table = create_score_table([incident_score])

        assert len(table) == 1
        assert 'IncidentId' in table.columns
        assert 'ImpactScore' in table.columns
        assert 'SeverityLevel' in table.columns
        assert table.iloc[0]['ImpactScore'] == 0.42

    def test_create_table_multiple(self):
        """测试创建多行表格"""
        scores = []
        for i in range(3):
            sensor_score = SensorScore(
                sensor_idx=0,
                occupancy_score=0.5,
                speed_score=0.4,
                volume_score=0.3,
                combined_score=0.42
            )

            scores.append(IncidentScore(
                incident_id=f"{i}",
                row_index=i,
                sensor_scores={0: sensor_score},
                overall_score=0.3 + i * 0.1,
                severity_level='MEDIUM',
                severity_label='中等影响'
            ))

        table = create_score_table(scores)

        assert len(table) == 3

    def test_create_table_empty(self):
        """测试空列表"""
        table = create_score_table([])

        assert len(table) == 0


class TestTimeTag:
    """测试时间标签（黄金时段分类）功能"""

    def test_time_tag_constants(self):
        """测试常量定义"""
        assert TIME_TAG_MORNING_RUSH == 1
        assert TIME_TAG_NOON_RUSH == 2
        assert TIME_TAG_EVENING_RUSH == 3
        assert TIME_TAG_NON_RUSH == 0
        assert TIME_TAG_ERROR == 999

    def test_morning_rush(self):
        """测试早高峰 (7:00-9:00)"""
        assert compute_time_tag(7) == TIME_TAG_MORNING_RUSH
        assert compute_time_tag(8) == TIME_TAG_MORNING_RUSH

    def test_morning_rush_boundary(self):
        """测试早高峰边界"""
        # 6:59 不是早高峰
        assert compute_time_tag(6) == TIME_TAG_NON_RUSH
        # 9:00 不是早高峰（左闭右开）
        assert compute_time_tag(9) == TIME_TAG_NON_RUSH

    def test_noon_rush(self):
        """测试午高峰 (11:00-13:00)"""
        assert compute_time_tag(11) == TIME_TAG_NOON_RUSH
        assert compute_time_tag(12) == TIME_TAG_NOON_RUSH

    def test_noon_rush_boundary(self):
        """测试午高峰边界"""
        assert compute_time_tag(10) == TIME_TAG_NON_RUSH
        assert compute_time_tag(13) == TIME_TAG_NON_RUSH

    def test_evening_rush(self):
        """测试晚高峰 (17:00-20:00)"""
        assert compute_time_tag(17) == TIME_TAG_EVENING_RUSH
        assert compute_time_tag(18) == TIME_TAG_EVENING_RUSH
        assert compute_time_tag(19) == TIME_TAG_EVENING_RUSH

    def test_evening_rush_boundary(self):
        """测试晚高峰边界"""
        assert compute_time_tag(16) == TIME_TAG_NON_RUSH
        assert compute_time_tag(20) == TIME_TAG_NON_RUSH

    def test_non_rush(self):
        """测试非高峰时段"""
        non_rush_hours = [0, 1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 15, 16, 20, 21, 22, 23]
        for hour in non_rush_hours:
            assert compute_time_tag(hour) == TIME_TAG_NON_RUSH, f"Hour {hour} should be non-rush"

    def test_invalid_hour(self):
        """测试无效小时数"""
        assert compute_time_tag(-1) == TIME_TAG_ERROR
        assert compute_time_tag(24) == TIME_TAG_ERROR
        assert compute_time_tag(100) == TIME_TAG_ERROR

    def test_none_hour(self):
        """测试 None 输入"""
        assert compute_time_tag(None) == TIME_TAG_ERROR

    def test_float_hour(self):
        """测试浮点数输入（应转换为整数）"""
        assert compute_time_tag(7.5) == TIME_TAG_MORNING_RUSH
        assert compute_time_tag(11.9) == TIME_TAG_NOON_RUSH

    def test_incident_score_time_tag(self):
        """测试 IncidentScore 中的 time_tag 字段"""
        sensor_score = SensorScore(
            sensor_idx=0,
            occupancy_score=0.5,
            speed_score=0.4,
            volume_score=0.3,
            combined_score=0.42
        )

        # 早高峰事故
        score = IncidentScore(
            incident_id="12345",
            row_index=0,
            sensor_scores={0: sensor_score},
            overall_score=0.42,
            severity_level='MEDIUM',
            severity_label='中等影响',
            time_tag=TIME_TAG_MORNING_RUSH
        )

        assert score.time_tag == TIME_TAG_MORNING_RUSH

    def test_scorer_with_incident_hour(self):
        """测试评分器计算时间标签"""
        occ_result = PercentileResult(
            data_type='occupancy',
            actual_values=np.array([0.5]),
            percentile_values={},
            deviations=np.array([0.1]),
            anomaly_scores=np.array([0.5])
        )

        sensor_result = SensorAnalysisResult(
            sensor_idx=0,
            occupancy=occ_result,
            speed=occ_result,
            volume=occ_result,
            overall_anomaly_score=0.5
        )

        analysis_result = AnalysisResult(
            incident_id="12345",
            sensor_results={0: sensor_result},
            errors=[]
        )

        scorer = IncidentScorer()

        # 测试不同时段
        result_morning = scorer.score(
            incident_id="12345", row_index=0,
            analysis_result=analysis_result, incident_hour=8
        )
        assert result_morning.time_tag == TIME_TAG_MORNING_RUSH

        result_evening = scorer.score(
            incident_id="12345", row_index=0,
            analysis_result=analysis_result, incident_hour=18
        )
        assert result_evening.time_tag == TIME_TAG_EVENING_RUSH

        result_non_rush = scorer.score(
            incident_id="12345", row_index=0,
            analysis_result=analysis_result, incident_hour=22
        )
        assert result_non_rush.time_tag == TIME_TAG_NON_RUSH

    def test_create_score_table_with_time_tag(self):
        """测试 create_score_table 包含 Time_tag 列"""
        sensor_score = SensorScore(
            sensor_idx=0,
            occupancy_score=0.5,
            speed_score=0.4,
            volume_score=0.3,
            combined_score=0.42
        )

        scores = [
            IncidentScore(
                incident_id="12345",
                row_index=0,
                sensor_scores={0: sensor_score},
                overall_score=0.42,
                severity_level='MEDIUM',
                severity_label='中等影响',
                time_tag=TIME_TAG_MORNING_RUSH
            ),
            IncidentScore(
                incident_id="67890",
                row_index=1,
                sensor_scores={0: sensor_score},
                overall_score=0.55,
                severity_level='HIGH',
                severity_label='较大影响',
                time_tag=TIME_TAG_EVENING_RUSH
            )
        ]

        table = create_score_table(scores)

        assert 'Time_tag' in table.columns
        assert table.iloc[0]['Time_tag'] == TIME_TAG_MORNING_RUSH
        assert table.iloc[1]['Time_tag'] == TIME_TAG_EVENING_RUSH


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
