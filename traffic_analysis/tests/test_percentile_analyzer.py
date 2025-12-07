# -*- coding: utf-8 -*-
"""
百分位分析模块单元测试

测试 processors/percentile_analyzer.py 中的所有类和函数
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.percentile_analyzer import (
    PercentileResult,
    SensorAnalysisResult,
    AnalysisResult,
    PercentileAnalyzer,
    analyze_incident,
    create_percentile_table,
    DEFAULT_PERCENTILES,
)
from processors.historical_sampler import HistoricalSample


class TestPercentileResult:
    """测试 PercentileResult 数据类"""

    def test_median_property(self):
        """测试中位数属性"""
        result = PercentileResult(
            data_type='occupancy',
            actual_values=np.array([0.5, 0.6, 0.7]),
            percentile_values={50: np.array([0.4, 0.5, 0.6])},
            deviations=np.array([0.25, 0.2, 0.17]),
            anomaly_scores=np.array([0.0, 0.0, 0.0])
        )

        assert len(result.median) == 3
        assert result.median[0] == 0.4

    def test_is_anomalous_true(self):
        """测试异常判定为真"""
        result = PercentileResult(
            data_type='occupancy',
            actual_values=np.array([0.9]),
            percentile_values={},
            deviations=np.array([1.0]),
            anomaly_scores=np.array([1.0])  # 严重异常
        )

        assert result.is_anomalous == True

    def test_is_anomalous_false(self):
        """测试异常判定为假"""
        result = PercentileResult(
            data_type='occupancy',
            actual_values=np.array([0.5]),
            percentile_values={},
            deviations=np.array([0.1]),
            anomaly_scores=np.array([0.0])  # 正常
        )

        assert result.is_anomalous == False


class TestSensorAnalysisResult:
    """测试 SensorAnalysisResult 数据类"""

    def test_create_result(self):
        """测试创建结果"""
        occ_result = PercentileResult(
            data_type='occupancy',
            actual_values=np.array([0.5]),
            percentile_values={},
            deviations=np.array([0.1]),
            anomaly_scores=np.array([0.0])
        )

        spd_result = PercentileResult(
            data_type='speed',
            actual_values=np.array([45.0]),
            percentile_values={},
            deviations=np.array([-0.1]),
            anomaly_scores=np.array([-0.5])
        )

        vol_result = PercentileResult(
            data_type='volume',
            actual_values=np.array([80.0]),
            percentile_values={},
            deviations=np.array([0.2]),
            anomaly_scores=np.array([0.5])
        )

        result = SensorAnalysisResult(
            sensor_idx=0,
            occupancy=occ_result,
            speed=spd_result,
            volume=vol_result,
            overall_anomaly_score=0.3
        )

        assert result.sensor_idx == 0
        assert result.overall_anomaly_score == 0.3


class TestAnalysisResult:
    """测试 AnalysisResult 数据类"""

    def test_success(self):
        """测试成功结果"""
        result = AnalysisResult(
            incident_id="12345",
            sensor_results={0: None},  # 简化测试
            percentiles_used=[10, 25, 50, 75, 90],
            errors=[]
        )

        assert result.success is True

    def test_failure_with_errors(self):
        """测试有错误的结果"""
        result = AnalysisResult(
            incident_id="12345",
            sensor_results={},
            percentiles_used=[],
            errors=["分析失败"]
        )

        assert result.success is False

    def test_failure_empty_results(self):
        """测试空结果"""
        result = AnalysisResult(
            incident_id="12345",
            sensor_results={},
            percentiles_used=[],
            errors=[]
        )

        assert result.success is False


class TestPercentileAnalyzer:
    """测试 PercentileAnalyzer 类"""

    @pytest.fixture
    def sample_incident_data(self):
        """创建测试用事故数据"""
        # 每个传感器3行（occ, spd, vol）
        rows = []
        for sensor in [0, 1]:
            for data_type in ['Occupancy_Rate', 'Speed', 'Volume']:
                row = {
                    'SensorNumber': sensor,
                    'Data_Type': data_type,
                }
                # 添加 Data1 到 Data25
                for i in range(25):
                    if data_type == 'Occupancy_Rate':
                        row[f'Data{i+1}'] = np.random.rand() * 0.8
                    elif data_type == 'Speed':
                        row[f'Data{i+1}'] = 30 + np.random.rand() * 35
                    else:
                        row[f'Data{i+1}'] = np.random.randint(50, 150)
                rows.append(row)

        return pd.DataFrame(rows)

    @pytest.fixture
    def sample_historical_samples(self):
        """创建测试用历史采样"""
        samples = {}
        for sensor_idx in [0, 1]:
            samples[sensor_idx] = HistoricalSample(
                sensor_idx=sensor_idx,
                weekday=4,
                sample_dates=[datetime(2023, 6, 1)] * 20,
                occupancy_samples=np.random.rand(20, 25) * 0.7,
                speed_samples=30 + np.random.rand(20, 25) * 35,
                volume_samples=50 + np.random.rand(20, 25) * 100
            )
        return samples

    def test_init(self):
        """测试初始化"""
        analyzer = PercentileAnalyzer()
        assert analyzer.percentiles == DEFAULT_PERCENTILES

        analyzer2 = PercentileAnalyzer(percentiles=[25, 50, 75])
        assert analyzer2.percentiles == [25, 50, 75]

    def test_analyze(self, sample_incident_data, sample_historical_samples):
        """测试分析"""
        analyzer = PercentileAnalyzer()
        result = analyzer.analyze(
            incident_id="12345",
            incident_data=sample_incident_data,
            historical_samples=sample_historical_samples
        )

        assert result.success is True
        assert len(result.sensor_results) == 2

    def test_analyze_empty_incident_data(self, sample_historical_samples):
        """测试空事故数据"""
        analyzer = PercentileAnalyzer()
        result = analyzer.analyze(
            incident_id="12345",
            incident_data=pd.DataFrame(),
            historical_samples=sample_historical_samples
        )

        assert result.success is False
        assert any('为空' in e for e in result.errors)

    def test_analyze_missing_historical_sample(self, sample_incident_data):
        """测试缺少历史采样"""
        analyzer = PercentileAnalyzer()
        result = analyzer.analyze(
            incident_id="12345",
            incident_data=sample_incident_data,
            historical_samples={}  # 空采样
        )

        # 没有历史数据就无法分析
        assert len(result.sensor_results) == 0

    def test_compute_anomaly_scores(self):
        """测试异常得分计算"""
        analyzer = PercentileAnalyzer()

        actual = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        percentiles = {
            10: np.array([0.15, 0.25, 0.45, 0.65, 0.85]),
            25: np.array([0.2, 0.3, 0.5, 0.7, 0.88]),
            50: np.array([0.3, 0.4, 0.55, 0.75, 0.9]),
            75: np.array([0.4, 0.5, 0.6, 0.8, 0.92]),
            90: np.array([0.5, 0.6, 0.7, 0.85, 0.95]),
        }

        scores = analyzer._compute_anomaly_scores(actual, percentiles)

        assert len(scores) == 5
        # 第一个值低于 P10，应该是 -1.0
        assert scores[0] == -1.0
        # 其他值根据百分位位置

    def test_compute_overall_score(self):
        """测试综合得分计算"""
        analyzer = PercentileAnalyzer()

        occ_result = PercentileResult(
            data_type='occupancy',
            actual_values=np.array([0.5]),
            percentile_values={},
            deviations=np.array([0.1]),
            anomaly_scores=np.array([0.5])
        )

        spd_result = PercentileResult(
            data_type='speed',
            actual_values=np.array([45.0]),
            percentile_values={},
            deviations=np.array([-0.1]),
            anomaly_scores=np.array([-0.5])
        )

        vol_result = PercentileResult(
            data_type='volume',
            actual_values=np.array([80.0]),
            percentile_values={},
            deviations=np.array([0.2]),
            anomaly_scores=np.array([0.5])
        )

        score = analyzer._compute_overall_score(occ_result, spd_result, vol_result)

        # 加权平均：(0.5*0.4 + 0.5*0.4 + 0.5*0.2) / (0.4+0.4+0.2)
        expected = (0.5 * 0.4 + 0.5 * 0.4 + 0.5 * 0.2)
        assert abs(score - expected) < 0.01


class TestAnalyzeIncident:
    """测试 analyze_incident 便捷函数"""

    def test_convenience_function(self):
        """测试便捷函数"""
        incident_data = pd.DataFrame({
            'SensorNumber': [0, 0, 0],
            'Data_Type': ['Occupancy_Rate', 'Speed', 'Volume'],
            'Data1': [0.5, 50.0, 100],
            'Data2': [0.6, 55.0, 110],
        })

        historical_samples = {
            0: HistoricalSample(
                sensor_idx=0,
                weekday=4,
                sample_dates=[datetime(2023, 6, 1)] * 10,
                occupancy_samples=np.random.rand(10, 2),
                speed_samples=np.random.rand(10, 2) * 65,
                volume_samples=np.random.randint(50, 150, (10, 2)).astype(float)
            )
        }

        result = analyze_incident(
            incident_id="12345",
            incident_data=incident_data,
            historical_samples=historical_samples
        )

        assert isinstance(result, AnalysisResult)


class TestCreatePercentileTable:
    """测试 create_percentile_table 函数"""

    def test_create_table(self):
        """测试创建表格"""
        occ_result = PercentileResult(
            data_type='occupancy',
            actual_values=np.array([0.5, 0.6]),
            percentile_values={50: np.array([0.4, 0.5])},
            deviations=np.array([0.25, 0.2]),
            anomaly_scores=np.array([0.5, 0.5])
        )

        spd_result = PercentileResult(
            data_type='speed',
            actual_values=np.array([45.0, 50.0]),
            percentile_values={50: np.array([50.0, 55.0])},
            deviations=np.array([-0.1, -0.09]),
            anomaly_scores=np.array([-0.5, -0.5])
        )

        vol_result = PercentileResult(
            data_type='volume',
            actual_values=np.array([80.0, 90.0]),
            percentile_values={50: np.array([75.0, 85.0])},
            deviations=np.array([0.07, 0.06]),
            anomaly_scores=np.array([0.0, 0.0])
        )

        sensor_result = SensorAnalysisResult(
            sensor_idx=0,
            occupancy=occ_result,
            speed=spd_result,
            volume=vol_result,
            overall_anomaly_score=0.33
        )

        analysis_result = AnalysisResult(
            incident_id="12345",
            sensor_results={0: sensor_result},
            percentiles_used=[50],
            errors=[]
        )

        table = create_percentile_table(analysis_result)

        assert len(table) == 3  # 3 data types
        assert 'IncidentId' in table.columns
        assert 'SensorNumber' in table.columns
        assert 'Data_Type' in table.columns

    def test_empty_result(self):
        """测试空结果"""
        analysis_result = AnalysisResult(
            incident_id="12345",
            sensor_results={},
            percentiles_used=[],
            errors=[]
        )

        table = create_percentile_table(analysis_result)

        assert len(table) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
