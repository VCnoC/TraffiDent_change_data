# -*- coding: utf-8 -*-
"""
传感器匹配模块单元测试

测试 processors/sensor_matcher.py 中的所有类和函数
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.sensor_matcher import (
    MatchResult,
    SensorMatcher,
    create_extracted_data,
)


class TestMatchResult:
    """测试 MatchResult 数据类"""

    def test_successful_match(self):
        """测试成功匹配"""
        sensors_df = pd.DataFrame({
            'station_id': [1001, 1002],
            'SensorNumber': [0, 1],
            'Fwy': [101, 101],
            'Direction': ['N', 'N'],
            'Abs PM': [15.0, 15.5]
        })

        result = MatchResult(
            incident_id="12345",
            row_index=0,
            matched_sensors=sensors_df,
            incident_info={'incident_id': '12345'},
            errors=[]
        )

        assert result.success is True
        assert result.sensor_count == 2

    def test_failed_match_with_errors(self):
        """测试有错误的匹配"""
        result = MatchResult(
            incident_id="12345",
            row_index=0,
            matched_sensors=pd.DataFrame(),
            incident_info={},
            errors=["没有找到传感器"]
        )

        assert result.success is False
        assert result.sensor_count == 0

    def test_failed_match_no_sensors(self):
        """测试没有匹配到传感器"""
        result = MatchResult(
            incident_id="12345",
            row_index=0,
            matched_sensors=pd.DataFrame(),
            incident_info={},
            errors=[]
        )

        assert result.success is False
        assert result.sensor_count == 0


class TestSensorMatcher:
    """测试 SensorMatcher 类"""

    @pytest.fixture
    def sample_sensors(self):
        """创建测试用传感器数据"""
        return pd.DataFrame({
            'station_id': [1001, 1002, 1003, 1004, 1005],
            'Fwy': [101, 101, 101, 405, 405],
            'Direction': ['N', 'N', 'S', 'N', 'S'],
            'Abs PM': [15.0, 15.3, 16.0, 20.0, 21.0],
            'Lat': [34.0, 34.1, 34.2, 34.3, 34.4],
            'Lng': [-118.0, -118.1, -118.2, -118.3, -118.4]
        })

    @pytest.fixture
    def node_order(self):
        """创建测试用节点顺序"""
        return np.array([1001, 1002, 1003, 1004, 1005])

    def test_init_with_node_order(self, sample_sensors, node_order):
        """测试使用 node_order 初始化"""
        matcher = SensorMatcher(sample_sensors, node_order)
        assert 'SensorNumber' in matcher.sensors.columns
        assert matcher.sensors.loc[matcher.sensors['station_id'] == 1001, 'SensorNumber'].iloc[0] == 0

    def test_init_without_node_order(self, sample_sensors):
        """测试不使用 node_order 初始化"""
        matcher = SensorMatcher(sample_sensors)
        assert 'SensorNumber' in matcher.sensors.columns
        # SensorNumber 应该是行索引
        assert matcher.sensors['SensorNumber'].iloc[0] == 0

    def test_match_success(self, sample_sensors, node_order):
        """测试成功匹配"""
        matcher = SensorMatcher(sample_sensors, node_order)

        incident = pd.Series({
            'incident_id': '12345',
            'Fwy': 101,
            'Freeway_direction': 'N',
            'Abs PM': 15.1,
            'dt': pd.Timestamp('2023-06-15 10:30:00'),
            'DESCRIPTION': 'Test incident'
        })

        result = matcher.match(incident, row_index=0)

        assert result.success is True
        assert result.sensor_count >= 1
        assert result.incident_id == '12345'

    def test_match_missing_fwy(self, sample_sensors, node_order):
        """测试缺少高速公路编号"""
        matcher = SensorMatcher(sample_sensors, node_order)

        incident = pd.Series({
            'incident_id': '12345',
            'Fwy': np.nan,
            'Freeway_direction': 'N',
            'Abs PM': 15.1,
            'dt': pd.Timestamp('2023-06-15 10:30:00')
        })

        result = matcher.match(incident, row_index=0)

        assert result.success is False
        assert len(result.errors) > 0
        assert any('Fwy' in e for e in result.errors)

    def test_match_missing_abs_pm(self, sample_sensors, node_order):
        """测试缺少里程标"""
        matcher = SensorMatcher(sample_sensors, node_order)

        incident = pd.Series({
            'incident_id': '12345',
            'Fwy': 101,
            'Freeway_direction': 'N',
            'Abs PM': np.nan,
            'dt': pd.Timestamp('2023-06-15 10:30:00')
        })

        result = matcher.match(incident, row_index=0)

        assert result.success is False
        assert len(result.errors) > 0
        assert any('Abs PM' in e for e in result.errors)

    def test_match_no_sensors_for_fwy(self, sample_sensors, node_order):
        """测试没有找到高速公路的传感器"""
        matcher = SensorMatcher(sample_sensors, node_order)

        incident = pd.Series({
            'incident_id': '12345',
            'Fwy': 999,  # 不存在的高速公路
            'Freeway_direction': 'N',
            'Abs PM': 15.1,
            'dt': pd.Timestamp('2023-06-15 10:30:00')
        })

        result = matcher.match(incident, row_index=0)

        assert result.success is False
        assert any('没有找到' in e for e in result.errors)

    def test_match_exceeds_max_distance(self, sample_sensors, node_order):
        """测试超过最大搜索距离"""
        matcher = SensorMatcher(sample_sensors, node_order, max_distance=0.1)

        incident = pd.Series({
            'incident_id': '12345',
            'Fwy': 101,
            'Freeway_direction': 'N',
            'Abs PM': 100.0,  # 远离所有传感器
            'dt': pd.Timestamp('2023-06-15 10:30:00')
        })

        result = matcher.match(incident, row_index=0)

        assert result.success is False

    def test_match_includes_adjacent_sensors(self, sample_sensors, node_order):
        """测试包含相邻传感器"""
        matcher = SensorMatcher(sample_sensors, node_order)

        incident = pd.Series({
            'incident_id': '12345',
            'Fwy': 101,
            'Freeway_direction': 'N',
            'Abs PM': 15.0,
            'dt': pd.Timestamp('2023-06-15 10:30:00')
        })

        result = matcher.match(incident, row_index=0)

        # MATLAB逻辑：匹配最近PM及下一个不同PM的传感器（closest_pm + closest_pm_xx）
        assert result.success is True
        assert result.sensor_count >= 1

    def test_match_batch(self, sample_sensors, node_order):
        """测试批量匹配"""
        matcher = SensorMatcher(sample_sensors, node_order)

        incidents = pd.DataFrame({
            'incident_id': ['001', '002', '003'],
            'Fwy': [101, 405, 101],
            'Freeway_direction': ['N', 'N', 'S'],
            'Abs PM': [15.1, 20.1, 16.0],
            'dt': pd.Timestamp('2023-06-15 10:30:00')
        })

        results = matcher.match_batch(incidents, start_idx=0, end_idx=3)

        assert len(results) == 3
        assert all(isinstance(r, MatchResult) for r in results)


class TestCreateExtractedData:
    """测试 create_extracted_data 函数"""

    def test_success_creates_dataframe(self):
        """测试成功匹配返回 DataFrame"""
        sensors_df = pd.DataFrame({
            'station_id': [1001],
            'SensorNumber': [0],
            'Fwy': [101],
            'Direction': ['N'],
            'Abs PM': [15.0]
        })

        match_result = MatchResult(
            incident_id="12345",
            row_index=0,
            matched_sensors=sensors_df,
            incident_info={'incident_id': '12345'},
            errors=[]
        )

        result = create_extracted_data(match_result)

        assert len(result) == 1
        assert 'station_id' in result.columns

    def test_failed_match_returns_empty(self):
        """测试失败匹配返回空 DataFrame"""
        match_result = MatchResult(
            incident_id="12345",
            row_index=0,
            matched_sensors=pd.DataFrame(),
            incident_info={},
            errors=["匹配失败"]
        )

        result = create_extracted_data(match_result)

        assert len(result) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
