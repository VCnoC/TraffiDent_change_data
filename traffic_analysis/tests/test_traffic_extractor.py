# -*- coding: utf-8 -*-
"""
流量数据提取模块单元测试

测试 processors/traffic_extractor.py 中的所有类和函数
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.traffic_extractor import (
    ExtractionResult,
    TrafficExtractor,
    extract_traffic_data,
)
from processors.sensor_matcher import MatchResult
from data.loader import TrafficMatrices


class TestExtractionResult:
    """测试 ExtractionResult 数据类"""

    def test_success_with_data(self):
        """测试有数据时成功"""
        result = ExtractionResult(
            incident_table=pd.DataFrame({'col': [1, 2, 3]}),
            common_table=pd.DataFrame({'col': [4, 5, 6]}),
            errors=[]
        )

        assert result.success is True

    def test_failure_with_errors(self):
        """测试有错误时失败"""
        result = ExtractionResult(
            incident_table=pd.DataFrame({'col': [1]}),
            common_table=pd.DataFrame(),
            errors=["提取失败"]
        )

        assert result.success is False

    def test_failure_empty_incident_table(self):
        """测试事故表为空时失败"""
        result = ExtractionResult(
            incident_table=pd.DataFrame(),
            common_table=pd.DataFrame({'col': [1]}),
            errors=[]
        )

        assert result.success is False


class TestTrafficExtractor:
    """测试 TrafficExtractor 类"""

    @pytest.fixture
    def sample_matrices(self):
        """创建测试用流量矩阵"""
        # 创建小型测试矩阵
        # 形状: (1000, 10) = (时间步, 传感器数)
        np.random.seed(42)
        return TrafficMatrices(
            occupancy=np.random.rand(1000, 10).astype(np.float32),
            speed=np.random.rand(1000, 10).astype(np.float32) * 65,  # 0-65 mph
            volume=np.random.randint(0, 100, (1000, 10)).astype(np.float32),
            node_order=np.arange(10)
        )

    @pytest.fixture
    def sample_match_result(self):
        """创建测试用匹配结果"""
        sensors_df = pd.DataFrame({
            'SensorNumber': [0, 1],
            'station_id': [1001, 1002],
            'Fwy': [101, 101],
            'Direction': ['N', 'N'],
            'Abs PM': [15.0, 15.3]
        })

        return MatchResult(
            incident_id="12345",
            row_index=0,
            matched_sensors=sensors_df,
            incident_info={
                'incident_id': '12345',
                'row_index': 0,
                'dt': pd.Timestamp('2023-01-02 10:30:00'),  # timestep ~462
                'DESCRIPTION': 'Test incident'
            },
            errors=[]
        )

    def test_init(self, sample_matrices):
        """测试初始化"""
        extractor = TrafficExtractor(sample_matrices, time_window=12)

        assert extractor.time_window == 12
        assert extractor.total_steps == 25  # 2 * 12 + 1
        assert extractor.max_timesteps == 1000
        assert extractor.num_sensors == 10

    def test_extract_success(self, sample_matrices, sample_match_result):
        """测试成功提取"""
        extractor = TrafficExtractor(sample_matrices, time_window=6, data_year=2023)
        result = extractor.extract(sample_match_result)

        assert result.success is True
        assert len(result.incident_table) > 0
        # 每个传感器3行数据（occ, spd, vol）
        assert len(result.incident_table) == 6  # 2 sensors * 3 types

    def test_extract_with_columns(self, sample_matrices, sample_match_result):
        """测试提取结果包含正确的列"""
        extractor = TrafficExtractor(sample_matrices, time_window=6, data_year=2023)
        result = extractor.extract(sample_match_result)

        expected_cols = ['row_index', 'IncidentId', 'SensorNumber', 'Data_Type']
        for col in expected_cols:
            assert col in result.incident_table.columns

    def test_extract_failed_match(self, sample_matrices):
        """测试失败的匹配结果"""
        failed_match = MatchResult(
            incident_id="12345",
            row_index=0,
            matched_sensors=pd.DataFrame(),
            incident_info={},
            errors=["匹配失败"]
        )

        extractor = TrafficExtractor(sample_matrices, time_window=6)
        result = extractor.extract(failed_match)

        assert result.success is False
        assert len(result.errors) > 0

    def test_extract_missing_dt(self, sample_matrices):
        """测试缺少事故时间"""
        sensors_df = pd.DataFrame({
            'SensorNumber': [0],
            'station_id': [1001]
        })

        match_result = MatchResult(
            incident_id="12345",
            row_index=0,
            matched_sensors=sensors_df,
            incident_info={
                'incident_id': '12345',
                'dt': pd.NaT  # 缺少时间
            },
            errors=[]
        )

        extractor = TrafficExtractor(sample_matrices, time_window=6)
        result = extractor.extract(match_result)

        assert result.success is False
        assert any('时间' in e for e in result.errors)

    def test_extract_common_data(self, sample_matrices, sample_match_result):
        """测试前后天数据提取"""
        extractor = TrafficExtractor(sample_matrices, time_window=6, data_year=2023)
        result = extractor.extract(sample_match_result)

        # common_table 应该包含前后天数据
        if len(result.common_table) > 0:
            assert 'Period' in result.common_table.columns
            periods = result.common_table['Period'].unique()
            assert 'Before' in periods or 'After' in periods

    def test_extract_sensor_data(self, sample_matrices):
        """测试单传感器数据提取"""
        extractor = TrafficExtractor(sample_matrices, time_window=6)

        data = extractor._extract_sensor_data(
            sample_matrices.occupancy,
            sensor_idx=0,
            start_step=100,
            end_step=112
        )

        assert len(data) == 13  # 100 to 112 inclusive
        assert all(isinstance(v, float) for v in data)


class TestExtractTrafficData:
    """测试 extract_traffic_data 便捷函数"""

    def test_convenience_function(self):
        """测试便捷函数"""
        np.random.seed(42)
        matrices = TrafficMatrices(
            occupancy=np.random.rand(1000, 10).astype(np.float32),
            speed=np.random.rand(1000, 10).astype(np.float32) * 65,
            volume=np.random.randint(0, 100, (1000, 10)).astype(np.float32),
            node_order=np.arange(10)
        )

        sensors_df = pd.DataFrame({
            'SensorNumber': [0],
            'station_id': [1001]
        })

        match_result = MatchResult(
            incident_id="12345",
            row_index=0,
            matched_sensors=sensors_df,
            incident_info={
                'incident_id': '12345',
                'dt': pd.Timestamp('2023-01-02 10:30:00')
            },
            errors=[]
        )

        result = extract_traffic_data(
            match_result,
            matrices,
            time_window=6,
            data_year=2023
        )

        assert isinstance(result, ExtractionResult)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
