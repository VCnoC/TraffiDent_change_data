# -*- coding: utf-8 -*-
"""
历史数据采样模块单元测试

测试 processors/historical_sampler.py 中的所有类和函数
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.historical_sampler import (
    HistoricalSample,
    SamplingResult,
    HistoricalSampler,
    sample_historical_data,
    compute_percentiles,
)
from data.loader import TrafficMatrices


class TestHistoricalSample:
    """测试 HistoricalSample 数据类"""

    def test_properties(self):
        """测试属性"""
        sample = HistoricalSample(
            sensor_idx=0,
            weekday=4,  # Thursday
            sample_dates=[datetime(2023, 6, 1), datetime(2023, 6, 8)],
            occupancy_samples=np.random.rand(2, 25),
            speed_samples=np.random.rand(2, 25),
            volume_samples=np.random.rand(2, 25)
        )

        assert sample.n_samples == 2
        assert sample.window_size == 25

    def test_empty_sample(self):
        """测试空采样"""
        sample = HistoricalSample(
            sensor_idx=0,
            weekday=4,
            sample_dates=[],
            occupancy_samples=np.array([]).reshape(0, 0),
            speed_samples=np.array([]).reshape(0, 0),
            volume_samples=np.array([]).reshape(0, 0)
        )

        assert sample.n_samples == 0
        assert sample.window_size == 0


class TestSamplingResult:
    """测试 SamplingResult 数据类"""

    def test_success(self):
        """测试成功结果"""
        sample = HistoricalSample(
            sensor_idx=0,
            weekday=4,
            sample_dates=[datetime(2023, 6, 1)],
            occupancy_samples=np.random.rand(1, 25),
            speed_samples=np.random.rand(1, 25),
            volume_samples=np.random.rand(1, 25)
        )

        result = SamplingResult(
            incident_dt=datetime(2023, 6, 15, 10, 30),
            samples={0: sample},
            excluded_dates=[],
            errors=[]
        )

        assert result.success is True

    def test_failure_with_errors(self):
        """测试有错误的结果"""
        result = SamplingResult(
            incident_dt=datetime(2023, 6, 15),
            samples={},
            excluded_dates=[],
            errors=["采样失败"]
        )

        assert result.success is False

    def test_failure_empty_samples(self):
        """测试空采样结果"""
        result = SamplingResult(
            incident_dt=datetime(2023, 6, 15),
            samples={},
            excluded_dates=[],
            errors=[]
        )

        assert result.success is False


class TestHistoricalSampler:
    """测试 HistoricalSampler 类"""

    @pytest.fixture
    def sample_matrices(self):
        """创建测试用流量矩阵"""
        # 创建全年数据 (105120 时间步)
        np.random.seed(42)
        n_timesteps = 105120
        n_sensors = 10

        return TrafficMatrices(
            occupancy=np.random.rand(n_timesteps, n_sensors).astype(np.float32),
            speed=np.random.rand(n_timesteps, n_sensors).astype(np.float32) * 65,
            volume=np.random.randint(0, 100, (n_timesteps, n_sensors)).astype(np.float32),
            node_order=np.arange(n_sensors)
        )

    def test_init(self, sample_matrices):
        """测试初始化"""
        sampler = HistoricalSampler(sample_matrices, time_window=12, data_year=2023)

        assert sampler.time_window == 12
        assert sampler.data_year == 2023
        assert sampler.total_steps == 25

    def test_sample_basic(self, sample_matrices):
        """测试基本采样"""
        sampler = HistoricalSampler(sample_matrices, time_window=6, data_year=2023)

        # 6月15日是星期四
        incident_dt = datetime(2023, 6, 15, 10, 30)
        sensor_indices = [0, 1, 2]

        result = sampler.sample(incident_dt, sensor_indices)

        assert result.success is True
        assert len(result.samples) >= 1
        assert len(result.excluded_dates) == 3  # 事故当天 ± 1天

    def test_sample_with_weeks_limit(self, sample_matrices):
        """测试限制周数采样"""
        sampler = HistoricalSampler(sample_matrices, time_window=6, data_year=2023)

        incident_dt = datetime(2023, 6, 15, 10, 30)
        sensor_indices = [0]

        result = sampler.sample(incident_dt, sensor_indices, weeks_limit=4)

        assert result.success is True

    def test_sample_excluded_dates(self, sample_matrices):
        """测试排除日期"""
        sampler = HistoricalSampler(sample_matrices, time_window=6, data_year=2023, exclude_days=2)

        incident_dt = datetime(2023, 6, 15, 10, 30)
        sensor_indices = [0]

        result = sampler.sample(incident_dt, sensor_indices)

        # 应排除 6/13, 6/14, 6/15, 6/16, 6/17
        assert len(result.excluded_dates) == 5

    def test_sample_invalid_sensor_index(self, sample_matrices):
        """测试无效的传感器索引"""
        sampler = HistoricalSampler(sample_matrices, time_window=6, data_year=2023)

        incident_dt = datetime(2023, 6, 15, 10, 30)
        sensor_indices = [999, -1]  # 无效索引

        result = sampler.sample(incident_dt, sensor_indices)

        # 应该没有采样到任何数据
        assert len(result.samples) == 0

    def test_get_excluded_dates(self, sample_matrices):
        """测试获取排除日期"""
        sampler = HistoricalSampler(sample_matrices, time_window=6, data_year=2023, exclude_days=1)

        incident_dt = datetime(2023, 6, 15, 10, 30)
        excluded = sampler._get_excluded_dates(incident_dt)

        assert len(excluded) == 3  # 6/14, 6/15, 6/16
        dates = [d.date() for d in excluded]
        assert datetime(2023, 6, 14).date() in dates
        assert datetime(2023, 6, 15).date() in dates
        assert datetime(2023, 6, 16).date() in dates

    def test_find_same_weekday_dates(self, sample_matrices):
        """测试找相同星期几的日期"""
        sampler = HistoricalSampler(sample_matrices, time_window=6, data_year=2023)

        incident_dt = datetime(2023, 6, 15, 10, 30)  # Thursday
        excluded = sampler._get_excluded_dates(incident_dt)
        dates = sampler._find_same_weekday_dates(incident_dt, excluded)

        # 应该找到一年中所有的星期四（排除事故当天）
        assert len(dates) >= 50  # 大约52周

        # 所有日期都应该是星期四
        for d in dates:
            assert d.isoweekday() == 4

    def test_is_valid_sample(self, sample_matrices):
        """测试数据有效性检查"""
        sampler = HistoricalSampler(sample_matrices, time_window=6, data_year=2023)

        # 有效数据
        valid = sampler._is_valid_sample(
            np.array([1.0, 2.0, 3.0]),
            np.array([10.0, 20.0, 30.0]),
            np.array([100.0, 200.0, 300.0])
        )
        assert valid is True

        # 全是 NaN
        invalid_nan = sampler._is_valid_sample(
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan])
        )
        assert invalid_nan is False

        # 全是 0
        invalid_zero = sampler._is_valid_sample(
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0])
        )
        assert invalid_zero is False


class TestSampleHistoricalData:
    """测试 sample_historical_data 便捷函数"""

    def test_convenience_function(self):
        """测试便捷函数"""
        np.random.seed(42)
        matrices = TrafficMatrices(
            occupancy=np.random.rand(105120, 10).astype(np.float32),
            speed=np.random.rand(105120, 10).astype(np.float32) * 65,
            volume=np.random.randint(0, 100, (105120, 10)).astype(np.float32),
            node_order=np.arange(10)
        )

        result = sample_historical_data(
            incident_dt=datetime(2023, 6, 15, 10, 30),
            sensor_indices=[0, 1],
            matrices=matrices,
            time_window=6,
            data_year=2023
        )

        assert isinstance(result, SamplingResult)


class TestComputePercentiles:
    """测试 compute_percentiles 函数"""

    def test_basic_percentiles(self):
        """测试基本百分位计算"""
        sample = HistoricalSample(
            sensor_idx=0,
            weekday=4,
            sample_dates=[datetime(2023, 6, 1)] * 10,
            occupancy_samples=np.random.rand(10, 25),
            speed_samples=np.random.rand(10, 25) * 65,
            volume_samples=np.random.randint(0, 100, (10, 25)).astype(float)
        )

        result = compute_percentiles(sample, [10, 25, 50, 75, 90])

        assert 'occupancy' in result
        assert 'speed' in result
        assert 'volume' in result

        for p in [10, 25, 50, 75, 90]:
            assert p in result['occupancy']
            assert len(result['occupancy'][p]) == 25

    def test_empty_sample(self):
        """测试空采样"""
        sample = HistoricalSample(
            sensor_idx=0,
            weekday=4,
            sample_dates=[],
            occupancy_samples=np.array([]).reshape(0, 0),
            speed_samples=np.array([]).reshape(0, 0),
            volume_samples=np.array([]).reshape(0, 0)
        )

        result = compute_percentiles(sample)

        assert result['occupancy'] == {}
        assert result['speed'] == {}
        assert result['volume'] == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
