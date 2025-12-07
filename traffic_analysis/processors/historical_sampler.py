# -*- coding: utf-8 -*-
"""
历史数据采样模块

根据事故发生的星期几，采样同一星期几的历史数据。
用于计算百分位基准，与事故时刻数据进行对比分析。

对应 MATLAB 代码：
- sort_function.m: 历史数据排序和采样

采样逻辑：
1. 获取事故发生时刻的星期几
2. 找出当年所有相同星期几的日期
3. 提取这些日期同一时段的流量数据
4. 排除事故当天及前后天的数据
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from ..utils.time_utils import (
        datetime_to_timestep,
        timestep_to_datetime,
        get_iso_weekday,
        TIMESTEPS_PER_DAY,
        DAYS_IN_YEAR,
        DAYS_IN_LEAP_YEAR,
        is_leap_year,
    )
    from ..data.loader import TrafficMatrices
except ImportError:
    # 支持直接运行模块
    from utils.time_utils import (
        datetime_to_timestep,
        timestep_to_datetime,
        get_iso_weekday,
        TIMESTEPS_PER_DAY,
        DAYS_IN_YEAR,
        DAYS_IN_LEAP_YEAR,
        is_leap_year,
    )
    from data.loader import TrafficMatrices


@dataclass
class HistoricalSample:
    """历史采样结果

    Attributes:
        sensor_idx: 传感器索引
        weekday: 星期几 (1-7, ISO 标准)
        sample_dates: 采样的日期列表
        occupancy_samples: 占用率采样数据 (n_samples, window_size)
        speed_samples: 速度采样数据 (n_samples, window_size)
        volume_samples: 流量采样数据 (n_samples, window_size)
    """
    sensor_idx: int
    weekday: int
    sample_dates: List[datetime]
    occupancy_samples: np.ndarray
    speed_samples: np.ndarray
    volume_samples: np.ndarray

    @property
    def n_samples(self) -> int:
        """采样数量"""
        return len(self.sample_dates)

    @property
    def window_size(self) -> int:
        """时间窗口大小"""
        if len(self.occupancy_samples) > 0:
            return self.occupancy_samples.shape[1]
        return 0


@dataclass
class SamplingResult:
    """采样结果集合

    Attributes:
        incident_dt: 事故发生时间
        samples: 各传感器的采样结果字典 {sensor_idx: HistoricalSample}
        excluded_dates: 被排除的日期（事故当天及前后）
        errors: 错误信息
    """
    incident_dt: datetime
    samples: Dict[int, HistoricalSample]
    excluded_dates: List[datetime] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """是否采样成功"""
        return len(self.samples) > 0 and len(self.errors) == 0


class HistoricalSampler:
    """历史数据采样器

    对应 MATLAB: sort_function

    根据事故发生的星期几，采样同一星期几的历史数据，
    用于计算百分位基准。

    Example:
        >>> sampler = HistoricalSampler(matrices, time_window=12)
        >>> result = sampler.sample(incident_dt, sensor_indices)
    """

    def __init__(
        self,
        matrices: TrafficMatrices,
        time_window: int = 12,
        data_year: int = 2023,
        exclude_days: int = 1
    ):
        """初始化采样器

        Args:
            matrices: 流量矩阵数据
            time_window: 事故时刻前后的时间步数（默认12，即前后各1小时）
            data_year: 数据年份
            exclude_days: 排除事故前后的天数（默认1天）
        """
        self.occupancy = matrices.occupancy
        self.speed = matrices.speed
        self.volume = matrices.volume
        self.time_window = time_window
        self.data_year = data_year
        self.exclude_days = exclude_days

        # 矩阵尺寸
        self.max_timesteps = self.occupancy.shape[0]
        self.num_sensors = self.occupancy.shape[1]

        # 总时间步数（2 * window + 1）
        self.total_steps = 2 * time_window + 1

    def sample(
        self,
        incident_dt: datetime,
        sensor_indices: List[int],
        weeks_limit: Optional[int] = None
    ) -> SamplingResult:
        """采样历史数据

        对应 MATLAB: sort_function

        Args:
            incident_dt: 事故发生时间
            sensor_indices: 传感器索引列表
            weeks_limit: 限制采样的周数（可选，用于限制历史范围）

        Returns:
            SamplingResult 包含各传感器的历史采样数据
        """
        errors = []

        if isinstance(incident_dt, pd.Timestamp):
            incident_dt = incident_dt.to_pydatetime()

        # 获取事故发生时刻的信息
        weekday = get_iso_weekday(incident_dt)
        incident_timestep = datetime_to_timestep(incident_dt, self.data_year)

        # 计算需要排除的日期范围
        excluded_dates = self._get_excluded_dates(incident_dt)

        # 找出同一星期几的所有日期
        sample_dates = self._find_same_weekday_dates(
            incident_dt, excluded_dates, weeks_limit
        )

        if len(sample_dates) == 0:
            errors.append("没有找到可用的历史采样日期")
            return SamplingResult(
                incident_dt=incident_dt,
                samples={},
                excluded_dates=excluded_dates,
                errors=errors
            )

        # 对每个传感器进行采样
        samples = {}
        for sensor_idx in sensor_indices:
            if sensor_idx < 0 or sensor_idx >= self.num_sensors:
                continue

            sample = self._sample_sensor(
                sensor_idx, incident_dt, sample_dates, weekday
            )
            if sample is not None:
                samples[sensor_idx] = sample

        return SamplingResult(
            incident_dt=incident_dt,
            samples=samples,
            excluded_dates=excluded_dates,
            errors=errors
        )

    def _get_excluded_dates(self, incident_dt: datetime) -> List[datetime]:
        """获取需要排除的日期列表

        排除事故当天及前后 exclude_days 天的数据

        Args:
            incident_dt: 事故发生时间

        Returns:
            需要排除的日期列表
        """
        excluded = []
        incident_date = incident_dt.date()

        for delta in range(-self.exclude_days, self.exclude_days + 1):
            excluded.append(
                datetime.combine(
                    incident_date + timedelta(days=delta),
                    datetime.min.time()
                )
            )

        return excluded

    def _find_same_weekday_dates(
        self,
        incident_dt: datetime,
        excluded_dates: List[datetime],
        weeks_limit: Optional[int] = None
    ) -> List[datetime]:
        """找出同一年中所有相同星期几的日期

        Args:
            incident_dt: 事故发生时间
            excluded_dates: 需要排除的日期
            weeks_limit: 限制周数

        Returns:
            可用的采样日期列表
        """
        target_weekday = incident_dt.isoweekday()
        excluded_set = {d.date() for d in excluded_dates}

        sample_dates = []

        # 从年初开始遍历
        year = self.data_year
        current_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)

        while current_date <= end_date:
            if (current_date.isoweekday() == target_weekday and
                current_date.date() not in excluded_set):
                sample_dates.append(current_date)
            current_date += timedelta(days=1)

        # 如果有周数限制，只取最近的 N 周
        if weeks_limit is not None and len(sample_dates) > weeks_limit:
            # 找到事故日期在列表中的位置
            incident_date = incident_dt.date()

            # 分成事故前和事故后的日期
            before_dates = [d for d in sample_dates if d.date() < incident_date]
            after_dates = [d for d in sample_dates if d.date() > incident_date]

            # 取事故前后各 weeks_limit/2 周
            half_limit = weeks_limit // 2
            selected_before = before_dates[-half_limit:] if len(before_dates) > half_limit else before_dates
            selected_after = after_dates[:half_limit] if len(after_dates) > half_limit else after_dates

            sample_dates = sorted(selected_before + selected_after)

        return sample_dates

    def _sample_sensor(
        self,
        sensor_idx: int,
        incident_dt: datetime,
        sample_dates: List[datetime],
        weekday: int
    ) -> Optional[HistoricalSample]:
        """对单个传感器进行历史采样

        Args:
            sensor_idx: 传感器索引
            incident_dt: 事故发生时间
            sample_dates: 采样日期列表
            weekday: 星期几

        Returns:
            HistoricalSample 或 None
        """
        # 准备存储采样数据
        occ_samples = []
        spd_samples = []
        vol_samples = []
        valid_dates = []

        # 事故发生的时分
        target_hour = incident_dt.hour
        target_minute = incident_dt.minute

        for sample_date in sample_dates:
            # 构造同一时刻
            sample_dt = sample_date.replace(
                hour=target_hour,
                minute=target_minute
            )

            # 计算时间步
            try:
                center_timestep = datetime_to_timestep(sample_dt, self.data_year)
            except:
                continue

            # 计算提取范围
            start_step = center_timestep - self.time_window
            end_step = center_timestep + self.time_window

            # 边界检查
            if start_step < 0 or end_step >= self.max_timesteps:
                continue

            # 提取数据
            occ_data = self.occupancy[start_step:end_step+1, sensor_idx]
            spd_data = self.speed[start_step:end_step+1, sensor_idx]
            vol_data = self.volume[start_step:end_step+1, sensor_idx]

            # 检查数据有效性（不全是 NaN 或 0）
            if self._is_valid_sample(occ_data, spd_data, vol_data):
                occ_samples.append(occ_data)
                spd_samples.append(spd_data)
                vol_samples.append(vol_data)
                valid_dates.append(sample_dt)

        if len(valid_dates) == 0:
            return None

        return HistoricalSample(
            sensor_idx=sensor_idx,
            weekday=weekday,
            sample_dates=valid_dates,
            occupancy_samples=np.array(occ_samples),
            speed_samples=np.array(spd_samples),
            volume_samples=np.array(vol_samples)
        )

    def _is_valid_sample(
        self,
        occ_data: np.ndarray,
        spd_data: np.ndarray,
        vol_data: np.ndarray
    ) -> bool:
        """检查采样数据是否有效

        有效条件：至少有一种数据不全是 NaN 或 0

        Args:
            occ_data: 占用率数据
            spd_data: 速度数据
            vol_data: 流量数据

        Returns:
            是否有效
        """
        # 检查是否全是 NaN
        if np.all(np.isnan(occ_data)) and np.all(np.isnan(spd_data)) and np.all(np.isnan(vol_data)):
            return False

        # 检查是否全是 0（可能表示传感器故障）
        # 只要有一个数据有非零值就认为有效
        occ_valid = not np.all(occ_data == 0)
        spd_valid = not np.all(spd_data == 0)
        vol_valid = not np.all(vol_data == 0)

        return occ_valid or spd_valid or vol_valid


def sample_historical_data(
    incident_dt: datetime,
    sensor_indices: List[int],
    matrices: TrafficMatrices,
    time_window: int = 12,
    data_year: int = 2023,
    weeks_limit: Optional[int] = None
) -> SamplingResult:
    """便捷函数：采样历史数据

    Args:
        incident_dt: 事故发生时间
        sensor_indices: 传感器索引列表
        matrices: 流量矩阵
        time_window: 时间窗口
        data_year: 数据年份
        weeks_limit: 限制采样周数

    Returns:
        SamplingResult
    """
    sampler = HistoricalSampler(matrices, time_window, data_year)
    return sampler.sample(incident_dt, sensor_indices, weeks_limit)


def compute_percentiles(
    sample: HistoricalSample,
    percentiles: List[int] = [10, 25, 50, 75, 90]
) -> Dict[str, np.ndarray]:
    """计算历史数据的百分位值

    对应 MATLAB: 百分位计算逻辑

    Args:
        sample: 历史采样数据
        percentiles: 要计算的百分位列表

    Returns:
        包含各数据类型百分位值的字典
        格式: {
            'occupancy': {10: [...], 25: [...], ...},
            'speed': {...},
            'volume': {...}
        }
    """
    result = {
        'occupancy': {},
        'speed': {},
        'volume': {}
    }

    if sample.n_samples == 0:
        return result

    # 计算各百分位
    for p in percentiles:
        result['occupancy'][p] = np.nanpercentile(
            sample.occupancy_samples, p, axis=0
        )
        result['speed'][p] = np.nanpercentile(
            sample.speed_samples, p, axis=0
        )
        result['volume'][p] = np.nanpercentile(
            sample.volume_samples, p, axis=0
        )

    return result
