# -*- coding: utf-8 -*-
"""
流量数据提取模块

从流量矩阵中提取事故时刻前后的流量数据。

对应 MATLAB 代码：
- p_trafficdata_function.m: 提取事故时刻数据
- p_trafficdata_two_function.m: 提取前后天数据

数据结构：
- 流量矩阵形状: (105120, 16972) = (时间步, 传感器数)
- 每个时间步 = 5 分钟
- 每天 = 288 个时间步
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

try:
    from ..utils.time_utils import (
        datetime_to_timestep,
        datetime_to_numeric,
        get_iso_weekday,
        TIMESTEPS_PER_DAY,
    )
    from ..data.loader import TrafficMatrices
    from .sensor_matcher import MatchResult
except ImportError:
    # 支持直接运行模块
    from utils.time_utils import (
        datetime_to_timestep,
        datetime_to_numeric,
        get_iso_weekday,
        TIMESTEPS_PER_DAY,
    )
    from data.loader import TrafficMatrices
    from processors.sensor_matcher import MatchResult


@dataclass
class ExtractionResult:
    """数据提取结果

    Attributes:
        incident_table: 事故时刻数据表（每个传感器3行：occupancy, speed, volume）
        common_table: 前后天数据表（每个传感器6行：3个before + 3个after）
        errors: 提取过程中的错误信息
    """
    incident_table: pd.DataFrame
    common_table: pd.DataFrame
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """是否提取成功"""
        return len(self.incident_table) > 0 and len(self.errors) == 0


class TrafficExtractor:
    """流量数据提取器

    对应 MATLAB: p_trafficdata_function + p_trafficdata_two_function

    Example:
        >>> extractor = TrafficExtractor(matrices, time_window=12)
        >>> result = extractor.extract(match_result)
    """

    def __init__(
        self,
        matrices: TrafficMatrices,
        time_window: int = 12,
        data_year: int = 2023
    ):
        """初始化提取器

        Args:
            matrices: 流量矩阵数据
            time_window: 事故时刻前后的时间步数（默认12，即前后各1小时）
            data_year: 数据年份
        """
        self.occupancy = matrices.occupancy
        self.speed = matrices.speed
        self.volume = matrices.volume
        self.time_window = time_window
        self.data_year = data_year

        # 矩阵尺寸
        self.max_timesteps = self.occupancy.shape[0]
        self.num_sensors = self.occupancy.shape[1]

        # 总时间步数（2 * window + 1）
        self.total_steps = 2 * time_window + 1

    def extract(self, match_result: MatchResult) -> ExtractionResult:
        """提取单个事故的流量数据

        对应 MATLAB: p_trafficdata_function + p_trafficdata_two_function

        Args:
            match_result: 传感器匹配结果

        Returns:
            ExtractionResult 包含事故时刻和前后天数据
        """
        errors = []

        if not match_result.success:
            return ExtractionResult(
                incident_table=pd.DataFrame(),
                common_table=pd.DataFrame(),
                errors=['传感器匹配失败，无法提取数据']
            )

        # 获取事故时间
        dt = match_result.incident_info.get('dt')
        if pd.isna(dt):
            errors.append('事故时间为空')
            return ExtractionResult(
                incident_table=pd.DataFrame(),
                common_table=pd.DataFrame(),
                errors=errors
            )

        try:
            # 计算事故时刻的时间步
            center_timestep = datetime_to_timestep(dt, self.data_year)
            numeric_time = datetime_to_numeric(dt)
            weekday = get_iso_weekday(dt)

            # 计算提取范围（不 clamp，让 _extract_sensor_data 处理边界）
            # 对应 MATLAB: y_down = y_shang - set_time; y_up = y_shang + set_time
            start_step = center_timestep - self.time_window
            end_step = center_timestep + self.time_window

        except Exception as e:
            errors.append(f'时间计算失败: {str(e)}')
            return ExtractionResult(
                incident_table=pd.DataFrame(),
                common_table=pd.DataFrame(),
                errors=errors
            )

        # 提取事故时刻数据
        incident_table = self._extract_incident_data(
            match_result, start_step, end_step, numeric_time, weekday
        )

        # 提取前后天数据
        # 注意：传入 dt 用于计算前后天的正确 Time 和 Week
        common_table = self._extract_common_data(
            match_result, center_timestep, dt
        )

        return ExtractionResult(
            incident_table=incident_table,
            common_table=common_table,
            errors=errors
        )

    def _extract_incident_data(
        self,
        match_result: MatchResult,
        start_step: int,
        end_step: int,
        numeric_time: int,
        weekday: int
    ) -> pd.DataFrame:
        """提取事故时刻数据

        对应 MATLAB: p_trafficdata_function

        每个传感器提取3行数据：
        - Occupancy_Rate
        - Speed
        - Volume

        Args:
            match_result: 匹配结果
            start_step: 起始时间步
            end_step: 结束时间步
            numeric_time: 数值格式时间
            weekday: 星期几

        Returns:
            事故时刻数据 DataFrame
        """
        rows = []
        sensors = match_result.matched_sensors
        incident_info = match_result.incident_info

        for _, sensor in sensors.iterrows():
            sensor_idx = int(sensor['SensorNumber'])

            # 验证传感器索引
            if sensor_idx < 0 or sensor_idx >= self.num_sensors:
                continue

            # 提取数据
            # 对应 MATLAB: data_rate.(field_rate{1})(j, x)
            occ_data = self._extract_sensor_data(
                self.occupancy, sensor_idx, start_step, end_step
            )
            spd_data = self._extract_sensor_data(
                self.speed, sensor_idx, start_step, end_step
            )
            vol_data = self._extract_sensor_data(
                self.volume, sensor_idx, start_step, end_step
            )

            # 创建基础行信息
            base_row = {
                'row_index': incident_info.get('row_index', -1),
                'IncidentId': incident_info.get('incident_id', ''),
                'Type': incident_info.get('Type', 'other'),  # 事故类型
                'SensorNumber': sensor_idx,
                'station_id': str(sensor.get('station_id', '')),
                'Time': numeric_time,
                'Week': weekday,
                'DESCRIPTION': incident_info.get('DESCRIPTION', '')
            }

            # 添加数据列（Data1, Data2, ... DataN）
            for data_type, data_values in [
                ('Occupancy_Rate', occ_data),
                ('Speed', spd_data),
                ('Volume', vol_data)
            ]:
                row = base_row.copy()
                row['Data_Type'] = data_type

                # 添加时间步数据
                for i, val in enumerate(data_values):
                    row[f'Data{i+1}'] = val

                rows.append(row)

        # 创建 DataFrame
        if rows:
            df = pd.DataFrame(rows)

            # 调整列顺序
            # 对应 MATLAB 的 movevars
            base_cols = ['row_index', 'IncidentId', 'Type', 'SensorNumber', 'station_id',
                         'Data_Type', 'Time', 'Week']
            data_cols = [f'Data{i+1}' for i in range(len(occ_data))]
            end_cols = ['DESCRIPTION']

            ordered_cols = base_cols + data_cols + end_cols
            existing_cols = [c for c in ordered_cols if c in df.columns]
            df = df[existing_cols]

            return df

        return pd.DataFrame()

    def _extract_common_data(
        self,
        match_result: MatchResult,
        center_timestep: int,
        dt
    ) -> pd.DataFrame:
        """提取前后天数据

        对应 MATLAB: p_trafficdata_two_function

        提取事故前一天和后一天同一时段的数据。
        每个传感器6行：3个before（occ/spd/vol）+ 3个after（occ/spd/vol）

        Args:
            match_result: 匹配结果
            center_timestep: 事故时刻的时间步
            dt: 事故发生的日期时间（用于计算前后天的 Time 和 Week）

        Returns:
            前后天数据 DataFrame
        """
        from datetime import timedelta

        rows = []
        sensors = match_result.matched_sensors
        incident_info = match_result.incident_info

        # 计算前一天和后一天的时间步
        before_center = center_timestep - TIMESTEPS_PER_DAY
        after_center = center_timestep + TIMESTEPS_PER_DAY

        # 计算前后天的实际日期时间、Time 和 Week
        # 对应 MATLAB: 前一天 Time/Week 使用 dt-1 天，后一天使用 dt+1 天
        before_dt = dt - timedelta(days=1)
        after_dt = dt + timedelta(days=1)

        before_numeric_time = datetime_to_numeric(before_dt)
        before_weekday = get_iso_weekday(before_dt)

        after_numeric_time = datetime_to_numeric(after_dt)
        after_weekday = get_iso_weekday(after_dt)

        for _, sensor in sensors.iterrows():
            sensor_idx = int(sensor['SensorNumber'])

            if sensor_idx < 0 or sensor_idx >= self.num_sensors:
                continue

            # 前一天数据
            if before_center >= 0:
                before_start = max(0, before_center - self.time_window)
                before_end = min(self.max_timesteps - 1, before_center + self.time_window)

                before_occ = self._extract_sensor_data(
                    self.occupancy, sensor_idx, before_start, before_end
                )
                before_spd = self._extract_sensor_data(
                    self.speed, sensor_idx, before_start, before_end
                )
                before_vol = self._extract_sensor_data(
                    self.volume, sensor_idx, before_start, before_end
                )
            else:
                # 边界情况：填充 NaN
                before_occ = [np.nan] * self.total_steps
                before_spd = [np.nan] * self.total_steps
                before_vol = [np.nan] * self.total_steps

            # 后一天数据
            if after_center < self.max_timesteps:
                after_start = max(0, after_center - self.time_window)
                after_end = min(self.max_timesteps - 1, after_center + self.time_window)

                after_occ = self._extract_sensor_data(
                    self.occupancy, sensor_idx, after_start, after_end
                )
                after_spd = self._extract_sensor_data(
                    self.speed, sensor_idx, after_start, after_end
                )
                after_vol = self._extract_sensor_data(
                    self.volume, sensor_idx, after_start, after_end
                )
            else:
                after_occ = [np.nan] * self.total_steps
                after_spd = [np.nan] * self.total_steps
                after_vol = [np.nan] * self.total_steps

            # 创建 before 数据的基础行信息
            # 对应 MATLAB: before 使用 dt-1 天的 Time 和 Week
            before_base_row = {
                'row_index': incident_info.get('row_index', -1),
                'IncidentId': incident_info.get('incident_id', ''),
                'Type': incident_info.get('Type', 'other'),  # 事故类型
                'SensorNumber': sensor_idx,
                'station_id': str(sensor.get('station_id', '')),
                'Time': before_numeric_time,
                'Week': before_weekday,
                'DESCRIPTION': incident_info.get('DESCRIPTION', '')
            }

            # 创建 after 数据的基础行信息
            # 对应 MATLAB: after 使用 dt+1 天的 Time 和 Week
            after_base_row = {
                'row_index': incident_info.get('row_index', -1),
                'IncidentId': incident_info.get('incident_id', ''),
                'Type': incident_info.get('Type', 'other'),  # 事故类型
                'SensorNumber': sensor_idx,
                'station_id': str(sensor.get('station_id', '')),
                'Time': after_numeric_time,
                'Week': after_weekday,
                'DESCRIPTION': incident_info.get('DESCRIPTION', '')
            }

            # 添加 before 数据（3行）
            # 对应 MATLAB: p_trafficdata_two_function.m 第105行
            # result_matrix(row_counter, 1) = {'before'};  使用小写
            for data_type, data_values in [
                ('Occupancy_Rate', before_occ),
                ('Speed', before_spd),
                ('Volume', before_vol)
            ]:
                row = before_base_row.copy()
                row['Data_Type'] = data_type
                row['Kind'] = 'before'  # MATLAB 使用小写 'before'
                row['Period'] = 'Before'

                for i, val in enumerate(data_values):
                    row[f'Data{i+1}'] = val

                rows.append(row)

            # 添加 after 数据（3行）
            # 对应 MATLAB: p_trafficdata_two_function.m 第123行
            # result_matrix(row_counter, 1) = {'after'};  使用小写
            for data_type, data_values in [
                ('Occupancy_Rate', after_occ),
                ('Speed', after_spd),
                ('Volume', after_vol)
            ]:
                row = after_base_row.copy()
                row['Data_Type'] = data_type
                row['Kind'] = 'after'  # MATLAB 使用小写 'after'
                row['Period'] = 'After'

                for i, val in enumerate(data_values):
                    row[f'Data{i+1}'] = val

                rows.append(row)

        # 创建 DataFrame
        if rows:
            df = pd.DataFrame(rows)
            return df

        return pd.DataFrame()

    def _extract_sensor_data(
        self,
        matrix: np.ndarray,
        sensor_idx: int,
        start_step: int,
        end_step: int,
        center_step: int = None
    ) -> List[float]:
        """从矩阵中提取单个传感器的数据（固定长度，边界 NaN 填充）

        对应 MATLAB p_trafficdata_function.m:
            num_time_steps = 2 * set_time + 1;
            occupancy_data = NaN(1, num_time_steps);  % 预先用 NaN 初始化
            for j = y_down:y_up
                if j > 0 && j <= max_columns
                    idx_time = j - y_down + 1;
                    occupancy_data(idx_time) = data_rate.(field_rate{1})(j, x);
                end
            end

        Args:
            matrix: 流量矩阵
            sensor_idx: 传感器索引（列索引）
            start_step: 起始时间步（可能为负或超出范围）
            end_step: 结束时间步（可能超出范围）
            center_step: 中心时间步（用于计算边界偏移）

        Returns:
            固定长度的数据值列表（边界外填充 NaN）
        """
        # 初始化固定长度的 NaN 数组 (对应 MATLAB: NaN(1, num_time_steps))
        result = [np.nan] * self.total_steps

        # 计算实际有效范围
        actual_start = max(0, start_step)
        actual_end = min(self.max_timesteps - 1, end_step)

        # 如果完全超出范围，返回全 NaN
        if actual_start > actual_end:
            return result

        # 提取有效范围内的数据
        data = matrix[actual_start:actual_end + 1, sensor_idx]

        # 计算填充的起始位置（对应 MATLAB: idx_time = j - y_down + 1）
        # 如果 start_step < 0，则从 result[|start_step|] 开始填充
        fill_start_idx = actual_start - start_step

        # 填充数据到正确位置
        for i, val in enumerate(data):
            result_idx = fill_start_idx + i
            if 0 <= result_idx < self.total_steps:
                result[result_idx] = float(val)

        return result


def extract_traffic_data(
    match_result: MatchResult,
    matrices: TrafficMatrices,
    time_window: int = 12,
    data_year: int = 2023
) -> ExtractionResult:
    """便捷函数：提取单个事故的流量数据

    Args:
        match_result: 传感器匹配结果
        matrices: 流量矩阵
        time_window: 时间窗口
        data_year: 数据年份

    Returns:
        ExtractionResult
    """
    extractor = TrafficExtractor(matrices, time_window, data_year)
    return extractor.extract(match_result)
