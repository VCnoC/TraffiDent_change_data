# -*- coding: utf-8 -*-
"""
传感器匹配模块

根据事故位置（高速公路编号 + 里程标）匹配最近的交通传感器。

对应 MATLAB 代码：
- Traffic_Function.m

匹配逻辑：
1. 按高速公路编号（Fwy）和方向（Direction）筛选传感器
2. 按绝对里程标（Abs PM）找到最近的传感器
3. 包含里程标相同或相邻的传感器（上下游）
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class MatchResult:
    """传感器匹配结果

    Attributes:
        incident_id: 事故ID
        row_index: 事故在数据中的行索引
        matched_sensors: 匹配到的传感器数据 DataFrame
        incident_info: 事故相关信息字典
        errors: 匹配过程中的错误信息列表
    """
    incident_id: str
    row_index: int
    matched_sensors: pd.DataFrame
    incident_info: Dict[str, Any]
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """是否匹配成功"""
        return len(self.matched_sensors) > 0 and len(self.errors) == 0

    @property
    def sensor_count(self) -> int:
        """匹配到的传感器数量"""
        return len(self.matched_sensors)


class SensorMatcher:
    """传感器匹配器

    对应 MATLAB: Traffic_Function

    Example:
        >>> matcher = SensorMatcher(sensors_df, node_order)
        >>> result = matcher.match(incident_row, row_index=0)
        >>> if result.success:
        ...     print(f"匹配到 {result.sensor_count} 个传感器")
    """

    def __init__(
        self,
        sensors: pd.DataFrame,
        node_order: Optional[np.ndarray] = None,
        max_distance: float = 5.0
    ):
        """初始化传感器匹配器

        Args:
            sensors: 传感器元数据 DataFrame
            node_order: 节点顺序数组（用于获取传感器在矩阵中的索引）
            max_distance: 最大搜索距离（里程标单位）
        """
        self.sensors = sensors.copy()
        self.node_order = node_order
        self.max_distance = max_distance

        # 预处理：添加 SensorNumber（矩阵列索引）
        self._preprocess_sensors()

    def _preprocess_sensors(self) -> None:
        """预处理传感器数据

        对应 MATLAB: sensor_data.SensorNumber = (1:height(sensor_data))';
        """
        # 如果有 node_order，使用它来确定 SensorNumber
        if self.node_order is not None:
            # 创建 station_id 到矩阵索引的映射
            station_to_idx = {
                int(sid): idx
                for idx, sid in enumerate(self.node_order)
            }
            self.sensors['SensorNumber'] = self.sensors['station_id'].apply(
                lambda x: station_to_idx.get(int(x), -1) if pd.notna(x) else -1
            )
        else:
            # 使用行索引作为 SensorNumber（从0开始，Python风格）
            self.sensors['SensorNumber'] = range(len(self.sensors))

        # 确保 Fwy 列是数值类型（处理可能的字符串）
        if 'Fwy' in self.sensors.columns:
            self.sensors['Fwy'] = pd.to_numeric(
                self.sensors['Fwy'], errors='coerce'
            )

    def match(
        self,
        incident: pd.Series,
        row_index: int
    ) -> MatchResult:
        """匹配单个事故到最近的传感器

        对应 MATLAB: Traffic_Function

        Args:
            incident: 事故数据行（Series）
            row_index: 事故行索引

        Returns:
            MatchResult 包含匹配结果和错误信息
        """
        errors = []
        matched_sensors = pd.DataFrame()

        # 提取事故信息
        try:
            incident_id = str(incident.get('incident_id', ''))
            fwy = incident.get('Fwy')
            direction = incident.get('Freeway_direction', '')
            abs_pm = incident.get('Abs PM')
            dt = incident.get('dt')
            description = incident.get('DESCRIPTION', '')

            # 获取事故类型（Type 或 Type_normalized）
            incident_type = incident.get('Type_normalized', incident.get('Type', 'other'))
            if pd.isna(incident_type) or str(incident_type).strip() == '':
                incident_type = 'other'
            else:
                incident_type = str(incident_type).strip().lower()

            incident_info = {
                'incident_id': incident_id,
                'row_index': row_index,
                'Type': incident_type,  # 事故类型
                'Fwy': fwy,
                'Direction': direction,
                'Abs PM': abs_pm,
                'dt': dt,
                'DESCRIPTION': description
            }

        except Exception as e:
            errors.append(f"提取事故信息失败: {str(e)}")
            return MatchResult(
                incident_id='Unknown',
                row_index=row_index,
                matched_sensors=pd.DataFrame(),
                incident_info={},
                errors=errors
            )

        # 验证必要字段
        if pd.isna(fwy):
            errors.append(f"事故 {incident_id}: 缺少高速公路编号 (Fwy)")
            return MatchResult(
                incident_id=incident_id,
                row_index=row_index,
                matched_sensors=pd.DataFrame(),
                incident_info=incident_info,
                errors=errors
            )

        if pd.isna(abs_pm):
            errors.append(f"事故 {incident_id}: 缺少里程标 (Abs PM)")
            return MatchResult(
                incident_id=incident_id,
                row_index=row_index,
                matched_sensors=pd.DataFrame(),
                incident_info=incident_info,
                errors=errors
            )

        # 第一步：按高速公路编号筛选
        # 对应 MATLAB: sensor_fwy_rows = strcmp(sensor_data.Fwy, road_name);
        fwy_mask = self.sensors['Fwy'] == fwy

        # 如果有方向信息，也按方向筛选
        if direction and 'Direction' in self.sensors.columns:
            direction_mask = self.sensors['Direction'] == direction
            combined_mask = fwy_mask & direction_mask
            filtered = self.sensors[combined_mask].copy()

            # 如果方向匹配无结果，退回到只按 Fwy 筛选
            if len(filtered) == 0:
                filtered = self.sensors[fwy_mask].copy()
        else:
            filtered = self.sensors[fwy_mask].copy()

        # 检查是否有匹配
        if len(filtered) == 0:
            errors.append(
                f"事故 {incident_id}: 没有找到道路 Fwy={fwy} 的传感器"
            )
            return MatchResult(
                incident_id=incident_id,
                row_index=row_index,
                matched_sensors=pd.DataFrame(),
                incident_info=incident_info,
                errors=errors
            )

        # 第二步：计算里程标距离，找最近的传感器
        # 对应 MATLAB: [min_diff, idx] = min(abs(sensor_abs_pm_values - input_abs_pm));
        filtered['_distance'] = (filtered['Abs PM'] - abs_pm).abs()

        # 找到最小距离
        min_distance = filtered['_distance'].min()

        # 检查最小距离是否在阈值内
        if min_distance > self.max_distance:
            errors.append(
                f"事故 {incident_id}: 最近传感器距离 {min_distance:.2f} 超过阈值 {self.max_distance}"
            )
            return MatchResult(
                incident_id=incident_id,
                row_index=row_index,
                matched_sensors=pd.DataFrame(),
                incident_info=incident_info,
                errors=errors
            )

        # 第三步：获取最近的传感器及相邻传感器
        # 对应 MATLAB 的上下寻找逻辑
        matched_sensors = self._find_adjacent_sensors(
            filtered, abs_pm, min_distance
        )

        # 清理临时列
        if '_distance' in matched_sensors.columns:
            matched_sensors = matched_sensors.drop(columns=['_distance'])

        # 添加事故信息到匹配结果
        matched_sensors = self._enrich_result(
            matched_sensors, incident_info
        )

        return MatchResult(
            incident_id=incident_id,
            row_index=row_index,
            matched_sensors=matched_sensors,
            incident_info=incident_info,
            errors=errors
        )

    def _find_adjacent_sensors(
        self,
        filtered: pd.DataFrame,
        target_pm: float,
        min_distance: float
    ) -> pd.DataFrame:
        """找到目标里程标处的所有传感器

        对应 MATLAB Traffic_Function.m 第 82-145 行的完整逻辑：
        1. 找到最近的传感器索引 idx
        2. 向前计数，找到所有与 idx 相同 PM 值的传感器
        3. closest_pm = 最近传感器的 PM 值
        4. closest_pm_xx = 下一个不同的 PM 值（如果存在）
        5. 向上搜索：收集所有 PM == closest_pm 的传感器
        6. 向下搜索：收集所有 PM == closest_pm_xx 的传感器（如果存在）
        7. 合并去重

        Args:
            filtered: 已按 Fwy 筛选的传感器数据
            target_pm: 目标里程标
            min_distance: 最小距离

        Returns:
            匹配的传感器 DataFrame
        """
        # 按 Abs PM 排序（对应 MATLAB 中 result = sensor_data(sensor_fwy_rows, :) 后的操作）
        sorted_df = filtered.sort_values('Abs PM').reset_index(drop=True)
        h = len(sorted_df)

        if h == 0:
            return pd.DataFrame()

        # 找到最近点的索引（对应 MATLAB 第 71 行）
        # [min_diff, idx] = min(abs(sensor_abs_pm_values - input_abs_pm));
        closest_idx = sorted_df['_distance'].idxmin()
        sensor_abs_pm_values = sorted_df['Abs PM'].values

        # 对应 MATLAB 第 83-96 行：向前计数找所有相同 PM 的传感器
        # flag = (idx < h); count = idx;
        flag = closest_idx < (h - 1)  # Python 0-based
        count = closest_idx

        if flag:
            # for i = idx:h, if PM[idx] == PM[i], count++, else break
            for i in range(closest_idx, h):
                if sensor_abs_pm_values[closest_idx] == sensor_abs_pm_values[i]:
                    count = i + 1  # count 指向下一个位置
                else:
                    break

        # 对应 MATLAB 第 98-117 行：获取 closest_pm 和 closest_pm_xx
        if flag and count >= 1 and count <= h:
            # closest_pm = sensor_abs_pm_values(count-1)
            # closest_pm_xx = sensor_abs_pm_values(count)
            closest_pm = sensor_abs_pm_values[count - 1]  # count-1 是 0-based
            closest_pm_xx = sensor_abs_pm_values[count] if count < h else None
        else:
            count = closest_idx + 1  # 调整为 1-based 风格用于后续计算
            closest_pm = sensor_abs_pm_values[closest_idx]
            closest_pm_xx = None

        result_indices = []

        # 对应 MATLAB 第 119-125 行：向上寻找与 closest_pm 相等的行
        # upper_row = count-1; while upper_row >= 1 && PM[upper_row] == closest_pm
        upper_row = count - 2  # Python 0-based, 从 count-1 (MATLAB) 转换
        while upper_row >= 0 and sensor_abs_pm_values[upper_row] == closest_pm:
            result_indices.append(upper_row)
            upper_row -= 1

        # 添加 count-1 位置（这是中心点）
        if count - 1 >= 0 and count - 1 < h:
            result_indices.append(count - 1)

        # 对应 MATLAB 第 127-135 行：向下寻找与 closest_pm_xx 相等的行（如果 flag）
        # lower_row = count; while lower_row <= h && PM[lower_row] == closest_pm_xx
        if flag and closest_pm_xx is not None:
            lower_row = count  # Python 0-based, count (MATLAB 1-based) = count (Python 0-based)
            while lower_row < h and sensor_abs_pm_values[lower_row] == closest_pm_xx:
                result_indices.append(lower_row)
                lower_row += 1

        # 去重并排序
        result_indices = sorted(set(result_indices))

        # 提取结果
        result = sorted_df.loc[result_indices].copy()

        # 按 SensorNumber 排序
        # 对应 MATLAB 第 152 行: closest_row_data = sortrows(closest_row_data, 'SensorNumber');
        result = result.sort_values('SensorNumber')

        return result

    def _enrich_result(
        self,
        sensors: pd.DataFrame,
        incident_info: Dict[str, Any]
    ) -> pd.DataFrame:
        """为匹配结果添加事故信息

        对应 MATLAB 第 158-191 行

        Args:
            sensors: 匹配的传感器数据
            incident_info: 事故信息字典

        Returns:
            增强的传感器数据
        """
        result = sensors.copy()

        # 添加事故信息列
        result['row_index'] = incident_info.get('row_index', -1)
        result['IncidentId'] = incident_info.get('incident_id', '')
        result['dt'] = incident_info.get('dt')
        result['DESCRIPTION'] = incident_info.get('DESCRIPTION', '')

        # 选择并重排列
        columns = [
            'row_index', 'IncidentId', 'SensorNumber', 'station_id',
            'Fwy', 'Direction', 'Abs PM', 'dt', 'DESCRIPTION'
        ]

        # 只保留存在的列
        existing_columns = [c for c in columns if c in result.columns]
        result = result[existing_columns]

        return result

    def match_batch(
        self,
        incidents: pd.DataFrame,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        progress_callback=None
    ) -> List[MatchResult]:
        """批量匹配事故到传感器

        Args:
            incidents: 事故数据 DataFrame
            start_idx: 起始索引
            end_idx: 结束索引（不包含）
            progress_callback: 进度回调函数

        Returns:
            MatchResult 列表
        """
        if end_idx is None:
            end_idx = len(incidents)

        results = []

        for idx in range(start_idx, end_idx):
            incident = incidents.iloc[idx]
            result = self.match(incident, row_index=idx)
            results.append(result)

            if progress_callback:
                progress_callback(idx, end_idx)

        return results


def create_extracted_data(match_result: MatchResult) -> pd.DataFrame:
    """从匹配结果创建提取数据表

    对应 MATLAB Traffic_Function 的 extracted_data 输出

    Args:
        match_result: 匹配结果

    Returns:
        提取的数据 DataFrame
    """
    if not match_result.success:
        return pd.DataFrame()

    return match_result.matched_sensors.copy()
