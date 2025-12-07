# -*- coding: utf-8 -*-
"""
百分位分析模块

对历史数据进行百分位分析，识别异常值，
计算事故时刻数据与历史百分位的偏差。

对应 MATLAB 代码：
- data_clean.m: 数据清洗和百分位分析

分析逻辑：
1. 从历史采样数据计算各百分位值
2. 将事故时刻数据与百分位值进行对比
3. 计算偏差和异常程度
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np

try:
    from .historical_sampler import HistoricalSample, compute_percentiles
except ImportError:
    from processors.historical_sampler import HistoricalSample, compute_percentiles


# 默认百分位值列表
DEFAULT_PERCENTILES = [10, 25, 50, 75, 90]

# MATLAB 风格 NaN 评分映射 (避免在循环中重复创建)
# 对应 MATLAB: nan_count=0→100, 1→60, 2→30, 3→0
MATLAB_NAN_SCORE_MAP = {0: 100, 1: 60, 2: 30, 3: 0}


@dataclass
class PercentileResult:
    """单个数据类型的百分位分析结果

    Attributes:
        data_type: 数据类型 (occupancy/speed/volume)
        actual_values: 事故时刻的实际值
        percentile_values: 各百分位值 {percentile: values}
        deviations: 与中位数的偏差
        anomaly_scores: 异常得分 (基于百分位位置)
        position_percentile: MATLAB风格位置百分位 (事故值在历史分布中的位置，0-1)
        median_value: 历史数据中位数
    """
    data_type: str
    actual_values: np.ndarray
    percentile_values: Dict[int, np.ndarray]
    deviations: np.ndarray
    anomaly_scores: np.ndarray
    position_percentile: float = -3.0  # MATLAB默认值：-3表示未找到
    median_value: float = 0.0

    @property
    def median(self) -> np.ndarray:
        """获取中位数 (P50)"""
        return self.percentile_values.get(50, np.array([]))

    @property
    def is_anomalous(self) -> bool:
        """是否存在异常（基于异常得分阈值）"""
        return np.any(np.abs(self.anomaly_scores) > 0.5)

    @property
    def matlab_percentile(self) -> float:
        """获取MATLAB风格的位置百分位值

        返回值含义：
        - 0.0-1.0: 正常百分位值 (如0.85表示高于85%的历史数据)
        - -1: 历史数据全为NaN
        - -2: 中位数无效
        - -3: 未找到匹配位置
        - -4: 对比值是NaN
        """
        return self.position_percentile


@dataclass
class SensorAnalysisResult:
    """单个传感器的分析结果

    Attributes:
        sensor_idx: 传感器索引
        occupancy: 占用率分析结果
        speed: 速度分析结果
        volume: 流量分析结果
        overall_anomaly_score: 综合异常得分
    """
    sensor_idx: int
    occupancy: PercentileResult
    speed: PercentileResult
    volume: PercentileResult
    overall_anomaly_score: float = 0.0


@dataclass
class AnalysisResult:
    """完整的百分位分析结果

    Attributes:
        incident_id: 事故ID
        incident_type: 事故类型 (hazard, fire, etc.)
        sensor_results: 各传感器的分析结果 {sensor_idx: SensorAnalysisResult}
        percentiles_used: 使用的百分位列表
        errors: 错误信息
    """
    incident_id: str
    sensor_results: Dict[int, SensorAnalysisResult]
    incident_type: str = 'other'  # 事故类型
    percentiles_used: List[int] = field(default_factory=lambda: DEFAULT_PERCENTILES.copy())
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """是否分析成功"""
        return len(self.sensor_results) > 0 and len(self.errors) == 0


class PercentileAnalyzer:
    """百分位分析器

    对应 MATLAB: data_clean

    将事故时刻的流量数据与历史数据的百分位进行对比分析。

    Example:
        >>> analyzer = PercentileAnalyzer()
        >>> result = analyzer.analyze(
        ...     incident_id="12345",
        ...     incident_data=incident_table,
        ...     historical_samples=sampling_result.samples
        ... )
    """

    def __init__(
        self,
        percentiles: List[int] = None,
        anomaly_threshold: float = 0.5,
        time_window: int = 12
    ):
        """初始化分析器

        Args:
            percentiles: 要计算的百分位列表
            anomaly_threshold: 异常判定阈值 (0-1)
            time_window: 时间窗口大小（用于确定中间点索引，默认12）
        """
        self.percentiles = percentiles or DEFAULT_PERCENTILES
        self.anomaly_threshold = anomaly_threshold
        self.time_window = time_window

    def analyze(
        self,
        incident_id: str,
        incident_data: pd.DataFrame,
        historical_samples: Dict[int, HistoricalSample]
    ) -> AnalysisResult:
        """执行百分位分析

        对应 MATLAB: data_clean 的主要逻辑

        Args:
            incident_id: 事故ID
            incident_data: 事故时刻数据表 (来自 TrafficExtractor)
            historical_samples: 历史采样数据 (来自 HistoricalSampler)

        Returns:
            AnalysisResult
        """
        errors = []
        sensor_results = {}

        if len(incident_data) == 0:
            errors.append("事故时刻数据为空")
            return AnalysisResult(
                incident_id=incident_id,
                sensor_results={},
                percentiles_used=self.percentiles,
                errors=errors
            )

        # 获取所有传感器索引
        sensor_indices = incident_data['SensorNumber'].unique()

        for sensor_idx in sensor_indices:
            # 获取该传感器的事故时刻数据
            sensor_data = incident_data[
                incident_data['SensorNumber'] == sensor_idx
            ]

            # 获取该传感器的历史采样
            if sensor_idx not in historical_samples:
                continue

            historical_sample = historical_samples[sensor_idx]

            # 分析该传感器
            try:
                result = self._analyze_sensor(
                    sensor_idx, sensor_data, historical_sample
                )
                if result is not None:
                    sensor_results[sensor_idx] = result
            except Exception as e:
                errors.append(f"传感器 {sensor_idx} 分析失败: {str(e)}")

        return AnalysisResult(
            incident_id=incident_id,
            sensor_results=sensor_results,
            percentiles_used=self.percentiles,
            errors=errors
        )

    def _analyze_sensor(
        self,
        sensor_idx: int,
        sensor_data: pd.DataFrame,
        historical_sample: HistoricalSample
    ) -> Optional[SensorAnalysisResult]:
        """分析单个传感器

        Args:
            sensor_idx: 传感器索引
            sensor_data: 该传感器的事故时刻数据
            historical_sample: 该传感器的历史采样

        Returns:
            SensorAnalysisResult 或 None
        """
        # 计算历史数据的百分位
        percentiles_data = compute_percentiles(
            historical_sample, self.percentiles
        )

        # 提取事故时刻的实际值
        occ_actual = self._extract_actual_values(
            sensor_data, 'Occupancy_Rate'
        )
        spd_actual = self._extract_actual_values(
            sensor_data, 'Speed'
        )
        vol_actual = self._extract_actual_values(
            sensor_data, 'Volume'
        )

        # 分析各数据类型 (传入历史采样数据用于MATLAB位置百分位计算)
        # 传递 time_window 以确保使用正确的中间点（Data13）
        occ_result = self._analyze_data_type(
            'occupancy', occ_actual, percentiles_data['occupancy'],
            historical_samples=historical_sample.occupancy_samples,
            time_window=self.time_window
        )
        spd_result = self._analyze_data_type(
            'speed', spd_actual, percentiles_data['speed'],
            historical_samples=historical_sample.speed_samples,
            time_window=self.time_window
        )
        vol_result = self._analyze_data_type(
            'volume', vol_actual, percentiles_data['volume'],
            historical_samples=historical_sample.volume_samples,
            time_window=self.time_window
        )

        # 计算综合异常得分
        overall_score = self._compute_overall_score(
            occ_result, spd_result, vol_result
        )

        return SensorAnalysisResult(
            sensor_idx=sensor_idx,
            occupancy=occ_result,
            speed=spd_result,
            volume=vol_result,
            overall_anomaly_score=overall_score
        )

    def _extract_actual_values(
        self,
        sensor_data: pd.DataFrame,
        data_type: str
    ) -> np.ndarray:
        """从传感器数据中提取实际值

        Args:
            sensor_data: 传感器数据
            data_type: 数据类型 (Occupancy_Rate/Speed/Volume)

        Returns:
            实际值数组
        """
        rows = sensor_data[sensor_data['Data_Type'] == data_type]
        if len(rows) == 0:
            return np.array([])

        # 验证：同一传感器同一数据类型应该只有一行
        # 如果有多行，记录警告并使用第一行（按现有数据格式，这种情况不应发生）
        if len(rows) > 1:
            import warnings
            warnings.warn(
                f"Found {len(rows)} rows for data_type '{data_type}', expected 1. "
                f"Using first row. This may indicate data format issues.",
                UserWarning
            )

        # 提取 Data1, Data2, ... DataN 列
        data_cols = [c for c in rows.columns if c.startswith('Data') and c[4:].isdigit()]
        if len(data_cols) == 0:
            return np.array([])

        data_cols = sorted(data_cols, key=lambda x: int(x[4:]))

        values = rows.iloc[0][data_cols].values.astype(float)
        return values

    def _analyze_data_type(
        self,
        data_type: str,
        actual_values: np.ndarray,
        percentile_values: Dict[int, np.ndarray],
        historical_samples: Optional[np.ndarray] = None,
        time_window: int = 12
    ) -> PercentileResult:
        """分析单个数据类型

        对应 MATLAB data_clean.m:
            contrast_occupancy = temp_table.Data13(position);
            (使用 Data13 即中间点，而非第一个非 NaN 值)

        Args:
            data_type: 数据类型名称
            actual_values: 事故时刻实际值
            percentile_values: 历史百分位值
            historical_samples: 历史采样原始数据 (用于MATLAB位置百分位计算)
            time_window: 时间窗口大小（用于计算中间点索引）

        Returns:
            PercentileResult
        """
        if len(actual_values) == 0 or 50 not in percentile_values:
            return PercentileResult(
                data_type=data_type,
                actual_values=actual_values,
                percentile_values=percentile_values,
                deviations=np.array([]),
                anomaly_scores=np.array([])
            )

        median = percentile_values[50]

        # 确保长度一致
        min_len = min(len(actual_values), len(median))
        actual_values = actual_values[:min_len]
        median = median[:min_len]

        # 计算与中位数的偏差
        # 避免除零
        with np.errstate(divide='ignore', invalid='ignore'):
            deviations = np.where(
                median != 0,
                (actual_values - median) / median,
                0
            )

        # 计算异常得分 (基于百分位位置)
        anomaly_scores = self._compute_anomaly_scores(
            actual_values, percentile_values
        )

        # 计算 MATLAB 风格的位置百分位
        # 对应 MATLAB: contrast_occupancy = temp_table.Data13(position)
        # 使用中间点（Data13 对应索引 12，即 time_window）作为对比值
        position_percentile = -3.0  # 默认：未找到
        median_value = 0.0

        if historical_samples is not None and len(historical_samples) > 0:
            # MATLAB 使用 Data13（中间点）作为对比值，索引 = time_window
            # 当 time_window=12 时，Data13 对应索引 12（0-based）
            center_idx = time_window  # 中间点索引

            # 确保索引在有效范围内
            if center_idx < len(actual_values):
                contrast_value = float(actual_values[center_idx])

                # 收集该时间点的所有历史数据
                # historical_samples 形状: (n_samples, window_size)
                if center_idx < historical_samples.shape[1]:
                    historical_column = historical_samples[:, center_idx]
                    position_percentile, median_value = self._compute_position_percentile(
                        contrast_value, historical_column
                    )

        return PercentileResult(
            data_type=data_type,
            actual_values=actual_values,
            percentile_values=percentile_values,
            deviations=deviations,
            anomaly_scores=anomaly_scores,
            position_percentile=position_percentile,
            median_value=median_value
        )

    def _compute_anomaly_scores(
        self,
        actual_values: np.ndarray,
        percentile_values: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """计算异常得分

        对应 MATLAB: 百分位位置判断逻辑

        得分范围：
        - 0: 正常 (在 P25-P75 范围内)
        - 0.5: 轻度异常 (在 P10-P25 或 P75-P90 范围内)
        - 1.0: 严重异常 (超出 P10 或 P90)

        负值表示低于正常，正值表示高于正常。

        Args:
            actual_values: 实际值
            percentile_values: 百分位值

        Returns:
            异常得分数组
        """
        # 获取各百分位值
        p10 = percentile_values.get(10, None)
        p25 = percentile_values.get(25, None)
        p50 = percentile_values.get(50, None)
        p75 = percentile_values.get(75, None)
        p90 = percentile_values.get(90, None)

        if p10 is None or p25 is None or p75 is None or p90 is None:
            return np.zeros(len(actual_values))

        # 对齐所有数组长度，避免尾部未比较的数据被视为正常
        min_len = min(len(actual_values), len(p10), len(p25), len(p75), len(p90))
        scores = np.zeros(min_len)

        for i in range(min_len):
            val = actual_values[i]

            # 处理 NaN 值：标记为 NaN 而不是默认 0（正常）
            if np.isnan(val):
                scores[i] = np.nan
                continue

            if val < p10[i]:
                # 严重低于正常
                scores[i] = -1.0
            elif val < p25[i]:
                # 轻度低于正常
                scores[i] = -0.5
            elif val <= p75[i]:
                # 正常范围
                scores[i] = 0.0
            elif val <= p90[i]:
                # 轻度高于正常
                scores[i] = 0.5
            else:
                # 严重高于正常
                scores[i] = 1.0

        return scores

    def _compute_position_percentile(
        self,
        contrast_value: float,
        historical_data: np.ndarray
    ) -> Tuple[float, float]:
        """计算MATLAB风格的位置百分位

        对应 MATLAB data_clean.m 的位置百分位计算逻辑：
        找到事故时刻值在排序后历史数据中的位置，返回该位置占总数的比例。

        MATLAB 原始逻辑：
        ```matlab
        if height_occupancy > 1
            for position_data = 1:(height_occupancy - 1)
                if clean_occupancy(position_data) <= contrast_occupancy && ...
                   clean_occupancy(position_data + 1) >= contrast_occupancy
                    account_occupancy = position_data / height_occupancy;
                    break;
                end
            end
        elseif height_occupancy == 1
            account_occupancy = 0.5;  % 只有一个数据点，返回 0.5
        end

        if height_occupancy == 0
            account_occupancy = -1;
            median_occupancy = -2;  % 空历史，中位数设为 -2
        end
        ```

        Args:
            contrast_value: 事故时刻的对比值
            historical_data: 历史采样数据数组

        Returns:
            Tuple[float, float]: (位置百分位, 中位数)
            位置百分位含义：
            - 0.0-1.0: 正常百分位值 (如0.85表示高于85%的历史数据)
            - -1: 历史数据全为NaN
            - -2: 中位数无效（空历史时的中位数值）
            - -3: 未找到匹配位置 (默认值)
            - -4: 对比值是NaN
        """
        # 检查对比值是否为 NaN
        # 对应 MATLAB: if isnan(contrast_occupancy) account_occupancy = -4; end
        if np.isnan(contrast_value):
            return -4.0, 0.0

        # 清洗历史数据：移除 NaN
        clean_data = historical_data[~np.isnan(historical_data)]

        # 检查清洗后是否有数据
        # 对应 MATLAB: if height_occupancy == 0 account=-1; median=-2; end
        if len(clean_data) == 0:
            return -1.0, -2.0  # 修复：空历史中位数返回 -2 而非 0

        # 计算中位数
        median_value = float(np.median(clean_data))

        # 对历史数据排序
        sorted_data = np.sort(clean_data)
        height = len(sorted_data)

        # 如果只有一个数据点
        # 对应 MATLAB: elseif height_occupancy == 1 account_occupancy = 0.5; end
        if height == 1:
            return 0.5, median_value  # 修复：单样本返回 0.5 而非 0/1

        # MATLAB 风格的位置百分位计算
        # 找到 contrast_value 在排序数据中的位置
        account = -3.0  # 默认：未找到

        for position in range(height - 1):
            # MATLAB: if clean(position) <= contrast && clean(position+1) >= contrast
            if sorted_data[position] <= contrast_value <= sorted_data[position + 1]:
                # MATLAB: account = position / height (1-based index)
                account = (position + 1) / height  # Python 0-based 转 1-based
                break

        # 边界情况处理
        if account == -3.0:
            if contrast_value < sorted_data[0]:
                # 小于最小值：0%
                account = 0.0
            elif contrast_value > sorted_data[-1]:
                # 大于最大值：100%
                account = 1.0

        return account, median_value

    def _compute_overall_score(
        self,
        occ_result: PercentileResult,
        spd_result: PercentileResult,
        vol_result: PercentileResult
    ) -> float:
        """计算综合异常得分

        加权平均各数据类型的异常得分

        Args:
            occ_result: 占用率分析结果
            spd_result: 速度分析结果
            vol_result: 流量分析结果

        Returns:
            综合异常得分 (0-1)
        """
        scores = []
        weights = []

        # 占用率权重 0.4
        if len(occ_result.anomaly_scores) > 0:
            scores.append(np.mean(np.abs(occ_result.anomaly_scores)))
            weights.append(0.4)

        # 速度权重 0.4
        if len(spd_result.anomaly_scores) > 0:
            scores.append(np.mean(np.abs(spd_result.anomaly_scores)))
            weights.append(0.4)

        # 流量权重 0.2
        if len(vol_result.anomaly_scores) > 0:
            scores.append(np.mean(np.abs(vol_result.anomaly_scores)))
            weights.append(0.2)

        if len(scores) == 0:
            return 0.0

        # 归一化权重
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        return weighted_sum / total_weight


def analyze_incident(
    incident_id: str,
    incident_data: pd.DataFrame,
    historical_samples: Dict[int, HistoricalSample],
    percentiles: List[int] = None,
    time_window: int = 12
) -> AnalysisResult:
    """便捷函数：分析单个事故

    Args:
        incident_id: 事故ID
        incident_data: 事故时刻数据
        historical_samples: 历史采样数据
        percentiles: 百分位列表
        time_window: 时间窗口大小（用于确定中间点索引，默认12）

    Returns:
        AnalysisResult
    """
    analyzer = PercentileAnalyzer(percentiles=percentiles, time_window=time_window)
    return analyzer.analyze(incident_id, incident_data, historical_samples)


def create_percentile_table(
    analysis_result: AnalysisResult,
    incident_type: str = None
) -> pd.DataFrame:
    """将分析结果转换为表格格式

    创建与 MATLAB 输出兼容的表格

    Args:
        analysis_result: 分析结果
        incident_type: 事故类型 (hazard, fire, etc.)，如果为 None 则从 analysis_result.incident_type 获取

    Returns:
        DataFrame 格式的分析结果
    """
    # 统一归一化 incident_type：优先使用参数，否则回退到 analysis_result.incident_type
    # 处理 None/NaN/空字符串，统一转为小写
    raw_type = incident_type if incident_type is not None else getattr(analysis_result, 'incident_type', None)
    if raw_type is None or (isinstance(raw_type, float) and np.isnan(raw_type)):
        normalized_type = 'other'
    else:
        normalized_type = str(raw_type).strip().lower()
        if normalized_type == '' or normalized_type == 'nan':
            normalized_type = 'other'

    rows = []

    for sensor_idx, sensor_result in analysis_result.sensor_results.items():
        for data_type, result in [
            ('Occupancy_Rate', sensor_result.occupancy),
            ('Speed', sensor_result.speed),
            ('Volume', sensor_result.volume)
        ]:
            if len(result.actual_values) == 0:
                continue

            # 计算 MATLAB 风格的 NaN 评分 (每个数据类型分别计算)
            # 对应 MATLAB: nan_count = sum(isnan(Data1_group)); switch nan_count...
            # Data1_group = temp_table.Data1(scoreIdx:scoreIdx+2) - 取前3个数据点
            actual_values = result.actual_values
            sample_values = actual_values[:min(3, len(actual_values))]
            nan_count = int(np.sum(np.isnan(sample_values)))
            # 如果数据点不足3个，补足NaN计数
            if len(sample_values) < 3:
                nan_count += (3 - len(sample_values))
            # 使用模块级常量 MATLAB_NAN_SCORE_MAP
            matlab_nan_score = MATLAB_NAN_SCORE_MAP.get(min(nan_count, 3), 0)

            row = {
                'IncidentId': analysis_result.incident_id,
                'Type': normalized_type,  # 事故类型（已归一化为小写）
                'SensorNumber': sensor_idx,
                'Data_Type': data_type,
                'Overall_Anomaly': sensor_result.overall_anomaly_score,
                # MATLAB 风格位置百分位
                'MatlabPercentile': result.position_percentile,
                'MedianValue': result.median_value,
                # MATLAB 风格 NaN 计数评分 (每个数据类型分别计算)
                'MatlabScore': matlab_nan_score,
                'NaNCount': nan_count
            }

            # 添加实际值
            for i, val in enumerate(result.actual_values):
                row[f'Actual_{i+1}'] = val

            # 添加中位数
            if 50 in result.percentile_values:
                for i, val in enumerate(result.percentile_values[50]):
                    row[f'P50_{i+1}'] = val

            # 添加偏差
            for i, val in enumerate(result.deviations):
                row[f'Deviation_{i+1}'] = val

            # 添加异常得分
            for i, val in enumerate(result.anomaly_scores):
                row[f'AnomalyScore_{i+1}'] = val

            rows.append(row)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.DataFrame(rows)
