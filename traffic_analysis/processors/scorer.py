# -*- coding: utf-8 -*-
"""
评分计算模块

基于百分位分析结果计算事故影响评分。
综合多个传感器的评分，生成最终的评分结果。

对应 MATLAB 代码：
- processGroup.m: 分组处理和评分计算

评分逻辑：
1. 计算各传感器的异常程度
2. 加权平均得到事故影响分数
3. 判定事故严重程度等级
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

try:
    from .percentile_analyzer import (
        AnalysisResult,
        SensorAnalysisResult,
        PercentileResult,
        MATLAB_NAN_SCORE_MAP,
    )
    from .traffic_extractor import ExtractionResult
except ImportError:
    from processors.percentile_analyzer import (
        AnalysisResult,
        SensorAnalysisResult,
        PercentileResult,
        MATLAB_NAN_SCORE_MAP,
    )
    from processors.traffic_extractor import ExtractionResult


# 评分等级定义
SEVERITY_LEVELS = {
    'NONE': (0.0, 0.1, '无影响'),
    'LOW': (0.1, 0.3, '轻微影响'),
    'MEDIUM': (0.3, 0.5, '中等影响'),
    'HIGH': (0.5, 0.7, '较大影响'),
    'SEVERE': (0.7, 1.0, '严重影响'),
}

# 时间标签定义（黄金时段分类）
# 对应 MATLAB processGroup.m 的 Time_tag 逻辑
TIME_TAG_MORNING_RUSH = 1    # 早高峰 7:00-9:00
TIME_TAG_NOON_RUSH = 2       # 午高峰 11:00-13:00
TIME_TAG_EVENING_RUSH = 3    # 晚高峰 17:00-20:00
TIME_TAG_NON_RUSH = 0        # 非高峰时段
TIME_TAG_ERROR = 999         # 解析错误


def compute_time_tag(hour: int) -> int:
    """计算时间标签（黄金时段分类）

    对应 MATLAB processGroup.m:
    ```matlab
    if (hours >= 7) && (hours < 9)
        tag = 1;
    elseif (hours >= 11) && (hours < 13)
        tag = 2;
    elseif (hours >= 17) && (hours < 20)
        tag = 3;
    else
        tag = 0;
    end
    ```

    Args:
        hour: 小时数 (0-23)

    Returns:
        int: 时间标签
            - 1: 早高峰 (7:00-9:00)
            - 2: 午高峰 (11:00-13:00)
            - 3: 晚高峰 (17:00-20:00)
            - 0: 非高峰时段
            - 999: 解析错误
    """
    if hour is None or not isinstance(hour, (int, float)):
        return TIME_TAG_ERROR

    hour = int(hour)

    if hour < 0 or hour > 23:
        return TIME_TAG_ERROR

    if 7 <= hour < 9:
        return TIME_TAG_MORNING_RUSH
    elif 11 <= hour < 13:
        return TIME_TAG_NOON_RUSH
    elif 17 <= hour < 20:
        return TIME_TAG_EVENING_RUSH
    else:
        return TIME_TAG_NON_RUSH


@dataclass
class SensorScore:
    """单个传感器的评分

    Attributes:
        sensor_idx: 传感器索引
        occupancy_score: 占用率评分 (0-1)
        speed_score: 速度评分 (0-1)
        volume_score: 流量评分 (0-1)
        combined_score: 综合评分 (0-1) - 即"事故影响分"，加权异常得分
        matlab_nan_score: MATLAB风格NaN评分 (100/60/30/0)
        nan_count: NaN数量 (0-3)
        time_series_pattern: 时间序列模式特征
    """
    sensor_idx: int
    occupancy_score: float
    speed_score: float
    volume_score: float
    combined_score: float
    matlab_nan_score: int = 100  # MATLAB: fs = 100/60/30/0
    nan_count: int = 0
    time_series_pattern: Dict[str, float] = field(default_factory=dict)

    @property
    def impact_score(self) -> float:
        """事故影响分（即 combined_score）

        基于加权异常得分：occ*0.4 + spd*0.4 + vol*0.2
        """
        return self.combined_score


@dataclass
class IncidentScore:
    """事故评分结果

    Attributes:
        incident_id: 事故ID
        row_index: 事故行索引
        incident_type: 事故类型（如 hazard, fire, carfire 等）
        sensor_scores: 各传感器评分 {sensor_idx: SensorScore}
        overall_score: 事故综合评分 (0-1) - 即"事故影响分"
        matlab_score: MATLAB风格平均NaN评分 (0-100)
        severity_level: 严重程度等级
        severity_label: 严重程度标签
        peak_time_offset: 峰值时间偏移（相对于事故时刻）
        duration_estimate: 影响持续时间估计（时间步数）
        time_tag: 时间标签（黄金时段分类）
            - 1: 早高峰 (7:00-9:00)
            - 2: 午高峰 (11:00-13:00)
            - 3: 晚高峰 (17:00-20:00)
            - 0: 非高峰时段
            - 999: 解析错误
    """
    incident_id: str
    row_index: int
    sensor_scores: Dict[int, SensorScore]
    overall_score: float
    incident_type: str = 'other'  # 事故类型
    matlab_score: float = 100.0  # MATLAB风格平均评分
    severity_level: str = 'NONE'
    severity_label: str = '无影响'
    peak_time_offset: int = 0
    duration_estimate: int = 0
    time_tag: int = TIME_TAG_ERROR  # 黄金时段分类

    @property
    def impact_score(self) -> float:
        """事故影响分（即 overall_score）

        基于所有传感器加权异常得分的平均值
        """
        return self.overall_score


class IncidentScorer:
    """事故评分器

    对应 MATLAB: processGroup

    基于百分位分析结果计算事故影响评分。

    Example:
        >>> scorer = IncidentScorer()
        >>> score = scorer.score(
        ...     incident_id="12345",
        ...     row_index=0,
        ...     analysis_result=analysis,
        ...     extraction_result=extraction
        ... )
    """

    def __init__(
        self,
        weights: Dict[str, float] = None,
        time_window: int = 12
    ):
        """初始化评分器

        Args:
            weights: 各数据类型的权重 {occupancy, speed, volume}
            time_window: 时间窗口大小
        """
        self.weights = weights or {
            'occupancy': 0.4,
            'speed': 0.4,
            'volume': 0.2
        }
        self.time_window = time_window

    def score(
        self,
        incident_id: str,
        row_index: int,
        analysis_result: AnalysisResult,
        extraction_result: Optional[ExtractionResult] = None,
        incident_hour: Optional[int] = None,
        incident_type: str = None
    ) -> IncidentScore:
        """计算事故评分

        对应 MATLAB: processGroup 的主要逻辑

        Args:
            incident_id: 事故ID
            row_index: 事故行索引
            analysis_result: 百分位分析结果
            extraction_result: 流量提取结果（用于时间序列分析）
            incident_hour: 事故发生时的小时数 (0-23)，用于计算时间标签
            incident_type: 事故类型（如 hazard, fire, carfire 等），
                          若为 None 则回退到 analysis_result.incident_type

        Returns:
            IncidentScore
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

        sensor_scores = {}
        all_impact_scores = []  # 事故影响分
        all_matlab_scores = []  # MATLAB NaN 评分

        for sensor_idx, sensor_result in analysis_result.sensor_results.items():
            sensor_score = self._score_sensor(sensor_result, extraction_result)
            sensor_scores[sensor_idx] = sensor_score
            all_impact_scores.append(sensor_score.combined_score)
            all_matlab_scores.append(sensor_score.matlab_nan_score)

        # 计算综合评分（所有传感器的平均）
        # 1. 事故影响分：加权异常得分的平均
        if len(all_impact_scores) > 0:
            overall_score = np.mean(all_impact_scores)
        else:
            overall_score = 0.0

        # 2. MATLAB 评分：NaN 评分的平均
        if len(all_matlab_scores) > 0:
            matlab_score = float(np.mean(all_matlab_scores))
        else:
            matlab_score = 0.0

        # 确定严重程度等级
        severity_level, severity_label = self._determine_severity(overall_score)

        # 估计峰值时间和持续时间
        peak_offset, duration = self._estimate_timing(sensor_scores)

        # 计算时间标签（黄金时段分类）
        time_tag = compute_time_tag(incident_hour)

        return IncidentScore(
            incident_id=incident_id,
            row_index=row_index,
            sensor_scores=sensor_scores,
            overall_score=overall_score,
            incident_type=normalized_type,  # 事故类型（已归一化为小写）
            matlab_score=matlab_score,
            severity_level=severity_level,
            severity_label=severity_label,
            peak_time_offset=peak_offset,
            duration_estimate=duration,
            time_tag=time_tag
        )

    def _score_sensor(
        self,
        sensor_result: SensorAnalysisResult,
        extraction_result: Optional[ExtractionResult] = None
    ) -> SensorScore:
        """计算单个传感器的评分

        Args:
            sensor_result: 传感器分析结果
            extraction_result: 流量提取结果

        Returns:
            SensorScore
        """
        # 计算各数据类型的评分（事故影响分的组成部分）
        occ_score = self._compute_type_score(sensor_result.occupancy)
        spd_score = self._compute_type_score(sensor_result.speed)
        vol_score = self._compute_type_score(sensor_result.volume)

        # 加权综合 → 事故影响分
        combined = (
            occ_score * self.weights['occupancy'] +
            spd_score * self.weights['speed'] +
            vol_score * self.weights['volume']
        )

        # 计算 MATLAB 风格 NaN 评分
        matlab_nan_score, nan_count = self._compute_matlab_nan_score(sensor_result)

        # 分析时间序列模式
        pattern = self._analyze_pattern(sensor_result)

        return SensorScore(
            sensor_idx=sensor_result.sensor_idx,
            occupancy_score=occ_score,
            speed_score=spd_score,
            volume_score=vol_score,
            combined_score=combined,
            matlab_nan_score=matlab_nan_score,
            nan_count=nan_count,
            time_series_pattern=pattern
        )

    def _compute_matlab_nan_score(
        self,
        sensor_result: SensorAnalysisResult
    ) -> Tuple[int, int]:
        """计算 MATLAB 风格的 NaN 评分

        对应 MATLAB processGroup.m 的评分逻辑：
        ```matlab
        Data1_group = temp_table.Data1(scoreIdx:scoreIdx+2);
        nan_count = sum(isnan(Data1_group));
        switch nan_count
            case 0: fs = 100;
            case 1: fs = 60;
            case 2: fs = 30;
            case 3: fs = 0;
        end
        ```

        **重要**：MATLAB 取的是同一时间点 (Data1) 跨 3 种数据类型 (occ/spd/vol) 的值，
        不是同一数据类型跨 3 个时间点！

        - scoreIdx 指向 Occupancy_Rate 行
        - scoreIdx+1 指向 Speed 行
        - scoreIdx+2 指向 Volume 行
        - Data1 是第一个时间步的数据列

        Args:
            sensor_result: 传感器分析结果

        Returns:
            Tuple[int, int]: (MATLAB评分, NaN数量)
        """
        # MATLAB: Data1_group = temp_table.Data1(scoreIdx:scoreIdx+2)
        # 取同一时间点 (Data1, 即 index 0) 跨 3 种数据类型的值
        data1_values = []

        # 从 occupancy 取 Data1 (index 0)
        if len(sensor_result.occupancy.actual_values) > 0:
            data1_values.append(sensor_result.occupancy.actual_values[0])
        else:
            data1_values.append(np.nan)

        # 从 speed 取 Data1 (index 0)
        if len(sensor_result.speed.actual_values) > 0:
            data1_values.append(sensor_result.speed.actual_values[0])
        else:
            data1_values.append(np.nan)

        # 从 volume 取 Data1 (index 0)
        if len(sensor_result.volume.actual_values) > 0:
            data1_values.append(sensor_result.volume.actual_values[0])
        else:
            data1_values.append(np.nan)

        # 计算 NaN 数量
        nan_count = int(np.sum(np.isnan(data1_values)))

        # 使用共享常量 MATLAB_NAN_SCORE_MAP (DRY 原则)
        matlab_score = MATLAB_NAN_SCORE_MAP.get(min(nan_count, 3), 0)

        return matlab_score, nan_count

    def _compute_type_score(self, result: PercentileResult) -> float:
        """计算单个数据类型的评分

        基于异常得分计算：
        - 绝对值越大表示越异常
        - 归一化到 0-1 范围

        Args:
            result: 百分位分析结果

        Returns:
            评分 (0-1)
        """
        if len(result.anomaly_scores) == 0:
            return 0.0

        # 取异常得分的绝对值平均
        mean_abs = np.mean(np.abs(result.anomaly_scores))

        # 归一化到 0-1
        # 异常得分范围是 [-1, 1]，绝对值是 [0, 1]
        return float(mean_abs)

    def _analyze_pattern(
        self,
        sensor_result: SensorAnalysisResult
    ) -> Dict[str, float]:
        """分析时间序列模式

        识别异常模式：
        - peak_position: 峰值位置（归一化到 0-1）
        - trend: 趋势方向 (-1 下降, 0 平稳, 1 上升)
        - volatility: 波动性

        Args:
            sensor_result: 传感器分析结果

        Returns:
            模式特征字典
        """
        pattern = {
            'peak_position': 0.5,
            'trend': 0.0,
            'volatility': 0.0
        }

        # 使用速度的异常得分分析（速度下降通常表示拥堵）
        scores = sensor_result.speed.anomaly_scores
        if len(scores) == 0:
            return pattern

        # 峰值位置（此处 len(scores) > 0 已由上方检查保证）
        peak_idx = np.argmax(np.abs(scores))
        pattern['peak_position'] = peak_idx / len(scores)

        # 趋势
        if len(scores) >= 3:
            first_half = np.mean(scores[:len(scores)//2])
            second_half = np.mean(scores[len(scores)//2:])
            pattern['trend'] = float(np.sign(second_half - first_half))

        # 波动性
        pattern['volatility'] = float(np.std(scores))

        return pattern

    def _determine_severity(self, score: float) -> Tuple[str, str]:
        """确定严重程度等级

        Args:
            score: 综合评分 (0-1)

        Returns:
            (等级代码, 等级标签)
        """
        for level, (low, high, label) in SEVERITY_LEVELS.items():
            if low <= score < high:
                return level, label

        # 默认返回最高级别
        return 'SEVERE', '严重影响'

    def _estimate_timing(
        self,
        sensor_scores: Dict[int, SensorScore]
    ) -> Tuple[int, int]:
        """估计影响时间特征

        Args:
            sensor_scores: 传感器评分

        Returns:
            (峰值时间偏移, 持续时间估计)
        """
        if len(sensor_scores) == 0:
            return 0, 0

        # 收集所有传感器的峰值位置
        peak_positions = [
            s.time_series_pattern.get('peak_position', 0.5)
            for s in sensor_scores.values()
        ]

        # 平均峰值位置转换为时间偏移
        avg_peak = np.mean(peak_positions)
        window = self.time_window * 2 + 1
        peak_offset = int((avg_peak - 0.5) * window)

        # 基于评分估计持续时间
        avg_score = np.mean([s.combined_score for s in sensor_scores.values()])

        # 简单启发式：评分越高，持续时间越长
        if avg_score < 0.2:
            duration = 2
        elif avg_score < 0.4:
            duration = 4
        elif avg_score < 0.6:
            duration = 8
        else:
            duration = 12

        return peak_offset, duration


def score_incident(
    incident_id: str,
    row_index: int,
    analysis_result: AnalysisResult,
    extraction_result: Optional[ExtractionResult] = None,
    weights: Dict[str, float] = None,
    incident_hour: Optional[int] = None,
    incident_type: str = None
) -> IncidentScore:
    """便捷函数：计算事故评分

    Args:
        incident_id: 事故ID
        row_index: 事故行索引
        analysis_result: 百分位分析结果
        extraction_result: 流量提取结果
        weights: 权重配置
        incident_hour: 事故发生时的小时数 (0-23)，用于计算时间标签
        incident_type: 事故类型（如 hazard, fire, carfire 等），
                      若为 None 则回退到 analysis_result.incident_type

    Returns:
        IncidentScore
    """
    scorer = IncidentScorer(weights=weights)
    return scorer.score(
        incident_id, row_index, analysis_result,
        extraction_result, incident_hour, incident_type
    )


def create_score_table(
    scores: List[IncidentScore]
) -> pd.DataFrame:
    """将评分结果转换为表格格式

    Args:
        scores: 评分结果列表

    Returns:
        DataFrame
    """
    rows = []

    for score in scores:
        row = {
            'row_index': score.row_index,
            'IncidentId': score.incident_id,
            'Type': score.incident_type,         # 事故类型
            'ImpactScore': score.overall_score,  # 事故影响分
            'MatlabScore': score.matlab_score,   # MATLAB NaN 评分
            'SeverityLevel': score.severity_level,
            'SeverityLabel': score.severity_label,
            'Time_tag': score.time_tag,          # 黄金时段分类
            'PeakTimeOffset': score.peak_time_offset,
            'DurationEstimate': score.duration_estimate,
            'SensorCount': len(score.sensor_scores)
        }

        # 添加各传感器评分
        for i, (sensor_idx, sensor_score) in enumerate(score.sensor_scores.items()):
            row[f'Sensor{i+1}_ID'] = sensor_idx
            row[f'Sensor{i+1}_OccScore'] = sensor_score.occupancy_score
            row[f'Sensor{i+1}_SpdScore'] = sensor_score.speed_score
            row[f'Sensor{i+1}_VolScore'] = sensor_score.volume_score
            row[f'Sensor{i+1}_ImpactScore'] = sensor_score.combined_score  # 事故影响分
            row[f'Sensor{i+1}_MatlabScore'] = sensor_score.matlab_nan_score  # MATLAB NaN 评分
            row[f'Sensor{i+1}_NaNCount'] = sensor_score.nan_count

        rows.append(row)

    return pd.DataFrame(rows)
