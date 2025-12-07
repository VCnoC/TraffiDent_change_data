# -*- coding: utf-8 -*-
"""
数据处理模块

包含：
- sensor_matcher.py: 传感器匹配（对应 Traffic_Function.m）
- traffic_extractor.py: 数据提取（对应 p_trafficdata_*.m）
- historical_sampler.py: 历史采样（对应 sort_function.m）
- percentile_analyzer.py: 百分位分析（对应 data_clean.m）
- scorer.py: 评分计算（对应 processGroup.m）
"""

from .sensor_matcher import (
    SensorMatcher,
    MatchResult,
    create_extracted_data,
)

from .traffic_extractor import (
    TrafficExtractor,
    ExtractionResult,
    extract_traffic_data,
)

from .historical_sampler import (
    HistoricalSampler,
    HistoricalSample,
    SamplingResult,
    sample_historical_data,
    compute_percentiles,
)

from .percentile_analyzer import (
    PercentileAnalyzer,
    PercentileResult,
    SensorAnalysisResult,
    AnalysisResult,
    analyze_incident,
    create_percentile_table,
)

from .scorer import (
    IncidentScorer,
    SensorScore,
    IncidentScore,
    score_incident,
    create_score_table,
)

from .exporter import (
    ResultExporter,
    ExportResult,
    export_results,
)

__all__ = [
    # sensor_matcher
    "SensorMatcher",
    "MatchResult",
    "create_extracted_data",
    # traffic_extractor
    "TrafficExtractor",
    "ExtractionResult",
    "extract_traffic_data",
    # historical_sampler
    "HistoricalSampler",
    "HistoricalSample",
    "SamplingResult",
    "sample_historical_data",
    "compute_percentiles",
    # percentile_analyzer
    "PercentileAnalyzer",
    "PercentileResult",
    "SensorAnalysisResult",
    "AnalysisResult",
    "analyze_incident",
    "create_percentile_table",
    # scorer
    "IncidentScorer",
    "SensorScore",
    "IncidentScore",
    "score_incident",
    "create_score_table",
    # exporter
    "ResultExporter",
    "ExportResult",
    "export_results",
]
