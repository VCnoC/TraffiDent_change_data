# -*- coding: utf-8 -*-
"""
多进程并行处理模块

使用 ProcessPoolExecutor 实现事故数据的并行处理。
配合共享内存管理器，实现高效的多核心并行计算。

对应 plan.md 优化计划:
- P0: 多进程并行改造 (动态 Workers: cpu_count-4) - 预期加速 50-80x

设计要点:
1. 主进程预加载数据到共享内存
2. 子进程通过共享内存访问数据（零拷贝）
3. 使用 ProcessPoolExecutor 管理 worker 池
4. 支持进度回调和错误处理

使用方法:
    processor = ParallelProcessor(
        data_dir='../data',
        num_workers=100
    )
    results = processor.process_incidents(incidents_df)
"""

import atexit
import logging
import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from shared_memory_manager import (
    SharedArrayInfo,
    SharedMemoryManager,
    SharedTrafficMatrices,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkerResult:
    """单个事故的处理结果

    Attributes:
        incident_id: 事故 ID
        row_index: 原始行索引
        success: 是否处理成功
        incident_table: 事故时刻数据表（成功时）
        common_table: 前后天数据表（成功时）
        score: 评分结果（成功时）
        analysis_result: 分析结果（成功时）
        error: 错误信息（失败时）
    """
    incident_id: str
    row_index: int
    success: bool
    incident_table: Optional[pd.DataFrame] = None
    common_table: Optional[pd.DataFrame] = None
    score: Optional[Any] = None
    analysis_result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class ParallelProcessingResult:
    """并行处理的总结果

    Attributes:
        incident_tables: 所有事故时刻数据表
        common_tables: 所有前后天数据表
        scores: 所有评分结果
        analysis_results: 所有分析结果
        errors: 所有错误记录
        total_processed: 总处理数
        total_success: 成功数
        total_failed: 失败数
        duration_seconds: 耗时（秒）
    """
    incident_tables: List[pd.DataFrame] = field(default_factory=list)
    common_tables: List[pd.DataFrame] = field(default_factory=list)
    scores: List[Any] = field(default_factory=list)
    analysis_results: List[Any] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)
    total_processed: int = 0
    total_success: int = 0
    total_failed: int = 0
    duration_seconds: float = 0.0


# ============================================================
# Worker 初始化和处理函数（必须在模块级别定义）
# ============================================================

# 全局变量，用于 worker 进程
_worker_initialized = False
_worker_matrices: Optional[SharedTrafficMatrices] = None
_worker_sensors: Optional[pd.DataFrame] = None
_worker_node_order: Optional[np.ndarray] = None
_worker_time_window: int = 12
_worker_year: int = 2023

# 处理器实例（延迟初始化）
_worker_matcher = None
_worker_extractor = None
_worker_sampler = None
_worker_analyzer = None
_worker_scorer = None


def _worker_init(
    array_info: Dict[str, SharedArrayInfo],
    sensors_data: bytes,
    node_order: Optional[np.ndarray],
    time_window: int,
    year: int
) -> None:
    """Worker 进程初始化函数

    在每个 worker 进程启动时调用一次，初始化共享数据访问。

    Args:
        array_info: 共享数组元信息
        sensors_data: 序列化的传感器数据
        node_order: 节点顺序数组
        time_window: 时间窗口
        year: 数据年份
    """
    global _worker_initialized, _worker_matrices, _worker_sensors
    global _worker_node_order, _worker_time_window, _worker_year
    global _worker_matcher, _worker_extractor, _worker_sampler
    global _worker_analyzer, _worker_scorer

    if _worker_initialized:
        return

    import pickle
    from processors.sensor_matcher import SensorMatcher
    from processors.traffic_extractor import TrafficExtractor
    from processors.historical_sampler import HistoricalSampler
    from processors.percentile_analyzer import PercentileAnalyzer
    from processors.scorer import IncidentScorer

    # 附加到共享内存
    _worker_matrices = SharedTrafficMatrices.from_shared_memory(
        array_info,
        node_order
    )

    # 反序列化传感器数据
    _worker_sensors = pickle.loads(sensors_data)

    _worker_node_order = node_order
    _worker_time_window = time_window
    _worker_year = year

    # 初始化处理器
    _worker_matcher = SensorMatcher(_worker_sensors, _worker_node_order)
    _worker_extractor = TrafficExtractor(
        _worker_matrices,
        _worker_time_window,
        _worker_year
    )
    _worker_sampler = HistoricalSampler(
        _worker_matrices,
        _worker_time_window,
        _worker_year
    )
    _worker_analyzer = PercentileAnalyzer()
    _worker_scorer = IncidentScorer()

    # 确保 worker 退出时关闭共享内存映射（不 unlink）
    def _close_shared_memory() -> None:
        if _worker_matrices is not None:
            _worker_matrices.close()

    atexit.register(_close_shared_memory)

    _worker_initialized = True


def _process_single_incident(
    incident_data: Dict[str, Any],
    row_index: int
) -> WorkerResult:
    """处理单个事故（Worker 函数）

    在 worker 进程中执行，处理单个事故数据。

    Args:
        incident_data: 事故数据字典
        row_index: 原始行索引

    Returns:
        WorkerResult 处理结果
    """
    global _worker_matcher, _worker_extractor, _worker_sampler
    global _worker_analyzer, _worker_scorer

    incident_id = str(incident_data.get('incident_id', ''))

    try:
        # 将字典转换为 Series
        incident = pd.Series(incident_data)

        # 1. 传感器匹配
        match_result = _worker_matcher.match(incident, row_index=row_index)
        if not match_result.success:
            return WorkerResult(
                incident_id=incident_id,
                row_index=row_index,
                success=False,
                error=f"sensor matching failed: {'; '.join(match_result.errors)}"
            )

        # 2. 流量数据提取
        extraction_result = _worker_extractor.extract(match_result)
        if not extraction_result.success:
            return WorkerResult(
                incident_id=incident_id,
                row_index=row_index,
                success=False,
                error=f"data extraction failed: {'; '.join(extraction_result.errors)}"
            )

        # 3. 历史数据采样
        sensor_indices = match_result.matched_sensors['SensorNumber'].tolist()
        sampling_result = _worker_sampler.sample(incident_data['dt'], sensor_indices)

        # 4. 百分位分析
        analysis_result = _worker_analyzer.analyze(
            incident_id=match_result.incident_id,
            incident_data=extraction_result.incident_table,
            historical_samples=sampling_result.samples
        )

        # 设置事故类型
        incident_type = match_result.incident_info.get('Type', 'other')
        analysis_result.incident_type = incident_type

        # 5. 评分计算
        score = _worker_scorer.score(
            incident_id=match_result.incident_id,
            row_index=row_index,
            analysis_result=analysis_result,
            extraction_result=extraction_result,
            incident_type=incident_type
        )

        return WorkerResult(
            incident_id=incident_id,
            row_index=row_index,
            success=True,
            incident_table=extraction_result.incident_table,
            common_table=extraction_result.common_table,
            score=score,
            analysis_result=analysis_result
        )

    except Exception:
        return WorkerResult(
            incident_id=incident_id,
            row_index=row_index,
            success=False,
            error=f"processing error: {traceback.format_exc()}"
        )


class ParallelProcessor:
    """多进程并行处理器

    使用 ProcessPoolExecutor 和共享内存实现高效的并行处理。

    Example:
        >>> processor = ParallelProcessor(
        ...     data_dir='../data',
        ...     num_workers=100
        ... )
        >>> result = processor.process_incidents(incidents_df, sensors_df)
        >>> print(f"处理完成: {result.total_success}/{result.total_processed}")
    """

    def __init__(
        self,
        data_dir: str,
        num_workers: Optional[int] = None,
        time_window: int = 12,
        year: int = 2023,
        chunk_size: int = 10
    ):
        """初始化并行处理器

        Args:
            data_dir: 数据目录路径
            num_workers: Worker 进程数量（默认: 自动 = CPU核心数-4，最少 1）
            time_window: 时间窗口大小
            year: 数据年份
            chunk_size: 每次提交的任务数量（用于进度控制）
        """
        self.data_dir = data_dir
        if num_workers is None:
            self.num_workers = max(1, mp.cpu_count() - 4)
            self._num_workers_source = "auto"
        else:
            self.num_workers = max(1, num_workers)
            self._num_workers_source = "user"
        self.time_window = time_window
        self.year = year
        self.chunk_size = chunk_size

        self._shm_manager: Optional[SharedMemoryManager] = None
        self._array_info: Optional[Dict[str, SharedArrayInfo]] = None
        self._node_order: Optional[np.ndarray] = None

        logger.info(
            f"初始化并行处理器: {self.num_workers} workers (source={self._num_workers_source})"
        )

    def _setup_shared_memory(self) -> None:
        """设置共享内存

        预加载流量矩阵数据到共享内存。
        """
        if self._shm_manager is not None:
            return

        logger.info("预加载数据到共享内存...")
        self._shm_manager = SharedMemoryManager()
        self._array_info = self._shm_manager.create_from_files(
            self.data_dir,
            self.year
        )
        self._node_order = self._shm_manager.get_node_order()

    def process_incidents(
        self,
        incidents: pd.DataFrame,
        sensors: pd.DataFrame,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> ParallelProcessingResult:
        """并行处理事故数据

        Args:
            incidents: 事故数据 DataFrame
            sensors: 传感器数据 DataFrame
            start_idx: 起始索引
            end_idx: 结束索引（不包含）
            progress_callback: 进度回调函数 (processed, total)

        Returns:
            ParallelProcessingResult 处理结果
        """
        import pickle

        # 设置共享内存
        self._setup_shared_memory()

        if end_idx is None:
            end_idx = len(incidents)

        total = end_idx - start_idx
        logger.info(f"开始并行处理 {total} 个事故 (workers={self.num_workers})")

        start_time = datetime.now()

        # 准备传递给 worker 的数据
        sensors_data = pickle.dumps(sensors)
        init_args = (
            self._array_info,
            sensors_data,
            self._node_order,
            self.time_window,
            self.year
        )

        # 收集结果
        result = ParallelProcessingResult()
        completed = 0

        # 使用 ProcessPoolExecutor 并行处理（带背压的分批提交）
        with ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_worker_init,
            initargs=init_args
        ) as executor:
            in_flight = {}
            submit_cursor = start_idx
            queue_limit = max(self.num_workers * 2, self.num_workers + 1)

            def submit_one(idx: int) -> None:
                incident = incidents.iloc[idx]
                incident_dict = incident.to_dict()
                if 'dt' in incident_dict and hasattr(incident_dict['dt'], 'isoformat'):
                    incident_dict['dt'] = incident_dict['dt']
                future = executor.submit(
                    _process_single_incident,
                    incident_dict,
                    idx
                )
                in_flight[future] = idx

            # 初始填充队列
            while submit_cursor < end_idx and len(in_flight) < queue_limit:
                submit_one(submit_cursor)
                submit_cursor += 1

            while in_flight:
                for future in as_completed(list(in_flight.keys())):
                    idx = in_flight.pop(future)
                    try:
                        worker_result = future.result()

                        if worker_result.success:
                            result.incident_tables.append(worker_result.incident_table)
                            result.common_tables.append(worker_result.common_table)
                            result.scores.append(worker_result.score)
                            result.analysis_results.append(worker_result.analysis_result)
                            result.total_success += 1
                        else:
                            result.errors.append({
                                'row_index': worker_result.row_index,
                                'incident_id': worker_result.incident_id,
                                'error': worker_result.error
                            })
                            result.total_failed += 1

                    except Exception as e:
                        tb = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                        result.errors.append({
                            'row_index': idx,
                            'incident_id': 'unknown',
                            'error': f"future error: {tb}"
                        })
                        result.total_failed += 1

                    completed += 1
                    result.total_processed = completed

                    # 进度回调
                    if progress_callback and self.chunk_size > 0 and completed % self.chunk_size == 0:
                        progress_callback(completed, total)

                    # 处理一个完成任务后尝试补充新的任务，保持有界队列
                    while submit_cursor < end_idx and len(in_flight) < queue_limit:
                        submit_one(submit_cursor)
                        submit_cursor += 1

                    # 跳出 as_completed 内层循环条件：让 while 判断 in_flight 重新拉取新 futures
                    if not in_flight:
                        break

        end_time = datetime.now()
        result.duration_seconds = (end_time - start_time).total_seconds()

        # 最终进度
        if progress_callback:
            progress_callback(completed, total)

        logger.info(f"并行处理完成: {result.total_success}/{total} 成功, "
                   f"耗时 {result.duration_seconds:.2f} 秒")

        return result

    def cleanup(self) -> None:
        """清理资源"""
        if self._shm_manager is not None:
            self._shm_manager.cleanup()
            self._shm_manager = None
            self._array_info = None
            self._node_order = None

    def __del__(self):
        """析构时清理"""
        self.cleanup()


def process_incidents_parallel(
    incidents: pd.DataFrame,
    sensors: pd.DataFrame,
    data_dir: str,
    num_workers: Optional[int] = None,
    time_window: int = 12,
    year: int = 2023,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> ParallelProcessingResult:
    """便捷函数：并行处理事故数据

    Args:
        incidents: 事故数据 DataFrame
        sensors: 传感器数据 DataFrame
        data_dir: 数据目录路径
        num_workers: Worker 进程数量
        time_window: 时间窗口大小
        year: 数据年份
        start_idx: 起始索引
        end_idx: 结束索引
        progress_callback: 进度回调函数

    Returns:
        ParallelProcessingResult 处理结果
    """
    processor = ParallelProcessor(
        data_dir=data_dir,
        num_workers=num_workers,
        time_window=time_window,
        year=year
    )

    try:
        return processor.process_incidents(
            incidents=incidents,
            sensors=sensors,
            start_idx=start_idx,
            end_idx=end_idx,
            progress_callback=progress_callback
        )
    finally:
        processor.cleanup()
