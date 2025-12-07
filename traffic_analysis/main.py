# -*- coding: utf-8 -*-
"""
交通事故数据预处理主程序

对应 MATLAB: main.m

功能：
1. 加载事故数据和流量矩阵
2. 对每个事故进行传感器匹配
3. 提取事故时刻前后的流量数据
4. 采样历史同期数据
5. 进行百分位分析
6. 计算事故影响评分
7. 导出结果到 Excel 文件

使用方法：
    python main.py --data-dir ../data --output-dir ./output

    # 处理指定范围的事故
    python main.py --start 0 --end 1000

    # 完整处理所有事故
    python main.py --data-dir ../data --full
"""

import argparse
import logging
import multiprocessing as mp
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import warnings
import re
import pandas as pd

# 忽略 numpy 的警告
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_config
from data.loader import load_incidents, load_sensors, load_traffic_matrices
from processors.sensor_matcher import SensorMatcher
from processors.traffic_extractor import TrafficExtractor
from processors.historical_sampler import HistoricalSampler
from processors.percentile_analyzer import PercentileAnalyzer
from processors.scorer import IncidentScorer
from processors.exporter import ResultExporter


def get_type_value(type_value: Any) -> str:
    """从 Type 字段获取可用于目录名的类别值。

    - 空值或无法解析 → "other"
    - 去除首尾空格，空白替换为下划线
    - 过滤常见非法路径字符，保证可作为目录名
    - 返回统一小写，避免大小写导致的过滤/分组不一致
    """
    if pd.isna(type_value):
        return "other"

    value = str(type_value).strip()
    if not value:
        return "other"

    # 将连续空白替换为单个下划线
    sanitized = re.sub(r"\s+", "_", value)
    # 替换文件系统不允许的字符
    sanitized = re.sub(r"[\\\\/:*?\"<>|]+", "_", sanitized)
    sanitized = sanitized.strip("._")

    if not sanitized:
        return "other"

    return sanitized.lower()


def group_incidents_by_type(
    incidents: pd.DataFrame,
    logger: logging.Logger
) -> Dict[str, pd.DataFrame]:
    """按 Type_normalized 列对事故进行分组（缺失时基于 Type 动态生成）

    Args:
        incidents: 事故数据 DataFrame
        logger: 日志器

    Returns:
        字典，键为 Type_normalized 类别值，值为该类别的事故 DataFrame
    """
    incidents = incidents.copy()

    if 'Type_normalized' not in incidents.columns:
        if 'Type' in incidents.columns:
            logger.info("未检测到 Type_normalized 列，基于 Type 创建规范化列供分组使用")
            incidents['Type_normalized'] = incidents['Type'].apply(get_type_value)
        else:
            logger.warning("事故数据中没有 Type 列，将所有事故归为 other 类别")
            incidents['Type_normalized'] = "other"

    categories = incidents['Type_normalized'].unique()
    grouped = {}

    logger.info(f"检测到 {len(categories)} 个 Type_normalized 类别:")
    for category in sorted(categories):
        category_df = incidents[incidents['Type_normalized'] == category].copy()
        grouped[category] = category_df
        logger.info(f"  - {category}: {len(category_df)} 条事故")

    return grouped


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """设置日志

    Args:
        log_level: 日志级别

    Returns:
        Logger 实例
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='交通事故数据预处理程序',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data',
        help='数据目录路径 (默认: ../data)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='输出目录路径 (默认: ./output)'
    )

    parser.add_argument(
        '--start',
        type=int,
        default=None,
        help='起始事故索引 (默认: 0)'
    )

    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='结束事故索引 (默认: 全部)'
    )

    parser.add_argument(
        '--time-window',
        type=int,
        default=12,
        help='时间窗口大小（时间步数，默认: 12，即前后各1小时）'
    )

    parser.add_argument(
        '--year',
        type=int,
        default=2023,
        help='数据年份 (默认: 2023)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='批处理大小 (默认: 100)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日志级别 (默认: INFO)'
    )

    parser.add_argument(
        '--full',
        action='store_true',
        help='处理所有事故（忽略 start/end 参数）'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='测试模式（只处理前10个事故）'
    )

    parser.add_argument(
        '--classify',
        action='store_true',
        help='按 Type 字段分类处理，每个类别输出到单独文件夹（内部先规范化并转为小写）'
    )

    parser.add_argument(
        '--type-filter',
        type=str,
        action='append',
        help='只处理指定 Type 值的事故 (可多次使用，例如 --type-filter Hazard，值会先规范化并转为小写)'
    )

    parser.add_argument(
        '--type-exclude',
        type=str,
        action='append',
        help='排除指定 Type 值的事故 (可多次使用，例如 --type-exclude Hazard，值会先规范化并转为小写)'
    )

    # ========================================
    # P0 优化参数：并行处理
    # ========================================
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='启用多进程并行处理模式 (P0 优化)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='并行处理的 Worker 数量 (默认: CPU核心数 - 25，最小为 4)'
    )

    parser.add_argument(
        '--max-cpus',
        type=int,
        default=None,
        help='实际可用的CPU核心数（用于容器环境，自动调整默认worker数）'
    )

    parser.add_argument(
        '--no-shared-memory',
        action='store_true',
        help='禁用共享内存优化（调试用）'
    )

    return parser.parse_args()


def process_batch(
    incidents,
    start_idx: int,
    end_idx: int,
    matcher: SensorMatcher,
    extractor: TrafficExtractor,
    sampler: HistoricalSampler,
    analyzer: PercentileAnalyzer,
    scorer: IncidentScorer,
    logger: logging.Logger
) -> tuple:
    """处理一批事故

    Args:
        incidents: 事故数据
        start_idx: 起始索引
        end_idx: 结束索引
        matcher: 传感器匹配器
        extractor: 流量提取器
        sampler: 历史采样器
        analyzer: 百分位分析器
        scorer: 评分器
        logger: 日志器

    Returns:
        (incident_tables, common_tables, scores, analysis_results, errors, success_count)
    """
    incident_tables = []
    common_tables = []
    scores = []
    analysis_results = []
    errors = []
    success_count = 0

    for idx in range(start_idx, end_idx):
        try:
            incident = incidents.iloc[idx]

            # 1. 传感器匹配
            match_result = matcher.match(incident, row_index=idx)
            if not match_result.success:
                errors.append({
                    'row_index': idx,
                    'incident_id': str(incident.get('incident_id', '')),
                    'error': 'sensor matching failed',
                    'details': '; '.join(match_result.errors)
                })
                continue

            # 2. 流量数据提取
            extraction_result = extractor.extract(match_result)
            if not extraction_result.success:
                errors.append({
                    'row_index': idx,
                    'incident_id': match_result.incident_id,
                    'error': 'data extraction failed',
                    'details': '; '.join(extraction_result.errors)
                })
                continue

            incident_tables.append(extraction_result.incident_table)
            common_tables.append(extraction_result.common_table)

            # 3. 历史数据采样
            sensor_indices = match_result.matched_sensors['SensorNumber'].tolist()
            sampling_result = sampler.sample(incident['dt'], sensor_indices)

            # 4. 百分位分析
            analysis_result = analyzer.analyze(
                incident_id=match_result.incident_id,
                incident_data=extraction_result.incident_table,
                historical_samples=sampling_result.samples
            )
            # 设置事故类型（从 match_result.incident_info 获取）
            incident_type = match_result.incident_info.get('Type', 'other')
            analysis_result.incident_type = incident_type
            analysis_results.append(analysis_result)

            # 5. 评分计算
            score = scorer.score(
                incident_id=match_result.incident_id,
                row_index=idx,
                analysis_result=analysis_result,
                extraction_result=extraction_result,
                incident_type=incident_type  # 传递事故类型
            )
            scores.append(score)

            success_count += 1

        except Exception as e:
            errors.append({
                'row_index': idx,
                'incident_id': str(incidents.iloc[idx].get('incident_id', '')),
                'error': 'processing error',
                'details': str(e)
            })

    return incident_tables, common_tables, scores, analysis_results, errors, success_count


def process_category(
    category: str,
    category_incidents: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    matcher: 'SensorMatcher',
    extractor: 'TrafficExtractor',
    sampler: 'HistoricalSampler',
    analyzer: 'PercentileAnalyzer',
    scorer: 'IncidentScorer',
    exporter: 'ResultExporter',
    batch_size: int,
    logger: logging.Logger
) -> tuple:
    """处理单个类别的所有事故

    Args:
        category: 类别名称
        category_incidents: 该类别的事故数据
        start_idx: 起始索引（在该类别内）
        end_idx: 结束索引（在该类别内）
        matcher: 传感器匹配器
        extractor: 流量提取器
        sampler: 历史采样器
        analyzer: 百分位分析器
        scorer: 评分器
        exporter: 导出器
        batch_size: 批处理大小
        logger: 日志器

    Returns:
        (total_success, total_errors, export_result)
    """
    all_incident_tables = []
    all_common_tables = []
    all_scores = []
    all_analysis_results = []
    all_errors = []
    total_success = 0

    num_to_process = end_idx - start_idx
    num_batches = (num_to_process + batch_size - 1) // batch_size

    logger.info(f"  处理范围: {start_idx} - {end_idx} (共 {num_to_process} 个事故)")

    for batch_num in range(num_batches):
        batch_start = start_idx + batch_num * batch_size
        batch_end = min(batch_start + batch_size, end_idx)

        logger.info(f"    批次 {batch_num + 1}/{num_batches}: "
                   f"事故 {batch_start} - {batch_end}")

        results = process_batch(
            category_incidents, batch_start, batch_end,
            matcher, extractor, sampler, analyzer, scorer, logger
        )

        incident_tables, common_tables, scores, analysis_results, errors, success = results

        all_incident_tables.extend(incident_tables)
        all_common_tables.extend(common_tables)
        all_scores.extend(scores)
        all_analysis_results.extend(analysis_results)
        all_errors.extend(errors)
        total_success += success

        # 打印批次进度
        processed = batch_end - start_idx
        pct = processed / num_to_process * 100
        logger.info(f"      完成: {success}/{batch_end - batch_start} 成功, "
                   f"进度: {processed}/{num_to_process} ({pct:.1f}%)")

    # 导出该类别的结果
    export_result = exporter.export(
        incident_tables=all_incident_tables if all_incident_tables else None,
        common_tables=all_common_tables if all_common_tables else None,
        scores=all_scores if all_scores else None,
        analysis_results=all_analysis_results if all_analysis_results else None,
        errors=all_errors if all_errors else None
    )

    return total_success, all_errors, export_result


def get_default_workers(max_workers: int = None) -> int:
    """获取默认的 Worker 数量

    基于 CPU 核心数计算，预留 25 核给系统和 I/O。
    最小为 4，最大为 CPU 核心数 - 25。
    如果指定了 max_workers，则不超过该值。

    Args:
        max_workers: 最大允许的 worker 数量（用于容器环境限制）

    Returns:
        推荐的 Worker 数量
    """
    cpu_count = mp.cpu_count()
    # 预留 25 核给系统和 I/O，但至少保留 4 个 worker
    workers = max(4, cpu_count - 25)

    # 如果指定了最大限制，不超过该值
    if max_workers:
        workers = min(workers, max_workers)

    return workers


def process_parallel(
    incidents: pd.DataFrame,
    sensors: pd.DataFrame,
    data_dir: str,
    output_dir: str,
    num_workers: int,
    time_window: int,
    year: int,
    start_idx: int,
    end_idx: int,
    exporter: 'ResultExporter',
    logger: logging.Logger
) -> tuple:
    """使用多进程并行处理事故数据

    P0 优化：多进程并行 + 共享内存

    Args:
        incidents: 事故数据
        sensors: 传感器数据
        data_dir: 数据目录
        output_dir: 输出目录
        num_workers: Worker 数量
        time_window: 时间窗口
        year: 数据年份
        start_idx: 起始索引
        end_idx: 结束索引
        exporter: 导出器
        logger: 日志器

    Returns:
        (total_success, all_errors, export_result)
    """
    from parallel_processor import ParallelProcessor

    logger.info("=" * 40)
    logger.info("🚀 P0 优化模式：多进程并行处理")
    logger.info("=" * 40)
    logger.info(f"  Worker 数量: {num_workers}")
    logger.info(f"  处理范围: {start_idx} - {end_idx}")
    logger.info(f"  共享内存: 启用")

    def progress_callback(processed: int, total: int):
        pct = processed / total * 100
        logger.info(f"  进度: {processed}/{total} ({pct:.1f}%)")

    # 创建并行处理器
    processor = ParallelProcessor(
        data_dir=data_dir,
        num_workers=num_workers,
        time_window=time_window,
        year=year
    )

    try:
        # 执行并行处理
        result = processor.process_incidents(
            incidents=incidents,
            sensors=sensors,
            start_idx=start_idx,
            end_idx=end_idx,
            progress_callback=progress_callback
        )

        # 导出结果
        export_result = exporter.export(
            incident_tables=result.incident_tables if result.incident_tables else None,
            common_tables=result.common_tables if result.common_tables else None,
            scores=result.scores if result.scores else None,
            analysis_results=result.analysis_results if result.analysis_results else None,
            errors=result.errors if result.errors else None
        )

        return result.total_success, result.errors, export_result

    finally:
        processor.cleanup()


def main():
    """主函数"""
    args = parse_args()
    logger = setup_logging(args.log_level)

    logger.info("=" * 60)
    logger.info("交通事故数据预处理程序")
    logger.info("=" * 60)

    # 显示并行处理模式
    if args.parallel:
        num_workers = args.workers if args.workers else get_default_workers(args.max_cpus)
        logger.info(f"🚀 并行处理模式已启用 (Workers: {num_workers})")

    start_time = datetime.now()

    # 设置路径
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"数据目录: {data_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"时间窗口: {args.time_window}")
    logger.info(f"数据年份: {args.year}")

    # 加载数据
    logger.info("-" * 40)
    logger.info("加载数据...")

    try:
        incidents_path = data_dir / 'incidents_y2023.csv'
        sensors_path = data_dir / 'sensor_meta_feature.csv'

        incidents = load_incidents(str(incidents_path))
        sensors = load_sensors(str(sensors_path))
        matrices = load_traffic_matrices(str(data_dir), use_mmap=True)

        logger.info(f"事故数据: {len(incidents)} 条")
        logger.info(f"传感器数据: {len(sensors)} 个")
        logger.info(f"流量矩阵形状: {matrices.occupancy.shape}")

        # ========================================
        # Type 字段规范化（解决数据一致性问题）
        # ========================================
        # 原因：过滤时使用原始值，但分类时使用规范化值，导致不一致
        # 例如：数据中 "Fire  " 无法匹配 --type-filter Fire
        # 解决：统一使用 get_type_value() 进行规范化
        # 变化： "Fire  " → "fire", "Car/Fire" → "car_fire", "" → "other"
        if 'Type' in incidents.columns:
            incidents['Type_normalized'] = incidents['Type'].apply(get_type_value)
        else:
            logger.warning("事故数据中没有 Type 列，创建 Type_normalized=other（小写）以保证后续逻辑可用")
            incidents['Type_normalized'] = "other"

        normalized_categories = incidents['Type_normalized'].unique()
        logger.info(
            "Type 规范化（已转小写）完成: %d 个类别 (示例: %s)" % (
                len(normalized_categories),
                ', '.join(sorted(normalized_categories[:5])) if len(normalized_categories) > 0 else 'NONE'
            )
        )

    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return 1

    # 过滤 Type
    if args.type_filter:
        logger.info(f"过滤保留 Type (规范化并小写后): {args.type_filter}")
        if 'Type_normalized' in incidents.columns:
            normalized_filter = [get_type_value(t) for t in args.type_filter]
            original_count = len(incidents)
            incidents = incidents[incidents['Type_normalized'].isin(normalized_filter)].copy()
            filtered_count = len(incidents)
            logger.info(f"  过滤结果: {original_count} -> {filtered_count} (移除 {original_count - filtered_count} 条)")
        else:
            logger.warning("事故数据中没有 Type_normalized 列，无法执行过滤")

    if args.type_exclude:
        logger.info(f"过滤排除 Type (规范化并小写后): {args.type_exclude}")
        if 'Type_normalized' in incidents.columns:
            normalized_exclude = [get_type_value(t) for t in args.type_exclude]
            original_count = len(incidents)
            incidents = incidents[~incidents['Type_normalized'].isin(normalized_exclude)].copy()
            filtered_count = len(incidents)
            logger.info(f"  过滤结果: {original_count} -> {filtered_count} (移除 {original_count - filtered_count} 条)")
        else:
            logger.warning("事故数据中没有 Type_normalized 列，无法执行过滤")

    if len(incidents) == 0:
        logger.warning("过滤后无事故数据，退出程序")
        return 0
        
    # 重置索引，确保后续处理逻辑正确
    incidents = incidents.reset_index(drop=True)

    # 确定处理范围
    total_incidents = len(incidents)

    if args.test:
        start_idx = 0
        end_idx = min(10, total_incidents)
        logger.info("测试模式：只处理前10个事故")
    elif args.full:
        start_idx = 0
        end_idx = total_incidents
        logger.info("完整模式：处理所有事故")
    else:
        start_idx = args.start if args.start is not None else 0
        end_idx = args.end if args.end is not None else min(start_idx + 100, total_incidents)

    logger.info(f"处理范围: {start_idx} - {end_idx} (共 {end_idx - start_idx} 个事故)")

    # 初始化处理器
    logger.info("-" * 40)
    logger.info("初始化处理器...")

    matcher = SensorMatcher(sensors, matrices.node_order)
    extractor = TrafficExtractor(matrices, args.time_window, args.year)
    sampler = HistoricalSampler(matrices, args.time_window, args.year)
    analyzer = PercentileAnalyzer()
    scorer = IncidentScorer()
    exporter = ResultExporter(str(output_dir))

    # 分批处理
    logger.info("-" * 40)
    logger.info("开始处理...")

    all_incident_tables = []
    all_common_tables = []
    all_scores = []
    all_analysis_results = []
    all_errors = []
    total_success = 0

    batch_size = args.batch_size

    # ========================================
    # 分类处理模式（--classify 启用时）
    # ========================================
    if args.classify:
        logger.info("=" * 40)
        logger.info("启用分类处理模式")
        logger.info("=" * 40)

        # 按 Type 分组
        grouped_incidents = group_incidents_by_type(incidents, logger)

        # 记录每个类别的统计信息
        category_stats = {}

        for category in sorted(grouped_incidents.keys()):
            category_incidents = grouped_incidents[category]
            category_total = len(category_incidents)

            # 计算该类别内的处理范围
            # 使用全局 start_idx/end_idx 的比例来确定类别内的范围
            if args.full:
                cat_start = 0
                cat_end = category_total
            elif args.test:
                cat_start = 0
                cat_end = min(10, category_total)
            else:
                # 按比例计算或使用全部
                cat_start = 0
                cat_end = category_total

            if cat_start >= cat_end:
                logger.info(f"\n跳过类别 [{category}]: 无事故需要处理")
                continue

            logger.info(f"\n{'='*40}")
            logger.info(f"处理类别: [{category}]")
            logger.info(f"  该类别共 {category_total} 条事故")
            logger.info(f"  处理范围: {cat_start} - {cat_end}")

            # 创建类别专属的输出目录
            category_output_dir = output_dir / category
            category_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"  输出目录: {category_output_dir}")

            # 创建类别专属的导出器
            category_exporter = ResultExporter(str(category_output_dir))

            # 处理该类别的事故
            cat_success, cat_errors, cat_export = process_category(
                category=category,
                category_incidents=category_incidents,
                start_idx=cat_start,
                end_idx=cat_end,
                matcher=matcher,
                extractor=extractor,
                sampler=sampler,
                analyzer=analyzer,
                scorer=scorer,
                exporter=category_exporter,
                batch_size=batch_size,
                logger=logger
            )

            # 记录统计信息
            category_stats[category] = {
                'total': category_total,
                'processed': cat_end - cat_start,
                'success': cat_success,
                'failed': len(cat_errors)
            }

            total_success += cat_success
            all_errors.extend(cat_errors)

            # 打印该类别的导出文件
            for file_path in cat_export.files_created:
                logger.info(f"    已创建: {file_path}")

        # 打印分类处理汇总
        logger.info("\n" + "=" * 40)
        logger.info("分类处理汇总")
        logger.info("=" * 40)
        for category, stats in sorted(category_stats.items()):
            success_rate = stats['success'] / stats['processed'] * 100 if stats['processed'] > 0 else 0
            logger.info(f"  [{category}]: {stats['success']}/{stats['processed']} 成功 ({success_rate:.1f}%)")

    # ========================================
    # 普通处理模式（默认）
    # ========================================
    else:
        # ========================================
        # P0 优化：并行处理模式
        # ========================================
        if args.parallel:
            num_workers = args.workers if args.workers else get_default_workers(args.max_cpus)

            total_success, all_errors, export_result = process_parallel(
                incidents=incidents,
                sensors=sensors,
                data_dir=str(data_dir),
                output_dir=str(output_dir),
                num_workers=num_workers,
                time_window=args.time_window,
                year=args.year,
                start_idx=start_idx,
                end_idx=end_idx,
                exporter=exporter,
                logger=logger
            )

            for file_path in export_result.files_created:
                logger.info(f"  已创建: {file_path}")

        # ========================================
        # 串行处理模式（原有逻辑）
        # ========================================
        else:
            num_batches = (end_idx - start_idx + batch_size - 1) // batch_size

            for batch_num in range(num_batches):
                batch_start = start_idx + batch_num * batch_size
                batch_end = min(batch_start + batch_size, end_idx)

                logger.info(f"处理批次 {batch_num + 1}/{num_batches}: "
                           f"事故 {batch_start} - {batch_end}")

                results = process_batch(
                    incidents, batch_start, batch_end,
                    matcher, extractor, sampler, analyzer, scorer, logger
                )

                incident_tables, common_tables, scores, analysis_results, errors, success = results

                all_incident_tables.extend(incident_tables)
                all_common_tables.extend(common_tables)
                all_scores.extend(scores)
                all_analysis_results.extend(analysis_results)
                all_errors.extend(errors)
                total_success += success

                # 打印批次进度
                processed = batch_end - start_idx
                total = end_idx - start_idx
                pct = processed / total * 100
                logger.info(f"  完成: {success}/{batch_end - batch_start} 成功, "
                           f"总进度: {processed}/{total} ({pct:.1f}%)")

            # 导出结果（仅串行模式需要在这里导出）
            logger.info("-" * 40)
            logger.info("导出结果...")

            export_result = exporter.export(
                incident_tables=all_incident_tables if all_incident_tables else None,
                common_tables=all_common_tables if all_common_tables else None,
                scores=all_scores if all_scores else None,
                analysis_results=all_analysis_results if all_analysis_results else None,
                errors=all_errors if all_errors else None
            )

            for file_path in export_result.files_created:
                logger.info(f"  已创建: {file_path}")

    # 导出处理摘要
    end_time = datetime.now()
    summary_path = exporter.export_summary(
        total_incidents=total_incidents,
        processed=end_idx - start_idx,
        successful=total_success,
        failed=len(all_errors),
        start_time=start_time,
        end_time=end_time
    )
    logger.info(f"  已创建: {summary_path}")

    # 打印摘要
    duration = (end_time - start_time).total_seconds()
    logger.info("-" * 40)
    logger.info("处理完成！")
    logger.info(f"  总事故数: {total_incidents}")
    logger.info(f"  处理事故: {end_idx - start_idx}")
    logger.info(f"  成功: {total_success}")
    logger.info(f"  失败: {len(all_errors)}")
    logger.info(f"  成功率: {total_success/(end_idx-start_idx)*100:.2f}%")
    logger.info(f"  耗时: {duration:.2f} 秒")
    logger.info(f"  平均每个事故: {duration/(end_idx-start_idx):.3f} 秒")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
