# -*- coding: utf-8 -*-
"""
结果导出模块

将处理结果导出为 Excel 文件。

对应 MATLAB 代码的输出部分：
- A_final_output.xlsx: 主输出表（事故时刻数据）
- A_final_score_table.xlsx: 评分表
- A_final_common_table.xlsx: 前后天对照数据
- A_error_traffic_table.xlsx: 错误记录

导出格式兼容原 MATLAB 输出。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime


def calculate_time_tag(time_value: Union[int, float, str]) -> int:
    """根据时间计算时段标签

    对应 MATLAB: processGroup.m 第 67-80 行

    时段定义:
        - 1: 早高峰 (7:00-9:00)
        - 2: 午间 (11:00-13:00)
        - 3: 晚高峰 (17:00-20:00)
        - 0: 其他时段

    Args:
        time_value: 时间戳，格式如 202001010730 (YYYYMMDDHHmm)

    Returns:
        tag: 0=其他, 1=早高峰, 2=午间, 3=晚高峰
    """
    try:
        time_str = str(int(time_value))
        # 时间格式: YYYYMMDDHHmm (12位)
        # 小时在索引 8-10 位置
        if len(time_str) >= 10:
            hour = int(time_str[8:10])
            if 7 <= hour < 9:
                return 1  # 早高峰
            elif 11 <= hour < 13:
                return 2  # 午间
            elif 17 <= hour < 20:
                return 3  # 晚高峰
        return 0  # 其他时段
    except (ValueError, TypeError):
        return 0  # 无法解析时返回默认值

try:
    from .traffic_extractor import ExtractionResult
    from .percentile_analyzer import AnalysisResult, create_percentile_table
    from .scorer import IncidentScore, create_score_table
except ImportError:
    from processors.traffic_extractor import ExtractionResult
    from processors.percentile_analyzer import AnalysisResult, create_percentile_table
    from processors.scorer import IncidentScore, create_score_table


@dataclass
class ExportResult:
    """导出结果

    Attributes:
        output_dir: 输出目录
        files_created: 创建的文件列表
        errors: 错误信息
    """
    output_dir: Path
    files_created: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """是否导出成功"""
        return len(self.files_created) > 0 and len(self.errors) == 0


class ResultExporter:
    """结果导出器

    将处理结果导出为与 MATLAB 兼容的 Excel 文件。

    Example:
        >>> exporter = ResultExporter(output_dir="./output")
        >>> result = exporter.export(
        ...     incident_tables=incident_tables,
        ...     common_tables=common_tables,
        ...     scores=scores,
        ...     errors=errors
        ... )
    """

    def __init__(
        self,
        output_dir: str = "./output",
        prefix: str = "A_"
    ):
        """初始化导出器

        Args:
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        self.output_dir = Path(output_dir)
        self.prefix = prefix

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        incident_tables: List[pd.DataFrame] = None,
        common_tables: List[pd.DataFrame] = None,
        scores: List[IncidentScore] = None,
        analysis_results: List[AnalysisResult] = None,
        errors: List[Dict] = None
    ) -> ExportResult:
        """导出所有结果

        Args:
            incident_tables: 事故时刻数据表列表
            common_tables: 前后天数据表列表
            scores: 评分结果列表
            analysis_results: 分析结果列表
            errors: 错误记录列表

        Returns:
            ExportResult
        """
        result = ExportResult(output_dir=self.output_dir)

        # 导出事故时刻数据
        if incident_tables:
            try:
                path = self._export_incident_data(incident_tables, analysis_results)
                result.files_created.append(str(path))
            except Exception as e:
                result.errors.append(f"导出事故时刻数据失败: {str(e)}")

        # 导出前后天数据
        if common_tables:
            try:
                path = self._export_common_data(common_tables, analysis_results)
                result.files_created.append(str(path))
            except Exception as e:
                result.errors.append(f"导出前后天数据失败: {str(e)}")

        # 导出评分表
        if scores:
            try:
                path = self._export_scores(scores)
                result.files_created.append(str(path))
            except Exception as e:
                result.errors.append(f"导出评分表失败: {str(e)}")

        # 导出百分位分析结果
        if analysis_results:
            try:
                path = self._export_analysis(analysis_results)
                result.files_created.append(str(path))
            except Exception as e:
                result.errors.append(f"导出分析结果失败: {str(e)}")

        # 导出错误记录
        if errors:
            try:
                path = self._export_errors(errors)
                result.files_created.append(str(path))
            except Exception as e:
                result.errors.append(f"导出错误记录失败: {str(e)}")

        return result

    def _export_incident_data(
        self,
        tables: List[pd.DataFrame],
        analysis_results: Optional[List[AnalysisResult]] = None
    ) -> Path:
        """导出事故时刻数据

        对应 MATLAB: A_final_output.xlsx

        Args:
            tables: 事故时刻数据表列表
            analysis_results: 百分位分析结果列表（用于添加 Percentage 和 Median 列）

        Returns:
            输出文件路径
        """
        # 合并所有表
        combined = pd.concat(tables, ignore_index=True)

        # 添加 Percentage 和 Median 列
        # 对应 MATLAB: data_clean.m 中的 Percentage 和 Median 列
        # 注意: 即使 analysis_results 为空，也必须添加这些列（使用默认值）
        # MATLAB 默认值: Percentage=-3（表示未找到）, Median=NaN
        if analysis_results:
            combined = self._merge_percentile_data(combined, analysis_results)
        else:
            # analysis_results 为空时，添加默认值的列
            combined = combined.copy()
            combined['Percentage'] = -3.0
            combined['Median'] = np.nan

        # 添加 tag 时段标签列
        # 对应 MATLAB: processGroup.m 中的 tag 计算
        if 'Time' in combined.columns:
            combined['tag'] = combined['Time'].apply(calculate_time_tag)

        # 输出文件路径
        output_path = self.output_dir / f"{self.prefix}final_output.csv"

        # 导出到 CSV（比 Excel 快 5-10 倍，适合大数据量）
        combined.to_csv(output_path, index=False, encoding='utf-8-sig')

        return output_path

    def _merge_percentile_data(
        self,
        incident_df: pd.DataFrame,
        analysis_results: List[AnalysisResult]
    ) -> pd.DataFrame:
        """将百分位分析结果合并到事故数据表中

        为每行添加 Percentage（位置百分位）和 Median（中位数）列。

        对应 MATLAB: data_clean.m 中的 Percentage 和 Median 列

        映射关系：
        - incident_df 中的 (IncidentId, SensorNumber, Data_Type) 对应
        - analysis_results 中的 (incident_id, sensor_results[sensor_idx], occupancy/speed/volume)

        Args:
            incident_df: 事故时刻数据表
            analysis_results: 百分位分析结果列表

        Returns:
            添加了 Percentage 和 Median 列的 DataFrame
        """
        # 创建 incident_id -> AnalysisResult 的映射
        results_map: Dict[str, AnalysisResult] = {}
        for result in analysis_results:
            results_map[result.incident_id] = result

        # Data_Type 到 PercentileResult 属性的映射
        data_type_map = {
            'Occupancy_Rate': 'occupancy',
            'Speed': 'speed',
            'Volume': 'volume'
        }

        # 初始化 Percentage 和 Median 列（默认值）
        # MATLAB 默认值：-3 表示未找到
        percentages = np.full(len(incident_df), -3.0)
        medians = np.full(len(incident_df), np.nan)

        # 遍历每一行，查找对应的分析结果
        for idx, row in incident_df.iterrows():
            incident_id = str(row.get('IncidentId', ''))
            sensor_number = row.get('SensorNumber', -1)
            data_type = row.get('Data_Type', '')

            # 查找对应的 AnalysisResult
            if incident_id not in results_map:
                continue

            analysis_result = results_map[incident_id]

            # 查找对应的 SensorAnalysisResult
            if sensor_number not in analysis_result.sensor_results:
                continue

            sensor_result = analysis_result.sensor_results[sensor_number]

            # 获取对应数据类型的 PercentileResult
            attr_name = data_type_map.get(data_type)
            if not attr_name:
                continue

            percentile_result = getattr(sensor_result, attr_name, None)
            if percentile_result is None:
                continue

            # 提取 Percentage（position_percentile）和 Median（median_value）
            percentages[idx] = percentile_result.position_percentile
            medians[idx] = percentile_result.median_value

        # 添加列到 DataFrame
        incident_df = incident_df.copy()
        incident_df['Percentage'] = percentages
        incident_df['Median'] = medians

        return incident_df

    def _export_common_data(
        self,
        tables: List[pd.DataFrame],
        analysis_results: Optional[List[AnalysisResult]] = None
    ) -> Path:
        """导出前后天数据

        对应 MATLAB: A_final_common_table.xlsx

        Args:
            tables: 前后天数据表列表
            analysis_results: 百分位分析结果列表（用于添加 Percentage 和 Median 列）

        Returns:
            输出文件路径
        """
        # 合并所有表
        combined = pd.concat(tables, ignore_index=True)

        # 添加 Percentage 和 Median 列
        # 对应 MATLAB: 前后天数据也需要包含百分位信息
        # 注意: 即使 analysis_results 为空，也必须添加这些列（使用默认值）
        # MATLAB 默认值: Percentage=-3（表示未找到）, Median=NaN
        if analysis_results:
            combined = self._merge_percentile_data(combined, analysis_results)
        else:
            # analysis_results 为空时，添加默认值的列
            combined = combined.copy()
            combined['Percentage'] = -3.0
            combined['Median'] = np.nan

        # Kind 列已经在 traffic_extractor.py 中直接生成（小写 'before'/'after'）
        # 对应 MATLAB: p_trafficdata_two_function.m 第105行和第123行
        # 不再需要从 Period 转换

        # 添加 tag 时段标签列
        if 'Time' in combined.columns:
            combined['tag'] = combined['Time'].apply(calculate_time_tag)

        # 输出文件路径
        output_path = self.output_dir / f"{self.prefix}final_common_table.csv"

        # 导出到 CSV（比 Excel 快 5-10 倍，适合大数据量）
        combined.to_csv(output_path, index=False, encoding='utf-8-sig')

        return output_path

    def _export_scores(
        self,
        scores: List[IncidentScore]
    ) -> Path:
        """导出评分表

        对应 MATLAB: A_final_score_table.xlsx

        Args:
            scores: 评分结果列表

        Returns:
            输出文件路径
        """
        # 使用 scorer 模块的表格创建函数
        table = create_score_table(scores)

        # 输出文件路径
        output_path = self.output_dir / f"{self.prefix}final_score_table.csv"

        # 导出到 CSV（比 Excel 快 5-10 倍，适合大数据量）
        table.to_csv(output_path, index=False, encoding='utf-8-sig')

        return output_path

    def _export_analysis(
        self,
        results: List[AnalysisResult]
    ) -> Path:
        """导出百分位分析结果

        Args:
            results: 分析结果列表

        Returns:
            输出文件路径
        """
        # 合并所有分析表，传递 incident_type
        tables = [create_percentile_table(r, r.incident_type) for r in results]
        tables = [t for t in tables if len(t) > 0]

        if len(tables) > 0:
            combined = pd.concat(tables, ignore_index=True)
        else:
            combined = pd.DataFrame()

        # 输出文件路径
        output_path = self.output_dir / f"{self.prefix}percentile_analysis.csv"

        # 导出到 CSV（比 Excel 快 5-10 倍，适合大数据量）
        combined.to_csv(output_path, index=False, encoding='utf-8-sig')

        return output_path

    def _export_errors(
        self,
        errors: List[Dict]
    ) -> Path:
        """导出错误记录

        对应 MATLAB: A_error_traffic_table.xlsx

        Args:
            errors: 错误记录列表

        Returns:
            输出文件路径
        """
        # 转换为 DataFrame
        df = pd.DataFrame(errors)

        # 输出文件路径
        output_path = self.output_dir / f"{self.prefix}error_traffic_table.csv"

        # 导出到 CSV（比 Excel 快 5-10 倍，适合大数据量）
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        return output_path

    def export_summary(
        self,
        total_incidents: int,
        processed: int,
        successful: int,
        failed: int,
        start_time: datetime,
        end_time: datetime
    ) -> Path:
        """导出处理摘要

        Args:
            total_incidents: 总事故数
            processed: 处理数
            successful: 成功数
            failed: 失败数
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            输出文件路径
        """
        duration = (end_time - start_time).total_seconds()

        summary = pd.DataFrame([{
            'Total_Incidents': total_incidents,
            'Processed': processed,
            'Successful': successful,
            'Failed': failed,
            'Success_Rate': f"{successful/processed*100:.2f}%" if processed > 0 else "N/A",
            'Start_Time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'End_Time': end_time.strftime("%Y-%m-%d %H:%M:%S"),
            'Duration_Seconds': duration,
            'Avg_Time_Per_Incident': f"{duration/processed:.3f}s" if processed > 0 else "N/A"
        }])

        output_path = self.output_dir / f"{self.prefix}processing_summary.csv"
        summary.to_csv(output_path, index=False, encoding='utf-8-sig')

        return output_path


def export_results(
    output_dir: str,
    incident_tables: List[pd.DataFrame] = None,
    common_tables: List[pd.DataFrame] = None,
    scores: List[IncidentScore] = None,
    analysis_results: List[AnalysisResult] = None,
    errors: List[Dict] = None
) -> ExportResult:
    """便捷函数：导出处理结果

    Args:
        output_dir: 输出目录
        incident_tables: 事故时刻数据表列表
        common_tables: 前后天数据表列表
        scores: 评分结果列表
        analysis_results: 分析结果列表
        errors: 错误记录列表

    Returns:
        ExportResult
    """
    exporter = ResultExporter(output_dir=output_dir)
    return exporter.export(
        incident_tables=incident_tables,
        common_tables=common_tables,
        scores=scores,
        analysis_results=analysis_results,
        errors=errors
    )
