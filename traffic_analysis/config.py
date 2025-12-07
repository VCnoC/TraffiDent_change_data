# -*- coding: utf-8 -*-
"""
配置管理模块

集中管理所有可配置参数，对应 MATLAB 中的硬编码常量。

对应 MATLAB 代码：
- main.m 第 19 行: set_time = 12;
- 各函数中的文件路径和参数
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import os


@dataclass
class PathConfig:
    """路径配置"""

    # 数据目录
    data_dir: Path = field(default_factory=lambda: Path("../data"))

    # 输入文件
    incidents_file: str = "incidents_y2023.csv"
    sensors_file: str = "sensor_meta_feature.csv"
    occupancy_file: str = "occupancy_2023_all.npy"
    speed_file: str = "speed_2023_all.npy"
    volume_file: str = "volume_2023_all.npy"
    node_order_file: str = "node_order.npy"

    # 输出目录
    output_dir: Path = field(default_factory=lambda: Path("./output"))

    # 输出文件
    output_main: str = "A_final_output.xlsx"
    output_score: str = "A_final_score_table.xlsx"
    output_common: str = "A_final_common_table.xlsx"
    output_error: str = "A_error_traffic_table.xlsx"

    def get_incidents_path(self) -> Path:
        """获取事故数据完整路径"""
        return self.data_dir / self.incidents_file

    def get_sensors_path(self) -> Path:
        """获取传感器数据完整路径"""
        return self.data_dir / self.sensors_file

    def get_output_path(self, filename: str) -> Path:
        """获取输出文件完整路径"""
        return self.output_dir / filename

    def ensure_output_dir(self) -> None:
        """确保输出目录存在"""
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ProcessingConfig:
    """数据处理配置"""

    # 时间窗口参数（对应 MATLAB: set_time = 12）
    # 表示事故时刻前后各取多少个时间步
    time_window: int = 12

    # 数据年份
    data_year: int = 2023

    # 传感器匹配参数
    # 最大搜索距离（里程标单位）
    max_search_distance: float = 5.0

    # 匹配传感器的最大数量
    max_sensors_per_incident: int = 5

    # 历史数据采样参数
    # 采样周数（同一星期几的历史数据周数）
    historical_weeks: int = 4

    # 百分位分析参数
    # 用于异常检测的百分位阈值
    percentile_thresholds: List[int] = field(
        default_factory=lambda: [10, 25, 50, 75, 90]
    )

    # 评分参数
    # 权重因子
    score_weights: dict = field(
        default_factory=lambda: {
            "occupancy": 0.4,
            "speed": 0.4,
            "volume": 0.2
        }
    )

    # 事故处理范围
    # 起始行索引（对应 MATLAB: for row_index = 80001:total_incidents）
    start_row: Optional[int] = None
    # 结束行索引
    end_row: Optional[int] = None

    @property
    def total_timesteps(self) -> int:
        """事故时刻周围的总时间步数"""
        return 2 * self.time_window + 1


@dataclass
class MemoryConfig:
    """内存管理配置"""

    # 是否使用内存映射加载大文件
    use_mmap: bool = True

    # 批处理大小（同时处理的事故数量）
    batch_size: int = 1000

    # 是否启用进度显示
    show_progress: bool = True

    # 日志级别
    log_level: str = "INFO"


@dataclass
class Config:
    """主配置类

    使用示例:
        config = Config()
        config.paths.data_dir = Path("/custom/data/path")
        config.processing.time_window = 15
    """

    paths: PathConfig = field(default_factory=PathConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def __post_init__(self):
        """初始化后处理"""
        # 确保路径是 Path 对象
        if isinstance(self.paths.data_dir, str):
            self.paths.data_dir = Path(self.paths.data_dir)
        if isinstance(self.paths.output_dir, str):
            self.paths.output_dir = Path(self.paths.output_dir)

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置

        支持的环境变量:
        - TRAFFIC_DATA_DIR: 数据目录
        - TRAFFIC_OUTPUT_DIR: 输出目录
        - TRAFFIC_TIME_WINDOW: 时间窗口
        - TRAFFIC_DATA_YEAR: 数据年份
        """
        config = cls()

        if data_dir := os.getenv("TRAFFIC_DATA_DIR"):
            config.paths.data_dir = Path(data_dir)

        if output_dir := os.getenv("TRAFFIC_OUTPUT_DIR"):
            config.paths.output_dir = Path(output_dir)

        if time_window := os.getenv("TRAFFIC_TIME_WINDOW"):
            config.processing.time_window = int(time_window)

        if data_year := os.getenv("TRAFFIC_DATA_YEAR"):
            config.processing.data_year = int(data_year)

        return config

    def validate(self) -> List[str]:
        """验证配置有效性

        Returns:
            错误信息列表，空列表表示配置有效
        """
        errors = []

        # 验证路径
        if not self.paths.data_dir.exists():
            errors.append(f"数据目录不存在: {self.paths.data_dir}")

        incidents_path = self.paths.get_incidents_path()
        if not incidents_path.exists():
            errors.append(f"事故数据文件不存在: {incidents_path}")

        sensors_path = self.paths.get_sensors_path()
        if not sensors_path.exists():
            errors.append(f"传感器数据文件不存在: {sensors_path}")

        # 验证处理参数
        if self.processing.time_window < 1:
            errors.append("时间窗口必须大于等于1")

        if self.processing.max_search_distance <= 0:
            errors.append("最大搜索距离必须大于0")

        return errors


# 默认配置实例
default_config = Config()


def get_config() -> Config:
    """获取配置实例

    优先尝试从环境变量加载，否则使用默认配置

    Returns:
        Config 实例
    """
    # 检查是否有环境变量设置
    if any(os.getenv(key) for key in [
        "TRAFFIC_DATA_DIR",
        "TRAFFIC_OUTPUT_DIR",
        "TRAFFIC_TIME_WINDOW",
        "TRAFFIC_DATA_YEAR"
    ]):
        return Config.from_env()

    return default_config
