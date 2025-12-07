# -*- coding: utf-8 -*-
"""
数据加载模块

负责加载：
- 事故数据 (incidents_y2023.csv)
- 传感器元数据 (sensor_meta_feature.csv)
- 流量矩阵 (occupancy/speed/volume .npy 文件)

对应 MATLAB 代码：
- main.m 第 7-16 行的数据加载逻辑
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, NamedTuple
import numpy as np
import pandas as pd


class TrafficMatrices(NamedTuple):
    """流量矩阵数据容器

    Attributes:
        occupancy: 占用率矩阵 (时间步 x 传感器数)
        speed: 速度矩阵 (时间步 x 传感器数)
        volume: 流量矩阵 (时间步 x 传感器数)
        node_order: 节点顺序数组 (可选)
    """
    occupancy: np.ndarray
    speed: np.ndarray
    volume: np.ndarray
    node_order: Optional[np.ndarray] = None


def load_incidents(file_path: str) -> pd.DataFrame:
    """加载事故数据

    对应 MATLAB: incident_data = readtable(incident_file);

    Args:
        file_path: CSV 文件路径 (incidents_y2023.csv)

    Returns:
        事故数据 DataFrame，包含以下列：
        - incident_id: 事故ID
        - duration: 持续时间（分钟）
        - Abs PM: 绝对里程标
        - Fwy: 高速公路编号
        - AREA: 区域
        - DESCRIPTION: 事故描述
        - LOCATION: 位置
        - dt: 事故发生时间
        - Latitude: 纬度
        - Longitude: 经度
        - Freeway_direction: 方向 (N/S/E/W)
        - Type: 事故类型

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式不正确
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"事故数据文件不存在: {file_path}")

    # 根据文件扩展名选择加载方式
    suffix = file_path.suffix.lower()

    if suffix == '.csv':
        # CSV 文件使用 tab 分隔符
        df = pd.read_csv(
            file_path,
            sep='\t',
            encoding='utf-8',
            low_memory=False  # 避免混合类型警告
        )
    elif suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")

    # 验证必需的列
    required_columns = ['incident_id', 'dt', 'Abs PM', 'Fwy', 'Freeway_direction']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"缺少必需的列: {missing_columns}")

    # 解析日期时间列
    if 'dt' in df.columns:
        df['dt'] = pd.to_datetime(df['dt'], format='%m/%d/%Y %H:%M:%S', errors='coerce')

    # 确保数值列的类型正确
    if 'Abs PM' in df.columns:
        df['Abs PM'] = pd.to_numeric(df['Abs PM'], errors='coerce')
    if 'Fwy' in df.columns:
        df['Fwy'] = pd.to_numeric(df['Fwy'], errors='coerce')
    if 'duration' in df.columns:
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    if 'Latitude' in df.columns:
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    if 'Longitude' in df.columns:
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    return df


def load_sensors(file_path: str) -> pd.DataFrame:
    """加载传感器元数据

    对应 MATLAB: sensor_file = 'new.xlsx';

    Args:
        file_path: CSV 文件路径 (sensor_meta_feature.csv)

    Returns:
        传感器元数据 DataFrame，包含以下关键列：
        - station_id: 传感器站点ID
        - Fwy Name: 高速公路名称
        - Fwy: 高速公路编号
        - Direction: 方向
        - Abs PM: 绝对里程标
        - Lat: 纬度
        - Lng: 经度

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式不正确
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"传感器元数据文件不存在: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == '.csv':
        # CSV 文件使用 tab 分隔符，处理引号和错误行
        df = pd.read_csv(
            file_path,
            sep='\t',
            encoding='utf-8',
            on_bad_lines='skip',
            quoting=3,  # QUOTE_NONE - 忽略引号
            low_memory=False
        )
    elif suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")

    # 验证必需的列
    required_columns = ['station_id', 'Abs PM', 'Fwy', 'Direction']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"缺少必需的列: {missing_columns}")

    # 确保数值列的类型正确
    if 'Abs PM' in df.columns:
        df['Abs PM'] = pd.to_numeric(df['Abs PM'], errors='coerce')
    if 'Fwy' in df.columns:
        df['Fwy'] = pd.to_numeric(df['Fwy'], errors='coerce')
    if 'station_id' in df.columns:
        df['station_id'] = pd.to_numeric(df['station_id'], errors='coerce')
    if 'Lat' in df.columns:
        df['Lat'] = pd.to_numeric(df['Lat'], errors='coerce')
    if 'Lng' in df.columns:
        df['Lng'] = pd.to_numeric(df['Lng'], errors='coerce')

    return df


def load_traffic_matrices(
    data_dir: str,
    use_mmap: bool = True,
    load_node_order: bool = True
) -> TrafficMatrices:
    """加载流量矩阵数据

    使用内存映射加载大型 numpy 数组，节省内存。

    对应 MATLAB:
        data_rate = load('All_Occupancy_Rate.mat');
        data_speed = load('ALL_done_Speed.mat');
        data_volume = load('ALL_Traffic_Volumn.mat');

    Args:
        data_dir: 数据目录路径
        use_mmap: 是否使用内存映射（推荐用于大文件）
        load_node_order: 是否加载节点顺序数组

    Returns:
        TrafficMatrices 命名元组，包含：
        - occupancy: 占用率矩阵
        - speed: 速度矩阵
        - volume: 流量矩阵
        - node_order: 节点顺序（如果 load_node_order=True）

    Raises:
        FileNotFoundError: 必需的数据文件不存在
    """
    data_dir = Path(data_dir)

    # 定义文件路径
    occupancy_path = data_dir / 'occupancy_2023_all.npy'
    speed_path = data_dir / 'speed_2023_all.npy'
    volume_path = data_dir / 'volume_2023_all.npy'
    node_order_path = data_dir / 'node_order.npy'

    # 检查必需文件
    for path, name in [
        (occupancy_path, '占用率矩阵'),
        (speed_path, '速度矩阵'),
        (volume_path, '流量矩阵')
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name}文件不存在: {path}")

    # 设置内存映射模式
    mmap_mode = 'r' if use_mmap else None

    # 加载矩阵
    occupancy = np.load(str(occupancy_path), mmap_mode=mmap_mode)
    speed = np.load(str(speed_path), mmap_mode=mmap_mode)
    volume = np.load(str(volume_path), mmap_mode=mmap_mode)

    # 加载节点顺序（可选）
    node_order = None
    if load_node_order and node_order_path.exists():
        node_order = np.load(str(node_order_path))

    return TrafficMatrices(
        occupancy=occupancy,
        speed=speed,
        volume=volume,
        node_order=node_order
    )


def get_sensor_index(
    node_order: np.ndarray,
    station_id: int
) -> Optional[int]:
    """根据 station_id 获取在流量矩阵中的索引

    Args:
        node_order: 节点顺序数组
        station_id: 传感器站点ID

    Returns:
        在矩阵中的列索引，如果未找到则返回 None
    """
    indices = np.where(node_order == station_id)[0]
    if len(indices) > 0:
        return int(indices[0])
    return None


def validate_data_directory(data_dir: str) -> Dict[str, bool]:
    """验证数据目录中所需文件是否存在

    Args:
        data_dir: 数据目录路径

    Returns:
        字典，键为文件名，值为是否存在
    """
    data_dir = Path(data_dir)

    required_files = {
        'incidents_y2023.csv': '事故数据',
        'sensor_meta_feature.csv': '传感器元数据',
        'occupancy_2023_all.npy': '占用率矩阵',
        'speed_2023_all.npy': '速度矩阵',
        'volume_2023_all.npy': '流量矩阵',
        'node_order.npy': '节点顺序'
    }

    result = {}
    for filename, description in required_files.items():
        path = data_dir / filename
        result[filename] = path.exists()

    return result


def get_matrix_info(data_dir: str) -> Dict[str, Tuple[int, ...]]:
    """获取流量矩阵的形状信息（不完全加载）

    Args:
        data_dir: 数据目录路径

    Returns:
        字典，键为矩阵名称，值为形状元组
    """
    data_dir = Path(data_dir)

    info = {}

    for name in ['occupancy_2023_all', 'speed_2023_all', 'volume_2023_all']:
        path = data_dir / f'{name}.npy'
        if path.exists():
            # 使用 mmap 模式获取形状信息
            arr = np.load(str(path), mmap_mode='r')
            info[name] = arr.shape

    return info
