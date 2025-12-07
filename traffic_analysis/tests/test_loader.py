# -*- coding: utf-8 -*-
"""
数据加载模块单元测试

测试 data/loader.py 中的所有函数
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import (
    TrafficMatrices,
    load_incidents,
    load_sensors,
    load_traffic_matrices,
    get_sensor_index,
    validate_data_directory,
    get_matrix_info,
)


class TestTrafficMatrices:
    """测试 TrafficMatrices 命名元组"""

    def test_create_traffic_matrices(self):
        """测试创建 TrafficMatrices"""
        occ = np.random.rand(100, 10)
        spd = np.random.rand(100, 10)
        vol = np.random.rand(100, 10)
        node_order = np.arange(10)

        matrices = TrafficMatrices(
            occupancy=occ,
            speed=spd,
            volume=vol,
            node_order=node_order
        )

        assert matrices.occupancy.shape == (100, 10)
        assert matrices.speed.shape == (100, 10)
        assert matrices.volume.shape == (100, 10)
        assert len(matrices.node_order) == 10

    def test_traffic_matrices_without_node_order(self):
        """测试不带 node_order 的 TrafficMatrices"""
        matrices = TrafficMatrices(
            occupancy=np.zeros((10, 5)),
            speed=np.zeros((10, 5)),
            volume=np.zeros((10, 5))
        )
        assert matrices.node_order is None


class TestLoadIncidents:
    """测试 load_incidents 函数"""

    def test_load_csv_file(self):
        """测试加载 CSV 文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("incident_id\tdt\tAbs PM\tFwy\tFreeway_direction\tduration\n")
            f.write("123\t01/15/2023 10:30:00\t15.5\t101\tN\t60\n")
            f.write("456\t01/16/2023 14:20:00\t20.3\t405\tS\t45\n")
            temp_path = f.name

        try:
            df = load_incidents(temp_path)
            assert len(df) == 2
            assert 'incident_id' in df.columns
            assert 'dt' in df.columns
            assert 'Abs PM' in df.columns
            assert df['Fwy'].iloc[0] == 101
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with pytest.raises(FileNotFoundError):
            load_incidents("/nonexistent/path/file.csv")

    def test_load_unsupported_format(self):
        """测试加载不支持的格式"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"test")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="不支持的文件格式"):
                load_incidents(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_missing_columns(self):
        """测试缺少必需列"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1\tcol2\n")
            f.write("a\tb\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="缺少必需的列"):
                load_incidents(temp_path)
        finally:
            os.unlink(temp_path)


class TestLoadSensors:
    """测试 load_sensors 函数"""

    def test_load_csv_file(self):
        """测试加载传感器 CSV 文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("station_id\tAbs PM\tFwy\tDirection\tLat\tLng\n")
            f.write("1001\t15.5\t101\tN\t34.05\t-118.25\n")
            f.write("1002\t20.3\t405\tS\t34.10\t-118.30\n")
            temp_path = f.name

        try:
            df = load_sensors(temp_path)
            assert len(df) == 2
            assert 'station_id' in df.columns
            assert 'Abs PM' in df.columns
            assert 'Direction' in df.columns
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with pytest.raises(FileNotFoundError):
            load_sensors("/nonexistent/path/file.csv")


class TestLoadTrafficMatrices:
    """测试 load_traffic_matrices 函数"""

    def test_load_matrices(self):
        """测试加载流量矩阵"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test matrices
            occ = np.random.rand(100, 10).astype(np.float32)
            spd = np.random.rand(100, 10).astype(np.float32)
            vol = np.random.rand(100, 10).astype(np.float32)
            node_order = np.arange(10)

            np.save(os.path.join(tmpdir, 'occupancy_2023_all.npy'), occ)
            np.save(os.path.join(tmpdir, 'speed_2023_all.npy'), spd)
            np.save(os.path.join(tmpdir, 'volume_2023_all.npy'), vol)
            np.save(os.path.join(tmpdir, 'node_order.npy'), node_order)

            matrices = load_traffic_matrices(tmpdir, use_mmap=False)

            assert matrices.occupancy.shape == (100, 10)
            assert matrices.speed.shape == (100, 10)
            assert matrices.volume.shape == (100, 10)
            assert len(matrices.node_order) == 10

    def test_load_matrices_with_mmap(self):
        """测试使用内存映射加载矩阵"""
        with tempfile.TemporaryDirectory() as tmpdir:
            occ = np.random.rand(100, 10).astype(np.float32)
            np.save(os.path.join(tmpdir, 'occupancy_2023_all.npy'), occ)
            np.save(os.path.join(tmpdir, 'speed_2023_all.npy'), occ)
            np.save(os.path.join(tmpdir, 'volume_2023_all.npy'), occ)

            matrices = load_traffic_matrices(tmpdir, use_mmap=True, load_node_order=False)
            assert matrices.occupancy.shape == (100, 10)

    def test_load_missing_file(self):
        """测试缺少必需文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_traffic_matrices(tmpdir)


class TestGetSensorIndex:
    """测试 get_sensor_index 函数"""

    def test_find_existing_sensor(self):
        """测试找到存在的传感器"""
        node_order = np.array([100, 200, 300, 400])
        idx = get_sensor_index(node_order, 200)
        assert idx == 1

    def test_find_first_sensor(self):
        """测试找到第一个传感器"""
        node_order = np.array([100, 200, 300])
        idx = get_sensor_index(node_order, 100)
        assert idx == 0

    def test_find_last_sensor(self):
        """测试找到最后一个传感器"""
        node_order = np.array([100, 200, 300])
        idx = get_sensor_index(node_order, 300)
        assert idx == 2

    def test_sensor_not_found(self):
        """测试找不到传感器"""
        node_order = np.array([100, 200, 300])
        idx = get_sensor_index(node_order, 999)
        assert idx is None


class TestValidateDataDirectory:
    """测试 validate_data_directory 函数"""

    def test_validate_with_all_files(self):
        """测试所有文件都存在"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create all required files
            files = [
                'incidents_y2023.csv',
                'sensor_meta_feature.csv',
                'occupancy_2023_all.npy',
                'speed_2023_all.npy',
                'volume_2023_all.npy',
                'node_order.npy'
            ]
            for f in files:
                Path(tmpdir, f).touch()

            result = validate_data_directory(tmpdir)
            assert all(result.values())

    def test_validate_with_missing_files(self):
        """测试缺少文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only some files
            Path(tmpdir, 'incidents_y2023.csv').touch()

            result = validate_data_directory(tmpdir)
            assert result['incidents_y2023.csv'] is True
            assert result['sensor_meta_feature.csv'] is False


class TestGetMatrixInfo:
    """测试 get_matrix_info 函数"""

    def test_get_info(self):
        """测试获取矩阵信息"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test matrices
            occ = np.random.rand(100, 10).astype(np.float32)
            np.save(os.path.join(tmpdir, 'occupancy_2023_all.npy'), occ)
            np.save(os.path.join(tmpdir, 'speed_2023_all.npy'), occ)
            np.save(os.path.join(tmpdir, 'volume_2023_all.npy'), occ)

            info = get_matrix_info(tmpdir)
            assert 'occupancy_2023_all' in info
            assert info['occupancy_2023_all'] == (100, 10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
