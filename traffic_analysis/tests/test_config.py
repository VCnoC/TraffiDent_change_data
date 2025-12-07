# -*- coding: utf-8 -*-
"""
配置管理模块单元测试

测试 config.py 中的所有类和函数
"""

import pytest
import os
from pathlib import Path
import sys
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PathConfig,
    ProcessingConfig,
    MemoryConfig,
    Config,
    default_config,
    get_config,
)


class TestPathConfig:
    """测试 PathConfig 类"""

    def test_default_values(self):
        """测试默认值"""
        config = PathConfig()
        assert config.data_dir == Path("../data")
        assert config.output_dir == Path("./output")
        assert config.incidents_file == "incidents_y2023.csv"
        assert config.sensors_file == "sensor_meta_feature.csv"
        assert config.occupancy_file == "occupancy_2023_all.npy"
        assert config.speed_file == "speed_2023_all.npy"
        assert config.volume_file == "volume_2023_all.npy"

    def test_get_incidents_path(self):
        """测试获取事故数据路径"""
        config = PathConfig(data_dir=Path("/test/data"))
        path = config.get_incidents_path()
        assert path == Path("/test/data/incidents_y2023.csv")

    def test_get_sensors_path(self):
        """测试获取传感器数据路径"""
        config = PathConfig(data_dir=Path("/test/data"))
        path = config.get_sensors_path()
        assert path == Path("/test/data/sensor_meta_feature.csv")

    def test_get_output_path(self):
        """测试获取输出文件路径"""
        config = PathConfig(output_dir=Path("/test/output"))
        path = config.get_output_path("result.xlsx")
        assert path == Path("/test/output/result.xlsx")

    def test_ensure_output_dir(self):
        """测试确保输出目录存在"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "new_output"
            config = PathConfig(output_dir=output_path)
            config.ensure_output_dir()
            assert output_path.exists()


class TestProcessingConfig:
    """测试 ProcessingConfig 类"""

    def test_default_values(self):
        """测试默认值"""
        config = ProcessingConfig()
        assert config.time_window == 12
        assert config.data_year == 2023
        assert config.max_search_distance == 5.0
        assert config.max_sensors_per_incident == 5
        assert config.historical_weeks == 4
        assert config.percentile_thresholds == [10, 25, 50, 75, 90]
        assert config.score_weights == {
            "occupancy": 0.4,
            "speed": 0.4,
            "volume": 0.2
        }

    def test_total_timesteps_property(self):
        """测试总时间步数属性"""
        config = ProcessingConfig(time_window=12)
        assert config.total_timesteps == 25  # 2 * 12 + 1

        config2 = ProcessingConfig(time_window=6)
        assert config2.total_timesteps == 13  # 2 * 6 + 1

    def test_start_end_row(self):
        """测试起始和结束行索引"""
        config = ProcessingConfig(start_row=0, end_row=1000)
        assert config.start_row == 0
        assert config.end_row == 1000


class TestMemoryConfig:
    """测试 MemoryConfig 类"""

    def test_default_values(self):
        """测试默认值"""
        config = MemoryConfig()
        assert config.use_mmap is True
        assert config.batch_size == 1000
        assert config.show_progress is True
        assert config.log_level == "INFO"

    def test_custom_values(self):
        """测试自定义值"""
        config = MemoryConfig(
            use_mmap=False,
            batch_size=500,
            show_progress=False,
            log_level="DEBUG"
        )
        assert config.use_mmap is False
        assert config.batch_size == 500
        assert config.show_progress is False
        assert config.log_level == "DEBUG"


class TestConfig:
    """测试 Config 主类"""

    def test_default_values(self):
        """测试默认值"""
        config = Config()
        assert isinstance(config.paths, PathConfig)
        assert isinstance(config.processing, ProcessingConfig)
        assert isinstance(config.memory, MemoryConfig)

    def test_post_init_path_conversion(self):
        """测试初始化后路径转换"""
        # Create with string paths
        paths = PathConfig()
        paths.data_dir = "/test/data"
        paths.output_dir = "/test/output"
        config = Config(paths=paths)

        # Post-init should convert strings to Path
        assert isinstance(config.paths.data_dir, Path)
        assert isinstance(config.paths.output_dir, Path)

    def test_from_env(self):
        """测试从环境变量创建配置"""
        # Set environment variables
        os.environ["TRAFFIC_DATA_DIR"] = "/env/data"
        os.environ["TRAFFIC_OUTPUT_DIR"] = "/env/output"
        os.environ["TRAFFIC_TIME_WINDOW"] = "10"
        os.environ["TRAFFIC_DATA_YEAR"] = "2024"

        try:
            config = Config.from_env()
            assert config.paths.data_dir == Path("/env/data")
            assert config.paths.output_dir == Path("/env/output")
            assert config.processing.time_window == 10
            assert config.processing.data_year == 2024
        finally:
            # Clean up environment variables
            del os.environ["TRAFFIC_DATA_DIR"]
            del os.environ["TRAFFIC_OUTPUT_DIR"]
            del os.environ["TRAFFIC_TIME_WINDOW"]
            del os.environ["TRAFFIC_DATA_YEAR"]

    def test_validate_with_missing_data_dir(self):
        """测试验证缺失的数据目录"""
        config = Config()
        config.paths.data_dir = Path("/nonexistent/path")
        errors = config.validate()
        assert len(errors) > 0
        assert any("数据目录不存在" in e for e in errors)

    def test_validate_with_invalid_time_window(self):
        """测试验证无效的时间窗口"""
        config = Config()
        config.processing.time_window = 0
        errors = config.validate()
        assert any("时间窗口必须大于等于1" in e for e in errors)

    def test_validate_with_invalid_search_distance(self):
        """测试验证无效的搜索距离"""
        config = Config()
        config.processing.max_search_distance = 0
        errors = config.validate()
        assert any("最大搜索距离必须大于0" in e for e in errors)


class TestDefaultConfig:
    """测试默认配置实例"""

    def test_default_config_exists(self):
        """测试默认配置存在"""
        assert default_config is not None
        assert isinstance(default_config, Config)


class TestGetConfig:
    """测试 get_config 函数"""

    def test_returns_config(self):
        """测试返回配置"""
        config = get_config()
        assert isinstance(config, Config)

    def test_returns_env_config_when_env_set(self):
        """测试设置环境变量时返回环境配置"""
        os.environ["TRAFFIC_DATA_DIR"] = "/env/test"
        try:
            config = get_config()
            assert config.paths.data_dir == Path("/env/test")
        finally:
            del os.environ["TRAFFIC_DATA_DIR"]

    def test_returns_default_when_no_env(self):
        """测试没有环境变量时返回默认配置"""
        # Ensure no relevant env vars are set
        for key in ["TRAFFIC_DATA_DIR", "TRAFFIC_OUTPUT_DIR",
                    "TRAFFIC_TIME_WINDOW", "TRAFFIC_DATA_YEAR"]:
            if key in os.environ:
                del os.environ[key]

        config = get_config()
        # Should return default_config or equivalent
        assert config.paths.data_dir == default_config.paths.data_dir


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
