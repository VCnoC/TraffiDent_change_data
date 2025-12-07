# -*- coding: utf-8 -*-
"""
结果导出模块单元测试

测试 processors/exporter.py 中的所有类和函数
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.exporter import (
    ExportResult,
    ResultExporter,
    export_results,
)
from processors.scorer import IncidentScore, SensorScore
from processors.percentile_analyzer import (
    AnalysisResult,
    SensorAnalysisResult,
    PercentileResult,
)


class TestExportResult:
    """测试 ExportResult 数据类"""

    def test_success(self):
        """测试成功结果"""
        result = ExportResult(
            output_dir=Path("/tmp"),
            files_created=["file1.xlsx", "file2.xlsx"],
            errors=[]
        )

        assert result.success is True

    def test_failure_with_errors(self):
        """测试有错误的结果"""
        result = ExportResult(
            output_dir=Path("/tmp"),
            files_created=["file1.xlsx"],
            errors=["导出失败"]
        )

        assert result.success is False

    def test_failure_no_files(self):
        """测试无文件的结果"""
        result = ExportResult(
            output_dir=Path("/tmp"),
            files_created=[],
            errors=[]
        )

        assert result.success is False


class TestResultExporter:
    """测试 ResultExporter 类"""

    @pytest.fixture
    def temp_output_dir(self):
        """创建临时输出目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_incident_tables(self):
        """创建测试用事故数据表"""
        return [
            pd.DataFrame({
                'row_index': [0, 0, 0],
                'IncidentId': ['12345', '12345', '12345'],
                'SensorNumber': [0, 0, 0],
                'Data_Type': ['Occupancy_Rate', 'Speed', 'Volume'],
                'Data1': [0.5, 45.0, 80],
                'Data2': [0.6, 50.0, 90]
            })
        ]

    @pytest.fixture
    def sample_common_tables(self):
        """创建测试用前后天数据表"""
        return [
            pd.DataFrame({
                'row_index': [0, 0],
                'IncidentId': ['12345', '12345'],
                'Period': ['Before', 'After'],
                'Data1': [0.4, 0.5],
                'Data2': [0.5, 0.6]
            })
        ]

    @pytest.fixture
    def sample_scores(self):
        """创建测试用评分"""
        sensor_score = SensorScore(
            sensor_idx=0,
            occupancy_score=0.5,
            speed_score=0.4,
            volume_score=0.3,
            combined_score=0.42
        )

        return [
            IncidentScore(
                incident_id="12345",
                row_index=0,
                sensor_scores={0: sensor_score},
                overall_score=0.42,
                severity_level='MEDIUM',
                severity_label='中等影响'
            )
        ]

    @pytest.fixture
    def sample_analysis_results(self):
        """创建测试用分析结果"""
        occ_result = PercentileResult(
            data_type='occupancy',
            actual_values=np.array([0.5, 0.6]),
            percentile_values={50: np.array([0.4, 0.5])},
            deviations=np.array([0.25, 0.2]),
            anomaly_scores=np.array([0.5, 0.5])
        )

        sensor_result = SensorAnalysisResult(
            sensor_idx=0,
            occupancy=occ_result,
            speed=occ_result,
            volume=occ_result,
            overall_anomaly_score=0.5
        )

        return [
            AnalysisResult(
                incident_id="12345",
                sensor_results={0: sensor_result},
                errors=[]
            )
        ]

    def test_init(self, temp_output_dir):
        """测试初始化"""
        exporter = ResultExporter(output_dir=temp_output_dir)

        assert exporter.output_dir == Path(temp_output_dir)
        assert exporter.prefix == "A_"

    def test_init_creates_directory(self):
        """测试初始化创建目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "new_output")
            exporter = ResultExporter(output_dir=new_dir)

            assert os.path.exists(new_dir)

    def test_export_incident_data(self, temp_output_dir, sample_incident_tables):
        """测试导出事故数据"""
        exporter = ResultExporter(output_dir=temp_output_dir)
        result = exporter.export(incident_tables=sample_incident_tables)

        assert result.success is True
        assert any('final_output.xlsx' in f for f in result.files_created)

    def test_export_common_data(self, temp_output_dir, sample_common_tables):
        """测试导出前后天数据"""
        exporter = ResultExporter(output_dir=temp_output_dir)
        result = exporter.export(common_tables=sample_common_tables)

        assert result.success is True
        assert any('common_table.xlsx' in f for f in result.files_created)

    def test_export_scores(self, temp_output_dir, sample_scores):
        """测试导出评分"""
        exporter = ResultExporter(output_dir=temp_output_dir)
        result = exporter.export(scores=sample_scores)

        assert result.success is True
        assert any('score_table.xlsx' in f for f in result.files_created)

    def test_export_analysis(self, temp_output_dir, sample_analysis_results):
        """测试导出分析结果"""
        exporter = ResultExporter(output_dir=temp_output_dir)
        result = exporter.export(analysis_results=sample_analysis_results)

        assert result.success is True
        assert any('percentile_analysis.xlsx' in f for f in result.files_created)

    def test_export_errors(self, temp_output_dir):
        """测试导出错误记录"""
        errors = [
            {'row_index': 0, 'error': '匹配失败'},
            {'row_index': 1, 'error': '数据缺失'}
        ]

        exporter = ResultExporter(output_dir=temp_output_dir)
        result = exporter.export(errors=errors)

        assert result.success is True
        assert any('error_traffic_table.xlsx' in f for f in result.files_created)

    def test_export_all(
        self,
        temp_output_dir,
        sample_incident_tables,
        sample_common_tables,
        sample_scores,
        sample_analysis_results
    ):
        """测试导出所有数据"""
        exporter = ResultExporter(output_dir=temp_output_dir)
        result = exporter.export(
            incident_tables=sample_incident_tables,
            common_tables=sample_common_tables,
            scores=sample_scores,
            analysis_results=sample_analysis_results
        )

        assert result.success is True
        assert len(result.files_created) == 4

    def test_export_summary(self, temp_output_dir):
        """测试导出摘要"""
        exporter = ResultExporter(output_dir=temp_output_dir)

        start_time = datetime(2023, 6, 15, 10, 0, 0)
        end_time = datetime(2023, 6, 15, 10, 30, 0)

        path = exporter.export_summary(
            total_incidents=100,
            processed=100,
            successful=95,
            failed=5,
            start_time=start_time,
            end_time=end_time
        )

        assert os.path.exists(path)
        assert 'summary.xlsx' in str(path)

        # 验证内容
        df = pd.read_excel(path)
        assert df.iloc[0]['Total_Incidents'] == 100
        assert df.iloc[0]['Successful'] == 95
        assert df.iloc[0]['Failed'] == 5

    def test_export_with_custom_prefix(self, temp_output_dir, sample_incident_tables):
        """测试自定义前缀"""
        exporter = ResultExporter(output_dir=temp_output_dir, prefix="B_")
        result = exporter.export(incident_tables=sample_incident_tables)

        assert any('B_final_output.xlsx' in f for f in result.files_created)

    def test_export_empty_data(self, temp_output_dir):
        """测试空数据"""
        exporter = ResultExporter(output_dir=temp_output_dir)
        result = exporter.export()

        # 没有数据则没有文件创建
        assert len(result.files_created) == 0


class TestExportResults:
    """测试 export_results 便捷函数"""

    def test_convenience_function(self):
        """测试便捷函数"""
        with tempfile.TemporaryDirectory() as tmpdir:
            incident_tables = [
                pd.DataFrame({
                    'row_index': [0],
                    'IncidentId': ['12345'],
                    'Data1': [0.5]
                })
            ]

            result = export_results(
                output_dir=tmpdir,
                incident_tables=incident_tables
            )

            assert isinstance(result, ExportResult)
            assert result.success is True


class TestFileContent:
    """测试导出文件内容"""

    def test_incident_table_content(self):
        """测试事故表内容"""
        with tempfile.TemporaryDirectory() as tmpdir:
            incident_tables = [
                pd.DataFrame({
                    'row_index': [0, 1],
                    'IncidentId': ['12345', '67890'],
                    'SensorNumber': [0, 1],
                    'Data_Type': ['Occupancy_Rate', 'Speed'],
                    'Data1': [0.5, 45.0],
                    'Data2': [0.6, 50.0]
                })
            ]

            exporter = ResultExporter(output_dir=tmpdir)
            exporter.export(incident_tables=incident_tables)

            # 读取并验证
            output_path = Path(tmpdir) / "A_final_output.xlsx"
            df = pd.read_excel(output_path)

            assert len(df) == 2
            assert 'IncidentId' in df.columns
            assert df.iloc[0]['Data1'] == 0.5

    def test_score_table_content(self):
        """测试评分表内容"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sensor_score = SensorScore(
                sensor_idx=0,
                occupancy_score=0.5,
                speed_score=0.4,
                volume_score=0.3,
                combined_score=0.42
            )

            scores = [
                IncidentScore(
                    incident_id="12345",
                    row_index=0,
                    sensor_scores={0: sensor_score},
                    overall_score=0.42,
                    severity_level='MEDIUM',
                    severity_label='中等影响'
                )
            ]

            exporter = ResultExporter(output_dir=tmpdir)
            exporter.export(scores=scores)

            # 读取并验证
            output_path = Path(tmpdir) / "A_final_score_table.xlsx"
            df = pd.read_excel(output_path)

            assert len(df) == 1
            assert df.iloc[0]['ImpactScore'] == 0.42
            assert df.iloc[0]['SeverityLevel'] == 'MEDIUM'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
