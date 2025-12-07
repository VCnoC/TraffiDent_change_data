# 交通事故数据分析系统

基于 Python 的交通事故影响评估系统，用于分析事故对周边交通流量的影响程度。

## 项目简介

本系统通过匹配事故位置附近的交通传感器，提取事故前后的流量数据，并与历史同期数据进行对比分析，计算事故影响评分。

**主要功能：**
- 传感器智能匹配（基于地理位置和道路拓扑）
- 事故时刻流量数据提取
- 历史同期数据采样
- 百分位统计分析
- 事故影响评分计算
- 分类处理与批量导出

## 快速开始

### 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 基本使用

```bash
# 处理所有事故
python main.py --data-dir ../data --output-dir ./output --full

# 测试模式（仅处理前10个事故）
python main.py --test

# 处理指定范围
python main.py --start 0 --end 1000

# 按类型分类处理
python main.py --classify --full

# 过滤特定类型
python main.py --type-filter accident --type-filter hazard
```

## 开发与构建

### 运行测试

```bash
# 运行所有测试
pytest

# 运行测试并生成覆盖率报告
pytest --cov=. --cov-report=html

# 运行特定测试文件
pytest tests/test_sensor_matcher.py
```

### 代码检查

```bash
# 使用 flake8 检查代码风格
flake8 . --max-line-length=100 --exclude=venv,__pycache__

# 使用 pylint 进行静态分析
pylint traffic_analysis
```

### 使用 Makefile（推荐）

```bash
# 查看所有可用命令
make help

# 安装依赖
make install

# 运行测试
make test

# 运行测试并生成覆盖率报告
make test-cov

# 代码检查
make lint

# 清理临时文件
make clean

# 运行程序（处理前100个事故）
make run

# 测试模式（仅10个事故）
make run-test

# 完整处理所有事故
make run-full

# 分类处理所有事故
make run-classify
```

### 项目结构

```
traffic_analysis/
├── main.py                 # 主程序入口
├── config.py              # 配置管理
├── requirements.txt       # 依赖列表
├── data/                  # 数据加载模块
│   └── loader.py
├── processors/            # 核心处理模块
│   ├── sensor_matcher.py      # 传感器匹配
│   ├── traffic_extractor.py   # 流量数据提取
│   ├── historical_sampler.py  # 历史数据采样
│   ├── percentile_analyzer.py # 百分位分析
│   ├── scorer.py              # 评分计算
│   └── exporter.py            # 结果导出
├── utils/                 # 工具函数
│   └── time_utils.py
└── tests/                 # 测试文件
    ├── test_*.py
    └── fixtures/
```

## 常见问题

### Q: 如何处理大规模数据？

A: 使用 `--parallel` 启用多进程，默认 worker 数自动设为 `CPU核数-4`；内部带背压（有界队列）避免一次性提交全部任务。可结合 `--workers` 精确控制并发；如仍受限，可配合 `--batch-size` 减小批次：

```bash
python main.py --batch-size 50 --full
```

### Q: 如何只处理特定类型的事故？

A: 使用 `--type-filter` 参数：

```bash
python main.py --type-filter accident --type-filter collision
```

### Q: 输出文件在哪里？

A: 默认输出到 `./output` 目录，可通过 `--output-dir` 参数自定义：

```bash
python main.py --output-dir ./results
```

输出文件包括：
- `A_final_output.xlsx` - 完整结果
- `A_final_score_table.xlsx` - 评分汇总
- `A_percentile_analysis.xlsx` - 百分位分析
- `A_processing_summary.xlsx` - 处理摘要

### Q: 如何调试处理失败的事故？

A: 使用 `--log-level DEBUG` 查看详细日志：

```bash
python main.py --log-level DEBUG --start 0 --end 10
```

失败记录会保存在 `A_error_traffic_table.xlsx` 中。

### Q: 测试覆盖率如何查看？

A: 运行测试后，打开 `htmlcov/index.html` 查看详细的覆盖率报告：

```bash
make test-cov
# 然后在浏览器中打开 htmlcov/index.html
```

### Q: 如何清理临时文件？

A: 使用 make 命令清理：

```bash
make clean
```

## 命令行参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data-dir` | 数据目录路径 | `../data` |
| `--output-dir` | 输出目录路径 | `./output` |
| `--start` | 起始事故索引 | `0` |
| `--end` | 结束事故索引 | `100` |
| `--time-window` | 时间窗口大小（时间步数） | `12` |
| `--parallel` | 是否启用多进程处理 | `False` |
| `--workers` | 并行 worker 数；不填则自动 = CPU 核数 - 4（最少 1） | `auto` |
| `--batch-size` | 批处理大小 | `100` |
| `--log-level` | 日志级别 | `INFO` |
| `--full` | 处理所有事故 | - |
| `--test` | 测试模式（仅10个事故） | - |
| `--classify` | 按类型分类处理 | - |
| `--type-filter` | 只处理指定类型 | - |
| `--type-exclude` | 排除指定类型 | - |

## 许可证

本项目仅供学术研究使用。

## 贡献

欢迎提交 Issue 和 Pull Request。
