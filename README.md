# Traffic Analysis - 交通事故数据预处理系统

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 将 MATLAB 交通事故数据预处理代码重构为 Python 实现，支持大规模数据处理（47 万+ 事故）和多进程并行计算，全部结果输出为 CSV（UTF-8-SIG）。

## 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [使用方法](#使用方法)
- [命令行参数](#命令行参数)
- [数据说明](#数据说明)
- [输出文件](#输出文件)
- [性能指标](#性能指标)
- [开发指南](#开发指南)
- [常见问题](#常见问题)
- [更多文档](#更多文档)

## 项目简介

本系统用于处理加州交通事故数据，主要功能包括：

1. **传感器匹配** - 根据事故位置（高速公路/方向/里程标）匹配附近的交通传感器
2. **流量数据提取** - 提取事故时刻前后的流量数据（占用率、速度、流量）
3. **历史数据采样** - 采样同一星期几的历史对比数据
4. **百分位分析** - 将当前数据与历史数据进行百分位对比（P10/P25/P50/P75/P90）
5. **影响评分** - 计算事故对交通的影响程度评分与时段标签（早/午/晚高峰）
6. **结果导出** - 生成 CSV 格式的分析报告，支持按事故类型分类输出

## 功能特性

- **高性能处理** - 支持多进程并行处理，可利用 100+ CPU 核心，50-80x 加速
- **内存优化** - 使用内存映射 (mmap) 处理 42GB+ 流量矩阵
- **共享内存** - 多进程间零拷贝数据共享，避免重复加载
- **分类处理** - 支持按事故类型（Type）分类输出结果
- **批量处理** - 支持指定范围和批次大小
- **类型规范化** - 自动处理 Type 字段的大小写和空格问题
- **完整日志** - 详细的处理进度和错误日志

## 项目结构

```
kaggle数据集/
├── README.md                  # 本文件
├── traffic_analysis/          # 主程序目录
│   ├── main.py                # 主程序入口（CLI）
│   ├── config.py              # 配置管理模块
│   ├── parallel_processor.py  # 多进程并行处理器
│   ├── shared_memory_manager.py # 共享内存管理器
│   ├── requirements.txt       # Python 依赖
│   ├── Makefile               # 常用命令快捷方式
│   ├── PROJECTWIKI.md         # 详细项目知识库
│   ├── CHANGELOG.md           # 版本变更日志
│   │
│   ├── data/                  # 数据加载模块
│   │   └── loader.py          # 数据加载器（支持 mmap）
│   │
│   ├── processors/            # 核心处理模块
│   │   ├── sensor_matcher.py      # 传感器匹配
│   │   ├── traffic_extractor.py   # 流量数据提取
│   │   ├── historical_sampler.py  # 历史数据采样
│   │   ├── percentile_analyzer.py # 百分位分析
│   │   ├── scorer.py              # 评分计算
│   │   └── exporter.py            # 结果导出
│   │
│   ├── utils/                 # 工具模块
│   │   └── time_utils.py      # 时间处理工具
│   │
│   ├── scripts/               # 辅助脚本
│   │   └── split_full_results_by_type.py  # 结果按类型拆分
│   │
│   ├── tests/                 # 测试用例
│   └── output/                # 输出目录（自动创建）
│
└── data/                      # 数据目录（需自行准备）
    ├── incidents_y2023.csv
    ├── sensor_meta_feature.csv
    ├── occupancy_2023_all.npy
    ├── speed_2023_all.npy
    ├── volume_2023_all.npy
    └── node_order.npy
```

## 快速开始

### 1. 安装依赖

```bash
cd traffic_analysis

# 推荐使用虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. 准备数据

确保数据目录（默认 `../data`）包含以下文件：

| 文件名 | 说明 | 大小 |
|--------|------|------|
| `incidents_y2023.csv` | 事故数据（476,768 条） | ~100MB |
| `sensor_meta_feature.csv` | 传感器元数据（16,972 个） | ~2MB |
| `occupancy_2023_all.npy` | 占用率矩阵 (105120, 16972) | ~14GB |
| `speed_2023_all.npy` | 速度矩阵 (105120, 16972) | ~14GB |
| `volume_2023_all.npy` | 流量矩阵 (105120, 16972) | ~14GB |
| `node_order.npy` | 节点顺序 | ~1MB |

### 3. 运行测试

```bash
# 测试模式（处理前 10 个事故）
python main.py --data-dir ../data --test
```

## 使用方法

### 基本用法

```bash
# 处理指定范围的事故
python main.py --data-dir ../data --start 0 --end 1000

# 完整处理所有事故
python main.py --data-dir ../data --full
```

### 并行处理（推荐用于大规模数据）

```bash
# 启用多进程并行处理（自动检测 CPU 核心数）
python main.py --data-dir ../data --full --parallel

# 指定 Worker 数量
python main.py --data-dir ../data --full --parallel --workers 100

# 在容器环境中指定实际 CPU 核心数
python main.py --data-dir ../data --full --parallel --max-cpus 128
```

### 分类处理

```bash
# 按事故类型分类处理，每个类型输出到单独文件夹
python main.py --data-dir ../data --full --classify

# 只处理特定类型的事故（大小写自动规范化）
python main.py --data-dir ../data --full --type-filter fire --type-filter hazard

# 排除特定类型
python main.py --data-dir ../data --full --type-exclude other --type-exclude construction
```

### 使用 Makefile

```bash
make install   # 安装依赖
make test      # 运行测试
make test-cov  # 测试并生成覆盖率报告
make lint      # flake8 代码检查
make run       # 处理前 100 条
make clean     # 清理输出文件
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | `../data` | 数据目录路径 |
| `--output-dir` | `./output` | 输出目录路径 |
| `--start` | `0` | 起始事故索引 |
| `--end` | `100` | 结束事故索引（未指定时默认 start+100） |
| `--time-window` | `12` | 时间窗口大小（时间步数，12 = 前后各 1 小时） |
| `--year` | `2023` | 数据年份 |
| `--batch-size` | `100` | 批处理大小 |
| `--log-level` | `INFO` | 日志级别（DEBUG/INFO/WARNING/ERROR） |
| `--test` | - | 测试模式（只处理前 10 个事故） |
| `--full` | - | 完整模式（处理所有事故） |
| `--parallel` | - | 启用多进程并行处理 |
| `--workers` | 自动 | 并行 Worker 数量（默认：CPU 核心数 - 25，最小 4） |
| `--max-cpus` | - | 实际可用 CPU 核心数（容器环境限制） |
| `--classify` | - | 按事故类型分类输出到子目录 |
| `--type-filter` | - | 只处理指定类型（可多次使用） |
| `--type-exclude` | - | 排除指定类型（可多次使用） |
| `--no-shared-memory` | - | 禁用共享内存优化（调试用） |

## 数据说明

### 时间定义

- **1 时间步** = 5 分钟
- **1 天** = 288 时间步
- **1 年** = 105,120 时间步（365 天）
- **时间窗口 12** = 前后各 12 个时间步 = 前后各 1 小时 = 共 25 个数据点

### 评分严重程度等级

| 等级 | 评分范围 | 标签 |
|------|---------|------|
| NONE | 0.0-0.1 | 无影响 |
| LOW | 0.1-0.3 | 轻微影响 |
| MEDIUM | 0.3-0.5 | 中等影响 |
| HIGH | 0.5-0.7 | 较大影响 |
| SEVERE | 0.7-1.0 | 严重影响 |

### 时段标签 (Time_tag)

| 标签 | 时段 | 说明 |
|------|------|------|
| 1 | 7:00-9:00 | 早高峰 |
| 2 | 11:00-13:00 | 午间 |
| 3 | 17:00-20:00 | 晚高峰 |
| 0 | 其他时段 | 非高峰 |

## 输出文件

处理完成后，输出目录包含以下文件（均为 CSV，UTF-8-SIG 编码）：

| 文件名 | 说明 |
|--------|------|
| `A_final_output.csv` | 事故时刻数据（每传感器 3 行：占用率/速度/流量） |
| `A_final_common_table.csv` | 前/后一天对照数据（含 Kind 标记） |
| `A_final_score_table.csv` | 事故影响评分表 |
| `A_percentile_analysis.csv` | 百分位分析结果 |
| `A_error_traffic_table.csv` | 处理错误记录 |
| `A_processing_summary.xlsx` | 处理摘要统计 |

### 分类输出结构

使用 `--classify` 时，按事故类型分目录输出：

```
output/
├── fire/
│   ├── A_final_output.csv
│   ├── A_final_common_table.csv
│   ├── A_final_score_table.csv
│   ├── A_percentile_analysis.csv
│   └── A_error_traffic_table.csv
├── hazard/
│   └── ...
├── other/
│   └── ...
└── A_processing_summary.xlsx
```

## 性能指标

| 指标 | 串行模式 | 并行模式（100 Workers） |
|------|----------|-------------------------|
| 单事故处理时间 | ~2.4 秒 | ~0.03 秒 |
| 内存占用 | ~500 MB | ~2 GB |
| 47 万事故预估时间 | ~13 天 | ~4 小时 |
| 加速比 | 1x | 50-80x |

### 性能优化建议

- **大规模处理**：使用 `--parallel` + 默认共享内存，可获得 50-80x 加速
- **大型服务器**：显式指定 `--workers 100` 或更高
- **内存紧张**：降低 `--workers` 数量或使用 `--no-shared-memory` 转回 mmap 模式
- **任务队列**：调整 `--batch-size` 避免一次提交过多任务

## 开发指南

### 运行测试

```bash
cd traffic_analysis

# 运行所有测试
pytest tests/ -v

# 运行测试并生成覆盖率报告
pytest tests/ --cov=. --cov-report=html

# 查看覆盖率报告
open htmlcov/index.html
```

### 环境变量配置

支持通过环境变量配置：

```bash
export TRAFFIC_DATA_DIR=/path/to/data
export TRAFFIC_OUTPUT_DIR=/path/to/output
export TRAFFIC_TIME_WINDOW=12
export TRAFFIC_DATA_YEAR=2023
```

### 代码模块对照

| Python 模块 | MATLAB 对应 | 功能 |
|-------------|-------------|------|
| `data/loader.py` | - | 数据加载（内存映射） |
| `utils/time_utils.py` | `isLeapYear`, `convertToISOWeekday` | 时间处理工具 |
| `processors/sensor_matcher.py` | `Traffic_Function.m` | 传感器匹配 |
| `processors/traffic_extractor.py` | `p_trafficdata_function.m` | 流量提取 |
| `processors/historical_sampler.py` | `sort_function.m` | 历史采样 |
| `processors/percentile_analyzer.py` | `data_clean.m` | 百分位分析 |
| `processors/scorer.py` | `processGroup.m` | 评分计算 |
| `processors/exporter.py` | - | 结果导出 |

## 常见问题

### Q: 输出还是 Excel 格式吗？
**A:** 不是，全部改为 CSV（UTF-8-SIG 编码），可直接用 WPS/Excel 打开，不会乱码。

### Q: Type 字段大小写/空格不一致怎么办？
**A:** 程序会自动规范化 Type（去除空格和非法字符、转换为小写），过滤和分类均使用规范化后的值。例如 `"Fire  "` 会被规范化为 `"fire"`。

### Q: 共享内存占满怎么办？
**A:** 可以：
1. 降低 `--workers` 数量
2. 使用 `--no-shared-memory` 转回 mmap 模式
3. 调整 `--batch-size` 减少并发任务数

### Q: 如何只处理特定类型的事故？
**A:** 使用 `--type-filter` 参数：
```bash
python main.py --data-dir ../data --full --type-filter hazard --type-filter fire
```

### Q: 如何拆分已有的汇总结果？
**A:** 使用附带的拆分脚本：
```bash
python scripts/split_full_results_by_type.py \
  --input-dir ../full_results_2025 \
  --incidents-file ../data/incidents_y2023.csv \
  --chunk-size 200000
```

### Q: 数据量太大，处理太慢怎么办？
**A:**
1. 使用并行模式：`--parallel --workers 100`
2. 先用测试模式验证：`--test`
3. 分批处理：`--start 0 --end 10000`

## 更多文档

- [traffic_analysis/PROJECTWIKI.md](./traffic_analysis/PROJECTWIKI.md) - 详细的项目知识库（架构设计、API 文档、ADR 等）
- [traffic_analysis/CHANGELOG.md](./traffic_analysis/CHANGELOG.md) - 版本变更日志
- [traffic_analysis/plan.md](./traffic_analysis/plan.md) - 开发计划和任务跟踪

## 许可证

MIT License

---

> 本项目源自 MATLAB 代码迁移，保持与原有逻辑的一致性。详细技术文档请参阅 [PROJECTWIKI.md](./traffic_analysis/PROJECTWIKI.md)。
