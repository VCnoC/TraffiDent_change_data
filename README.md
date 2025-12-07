# 交通事故数据分析/预处理系统

面向批量交通事故的影响评估工具链，已将 MATLAB 逻辑迁移为 Python，并加入并行与共享内存优化，全部结果输出为 CSV（UTF-8-SIG）。适合一次性处理 47 万+ 事故。

## 你会得到什么
- 事故附近传感器智能匹配（高速公路/方向/里程容差）
- 事故时刻前后流量提取 + 前/后一天对照表
- 历史同期采样 + 百分位分析（P10/25/50/75/90）
- 影响评分与时段标签（早/午/晚高峰/其他）
- 并行处理 + 共享内存（P0 优化），可按 Type 分类输出
- 结果按 CSV 导出；附 Type 拆分辅助脚本

## 环境与数据
- Python 3.8+，建议使用虚拟环境
- 必要数据（默认 `--data-dir ../data`）：
  - `incidents_y2023.csv`（事故）
  - `sensor_meta_feature.csv`（传感器元数据）
  - `occupancy_2023_all.npy` / `speed_2023_all.npy` / `volume_2023_all.npy`（三大流量矩阵，支持 mmap）

## 快速开始
```bash
# 安装依赖
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 测试模式：前 10 条
python main.py --data-dir ../data --test

# 指定范围
python main.py --data-dir ../data --start 0 --end 1000

# 全量处理
python main.py --data-dir ../data --full

# 并行 + 共享内存（推荐大规模处理）
python main.py --data-dir ../data --full --parallel --workers 100

# 按 Type 分类写出子目录
python main.py --data-dir ../data --full --classify

# 仅保留/排除特定 Type（大小写与空格自动规范化）
python main.py --type-filter hazard --type-exclude construction
```

### 常用 Make 命令
```bash
make install   # 安装依赖
make test      # 运行测试
make test-cov  # 测试并生成覆盖率报告
make lint      # flake8 检查
make run       # 处理前 100 条
```

## 命令行参数（摘自 `main.py`）
| 参数 | 说明 | 默认值 |
|---|---|---|
| `--data-dir` | 数据目录 | `../data` |
| `--output-dir` | 输出目录 | `./output` |
| `--start` / `--end` | 起止索引（未指定 end 时默认 start+100） | `None` |
| `--time-window` | 时间窗口（时间步，5 分钟/步） | `12` |
| `--year` | 数据年份 | `2023` |
| `--batch-size` | 单批处理大小 | `100` |
| `--log-level` | 日志级别 | `INFO` |
| `--full` | 处理全部事故 | - |
| `--test` | 测试模式（前 10 条） | - |
| `--classify` | 按 Type 分类输出到子目录 | - |
| `--type-filter` | 仅处理指定 Type（可多次） | - |
| `--type-exclude` | 排除指定 Type（可多次） | - |
| `--parallel` | 启用多进程 + 共享内存 | - |
| `--workers` | 并行 worker 数（未填时=CPU核数-25，至少 4） | `auto` |
| `--max-cpus` | 限制可用 CPU 上限（容器环境） | `None` |
| `--no-shared-memory` | 禁用共享内存（调试用） | - |

## 输出文件（均为 CSV, UTF-8-SIG）
- `A_final_output.csv`：事故时刻数据
- `A_final_common_table.csv`：前/后一天对照数据（含 Kind 标记）
- `A_final_score_table.csv`：评分结果
- `A_percentile_analysis.csv`：百分位分析
- `A_error_traffic_table.csv`：失败记录
- `A_processing_summary.csv`：处理摘要
- 分类模式下：文件写入 `output/<type>/` 子目录

## 性能与并行建议
- P0 模式：`--parallel` + 默认共享内存，可 50~80x 加速；大机型可显式 `--workers 100`
- 共享内存自动启用（除非 `--no-shared-memory`），避免重复加载 3×14GB 矩阵
- 批大小可调（`--batch-size`），避免一次提交过多任务

## 附加工具
- Type 拆分现有汇总结果：
```bash
python scripts/split_full_results_by_type.py \
  --input-dir ../full_results_2025 \
  --incidents-file ../data/incidents_y2023.csv \
  --chunk-size 200000
```

## 常见问题
- **输出仍是 Excel 吗？** 不再使用 Excel，全部改为 CSV（utf-8-sig），WPS/Excel 可直接打开。
- **Type 大小写/空格不一致怎么办？** 程序会规范化 Type（去空格、非法字符，转小写），过滤和分类均使用规范化值。
- **共享内存占满怎么办？** 可降低 `--workers` 或加上 `--no-shared-memory` 转回 mmap；另可调整 `--batch-size`。

## 项目目录速览
```
traffic_analysis/
├── main.py                # CLI 入口，支持并行/分类
├── processors/            # 匹配、提取、采样、分析、评分、导出
├── data/                  # 数据加载
├── utils/time_utils.py    # 时间步工具
├── scripts/split_full_results_by_type.py
├── parallel_processor.py  # 多进程+共享内存实现
├── shared_memory_manager.py
├── requirements.txt
└── tests/                 # 单元测试
```

## 开发者提示
- 变更同步：文档详情以 `PROJECTWIKI.md` 为准；更新代码时记得对齐 README/CHANGELOG。
- 数据量大时优先使用并行模式；小样本调试可用 `--test`。
