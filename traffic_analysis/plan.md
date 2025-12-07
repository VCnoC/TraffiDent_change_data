# Traffic Analysis 高性能优化计划

> 硬件配置：RTX 5090 × 5 | CPU 125核 | 内存 450GB

## 概述

针对 **476,768 条事故数据** 的高速处理优化方案。

---

## 1. 数据分片策略（5-GPU 并行）

- [ ] **任务分配**

| GPU 编号 | 起始索引 | 结束索引 | 数据量 |
|----------|----------|----------|--------|
| GPU 0    | 0        | 95,353   | 95,353 |
| GPU 1    | 95,354   | 190,706  | 95,353 |
| GPU 2    | 190,707  | 286,059  | 95,353 |
| GPU 3    | 286,060  | 381,412  | 95,353 |
| GPU 4    | 381,413  | 476,767  | 95,355 |

启动命令示例：
```bash
# GPU 0
python main.py --start 0 --end 95353 --output output_gpu0

# GPU 1
python main.py --start 95354 --end 190706 --output output_gpu1

# ... 以此类推
```

---

## 2. 多进程并行优化（CPU 125核）

- [x] **实现 `ProcessPoolExecutor`** ✅ 2024-12-06
  - 推荐 Worker 数量：**100** (预留 25 核给系统和 I/O)
  - 预期加速：**50-80x**
  - **已实现文件：** `parallel_processor.py`

- [x] 代码改造位置：`main.py` 的 `process_incidents()` 函数 ✅
  - **新增参数：** `--parallel` 启用并行模式
  - **新增参数：** `--workers` 指定 Worker 数量

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def process_incidents_parallel(incidents_df, num_workers=100):
    """多进程并行处理事故数据"""
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_single_incident, row)
            for _, row in incidents_df.iterrows()
        ]
        results = [f.result() for f in futures]
    return results
```

---

## 3. 内存预加载优化（450GB 内存）

- [x] **一次性加载所有 .npy 数据到内存** ✅ 2024-12-06
  - occupancy_2023_all.npy
  - speed_2023_all.npy
  - volume_2023_all.npy
  - 预期加速：**2-3x**（减少磁盘 I/O）
  - **已实现文件：** `shared_memory_manager.py`

- [x] **使用共享内存 (`multiprocessing.shared_memory`)** ✅ 2024-12-06
  - 避免进程间数据复制
  - 代码改造位置：数据加载模块
  - **已实现：** `SharedMemoryManager` 类管理共享内存生命周期

```python
import numpy as np
from multiprocessing import shared_memory

def preload_to_shared_memory(file_path, shm_name):
    """将 numpy 数组预加载到共享内存"""
    data = np.load(file_path)
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes, name=shm_name)
    shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    shared_array[:] = data[:]
    return shm, shared_array
```

---

## 4. 输出格式优化

- [x] **替换 Excel 为 CSV 格式** ✅ 2024-12-06
  - 预期 I/O 加速：**5-10x**
  - 文件体积减少：**30-50%**
  - **优势：WPS/Excel 可直接打开，兼容性好**
  - **已实现文件：** `processors/exporter.py`

- [x] 代码改造位置：`processors/exporter.py` ✅ 2024-12-06

```python
# 原 Excel 导出（慢）
# df.to_excel(output_path, index=False, engine='openpyxl')

# 现 CSV 导出（快 5-10 倍）
df.to_csv(output_path, index=False, encoding='utf-8-sig')
```

**输出文件变更：**
| 原文件名 | 新文件名 |
|---------|---------|
| A_final_output.xlsx | A_final_output.csv |
| A_final_score_table.xlsx | A_final_score_table.csv |
| A_final_common_table.xlsx | A_final_common_table.csv |
| A_percentile_analysis.xlsx | A_percentile_analysis.csv |
| A_error_traffic_table.xlsx | A_error_traffic_table.csv |
| A_processing_summary.xlsx | A_processing_summary.csv |

---

## 5. GPU 加速（RTX 5090 × 5）

- [ ] **引入 cuDF 替换 pandas**
  - 适用于数据筛选、聚合操作
  - 预期加速：**10-100x**

- [ ] **引入 cuPy 替换 numpy**
  - 适用于百分位计算、矩阵运算

```python
import cudf
import cupy as cp

# 使用 cuDF 读取数据
gdf = cudf.read_csv('incidents.csv')

# 使用 cuPy 进行数值计算
arr_gpu = cp.asarray(arr_cpu)
percentile = cp.percentile(arr_gpu, [25, 50, 75])
```

---

## 6. 综合预期效果

| 优化项 | 预期加速倍数 | 优先级 |
|--------|-------------|--------|
| 多进程并行 (100 workers) | 50-80x | P0 (必做) |
| 内存预加载 + 共享内存 | 2-3x | P0 (必做) |
| Parquet 替换 Excel | 10-50x (I/O) | P1 (推荐) |
| GPU cuDF/cuPy | 10-100x (计算) | P2 (可选) |

**综合加速：100-500x**

原处理时间（假设单进程 10 小时）→ 优化后预计 **6-60 分钟**

---

## 变更日志

| 日期 | 内容 | 状态 |
|------|------|------|
| 2024-12-06 | 初始计划制定 | 待实施 |

---

# 按 Type 拆分 full_results_2025 输出（2025-12-07）

> 需求：将 `full_results_2025` 中的 5 个汇总 CSV（A_final_output/common_table/score_table/percentile_analysis/error_traffic_table）按事故 Type 分组，生成各自子目录及同名子文件。

## 计划任务
- [x] 清点表头与关键列：确认 5 个文件的 Type/IncidentId 字段可用（error 表通过 incidents 映射补齐 Type）。
  完成：阅览 full_results_2025 五表表头，确认 Type 存在但 error 表缺失需映射。
  摘要：A_error_traffic_table 仅 incident_id/error，Type 要用 incidents_y2023.csv 补充。
- [x] 设计拆分方案：确定规范化规则、分块大小、输出目录结构 `<type>/<原文件名>`，空类别处理。
  完成：采用 normalize_type 去空/非法字符转小写，分块默认 200k，覆盖式输出。
  摘要：默认不生成空文件，目录保持 input-dir/类型，同名文件写入。
- [x] 实现脚本/模块：新增独立脚本或工具函数，支持参数（input-dir、incidents、chunk-size），覆盖 5 个文件。
  完成：新增 `scripts/split_full_results_by_type.py`，支持参数与行数汇总。
  摘要：按 Type 分组分块写入，error 表通过 incidents 映射 Type，UTF-8-SIG 输出。
- [x] 校验与测试：准备小样本单测验证行数一致、空类型行为；必要时做一次真实目录跑通的行数对齐检查。
  完成：新增 `tests/test_split_full_results_by_type.py`，pytest 单测通过。
  摘要：验证三类型拆分、error 表 Type 映射、行数一致。
- [x] 文档更新：在 PROJECTWIKI.md 使用指南/模块文档补充拆分脚本说明；CHANGELOG.md Unreleased 记录新增功能。
  完成：PROJECTWIKI.md 增加 9.4 拆分脚本用法示例；CHANGELOG.md Unreleased 新增脚本与单测条目。
  摘要：文档对齐最新功能，日志记录新增脚本和测试。

## 评审/备注
- 简单按 Type 分组，保持原列顺序与编码（utf-8-sig），默认覆盖已存在的目标文件。
- error 表缺失 Type 时，优先用 incidents_y2023.csv 的 incident_id→Type 映射；缺失则归入 `unknown`。

---

# 并行与内存问题修复计划（2025-12-07）

## 任务列表
- [x] 阅读并确认现状：评估共享内存 GC/泄漏、背压缺失、日志栈缺失（完成摘要：确定 attach 句柄保存但未在 worker 退出时 close，任务提交无背压）
- [x] 任务分批与背压：在 `ParallelProcessor.process_incidents` 引入有界队列/分批提交，避免一次性排队 47 万任务占满内存（完成摘要：使用 queue_limit=workers*2 的增量提交，as_completed 补投）
- [x] 动态 worker 数：默认值改为基于 `cpu_count()-4` 的动态计算，可通过参数覆盖，并记录到日志（完成摘要：auto 来源记录在日志 source=auto/user）
- [x] 共享内存关闭钩子：worker 退出前调用 `_worker_matrices.close()`，仅 close 不 unlink，防止句柄泄漏（完成摘要：worker 初始化注册 atexit 关闭映射）
- [x] 异常日志堆栈：`_process_single_incident` 与 future 捕获处记录 `traceback.format_exc()`，保留 incident_id/row_index（完成摘要：错误返回带 traceback 字符串）
- [x] 文档与变更同步：更新 `PROJECTWIKI.md`（补齐 API/章节编号，记录并行/共享内存变更）、`CHANGELOG.md`（Unreleased 添加本次修复），补充本计划完成状态（完成摘要：新增 API 手册章节、顺序编号 14 节，Changelog 记录并行/GC/traceback 变更）

## 验收标准（DoD）
- 并行任务提交峰值内存不再与任务总数线性增长（通过分批提交实现）。
- 默认 worker 数随 CPU 自动调整；用户可覆盖，日志显示最终值。
- worker 退出时共享内存映射关闭，无悬挂句柄（可通过小规模运行后检查）。
- 异常日志包含堆栈，错误定位信息含 incident_id/row_index。
- `PROJECTWIKI.md` 至少具备模板 12 个一级章节，新增并行/共享内存设计说明；`CHANGELOG.md` 记录本次修复。

---

## 7. Gemini 3 深度可行性分析

> 分析日期：2024-12-06 | 模型：gemini-3-pro-preview-thinking | 置信度：High

### 7.1 实际数据规模（关键发现）

| 数据文件 | 大小 | 备注 |
|---------|------|------|
| occupancy_2023_all.npy | **14 GB** | 占用率矩阵 |
| speed_2023_all.npy | **14 GB** | 速度矩阵 |
| volume_2023_all.npy | **14 GB** | 流量矩阵 |
| **总计** | **42 GB** | 仅占 450GB 内存的 **9.3%** |

**结论：450GB 内存完全足够一次性加载所有核心数据！**

### 7.2 优化方案可行性评估

| 优化方案 | 可行性 | 预期加速 | 实施难度 | 优先级 |
|----------|--------|----------|----------|--------|
| 多进程并行 (100 Workers) | ✅ 极高 | 50-80x | 低 | **P0** |
| 内存预加载 + 共享内存 | ✅ 极高 | 2-3x | 低 | **P0** |
| Parquet 替换 Excel | ✅ 高 | 10-50x (I/O) | 低 | **P1** |
| GPU cuDF/cuPy | ⚠️ 中等 | 10-100x | **高** | P2 |
| 5-GPU 数据分片 | ✅ 高 | 5x | 低 | **P0** |

### 7.3 瓶颈分析

```
当前瓶颈排序：
┌─────────────────────────────────────────────┐
│ 1. 🔴 CPU 串行处理 (主要)                    │
│    - main.py 使用 for 循环逐条处理            │
│    - 125 核只用了 1 核                        │
├─────────────────────────────────────────────┤
│ 2. 🟠 Excel 导出 I/O (次要)                  │
│    - openpyxl 写入速度慢                      │
│    - 476K 条数据导出耗时严重                  │
├─────────────────────────────────────────────┤
│ 3. 🟢 内存未充分利用                          │
│    - 42GB 数据，450GB 内存                   │
│    - 建议全部预加载到内存                     │
└─────────────────────────────────────────────┘
```

### 7.4 推荐实施顺序

**阶段 1 (P0 - 必做，预期 50-100x)：**
1. [x] 多进程并行改造 (100 Workers)
2. [x] 42GB 数据一次性预加载到内存
3. [x] 使用 `multiprocessing.shared_memory` 避免进程间复制

**阶段 2 (P1 - 推荐，额外 10-50x I/O 加速)：**
4. [ ] Excel → Parquet 输出格式

**阶段 3 (P2 - 可选，需大量改造)：**
5. [ ] GPU cuDF/cuPy 加速（ROI 待评估）

### 7.5 风险提示

| 风险点 | 说明 | 缓解措施 |
|--------|------|----------|
| 进程间数据复制开销 | 多进程可能复制大数组 | 使用 shared_memory |
| GPU 数据传输开销 | CPU↔GPU 传输可能抵消收益 | 先 Profile 再决定 |
| 下游兼容性 | Parquet 格式需确认下游支持 | 保留 Excel 导出选项 |

### 7.6 预期效果对比

```
原始方案（单进程）:
  476,768 条 × ~0.5秒/条 ≈ 66 小时

优化后（P0 + P1）:
  多进程 100x + 内存预加载 3x + Parquet 10x
  ≈ 66 小时 / (100 × 3) ≈ 13 分钟

保守估计: 30-60 分钟完成全部数据
```

---

## 变更日志

| 日期 | 内容 | 状态 |
|------|------|------|
| 2024-12-06 | 初始计划制定 | 待实施 |
| 2024-12-06 | 添加 Gemini 3 深度可行性分析 | ✅ 完成 |
| 2024-12-06 | **P0 阶段实现完成** | ✅ 完成 |
|            | - 创建 `shared_memory_manager.py` 共享内存管理模块 | |
|            | - 创建 `parallel_processor.py` 多进程并行处理模块 | |
|            | - 修改 `main.py` 添加 `--parallel`、`--workers` 参数 | |
|            | - 预期加速: 50-100x (100 Workers + 共享内存) | |
| 2024-12-06 | **P1 输出格式优化完成** | ✅ 完成 |
|            | - 修改 `processors/exporter.py` 将 Excel 改为 CSV | |
|            | - 预期 I/O 加速: 5-10x | |
|            | - 兼容 WPS/Excel 直接打开 | |

---

## 评审备注

**Gemini 3 分析结论：**
- 硬件配置完全满足需求，42GB 数据仅占 9.3% 内存
- 主要瓶颈：串行处理 + Excel I/O
- 推荐优先实施 P0（多进程 + 内存预加载），性价比最高
- GPU 加速需要较大代码改造，建议在 P0/P1 实施后再评估 ROI
