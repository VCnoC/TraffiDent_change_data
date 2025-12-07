# 变更日志 (Changelog)

所有重要变更均记录于此文件。

本文件格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
并遵循 [语义化版本号](https://semver.org/lang/zh-CN/) 规范。

## [Unreleased]

### Added（新增）
- **scripts/split_full_results_by_type.py**: 新增按 Type 将 `full_results_2025` 五个汇总 CSV 拆分到各自子目录的脚本，支持分块与 incidents 映射补 Type。
- **tests/test_split_full_results_by_type.py**: 覆盖拆分脚本的小样本单测，验证分类输出与错误表 Type 映射。

### Changed（变更）
- **main.py**: 将分类逻辑切换为 `group_incidents_by_type`，直接使用 `Type` 列分组；缺失 Type 列或空值回退 `other`，日志文案改为 Type；保留 `get_type_value` 进行空白/非法字符清洗以确保目录安全。
- **CLI**: `--classify` 帮助文本更新为“按 Type 字段分类处理”，与当前分类逻辑保持一致。
- **main.py**: 数据加载后新增 `Type_normalized` 列（保留原始 `Type`），过滤与分类统一使用规范化值；过滤参数同样规范化以避免空格/大小写/非法字符导致的不一致。
- **main.py / CLI**: `Type_normalized` 结果以及过滤/分组输入统一转为小写；日志与 CLI 帮助明确“规范化并转小写”，避免 `Fire`/`fire`/`FIRE` 等大小写差异导致过滤失效。
- **parallel_processor.py**: 默认 worker 数改为自动 `CPU核数-4`（最少 1），日志记录来源；并行任务采用有界队列分批提交（上限约 `workers*2`），避免一次性提交 47 万任务导致内存峰值。
- **PROJECTWIKI.md / README.md**: 补充并行/共享内存接口说明，新增 API 手册章节，更新 CLI 表中 `--parallel`、`--workers` 参数及背压说明。

### Fixed（修复）
- **processors/traffic_extractor.py**: 为 `common_table` 补充 `Period` 列（Before/After），与前后天数据标记和测试预期一致。
- **parallel_processor.py**: worker 异常和 future 异常现在返回完整 traceback，附带 incident_id/row_index；并行收集过程中不再一次性占用 tasks 列表内存。
- **shared_memory_manager.py / parallel_processor.py**: worker 进程在退出时主动 `close()` 共享内存映射，避免句柄泄漏；attach 返回的句柄在对象生命周期内持有，防 GC 回收。
- **shared_memory_manager.py**: 共享内存创建/附加失败改用 `logger.exception` 输出完整堆栈，同时保留异常清理逻辑，便于排查资源释放问题。

## [1.0.2] - 2024-12-03

### Fixed (修复)

- **processors/sensor_matcher.py**: 重大修复 - 完全重写 `_find_adjacent_sensors()` 方法以匹配 MATLAB 逻辑
  - **问题**：之前错误地使用了 0.5 英里容差匹配，这并非原始 MATLAB 代码的行为
  - **原因**：错误理解 MATLAB `Traffic_Function.m` 第 82-145 行的逻辑
  - **修复内容**：
    1. 移除错误的 0.5 英里容差判断
    2. 实现 MATLAB 的精确匹配逻辑：
       - 找到最近传感器的 PM 值 (`closest_pm`)
       - 向上搜索：收集所有 `Abs PM == closest_pm` 的传感器
       - 向下搜索：收集所有 `Abs PM == closest_pm_xx` 的传感器（下一个不同的 PM 值）
    3. 同一位置的多个传感器（如不同车道）现在能正确匹配
    4. 相邻位置的传感器（closest_pm_xx）也会被包含
  - **影响范围**：`_find_adjacent_sensors()` 方法
  - **对应 MATLAB**：`Traffic_Function.m` 第 119-145 行

## [1.0.1] - 2024-12-03

### Added (新增)

- **tests/**: 添加完整的单元测试套件
  - 183 个测试用例
  - 核心模块覆盖率 92%
  - 覆盖所有处理器模块和工具模块

## [1.0.0] - 2024-12-03

### Added (新增)

#### 核心模块
- **data/loader.py**: 数据加载模块
  - 支持内存映射 (mmap) 加载大型矩阵
  - 自动处理 Tab 分隔的 CSV 文件
  - `TrafficMatrices` 命名元组封装矩阵数据

- **utils/time_utils.py**: 时间工具模块
  - `datetime_to_timestep()`: 日期时间转时间步索引
  - `timestep_to_datetime()`: 时间步索引转日期时间
  - `get_iso_weekday()`: 获取 ISO 标准星期几
  - MATLAB 1-based 到 Python 0-based 索引转换

- **config.py**: 配置管理模块
  - `PathConfig`: 路径配置
  - `ProcessingConfig`: 处理参数配置
  - `MemoryConfig`: 内存管理配置
  - 支持环境变量配置

#### 处理器模块
- **processors/sensor_matcher.py**: 传感器匹配模块
  - 对应 MATLAB `Traffic_Function.m`
  - 按 Fwy + Direction + Abs PM 匹配传感器
  - 支持相邻传感器容差匹配

- **processors/traffic_extractor.py**: 流量数据提取模块
  - 对应 MATLAB `p_trafficdata_function.m`
  - 提取事故时刻前后流量数据
  - 提取前后天对照数据

- **processors/historical_sampler.py**: 历史数据采样模块
  - 对应 MATLAB `sort_function.m`
  - 采样同一星期几的历史数据
  - 排除事故当天及前后日期

- **processors/percentile_analyzer.py**: 百分位分析模块
  - 对应 MATLAB `data_clean.m`
  - 计算 P10, P25, P50, P75, P90 百分位
  - 异常得分计算

- **processors/scorer.py**: 评分计算模块
  - 对应 MATLAB `processGroup.m`
  - 加权综合评分 (occ:0.4, spd:0.4, vol:0.2)
  - 五级严重程度分类

- **processors/exporter.py**: 结果导出模块
  - Excel 文件导出
  - 兼容原 MATLAB 输出格式

#### 主程序
- **main.py**: 主程序入口
  - 对应 MATLAB `main.m`
  - 命令行参数支持
  - 批处理模式
  - 进度日志

### Technical Details (技术细节)

- **索引转换**: 所有时间步索引从 MATLAB 1-based 转换为 Python 0-based
- **内存优化**: 使用 numpy mmap 模式加载 ~14GB 矩阵
- **性能**: 单事故处理约 2.4 秒，支持批量处理
- **兼容性**: 输出 Excel 格式与原 MATLAB 保持一致

### Dependencies (依赖)

```
numpy>=1.21.0
pandas>=1.3.0
openpyxl>=3.0.0
```

---

## 版本对照

| 版本 | 日期 | 主要变更 |
|------|------|---------|
| 1.0.2 | 2024-12-03 | 修复传感器匹配逻辑以完全匹配 MATLAB |
| 1.0.1 | 2024-12-03 | 添加完整单元测试套件 |
| 1.0.0 | 2024-12-03 | MATLAB → Python 完整迁移 |

<!-- 比对链接 -->
[Unreleased]: #
[1.0.0]: #

<!--
归类指引（Conventional Commits → Changelog 分区）
feat: Added（新增）
fix: Fixed（修复）
perf / refactor / style / chore / docs / test: Changed（变更）
deprecate: Deprecated（弃用）
remove / breaking: Removed（移除）
security: Security（安全）
-->
