#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 Type 将 full_results_2025 目录下的汇总 CSV 拆分到各自子目录。

输出结构：
    <output_dir>/<type>/A_final_output.csv
    <output_dir>/<type>/A_final_common_table.csv
    <output_dir>/<type>/A_final_score_table.csv
    <output_dir>/<type>/A_percentile_analysis.csv
    <output_dir>/<type>/A_error_traffic_table.csv

用法示例：
    python scripts/split_full_results_by_type.py \
        --input-dir ../full_results_2025 \
        --incidents-file ../data/incidents_y2023.csv \
        --chunk-size 200000
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


# 文件清单（按需求固定五个输出）
TARGET_FILES = [
    "A_final_output.csv",
    "A_final_common_table.csv",
    "A_final_score_table.csv",
    "A_percentile_analysis.csv",
    "A_error_traffic_table.csv",
]


def normalize_type(type_value) -> str:
    """规范化 Type 值，用于目录名和分组键。"""
    if pd.isna(type_value):
        return "other"

    value = str(type_value).strip()
    if not value:
        return "other"

    sanitized = re.sub(r"\s+", "_", value)  # 连续空白 → 下划线
    sanitized = re.sub(r"[\\\\/:*?\"<>|]+", "_", sanitized)  # 非法路径字符替换
    sanitized = sanitized.strip("._")

    return sanitized.lower() if sanitized else "other"


def load_incident_type_map(incidents_path: Path) -> Dict[str, str]:
    """加载 incident_id → Type 的映射，用于 error 表补 Type。"""
    if not incidents_path.exists():
        raise FileNotFoundError(f"找不到事故数据文件: {incidents_path}")

    df = pd.read_csv(incidents_path, sep=None, engine="python")
    if "incident_id" not in df.columns or "Type" not in df.columns:
        raise ValueError("事故数据需要包含 incident_id 和 Type 列")

    mapping = (
        df[["incident_id", "Type"]]
        .dropna(subset=["incident_id"])
        .assign(Type=lambda d: d["Type"].apply(normalize_type))
    )
    return dict(zip(mapping["incident_id"].astype(str), mapping["Type"]))


def write_chunk(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str,
    type_value: str,
    header_written: Dict[str, Dict[str, bool]],
) -> None:
    """将分组后的块写入对应文件，首块写表头。"""
    type_dir = output_dir / type_value
    type_dir.mkdir(parents=True, exist_ok=True)

    file_path = type_dir / filename
    file_headers = header_written.setdefault(filename, {})
    has_header = file_headers.get(type_value, False)

    df.to_csv(
        file_path,
        mode="a",
        index=False,
        header=not has_header,
        encoding="utf-8-sig",
    )

    file_headers[type_value] = True


def split_file(
    file_path: Path,
    incidents_map: Optional[Dict[str, str]],
    output_dir: Path,
    chunk_size: int,
    header_written: Dict[str, Dict[str, bool]],
) -> Dict[str, int]:
    """按 Type 拆分单个文件，返回类型计数。"""
    counts: Dict[str, int] = defaultdict(int)
    filename = file_path.name

    for chunk in pd.read_csv(file_path, chunksize=chunk_size, encoding="utf-8-sig"):
        # 复制一份避免修改原 chunk 引用
        df = chunk.copy()

        if "Type" in df.columns:
            df["__type__"] = df["Type"].apply(normalize_type)
        elif filename == "A_error_traffic_table.csv":
            # error 表没有 Type，用 incidents 映射补充
            if incidents_map is None:
                raise ValueError("拆分错误表需要提供 incidents 映射文件")
            incident_col = "incident_id" if "incident_id" in df.columns else "IncidentId"
            if incident_col not in df.columns:
                raise ValueError("错误表需要 incident_id/IncidentId 列")
            df["__type__"] = df[incident_col].astype(str).map(incidents_map).fillna("unknown")
        else:
            df["__type__"] = "unknown"

        for type_value, group in df.groupby("__type__"):
            group = group.drop(columns="__type__")
            write_chunk(group, output_dir, filename, type_value, header_written)
            counts[type_value] += len(group)

    return counts


def split_full_results_by_type(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    incidents_path: Optional[Path] = None,
    chunk_size: int = 200_000,
) -> Dict[str, Dict[str, int]]:
    """拆分 full_results_2025 汇总文件。

    Returns:
        dict: {filename: {type: count}}
    """
    input_dir = input_dir.resolve()
    output_dir = (output_dir or input_dir).resolve()

    incidents_map = load_incident_type_map(incidents_path) if incidents_path else None
    header_written: Dict[str, Dict[str, bool]] = {}
    summary: Dict[str, Dict[str, int]] = {}

    for filename in TARGET_FILES:
        file_path = input_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"未找到文件: {file_path}")

        summary[filename] = split_file(
            file_path=file_path,
            incidents_map=incidents_map,
            output_dir=output_dir,
            chunk_size=chunk_size,
            header_written=header_written,
        )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 Type 拆分 full_results_2025 汇总 CSV")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../full_results_2025"),
        help="汇总 CSV 所在目录（默认: ../full_results_2025）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出根目录（默认与 input-dir 相同）",
    )
    parser.add_argument(
        "--incidents-file",
        type=Path,
        default=Path("../data/incidents_y2023.csv"),
        help="包含 incident_id 和 Type 的事故文件，用于 error 表补 Type",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="分块大小（默认 200000 行）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    summary = split_full_results_by_type(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        incidents_path=args.incidents_file,
        chunk_size=args.chunk_size,
    )

    print("拆分完成，行数汇总：")
    for filename, type_counts in summary.items():
        total = sum(type_counts.values())
        print(f"- {filename}: {total} 行，类别数 {len(type_counts)}")


if __name__ == "__main__":
    main()
