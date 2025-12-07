# -*- coding: utf-8 -*-
"""
验证传感器匹配逻辑是否与 MATLAB 一致

测试用例：
- 传感器 PM 列表（已排序）: [10.0, 10.0, 10.5, 10.5, 10.5, 11.0]
- 目标 PM: 10.4
- 期望结果：找到 10.5×3 + 11.0×1 = 4 个传感器
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from processors.sensor_matcher import SensorMatcher, MatchResult


def test_matlab_logic():
    """验证 MATLAB 逻辑的完整测试"""

    print("=" * 60)
    print("传感器匹配逻辑验证 - 对照 MATLAB 代码")
    print("=" * 60)

    # 创建测试数据：[10.0, 10.0, 10.5, 10.5, 10.5, 11.0]
    sensors_df = pd.DataFrame({
        'station_id': [1001, 1002, 1003, 1004, 1005, 1006],
        'Fwy': [101, 101, 101, 101, 101, 101],
        'Direction': ['N', 'N', 'N', 'N', 'N', 'N'],
        'Abs PM': [10.0, 10.0, 10.5, 10.5, 10.5, 11.0],
        'Lat': [34.0] * 6,
        'Lng': [-118.0] * 6
    })

    print("\n📊 测试数据（传感器列表，按 Abs PM 排序）：")
    print("-" * 50)
    for i, row in sensors_df.iterrows():
        print(f"  索引 {i+1} (MATLAB 1-based): station_id={row['station_id']}, Abs PM={row['Abs PM']}")

    # 创建事故数据
    incident = pd.Series({
        'incident_id': 'TEST001',
        'Fwy': 101,
        'Freeway_direction': 'N',
        'Abs PM': 10.4,  # 目标 PM
        'dt': pd.Timestamp('2023-06-15 10:30:00'),
        'DESCRIPTION': 'Test incident'
    })

    print(f"\n🎯 目标 Abs PM: {incident['Abs PM']}")

    # 计算各传感器与目标的距离
    print("\n📏 距离计算：")
    print("-" * 50)
    for i, pm in enumerate(sensors_df['Abs PM']):
        dist = abs(pm - incident['Abs PM'])
        marker = " ← 最近" if dist == min(abs(sensors_df['Abs PM'] - incident['Abs PM'])) else ""
        print(f"  |{pm} - {incident['Abs PM']}| = {dist:.1f}{marker}")

    # 执行匹配
    matcher = SensorMatcher(sensors_df)
    result = matcher.match(incident, row_index=0)

    print("\n" + "=" * 60)
    print("🔍 匹配结果")
    print("=" * 60)

    if result.success:
        print(f"✅ 匹配成功！找到 {result.sensor_count} 个传感器")
        print("\n匹配到的传感器：")
        print("-" * 50)
        for _, row in result.matched_sensors.iterrows():
            print(f"  station_id={row['station_id']}, Abs PM={row['Abs PM']}")

        matched_pms = result.matched_sensors['Abs PM'].tolist()
        print(f"\n匹配到的 PM 值列表: {matched_pms}")

        # 统计各 PM 值的数量
        from collections import Counter
        pm_counts = Counter(matched_pms)
        print(f"PM 值统计: {dict(pm_counts)}")

    else:
        print(f"❌ 匹配失败！错误: {result.errors}")

    # MATLAB 预期结果
    print("\n" + "=" * 60)
    print("📋 与 MATLAB 预期对比")
    print("=" * 60)

    print("""
MATLAB 代码逻辑追踪：
1. idx = 3 (第一个 10.5 的位置，1-based)
2. 循环后 count = 6 (指向 11.0)
3. closest_pm = PM[count-1] = PM[5] = 10.5
4. closest_pm_xx = PM[count] = PM[6] = 11.0
5. 向上搜索：从 count-1=5 开始，找 PM == 10.5
   - PM[5]=10.5 ✓, PM[4]=10.5 ✓, PM[3]=10.5 ✓, PM[2]=10.0 ✗
   - 找到 3 个 (索引 5,4,3)
6. 向下搜索：从 count=6 开始，找 PM == 11.0
   - PM[6]=11.0 ✓, 超出范围
   - 找到 1 个 (索引 6)
7. 总计: 4 个传感器 (10.5×3 + 11.0×1)
""")

    expected_count = 4
    expected_pms = [10.5, 10.5, 10.5, 11.0]

    actual_count = result.sensor_count
    actual_pms = sorted(result.matched_sensors['Abs PM'].tolist()) if result.success else []

    print(f"MATLAB 预期传感器数量: {expected_count}")
    print(f"Python 实际传感器数量: {actual_count}")
    print(f"MATLAB 预期 PM 值: {sorted(expected_pms)}")
    print(f"Python 实际 PM 值: {actual_pms}")

    # 验证结果
    print("\n" + "=" * 60)
    if actual_count == expected_count and sorted(actual_pms) == sorted(expected_pms):
        print("✅ 验证通过！Python 实现与 MATLAB 逻辑一致！")
    else:
        print("❌ 验证失败！Python 实现与 MATLAB 逻辑不一致！")
        print(f"   数量差异: 预期 {expected_count}, 实际 {actual_count}")
        print(f"   PM 差异: 预期 {sorted(expected_pms)}, 实际 {actual_pms}")
    print("=" * 60)

    return actual_count == expected_count and sorted(actual_pms) == sorted(expected_pms)


def test_edge_case_last_sensor():
    """测试边界情况：最近传感器已是最后一个"""

    print("\n\n" + "=" * 60)
    print("边界测试：最近传感器是最后一个 (flag=false)")
    print("=" * 60)

    # 传感器列表：[10.0, 10.0, 10.5]，目标 PM=10.6
    sensors_df = pd.DataFrame({
        'station_id': [1001, 1002, 1003],
        'Fwy': [101, 101, 101],
        'Direction': ['N', 'N', 'N'],
        'Abs PM': [10.0, 10.0, 10.5],
        'Lat': [34.0] * 3,
        'Lng': [-118.0] * 3
    })

    incident = pd.Series({
        'incident_id': 'TEST002',
        'Fwy': 101,
        'Freeway_direction': 'N',
        'Abs PM': 10.6,
        'dt': pd.Timestamp('2023-06-15 10:30:00'),
        'DESCRIPTION': 'Test incident'
    })

    print(f"传感器 PM 列表: {sensors_df['Abs PM'].tolist()}")
    print(f"目标 PM: {incident['Abs PM']}")

    matcher = SensorMatcher(sensors_df)
    result = matcher.match(incident, row_index=0)

    print(f"\n结果: 找到 {result.sensor_count} 个传感器")
    if result.success:
        print(f"匹配到的 PM 值: {result.matched_sensors['Abs PM'].tolist()}")

    # MATLAB 预期：idx=3, flag=false, 只向上找 PM==10.5，找到 1 个
    expected = 1
    print(f"\nMATLAB 预期: {expected} 个传感器 (只有 10.5)")

    if result.sensor_count == expected:
        print("✅ 边界测试通过！")
        return True
    else:
        print("❌ 边界测试失败！")
        return False


if __name__ == '__main__':
    test1 = test_matlab_logic()
    test2 = test_edge_case_last_sensor()

    print("\n\n" + "=" * 60)
    print("📊 总结")
    print("=" * 60)
    print(f"主测试（MATLAB 逻辑）: {'✅ 通过' if test1 else '❌ 失败'}")
    print(f"边界测试（最后一个传感器）: {'✅ 通过' if test2 else '❌ 失败'}")

    if test1 and test2:
        print("\n🎉 所有验证通过！Python 实现与 MATLAB 完全一致！")
    else:
        print("\n⚠️ 存在不一致，需要检查！")
