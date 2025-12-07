# -*- coding: utf-8 -*-
"""
时间处理工具模块

提供日期时间与流量矩阵时间步索引之间的转换功能。

对应 MATLAB 代码：
- p_trafficdata_function.m 中的 isLeapYear, convertToISOWeekday 函数
- 时间步计算逻辑 (y = y_month*24*60 + (days-1)*60*24 + hours*60 + minutes)

时间步定义：
- 每个时间步 = 5 分钟
- 每天 = 288 个时间步 (24小时 × 12个5分钟)
- 每年 = 105,120 时间步 (平年) 或 105,408 时间步 (闰年)
- 数据矩阵维度: (时间步数, 传感器数) = (105120, 16972)

索引约定：
- MATLAB 使用 1-based 索引
- Python 使用 0-based 索引
- 本模块返回的时间步索引是 0-based (Python 风格)
"""

from datetime import datetime, date, timedelta
from typing import Union, Tuple, Optional
import numpy as np
import pandas as pd


# 常量定义
MINUTES_PER_TIMESTEP = 5
TIMESTEPS_PER_HOUR = 60 // MINUTES_PER_TIMESTEP  # 12
TIMESTEPS_PER_DAY = 24 * TIMESTEPS_PER_HOUR  # 288
DAYS_IN_YEAR = 365
DAYS_IN_LEAP_YEAR = 366

# 每月天数
DAYS_IN_MONTH_NORMAL = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
DAYS_IN_MONTH_LEAP = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def is_leap_year(year: int) -> bool:
    """判断是否为闰年

    对应 MATLAB: isLeapYear(year)

    闰年规则：
    - 能被4整除且不能被100整除，或者
    - 能被400整除

    Args:
        year: 年份

    Returns:
        是否为闰年
    """
    if year % 4 == 0:
        if year % 100 != 0 or year % 400 == 0:
            return True
    return False


def matlab_to_iso_weekday(matlab_weekday: int) -> int:
    """将 MATLAB 星期几转换为 ISO 8601 标准

    对应 MATLAB: convertToISOWeekday(matlab_day)

    MATLAB 约定：
    - 1 = 周日
    - 2 = 周一
    - ...
    - 7 = 周六

    ISO 8601 标准 (Python datetime.isoweekday())：
    - 1 = 周一
    - 2 = 周二
    - ...
    - 7 = 周日

    Args:
        matlab_weekday: MATLAB 的星期几 (1-7, 1=周日)

    Returns:
        ISO 8601 标准的星期几 (1-7, 1=周一)
    """
    if matlab_weekday == 1:
        return 7  # 周日
    else:
        return matlab_weekday - 1


def get_iso_weekday(dt: Union[datetime, pd.Timestamp]) -> int:
    """获取日期的 ISO 星期几

    Args:
        dt: datetime 或 pandas Timestamp

    Returns:
        ISO 8601 标准的星期几 (1=周一, 7=周日)
    """
    if isinstance(dt, pd.Timestamp):
        return dt.isoweekday()
    return dt.isoweekday()


def datetime_to_numeric(dt: Union[datetime, pd.Timestamp]) -> int:
    """将日期时间转换为数值格式

    对应 MATLAB:
        numeric_time = years * 1e8 + months * 1e6 + days * 1e4 + hours * 1e2 + minutes

    格式: YYYYMMDDHHmm (例如: 202301021520 表示 2023-01-02 15:20)

    Args:
        dt: datetime 或 pandas Timestamp

    Returns:
        数值格式的时间戳
    """
    if pd.isna(dt):
        return 0

    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()

    return (dt.year * 100000000 +
            dt.month * 1000000 +
            dt.day * 10000 +
            dt.hour * 100 +
            dt.minute)


def datetime_to_timestep(
    dt: Union[datetime, pd.Timestamp],
    base_year: Optional[int] = None
) -> int:
    """将日期时间转换为流量矩阵中的时间步索引 (0-based)

    对应 MATLAB 逻辑:
        y_month = 0;
        if months > 1
            y_month = sum(year_day(1:months-1));
        end
        y = y_month*24*60 + (days-1)*60*24 + hours*60 + minutes;
        y_shang = floor(y / 5);

    计算公式（转换为0-based）：
    1. 计算当年1月1日0:00到目标时刻的总分钟数
    2. 除以5得到时间步索引

    Args:
        dt: datetime 或 pandas Timestamp
        base_year: 基准年份，如果不指定则使用 dt 的年份

    Returns:
        时间步索引 (0-based)，范围 [0, 105119] (平年) 或 [0, 105407] (闰年)

    Raises:
        ValueError: 如果时间无效
    """
    if pd.isna(dt):
        raise ValueError("日期时间不能为空")

    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()

    year = base_year if base_year else dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute

    # 选择正确的每月天数表
    days_in_month = DAYS_IN_MONTH_LEAP if is_leap_year(year) else DAYS_IN_MONTH_NORMAL

    # 计算当月之前所有月份的总天数
    days_before_month = sum(days_in_month[:month - 1]) if month > 1 else 0

    # 计算从年初到当前时刻的总分钟数
    # MATLAB: y = y_month*24*60 + (days-1)*60*24 + hours*60 + minutes
    total_minutes = (
        days_before_month * 24 * 60 +  # 前几个月的分钟数
        (day - 1) * 24 * 60 +          # 当月前几天的分钟数
        hour * 60 +                     # 当天前几小时的分钟数
        minute                          # 分钟
    )

    # 计算时间步索引 (0-based)
    # MATLAB: y_shang = floor(y / 5)
    # 注意：MATLAB 是 1-based，结果再 +1 使用
    # Python 我们直接用 0-based
    timestep = total_minutes // MINUTES_PER_TIMESTEP

    return timestep


def timestep_to_datetime(
    timestep: int,
    year: int
) -> datetime:
    """将时间步索引转换回日期时间

    Args:
        timestep: 时间步索引 (0-based)
        year: 年份

    Returns:
        对应的 datetime
    """
    # 计算从年初开始的总分钟数
    total_minutes = timestep * MINUTES_PER_TIMESTEP

    # 创建年初的 datetime
    base_dt = datetime(year, 1, 1, 0, 0)

    # 加上分钟数
    result = base_dt + timedelta(minutes=total_minutes)

    return result


def get_timestep_range(
    dt: Union[datetime, pd.Timestamp],
    before_steps: int,
    after_steps: int,
    max_timesteps: int = 105120
) -> Tuple[int, int]:
    """计算以某时刻为中心的时间步范围

    对应 MATLAB:
        y_up = min(max_columns, y_shang + set_time)
        y_down = max(1, y_shang - set_time)

    Args:
        dt: 中心时刻
        before_steps: 向前的时间步数
        after_steps: 向后的时间步数
        max_timesteps: 最大时间步数（默认为一年的时间步数）

    Returns:
        (start_index, end_index) - 均为 0-based，左闭右闭区间
    """
    center = datetime_to_timestep(dt)

    # MATLAB: y_down = max(1, y_shang - set_time)
    # 转换为 0-based: max(0, center - before_steps)
    start_index = max(0, center - before_steps)

    # MATLAB: y_up = min(max_columns, y_shang + set_time)
    # 转换为 0-based: min(max_timesteps - 1, center + after_steps)
    end_index = min(max_timesteps - 1, center + after_steps)

    return (start_index, end_index)


def get_same_weekday_timesteps(
    dt: Union[datetime, pd.Timestamp],
    year: int,
    window_before: int = 12,
    window_after: int = 12
) -> list:
    """获取同一年中所有相同星期几的对应时间步

    用于历史数据采样：找出过去同一星期几、同一时段的数据点

    Args:
        dt: 参考时刻
        year: 年份
        window_before: 向前的时间步数
        window_after: 向后的时间步数

    Returns:
        时间步索引列表 (0-based)
    """
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()

    target_weekday = dt.isoweekday()  # 1-7

    # 获取参考时刻的时间步
    ref_timestep = datetime_to_timestep(dt, year)

    # 找出同一年中所有相同星期几的日期
    results = []
    current_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)

    while current_date <= end_date:
        if current_date.isoweekday() == target_weekday:
            # 同一星期几，获取对应时间步
            same_time = current_date.replace(
                hour=dt.hour,
                minute=dt.minute
            )
            center_step = datetime_to_timestep(same_time, year)

            # 添加窗口范围内的时间步
            max_steps = DAYS_IN_LEAP_YEAR * TIMESTEPS_PER_DAY if is_leap_year(year) else DAYS_IN_YEAR * TIMESTEPS_PER_DAY
            for offset in range(-window_before, window_after + 1):
                step = center_step + offset
                if 0 <= step < max_steps:
                    results.append(step)

        current_date += timedelta(days=1)

    return sorted(set(results))


def date_to_day_of_year(dt: Union[datetime, date, pd.Timestamp]) -> int:
    """获取日期在当年中的天数 (1-based)

    Args:
        dt: 日期

    Returns:
        当年第几天 (1-366)
    """
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()

    if isinstance(dt, datetime):
        dt = dt.date()

    return dt.timetuple().tm_yday


def validate_timestep(timestep: int, year: int) -> bool:
    """验证时间步索引是否有效

    Args:
        timestep: 时间步索引 (0-based)
        year: 年份

    Returns:
        是否有效
    """
    max_steps = DAYS_IN_LEAP_YEAR * TIMESTEPS_PER_DAY if is_leap_year(year) else DAYS_IN_YEAR * TIMESTEPS_PER_DAY
    return 0 <= timestep < max_steps
