# -*- coding: utf-8 -*-
"""
时间工具模块单元测试

测试 utils/time_utils.py 中的所有函数
"""

import pytest
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.time_utils import (
    MINUTES_PER_TIMESTEP,
    TIMESTEPS_PER_HOUR,
    TIMESTEPS_PER_DAY,
    DAYS_IN_YEAR,
    DAYS_IN_LEAP_YEAR,
    DAYS_IN_MONTH_NORMAL,
    DAYS_IN_MONTH_LEAP,
    is_leap_year,
    matlab_to_iso_weekday,
    get_iso_weekday,
    datetime_to_numeric,
    datetime_to_timestep,
    timestep_to_datetime,
    get_timestep_range,
    get_same_weekday_timesteps,
    date_to_day_of_year,
    validate_timestep,
)


class TestConstants:
    """测试常量定义"""

    def test_minutes_per_timestep(self):
        """测试每个时间步的分钟数"""
        assert MINUTES_PER_TIMESTEP == 5

    def test_timesteps_per_hour(self):
        """测试每小时的时间步数"""
        assert TIMESTEPS_PER_HOUR == 12

    def test_timesteps_per_day(self):
        """测试每天的时间步数"""
        assert TIMESTEPS_PER_DAY == 288

    def test_days_in_year(self):
        """测试平年天数"""
        assert DAYS_IN_YEAR == 365

    def test_days_in_leap_year(self):
        """测试闰年天数"""
        assert DAYS_IN_LEAP_YEAR == 366

    def test_days_in_month_normal(self):
        """测试平年每月天数"""
        assert len(DAYS_IN_MONTH_NORMAL) == 12
        assert sum(DAYS_IN_MONTH_NORMAL) == 365
        assert DAYS_IN_MONTH_NORMAL[1] == 28  # February

    def test_days_in_month_leap(self):
        """测试闰年每月天数"""
        assert len(DAYS_IN_MONTH_LEAP) == 12
        assert sum(DAYS_IN_MONTH_LEAP) == 366
        assert DAYS_IN_MONTH_LEAP[1] == 29  # February


class TestIsLeapYear:
    """测试 is_leap_year 函数"""

    def test_leap_year_divisible_by_4(self):
        """测试能被4整除的年份"""
        assert is_leap_year(2024) is True
        assert is_leap_year(2020) is True

    def test_not_leap_year_divisible_by_100(self):
        """测试能被100整除但不能被400整除的年份"""
        assert is_leap_year(1900) is False
        assert is_leap_year(2100) is False

    def test_leap_year_divisible_by_400(self):
        """测试能被400整除的年份"""
        assert is_leap_year(2000) is True
        assert is_leap_year(1600) is True

    def test_non_leap_year(self):
        """测试普通年份"""
        assert is_leap_year(2023) is False
        assert is_leap_year(2021) is False


class TestMatlabToIsoWeekday:
    """测试 matlab_to_iso_weekday 函数"""

    def test_sunday(self):
        """测试周日转换"""
        # MATLAB: 1 = 周日 -> ISO: 7
        assert matlab_to_iso_weekday(1) == 7

    def test_weekdays(self):
        """测试工作日转换"""
        # MATLAB: 2 = 周一 -> ISO: 1
        assert matlab_to_iso_weekday(2) == 1
        # MATLAB: 3 = 周二 -> ISO: 2
        assert matlab_to_iso_weekday(3) == 2
        # MATLAB: 6 = 周五 -> ISO: 5
        assert matlab_to_iso_weekday(6) == 5

    def test_saturday(self):
        """测试周六转换"""
        # MATLAB: 7 = 周六 -> ISO: 6
        assert matlab_to_iso_weekday(7) == 6


class TestGetIsoWeekday:
    """测试 get_iso_weekday 函数"""

    def test_datetime_monday(self):
        """测试 datetime 周一"""
        dt = datetime(2023, 1, 2)  # Monday
        assert get_iso_weekday(dt) == 1

    def test_datetime_sunday(self):
        """测试 datetime 周日"""
        dt = datetime(2023, 1, 1)  # Sunday
        assert get_iso_weekday(dt) == 7

    def test_pandas_timestamp(self):
        """测试 pandas Timestamp"""
        ts = pd.Timestamp('2023-01-02')  # Monday
        assert get_iso_weekday(ts) == 1


class TestDatetimeToNumeric:
    """测试 datetime_to_numeric 函数"""

    def test_basic_conversion(self):
        """测试基本转换"""
        dt = datetime(2023, 1, 2, 15, 20)
        result = datetime_to_numeric(dt)
        assert result == 202301021520

    def test_midnight(self):
        """测试午夜"""
        dt = datetime(2023, 12, 31, 0, 0)
        result = datetime_to_numeric(dt)
        assert result == 202312310000

    def test_pandas_timestamp(self):
        """测试 pandas Timestamp"""
        ts = pd.Timestamp('2023-06-15 10:30')
        result = datetime_to_numeric(ts)
        assert result == 202306151030

    def test_nan_value(self):
        """测试 NaN 值"""
        result = datetime_to_numeric(pd.NaT)
        assert result == 0


class TestDatetimeToTimestep:
    """测试 datetime_to_timestep 函数"""

    def test_midnight_jan_1(self):
        """测试1月1日午夜"""
        dt = datetime(2023, 1, 1, 0, 0)
        result = datetime_to_timestep(dt, 2023)
        assert result == 0

    def test_first_timestep(self):
        """测试第一个时间步"""
        dt = datetime(2023, 1, 1, 0, 5)
        result = datetime_to_timestep(dt, 2023)
        assert result == 1

    def test_midday(self):
        """测试中午"""
        dt = datetime(2023, 1, 1, 12, 0)
        # 12 hours * 12 timesteps/hour = 144
        result = datetime_to_timestep(dt, 2023)
        assert result == 144

    def test_end_of_day(self):
        """测试一天结束"""
        dt = datetime(2023, 1, 1, 23, 55)
        # 23*12 + 11 = 276 + 11 = 287
        result = datetime_to_timestep(dt, 2023)
        assert result == 287

    def test_second_day(self):
        """测试第二天"""
        dt = datetime(2023, 1, 2, 0, 0)
        result = datetime_to_timestep(dt, 2023)
        assert result == 288

    def test_february_non_leap(self):
        """测试平年2月"""
        dt = datetime(2023, 3, 1, 0, 0)
        # January (31 days) + February (28 days) = 59 days
        result = datetime_to_timestep(dt, 2023)
        assert result == 59 * 288

    def test_february_leap(self):
        """测试闰年2月"""
        dt = datetime(2024, 3, 1, 0, 0)
        # January (31 days) + February (29 days) = 60 days
        result = datetime_to_timestep(dt, 2024)
        assert result == 60 * 288

    def test_pandas_timestamp(self):
        """测试 pandas Timestamp"""
        ts = pd.Timestamp('2023-01-01 00:10')
        result = datetime_to_timestep(ts, 2023)
        assert result == 2

    def test_nan_raises_error(self):
        """测试 NaN 值抛出异常"""
        with pytest.raises(ValueError):
            datetime_to_timestep(pd.NaT)


class TestTimestepToDatetime:
    """测试 timestep_to_datetime 函数"""

    def test_timestep_zero(self):
        """测试时间步0"""
        result = timestep_to_datetime(0, 2023)
        assert result == datetime(2023, 1, 1, 0, 0)

    def test_timestep_one(self):
        """测试时间步1"""
        result = timestep_to_datetime(1, 2023)
        assert result == datetime(2023, 1, 1, 0, 5)

    def test_end_of_day(self):
        """测试一天结束"""
        result = timestep_to_datetime(287, 2023)
        assert result == datetime(2023, 1, 1, 23, 55)

    def test_second_day(self):
        """测试第二天"""
        result = timestep_to_datetime(288, 2023)
        assert result == datetime(2023, 1, 2, 0, 0)

    def test_roundtrip(self):
        """测试往返转换"""
        dt = datetime(2023, 6, 15, 14, 30)
        timestep = datetime_to_timestep(dt, 2023)
        result = timestep_to_datetime(timestep, 2023)
        assert result == dt


class TestGetTimestepRange:
    """测试 get_timestep_range 函数"""

    def test_basic_range(self):
        """测试基本范围"""
        dt = datetime(2023, 6, 15, 12, 0)
        start, end = get_timestep_range(dt, 12, 12)
        center = datetime_to_timestep(dt)
        assert start == center - 12
        assert end == center + 12

    def test_boundary_start(self):
        """测试起始边界"""
        dt = datetime(2023, 1, 1, 0, 30)  # Early in the year
        start, end = get_timestep_range(dt, 12, 12)
        assert start >= 0

    def test_boundary_end(self):
        """测试结束边界"""
        dt = datetime(2023, 12, 31, 23, 30)  # Late in the year
        start, end = get_timestep_range(dt, 12, 12, 105120)
        assert end < 105120


class TestGetSameWeekdayTimesteps:
    """测试 get_same_weekday_timesteps 函数"""

    def test_finds_same_weekday(self):
        """测试找到相同星期几"""
        dt = datetime(2023, 6, 15, 12, 0)  # Thursday
        result = get_same_weekday_timesteps(dt, 2023, 0, 0)

        # Should find all Thursdays in 2023
        assert len(result) > 50  # At least 52 Thursdays in a year

    def test_with_window(self):
        """测试带时间窗口"""
        dt = datetime(2023, 6, 15, 12, 0)
        result = get_same_weekday_timesteps(dt, 2023, 2, 2)

        # With window, should have more timesteps
        assert len(result) > 0


class TestDateToDayOfYear:
    """测试 date_to_day_of_year 函数"""

    def test_jan_1(self):
        """测试1月1日"""
        dt = datetime(2023, 1, 1)
        assert date_to_day_of_year(dt) == 1

    def test_jan_31(self):
        """测试1月31日"""
        dt = datetime(2023, 1, 31)
        assert date_to_day_of_year(dt) == 31

    def test_feb_1(self):
        """测试2月1日"""
        dt = datetime(2023, 2, 1)
        assert date_to_day_of_year(dt) == 32

    def test_dec_31_non_leap(self):
        """测试平年12月31日"""
        dt = datetime(2023, 12, 31)
        assert date_to_day_of_year(dt) == 365

    def test_dec_31_leap(self):
        """测试闰年12月31日"""
        dt = datetime(2024, 12, 31)
        assert date_to_day_of_year(dt) == 366

    def test_date_object(self):
        """测试 date 对象"""
        d = date(2023, 6, 15)
        result = date_to_day_of_year(d)
        assert result == 166

    def test_pandas_timestamp(self):
        """测试 pandas Timestamp"""
        ts = pd.Timestamp('2023-01-15')
        result = date_to_day_of_year(ts)
        assert result == 15


class TestValidateTimestep:
    """测试 validate_timestep 函数"""

    def test_valid_timestep(self):
        """测试有效时间步"""
        assert validate_timestep(0, 2023) is True
        assert validate_timestep(100, 2023) is True
        assert validate_timestep(105119, 2023) is True

    def test_invalid_negative(self):
        """测试负数时间步"""
        assert validate_timestep(-1, 2023) is False

    def test_invalid_too_large(self):
        """测试过大的时间步"""
        assert validate_timestep(105120, 2023) is False  # Exactly at limit
        assert validate_timestep(200000, 2023) is False

    def test_leap_year_max(self):
        """测试闰年最大时间步"""
        # Leap year has 366 * 288 = 105408 timesteps
        assert validate_timestep(105407, 2024) is True
        assert validate_timestep(105408, 2024) is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
