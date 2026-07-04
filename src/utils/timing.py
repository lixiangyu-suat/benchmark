import time
from datetime import datetime


def get_time_format():
    """
    获取当前时间的自定义格式字符串。
    格式形如: 2026-01-01_01_02_03
    """
    return datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
