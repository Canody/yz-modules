import numpy as np


def hex2rgb(hex_color):
    """将 #RRGGBB 格式的十六进制颜色转换为 RGB 元组"""
    hex_color = hex_color.lstrip('#')  # 去除开头的 #
    if len(hex_color) != 6:
        raise ValueError("颜色格式应为 #RRGGBB")
    r = int(hex_color[0:2], 16) 
    g = int(hex_color[2:4], 16) 
    b = int(hex_color[4:6], 16) 
    return np.array([r, g, b])