import bisect


def grad_cmap(colors, positions, n):
    """
    根据颜色列表、位置列表和长度n生成一个渐变颜色列表。

    参数:
        colors (list of tuples): 颜色列表，每个颜色为三元组(R, G, B)，数值范围建议0-255或0.0-1.0。
        positions (list of float): 对应颜色位置列表，数值将被归一化到0-1范围。
        n (int): 生成的渐变颜色列表长度，必须为正整数。

    返回:
        list of tuples: 生成的渐变颜色列表，每个颜色为三元组(R, G, B)，数值类型与输入一致。

    异常:
        ValueError: 输入不符合要求时抛出。
    """
    # 输入验证
    if len(colors) != len(positions):
        raise ValueError("颜色列表和位置列表的长度必须相同。")
    if n <= 0:
        raise ValueError("n必须是正整数。")
    for color in colors:
        if len(color) != 3 or not all(isinstance(c, (int, float)) for c in color):
            raise ValueError("颜色必须为包含三个数值的三元组。")
    for pos in positions:
        if not isinstance(pos, (int, float)):
            raise ValueError("位置必须是数值类型。")

    # 归一化位置处理
    min_pos = min(positions)
    max_pos = max(positions)
    if min_pos == max_pos:
        raise ValueError("所有位置相同，无法生成渐变。")
    normalized_positions = [(p - min_pos) / (max_pos - min_pos) for p in positions]

    # 按位置排序颜色
    sorted_pairs = sorted(zip(normalized_positions, colors), key=lambda x: x[0])
    sorted_positions, sorted_colors = zip(*sorted_pairs)
    sorted_positions = list(sorted_positions)
    sorted_colors = list(sorted_colors)

    # 检查位置是否严格递增
    for i in range(1, len(sorted_positions)):
        if sorted_positions[i] <= sorted_positions[i - 1]:
            raise ValueError("位置列表中存在重复或非递增的位置。")

    # 生成等间距采样点
    x_values = [i / (n - 1) for i in range(n)] if n > 1 else [0.5]

    gradient = []
    for x in x_values:
        if x <= sorted_positions[0]:
            gradient.append(sorted_colors[0])
        elif x >= sorted_positions[-1]:
            gradient.append(sorted_colors[-1])
        else:
            # 查找插值区间
            i = bisect.bisect_left(sorted_positions, x)
            left_pos = sorted_positions[i - 1]
            right_pos = sorted_positions[i]
            left_color = sorted_colors[i - 1]
            right_color = sorted_colors[i]

            # 计算插值比例
            t = (x - left_pos) / (right_pos - left_pos)

            # 线性插值计算每个通道
            interpolated = [
                left_color[0] + t * (right_color[0] - left_color[0]),
                left_color[1] + t * (right_color[1] - left_color[1]),
                left_color[2] + t * (right_color[2] - left_color[2])
            ]
            gradient.append(tuple(interpolated))

    return gradient
