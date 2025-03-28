import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def bar_grad(
    ax,
    x,
    height,
    width=0.8,
    bottom=0,
    *,
    align='center',
    orientation='vertical',
    cmap='viridis',
    yerr=None,
    xerr=None,
    capsize=3,
    ecolor='black',
    **kwargs
):
    lw = 1
    if ('lw' in kwargs) or ('linewidth' in kwargs):
        lw = kwargs['lw'] if 'lw' in kwargs else kwargs['linewidth']
    if not isinstance(cmap, str):
        cmap = ListedColormap(cmap)
    # cmap = ListedColormap(cmap)
    x = np.atleast_1d(x).astype(float)
    height = np.atleast_1d(height).astype(float)
    bottom = np.atleast_1d(bottom).astype(float)

    if len(x) != len(height):
        raise ValueError("x 和 height 的长度必须相同。")
    if len(bottom) == 1:
        bottom = np.full_like(x, bottom[0])
    elif len(bottom) != len(x):
        raise ValueError("bottom 要么是标量，要么与 x 长度相同。")

    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    grad = np.concatenate([grad, grad], axis=1)

    bars = []

    if orientation == 'vertical':
        bar_centers = []
        bar_tops = []

        for i, (xi, hi, bi) in enumerate(zip(x, height, bottom)):
            if align == 'center':
                left = xi - width / 2
                bar_center = xi
            else:
                left = xi
                bar_center = xi + width / 2

            bottom_val, top = (bi, bi + hi) if hi >= 0 else (bi + hi, bi)
            grad_img = grad if hi >= 0 else grad[::-1, :]

            im = ax.imshow(
                grad_img,
                extent=(left, left + width, bottom_val, top),
                origin='lower',
                aspect='auto',
                cmap=cmap
            )
            bars.append(im)
            bar_centers.append(bar_center)
            bar_tops.append(top)

        # 添加误差棒（垂直）
        if yerr is not None:
            yerr = np.atleast_1d(yerr)
            if len(yerr) == 1:
                yerr = np.full_like(height, yerr[0])
            ax.errorbar(
                bar_centers,
                bar_tops,
                yerr=yerr,
                fmt='none',
                capsize=capsize,
                ecolor=ecolor,
                elinewidth=lw,
                **kwargs
            )

    elif orientation == 'horizontal':
        bar_centers = []
        bar_ends = []

        grad_h = grad.T
        for i, (yi, wi, bi) in enumerate(zip(x, height, bottom)):
            if align == 'center':
                low = yi - width / 2
                bar_center = yi
            else:
                low = yi
                bar_center = yi + width / 2

            left, right = (bi, bi + wi) if wi >= 0 else (bi + wi, bi)
            grad_img = grad_h if wi >= 0 else grad_h[::-1, :]

            im = ax.imshow(
                grad_img,
                extent=(left, right, low, low + width),
                origin='lower',
                aspect='auto',
                cmap=cmap
            )
            bars.append(im)
            bar_centers.append(bar_center)
            bar_ends.append(right)

        # 添加误差棒（水平）
        if xerr is not None:
            xerr = np.atleast_1d(xerr)
            if len(xerr) == 1:
                xerr = np.full_like(height, xerr[0])
            ax.errorbar(
                bar_ends,
                bar_centers,
                xerr=xerr,
                fmt='none',
                capsize=capsize,
                ecolor=ecolor,
                elinewidth=lw,
                **kwargs
            )

    else:
        raise ValueError("orientation 只能是 'vertical' 或 'horizontal'。")

    return bars


if __name__ == '__main__':
    fig, ax = plt.subplots()

    x = [0, 1, 2, 3, 4]
    y = [3, 5, 2, 6, 4]
    yerr = [0.5, 1, 0.3, 0.7, 0.4]

    c = [0.7, 0, 0.1]
    cmap = np.array([np.linspace(1, c[0], 256).T, np.linspace(1, c[1], 256).T, np.linspace(1, c[2], 256).T]).T
    bar_grad(ax, x, y, width=0.8, bottom=0, cmap=cmap, yerr=yerr, capsize=5, lw=1)

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0, 8)
    ax.set_xticks(x)
    ax.set_ylabel("Value")
    ax.set_title("Gradient Bar Chart with Error Bars")

    plt.show()
