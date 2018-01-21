# -*- coding: utf-8 -*-
"""
This file contains visualization methods.
"""
import matplotlib.pyplot as plt

def render_points(data, width=1, height=1, margin=0.00):
    """Render text points to a pylab figure.

    Parameters
    ----------
    points : [(str, (str, float, float))]
        data points to render, having the form [(color, (title, x, y))]
    width : int
        width of the graph in inches
    height : int
        height of the graph in inches
    margin : float
        amount of extra whitespace added at the edges
    """
    plt.figure(figsize=(width, height), tight_layout=True)
    ax = plt.gca()

    minx = 0
    maxx = 0
    miny = 0
    maxy = 0

    for _, points in data:
        # get min and max coordinates of the figure
        for (title, x, y) in points:
            if minx > x: minx = x
            if maxx < x: maxx = x
            if miny > y: miny = y
            if maxy < y: maxy = y

    dx = maxx - minx
    dy = maxy - miny
    assert dx > 0
    assert dy > 0
    minx -= dx * margin
    miny -= dy * margin
    maxx += dx * margin
    maxy += dy * margin

    ax.set_autoscale_on(False)

    minx_pos = 50000000
    maxx_pos = -50000000
    miny_pos = 50000000
    maxy_pos = -50000000

    for color, points in data:
        # render the single points
        for pt in points:
            (title, x, y) = pt
            x = 1. * (x - minx) / (maxx - minx)
            y = 1. * (y - miny) / (maxy - miny)

            minx_pos = min(minx_pos, x)
            maxx_pos = max(maxx_pos, x)
            miny_pos = min(miny_pos, y)
            maxy_pos = max(maxy_pos, y)
            pos = (x, y)

            plt.annotate(title, pos, color=color)

    ax.set_xlim([minx_pos, maxx_pos])
    ax.set_ylim([miny_pos, maxy_pos])
