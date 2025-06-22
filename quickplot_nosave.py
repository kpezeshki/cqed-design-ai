# TODO: Allow toggling of layers with 1-9 keypresses
# TODO: alias font should be slightly transparent


import sys
import warnings

import gdspy
import numpy as np

import phidl
from phidl.device_layout import (
    CellArray,
    Device,
    DeviceReference,
    Layer,
    Path,
    Polygon,
    _rotate_points,
)

_SUBPORT_RGB = (0, 120, 120)
_PORT_RGB = (190, 0, 0)


try:
    from PyQt5 import QtCore, QtGui
    from PyQt5.QtCore import (
        QCoreApplication,
        QLineF,
        QPoint,
        QPointF,
        QRect,
        QRectF,
        QSize,
        QSizeF,
        Qt,
    )
    from PyQt5.QtGui import QColor, QPen, QPolygonF
    from PyQt5.QtWidgets import (
        QApplication,
        QGraphicsItem,
        QGraphicsScene,
        QGraphicsView,
        QLabel,
        QMainWindow,
        QMessageBox,
        QRubberBand,
    )

    PORT_COLOR = QColor(*_PORT_RGB)
    SUBPORT_COLOR = QColor(*_SUBPORT_RGB)
    OUTLINE_PEN = QColor(200, 200, 200)
    qt_imported = True
except ImportError:
    QMainWindow = object
    QGraphicsView = object
    qt_imported = False

_quickplot_options = dict(
    show_ports=True,
    show_subports=True,
    label_aliases=False,
    new_window=False,
    blocking=False,
    zoom_factor=1.4,
    interactive_zoom=None,
)


def _zoom_factory(axis, scale_factor=1.4):
    """returns zooming functionality to axis.
    From https://gist.github.com/tacaswell/3144287"""

    def zoom_fun(event, ax, scale):
        """zoom when scrolling"""
        if event.inaxes == axis:
            scale_factor = np.power(scale, -event.step)
            xdata = event.xdata
            ydata = event.ydata
            x_left = xdata - ax.get_xlim()[0]
            x_right = ax.get_xlim()[1] - xdata
            y_top = ydata - ax.get_ylim()[0]
            y_bottom = ax.get_ylim()[1] - ydata

            ax.set_xlim([xdata - x_left * scale_factor, xdata + x_right * scale_factor])
            ax.set_ylim([ydata - y_top * scale_factor, ydata + y_bottom * scale_factor])
            ax.figure.canvas.draw()
            # Update toolbar so back/forward buttons work
            fig.canvas.toolbar.push_current()

    fig = axis.get_figure()
    fig.canvas.mpl_connect(
        "scroll_event", lambda event: zoom_fun(event, axis, scale_factor)
    )


_qp_objects = {}


def _rectangle_selector_factory(fig, ax):
    from matplotlib.widgets import RectangleSelector

    def line_select_callback(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        left = min(x1, x2)
        right = max(x1, x2)
        bottom = min(y1, y2)
        top = max(y1, y2)
        ax.set_xlim([left, right])
        ax.set_ylim([bottom, top])
        ax.figure.canvas.draw()
        # Update toolbar so back/forward buttons work
        fig.canvas.toolbar.push_current()

    rs = RectangleSelector(
        ax,
        line_select_callback,
        drawtype="box",
        useblit=True,
        button=[1, 3],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=False,
    )
    return rs


def set_quickplot_options(
    show_ports=None,
    show_subports=None,
    label_aliases=None,
    new_window=None,
    blocking=None,
    zoom_factor=None,
    interactive_zoom=None,
):
    """Sets plotting options for quickplot()

    Parameters
    ----------
    show_ports : bool
        Sets whether ports are drawn
    show_subports : bool
        Sets whether subports (ports that belong to references) are drawn
    label_aliases : bool
        Sets whether aliases are labeled with a text name
    new_window : bool
        If True, each call to quickplot() will generate a separate window
    blocking : bool
        If True, calling quickplot() will pause execution of ("block") the
        remainder of the python code until the quickplot() window is closed.  If
        False, the window will be opened and code will continue to run.
    zoom_factor : float
        Sets the scaling factor when zooming the quickplot window with the
        mousewheel/trackpad
    interactive_zoom : bool
        Enables/disables the ability to use mousewheel/trackpad to zoom
    """
    if show_ports is not None:
        _quickplot_options["show_ports"] = show_ports
    if show_subports is not None:
        _quickplot_options["show_subports"] = show_subports
    if label_aliases is not None:
        _quickplot_options["label_aliases"] = label_aliases
    if new_window is not None:
        _quickplot_options["new_window"] = new_window
    if blocking is not None:
        _quickplot_options["blocking"] = blocking
    if zoom_factor is not None:
        _quickplot_options["zoom_factor"] = zoom_factor
    if interactive_zoom is not None:
        _quickplot_options["interactive_zoom"] = interactive_zoom


def quickplot_noshow(items):  # noqa: C901
    """Takes a list of devices/references/polygons or single one of those, and
    plots them. Use `set_quickplot_options()` to modify the viewer behavior
    (e.g. displaying ports, creating new windows, etc)

    Parameters
    ----------
    items : PHIDL object or list of PHIDL objects
        The item(s) which are to be plotted

    Examples
    --------
    >>> R = pg.rectangle()
    >>> quickplot(R)

    >>> R = pg.rectangle()
    >>> E = pg.ellipse()
    >>> quickplot([R, E])
    """

    try:
        from matplotlib import pyplot as plt

        matplotlib_imported = True
    except ImportError:
        matplotlib_imported = False

    # Override default options with _quickplot_options
    show_ports = _quickplot_options["show_ports"]
    show_subports = _quickplot_options["show_subports"]
    label_aliases = _quickplot_options["label_aliases"]
    new_window = _quickplot_options["new_window"]
    blocking = _quickplot_options["blocking"]

    if not matplotlib_imported:
        raise ImportError(
            "PHIDL tried to import matplotlib but it failed. PHIDL "
            + "will still work but quickplot() will not"
        )

    if new_window:
        fig, ax = plt.subplots(1)
        ax.autoscale(enable=True, tight=True)
    else:
        if plt.fignum_exists(num="PHIDL quickplot"):
            fig = plt.figure("PHIDL quickplot")
            plt.clf()  # Erase figure so toolbar at top works correctly
            ax = fig.add_subplot(111)
        else:
            fig, ax = plt.subplots(num="PHIDL quickplot")

    ax.axis("equal")
    ax.grid(True, which="both", alpha=0.4)
    ax.axhline(y=0, color="k", alpha=0.2, linewidth=1)
    ax.axvline(x=0, color="k", alpha=0.2, linewidth=1)
    bbox = None

    # Iterate through each each Device/DeviceReference/Polygon
    if not isinstance(items, list):
        items = [items]
    for item in items:
        if isinstance(item, (Device, DeviceReference, CellArray)):
            polygons_spec = item.get_polygons(by_spec=True, depth=None)
            for key in sorted(polygons_spec):
                polygons = polygons_spec[key]
                layerprop = _get_layerprop(layer=key[0], datatype=key[1])
                new_bbox = _draw_polygons(
                    polygons,
                    ax,
                    facecolor=layerprop["color"],
                    edgecolor="k",
                    alpha=layerprop["alpha"],
                )
                bbox = _update_bbox(bbox, new_bbox)
            # If item is a Device or DeviceReference, draw ports
            if isinstance(item, (Device, DeviceReference)) and show_ports is True:
                for name, port in item.ports.items():
                    if (port.width is None) or (port.width == 0):
                        new_bbox = _draw_port_as_point(ax, port)
                    else:
                        new_bbox = _draw_port(ax, port, is_subport=False, color="r")
                    bbox = _update_bbox(bbox, new_bbox)
            if isinstance(item, Device) and show_subports is True:
                for sd in item.references:
                    if not isinstance(sd, (gdspy.CellArray)):
                        for name, port in sd.ports.items():
                            new_bbox = _draw_port(
                                ax,
                                port,
                                is_subport=True,
                                color=np.array(_SUBPORT_RGB) / 255,
                            )
                            bbox = _update_bbox(bbox, new_bbox)
            if isinstance(item, Device) and label_aliases is True:
                for name, ref in item.aliases.items():
                    ax.text(
                        ref.x,
                        ref.y,
                        str(name),
                        style="italic",
                        color="blue",
                        weight="bold",
                        size="large",
                        ha="center",
                        fontsize=14,
                    )
        elif isinstance(item, Polygon):
            polygons = item.polygons
            layerprop = _get_layerprop(item.layers[0], item.datatypes[0])
            new_bbox = _draw_polygons(
                polygons,
                ax,
                facecolor=layerprop["color"],
                edgecolor="k",
                alpha=layerprop["alpha"],
            )
            bbox = _update_bbox(bbox, new_bbox)
        elif isinstance(item, Path):
            points = item.points
            new_bbox = _draw_line(
                x=points[:, 0],
                y=points[:, 1],
                ax=ax,
                linestyle="--",
                linewidth=2,
                color="b",
            )
            _draw_arrow(ax=ax, x=points[-1, 0], y=points[-1, 1], angle=item.end_angle)
            _draw_dot(ax=ax, x=points[0, 0], y=points[0, 1])
            bbox = _update_bbox(bbox, new_bbox)

    if bbox is None:
        bbox = [-1, -1, 1, 1]
    xmargin = (bbox[2] - bbox[0]) * 0.1 + 1e-9
    ymargin = (bbox[3] - bbox[1]) * 0.1 + 1e-9
    ax.set_xlim([bbox[0] - xmargin, bbox[2] + xmargin])
    ax.set_ylim([bbox[1] - ymargin, bbox[3] + ymargin])

    # When using inline Jupyter notebooks, this may fail so allow it to fail gracefully
    try:
        if _use_interactive_zoom():
            _zoom_factory(ax, scale_factor=_quickplot_options["zoom_factor"])
        # Need to hang on to RectangleSelector so it doesn't get garbage collected
        _qp_objects["rectangle_selector"] = _rectangle_selector_factory(fig, ax)
        # Update matplotlib toolbar so the Home button works
        fig.canvas.toolbar.update()
        fig.canvas.toolbar.push_current()
    except Exception:
        pass

    plt.draw()
    #plt.show(block=blocking)


def _draw_arrow(ax, x, y, angle):
    from matplotlib.markers import MarkerStyle

    rotated_marker = MarkerStyle(marker=9)
    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(angle)
    ax.scatter(x, y, marker=rotated_marker, s=50, facecolors="b", alpha=0.5)


def _draw_dot(ax, x, y):
    ax.scatter(x, y, marker=".", color="b", s=50, alpha=0.9)


def _use_interactive_zoom():
    """Checks whether the current matplotlib backend is compatible with
    interactive zoom"""
    import matplotlib

    if _quickplot_options["interactive_zoom"] is not None:
        return _quickplot_options["interactive_zoom"]
    forbidden_backends = ["nbagg"]
    backend = matplotlib.get_backend().lower()
    usable = not any([(fb.lower() in backend) for fb in forbidden_backends])
    return usable


def _update_bbox(bbox, new_bbox):
    if bbox is None:
        return new_bbox
    if new_bbox[0] < bbox[0]:
        bbox[0] = new_bbox[0]  # xmin
    if new_bbox[1] < bbox[1]:
        bbox[1] = new_bbox[1]  # ymin
    if new_bbox[2] > bbox[2]:
        bbox[2] = new_bbox[2]  # xmin
    if new_bbox[3] > bbox[3]:
        bbox[3] = new_bbox[3]  # ymin
    return bbox


def _get_layerprop(layer, datatype):
    # Colors generated from here: http://phrogz.net/css/distinct-colors.html
    layer_colors = [
        "#3dcc5c",
        "#2b0fff",
        "#cc3d3d",
        "#e5dd45",
        "#7b3dcc",
        "#cc860c",
        "#73ff0f",
        "#2dccb4",
        "#ff0fa3",
        "#0ec2e6",
        "#3d87cc",
        "#e5520e",
    ]

    _layer = Layer.layer_dict.get((layer, datatype))
    if _layer is not None:
        color = _layer.color
        alpha = _layer.alpha
        if color is None:
            color = layer_colors[np.mod(layer, len(layer_colors))]
    else:
        color = layer_colors[np.mod(layer, len(layer_colors))]
        alpha = 0.6
    return {"color": color, "alpha": alpha}


def _draw_polygons(polygons, ax, **kwargs):
    from matplotlib.collections import PolyCollection

    coll = PolyCollection(polygons, **kwargs)
    ax.add_collection(coll)
    stacked_polygons = np.vstack(polygons)
    xmin, ymin = np.min(stacked_polygons, axis=0)
    xmax, ymax = np.max(stacked_polygons, axis=0)
    bbox = [xmin, ymin, xmax, ymax]
    return bbox


def _draw_line(x, y, ax, **kwargs):
    from matplotlib.lines import Line2D

    line = Line2D(x, y, **kwargs)
    ax.add_line(line)
    xmin, ymin = np.min(x), np.min(y)
    xmax, ymax = np.max(x), np.max(y)
    bbox = [xmin, ymin, xmax, ymax]
    return bbox


def _port_marker(port, is_subport):
    if is_subport:
        arrow_scale = 0.75
        rad = (port.orientation + 45) * np.pi / 180
        pm = +1
    else:
        arrow_scale = 1
        rad = (port.orientation - 45) * np.pi / 180
        pm = -1
    arrow_points = (
        np.array([[0, 0], [10, 0], [6, pm * 4], [6, pm * 2], [0, pm * 2]])
        / 35
        * port.width
        * arrow_scale
    )
    arrow_points += port.midpoint
    arrow_points = _rotate_points(
        arrow_points, angle=port.orientation, center=port.center
    )
    text_pos = np.array([np.cos(rad), np.sin(rad)]) * port.width / 3 + port.center
    return arrow_points, text_pos


def _draw_port(ax, port, is_subport, color):
    xbound, ybound = np.column_stack(port.endpoints)
    # plt.plot(x, y, 'rp', markersize = 12) # Draw port midpoint
    arrow_points, text_pos = _port_marker(port, is_subport)
    xmin, ymin = np.min(np.vstack([arrow_points, port.endpoints]), axis=0)
    xmax, ymax = np.max(np.vstack([arrow_points, port.endpoints]), axis=0)
    ax.plot(xbound, ybound, alpha=0.5, linewidth=3, color=color)  # Draw port edge
    ax.plot(
        arrow_points[:, 0], arrow_points[:, 1], alpha=0.8, linewidth=2, color=color
    )  # Draw port edge

    bbox = [xmin, ymin, xmax, ymax]
    return bbox


def _draw_port_as_point(ax, port, **kwargs):
    from matplotlib import pyplot as plt

    x = port.midpoint[0]
    y = port.midpoint[1]
    plt.plot(x, y, "r+", alpha=0.5, markersize=15, markeredgewidth=2)  # Draw port edge
    bbox = [
        x - port.width / 2,
        y - port.width / 2,
        x + port.width / 2,
        y + port.width / 2,
    ]

    return bbox
