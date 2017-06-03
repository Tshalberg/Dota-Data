from bokeh.layouts import column
from bokeh.models import CustomJS, ColumnDataSource, Slider
from bokeh.plotting import Figure, output_file, show
from bokeh.models.widgets import Select
import numpy as np
output_file("callback.html")

foo = [x*0.005 for x in range(0, 200)]
bar = [i*0.1 for i in range(0, 200)]
baz = [np.sin(b) for b in bar]
quux = [np.cos(b) for b in bar]

source = ColumnDataSource(data=dict(x=foo, y=foo, foo=foo, bar=bar, baz=baz, quux=quux))

plot = Figure(plot_width=400, plot_height=400)
plot.scatter('x', 'y', source=source, line_width=3, line_alpha=0.6)

def callbacky(source=source, window=None):
    data = source.data
    f = cb_obj.value
    x, y = data['x'], data['y']
    data['y'] = data[f]

    source.trigger('change')

def callbackx(source=source, window=None):
    data = source.data
    f = cb_obj.value
    x, y = data['x'], data['y']
    data['x'] = data[f]
    source.trigger('change')



selecty = Select(title="Y-axis:", value="foo", options=["foo", "bar", "baz", "quux"], callback=CustomJS.from_py_func(callbacky))
selectx = Select(title="X-axis:", value="foo", options=["foo", "bar", "baz", "quux"], callback=CustomJS.from_py_func(callbackx))

layout = column(selecty, selectx, plot)

show(layout)