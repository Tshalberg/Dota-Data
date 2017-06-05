from bokeh.models.widgets import Select, Button, DataTable, DateFormatter, TableColumn
from bokeh.layouts import widgetbox, column
from bokeh.io import curdoc
from bokeh.plotting import Figure
from bokeh.models import ColumnDataSource

import pandas as pd
import numpy as np


x = list(range(10))
y = [x**2 for x in x]
z = [x**3 for x in x]
k = [x**0.5 for x in x]

df = pd.DataFrame(dict(X=x, Y=y, x=x, y=y, z=z, k=k))

s1 = ColumnDataSource(df)

options =  ['x', 'y', 'z', 'k']

def get_new_data(new):
    data = s1.data
    df = pd.DataFrame(data)
    df['Y'] = df[new]
    return ColumnDataSource(data=df)
    

def update(attr, old, new):
    stuff = get_new_data(new)
    s1.data.update(stuff.data)
    

select = Select(title='select y-axis', options=options, value='x')
select.on_change('value', update)


plot = Figure(plot_height=400, plot_width=400)
plot.line('X', 'Y', source=s1, line_width=3, line_alpha=0.6)

#update('value', 'Y', 'z')

layout = column(select, plot)

curdoc().add_root(layout)
