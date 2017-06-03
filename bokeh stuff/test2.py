import numpy as np

from bokeh.models import ColumnDataSource
from bokeh.plotting import Figure

from bokeh.models.widgets import Select,TextInput
from bokeh.models.layouts import Column, Row
import bokeh.io 
from bokeh.models import CustomJS

N = 200

# Define the data to be used
x = np.linspace(0,4.*np.pi,N)
y = 3*np.cos(2*np.pi*x + np.pi*0.2)
z = 0.5*np.sin(2*np.pi*0.8*x + np.pi*0.4)

source = ColumnDataSource(data={'x':x,'y':y, 'X': x, 'cos':y,'sin':z})


code="""
        var data = source.get('data');
        var r = data[cb_obj.get('value')];
        var {var} = data[cb_obj.get('value')];
        //window.alert( "{var} " + cb_obj.get('value') + {var}  );
        for (i = 0; i < r.length; i++) {{
            {var}[i] = r[i] ;
            data['{var}'][i] = r[i];
        }}
        source.trigger('change');
    """

callbackx = CustomJS(args=dict(source=source), code=code.format(var="x"))
callbacky = CustomJS(args=dict(source=source), code=code.format(var="y"))

# create a new plot 
plot = Figure(title=None)

# Make a line and connect to data source
plot.line(x="x", y="y", line_color="#F46D43", line_width=6, line_alpha=0.6, source=source)


# Add list boxes for selecting which columns to plot on the x and y axis
yaxis_select = Select(title="Y axis:", value="cos",
                           options=['X','cos','sin'], callback=callbacky)


xaxis_select = Select(title="X axis:", value="x",
                           options=['X','cos','sin'], callback=callbackx)


# Text input as a title
text = TextInput(title="title", value='my sine wave plotter')

# Layout widgets next to the plot                     
controls = Column(text,yaxis_select,xaxis_select)

layout = Row(controls,plot,width=800)

bokeh.io.show(layout)