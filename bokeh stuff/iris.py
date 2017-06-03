from bokeh.plotting import figure, output_file, show
from bokeh.models import CustomJS
from bokeh.models.sources import ColumnDataSource
from bokeh.layouts import column
from bokeh.models.widgets import DataTable, TableColumn, Button

output_file("subset_example.html")

data = dict(
        index = list(range(10)),
        x = list(range(10)),
        y = list(range(10)),
        z = ['some other data'] * 10
    )

filtered_index = [i for i, y in enumerate(data['y']) if y > 5]
filtered_data = dict(
        index = filtered_index,
        x = [x for i, x in enumerate(data['x']) if i in filtered_index],
        y = [y for i, y in enumerate(data['y']) if i in filtered_index],
        z = ['some other data'] * len(filtered_index)
)

source1 = ColumnDataSource(data)
source2 = ColumnDataSource(filtered_data)

fig1 = figure(plot_width=300, plot_height=300)
fig1.circle(x='x', y='y', size=10, source=source1)

columns = [
        TableColumn(field="y", title="Y"),
        TableColumn(field="z", title="Text"),
    ]
data_table = DataTable(source=source2, columns=columns, width=400, height=280)

button = Button(label="Select")
button.callback = CustomJS(args=dict(source1=source1, source2=source2), code="""
        var inds_in_source2 = source2['selected']['1d'].indices;

        var d = source2['data'];
        var inds = []

        if (inds_in_source2.length == 0) { return; }

        for (i = 0; i < inds_in_source2.length; i++) {
            ind2 = inds_in_source2[i]
            inds.push(d['index'][ind2])
        }

        source1['selected']['1d'].indices = inds
        source1.trigger('change');
    """)

show(column(fig1, data_table, button))