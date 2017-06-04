import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column
from bokeh.models import CustomJS, ColumnDataSource, Slider
from bokeh.plotting import Figure, curdoc
from bokeh.models.widgets import Select, Button
from bokeh.layouts import widgetbox
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from sklearn import preprocessing
output_file("callback.html")

try:
    engine = create_engine("postgresql://postgres:dota2db@localhost/dota")
    pstats = pd.read_sql_table('3342844_pStats_full', engine, schema='dota_data')
    mstats = pd.read_sql_table('3342844_mStats_full', engine, schema='dota_data')
except:
    pass
    pstats = pd.read_excel('pstats.xlsx')
    mstats = pd.read_excel('mstats.xlsx')



pstats.index = pstats['index']
mstats.index = mstats['index']

output_file("lines.html")


pcols = ['gold_per_min', 'xp_per_min', 'kills', 'deaths', 'assists', 'hero_name', 'player_slot']
mcols = ['radiant_win', 'duration']

tommy = pd.concat([pstats[pcols], mstats[mcols]], axis=1)
tommy.dropna(inplace=True)
victory = []
faction = []
kda = []
for ind in tommy.index:
    if tommy.player_slot[ind] < 5:
        faction.append('radiant')
        if tommy.radiant_win[ind] == True:
            victory.append(True)
        else:
            victory.append(False)
    else:
        faction.append('dire')
        if tommy.radiant_win[ind] == True:
            victory.append(False)
        else:
            victory.append(True)
    if tommy.deaths[ind] == 0:
        kda.append((tommy.kills[ind]+tommy.assists[ind]))
    else:
        kda.append((tommy.kills[ind]+tommy.assists[ind])/tommy.deaths[ind])



tommy['victory'] = victory
tommy['faction'] = faction
tommy['kda'] = kda

wins = tommy.victory.value_counts()[1]
losses = tommy.victory.value_counts()[0]

print('winrate = ', wins/len(tommy))

tgpm_w = sum(tommy["gold_per_min"][tommy['victory']])
tgpm_l = sum(tommy["gold_per_min"][~tommy['victory']])


print('avg gpm when wining: ', tgpm_w/wins, '\navg gpm when losing: ', tgpm_l/losses)

tommy['duration'] = [d/60 for d in tommy['duration']]

tommy = tommy[(tommy['duration'] > 15) & (tommy.gold_per_min < 1300)]


cols = [col for col in tommy.columns if tommy[col].dtype in (np.dtype('int64'), np.dtype('float64'))]

colors = []
for i in range(len(tommy)):
    if tommy.victory.iloc[i] == True:
        colors.append('blue')
#    colors.append("#%02x%02x%02x" % (int(tommy.age.iloc[i]*255), int(tommy.age.iloc[i]*100), int(tommy.age.iloc[i]*150)))
    else:
        colors.append('red')

unique_heroes = ['All Heroes']+sorted(list(tommy.hero_name.unique()))
heroes = list(tommy.hero_name)
data = dict(x=list(tommy[cols[0]]), y=list(tommy[cols[0]]), colors=colors, heroes=heroes)
for col in cols:
    data[col] = list(tommy[col])


source = ColumnDataSource(data=data)

source1 = ColumnDataSource(data=data)

def callbacky(source1=source1, window=None):
    data = source1.data
    f = cb_obj.value
    x, y = data['x'], data['y']
    data['y'] = data[f]
    source1.trigger('change')

def callbackx(source1=source1, window=None):
    data = source1.data
    f = cb_obj.value
    x, y = data['x'], data['y']
    data['x'] = data[f]
    source1.trigger('change')

selecty = Select(title="Y-axis:", value=cols[0], options=cols, callback=CustomJS.from_py_func(callbacky))
selectx = Select(title="X-axis:", value=cols[0], options=cols, callback=CustomJS.from_py_func(callbackx))


def callback_hero(source=source, source1=source1, selecty=selecty, selectx=selectx, window=None):
    data = source.data
    data1 = source1.data
    f = cb_obj.value
    x, y, c = data[selectx.value], data[selecty.value], data['colors']
    xnew = []
    ynew = []
    colorsnew = []
    for i in range(len(x)):
        if f == 'All Heroes':
            xnew = x
            ynew = y
            colorsnew = c
            break
        elif data['heroes'][i] == f:
            xnew.append(x[i])
            ynew.append(y[i])
            colorsnew.append(c[i])
    data1['x'] = xnew
    data1['y'] = ynew
    data1['colors'] = colorsnew
    source1.trigger('change')
 
select_hero = Select(title="Hero:", value=unique_heroes[0], options=unique_heroes, callback=CustomJS.from_py_func(callback_hero))

def print_stuff(attr, old, new, select_hero=select_hero, selecty=selecty, selectx=selectx):
    print(select_hero.value, selecty.value, selectx.value)



plot = Figure(plot_width=400, plot_height=400)
plot.circle('x', 'y', source=source1, line_alpha=0.6, color='colors')

data_table = dict()
filter_list = ['kills', 'xp_per_min', 'assists', 'duration', 'kda', 'deaths', 'gold_per_min']
data_table['hero'] = [select_hero.value]
for key in source1.data.keys():
    if key in filter_list:
        data_table[key] = [round(np.mean(data[key]),2)]
        
source_table = ColumnDataSource(data=data_table)

columns = [  
        TableColumn(field="hero", title="hero"),
        TableColumn(field="xp_per_min", title="xpm"),
        TableColumn(field="gold_per_min", title="gpm"),
        TableColumn(field="kills", title="kills"),
        TableColumn(field="deaths", title="deaths"),
        TableColumn(field="assists", title="assists"),
        TableColumn(field="kda", title="kda"),
        TableColumn(field="duration", title="duration"), 
    ]
table = DataTable(source=source_table, columns=columns, width=600, height=280)


#def update_table(dict(source=source, source_table=source_table, select_hero=select_hero, filter_list=filter_list, window=None)):
#dawdaw = dict(source=source, source_table=source_table, select_hero=select_hero, filter_list=filter_list)
#def update_table(source=source, source_table=source_table, select_hero=select_hero, filter_list=filter_list, window=None):
#    print(new)
#    new = select_hero.value
#    data = source    
#    data2 = source_table
#    df = pd.DataFrame(data)
#    df = df[df['heroes'] == new]
#    df = df[filter_list]
#    data2['hero'] = [new]
#    for key in df:
#        data2[key] = [df[key].mean()] 
#    source_table.trigger('change')
#    print('LEL')
#
#button = Button(label="Update Table", button_type="success", callback=CustomJS.from_py_func(update_table))

#['kills', 'xp_per_min', 'assists', 'duration', 'kda', 'deaths', 'gold_per_min']

selectx.on_change('value', print_stuff)
selecty.on_change('value', print_stuff)
select_hero.on_change('value', print_stuff)
#select_hero.on_change('value', CustomJS.from_py_func(update_table))






layout = column(selecty, selectx, select_hero, plot,table)

#show(layout)
curdoc().add_root(layout)