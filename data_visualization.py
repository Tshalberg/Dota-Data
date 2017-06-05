import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column, row
from bokeh.models import CustomJS, ColumnDataSource, Slider
from bokeh.plotting import Figure, curdoc
from bokeh.models.widgets import Select, Button
from bokeh.layouts import widgetbox
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from sklearn import preprocessing
from bokeh.models import HoverTool, BoxSelectTool

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
tommy['match_id'] = tommy.index

wins = tommy.victory.value_counts()[1]
losses = tommy.victory.value_counts()[0]

print('winrate = ', wins/len(tommy))

tgpm_w = sum(tommy["gold_per_min"][tommy['victory']])
tgpm_l = sum(tommy["gold_per_min"][~tommy['victory']])


print('avg gpm when wining: ', tgpm_w/wins, '\navg gpm when losing: ', tgpm_l/losses)

tommy['duration'] = [d/60 for d in tommy['duration']]

tommy = tommy[(tommy['duration'] > 15) & (tommy.gold_per_min < 1300)]


cols = [col for col in tommy.columns if tommy[col].dtype in (np.dtype('int64'), np.dtype('float64')) and col != 'player_slot']

colors = []
for i in range(len(tommy)):
    if tommy.victory.iloc[i] == True:
        colors.append('blue')
#    colors.append("#%02x%02x%02x" % (int(tommy.age.iloc[i]*255), int(tommy.age.iloc[i]*100), int(tommy.age.iloc[i]*150)))
    else:
        colors.append('red')
tommy['colors'] = colors
unique_heroes = ['All Heroes']+sorted(list(tommy.hero_name.unique()))
tommy.rename(columns={'hero_name':'heroes'}, inplace=True)
tommy['x'] = tommy[cols[0]]
tommy['y'] = tommy[cols[0]]

source = ColumnDataSource(data=tommy)

source1 = ColumnDataSource(data=tommy)


def updatey(attr, old, new, source1=source1):
    data = source1.data
    data['y'] = data[new]
    source1.data = data

def updatex(attr, old, new, source1=source1):
    data = source1.data
    data['x'] = data[new]
    source1.data = data

selecty = Select(title="Y-axis:", value=cols[0], options=cols)
selectx = Select(title="X-axis:", value=cols[0], options=cols)
selecty.on_change('value', updatey)
selectx.on_change('value', updatex)

table_cols = ['hero', 'gold_per_min', 'xpm_per_min', 'kills', "deaths", "assists", "duration", "winrate", "matches"]
data_table = dict()
for key in table_cols:
    data_table[key] = []

source_table = ColumnDataSource(data=data_table)

select_hero = Select(title="Hero:", value=unique_heroes[0], options=unique_heroes)

factions = ['both', 'radiant', 'dire']
select_faction = Select(title="Faction:", value=factions[0], options=factions)
  
def updatehero(attr, old, new, source=source, source1=source1, select_hero=select_hero, selecty=selecty, selectx=selectx, select_faction=select_faction):
    df = pd.DataFrame(source.data)
    hero = select_hero.value
    faction = select_faction.value
    if hero != 'All Heroes':
        df = df[df['heroes'] == hero]
    if faction != 'both':
        df = df[df['faction'] == faction]
    df['x'] = df[selectx.value]
    df['y'] = df[selecty.value]
    source1.data = ColumnDataSource(df).data

def print_stuff(attr, old, new, select_hero=select_hero, selecty=selecty, selectx=selectx):
    print(select_hero.value, selecty.value, selectx.value)


hover = HoverTool()
hover.tooltips = [
    ("(x,y)", "($x, $y)"),
    ("hero", "@heroes"),
    ("kills", "@kills"),
    ("deaths", "@deaths"),
    ("assists", "@assists"),
]

TOOLS = [BoxSelectTool(), hover]

plot = Figure(plot_width=400, plot_height=400, tools=TOOLS)
plot.circle('x', 'y', source=source1, line_alpha=0.6, color='colors')



columns = [  
        TableColumn(field="hero", title="hero"),
        TableColumn(field="gold_per_min", title="gpm"),
        TableColumn(field="xp_per_min", title="xpm"),
        TableColumn(field="kills", title="kills"),
        TableColumn(field="deaths", title="deaths"),
        TableColumn(field="assists", title="assists"),
        TableColumn(field="kda", title="kda"),
        TableColumn(field="duration", title="duration"),
        TableColumn(field="winrate", title="winrate"),
        TableColumn(field="matches", title="matches"),
    ]

table = DataTable(source=source_table, columns=columns, width=1000, height=280)

def table_data(df, hero, faction):
    if hero != 'All Heroes':
        df = df[df['heroes'] == hero]
    if faction != 'both':
        df = df[df['faction'] == faction]
#    df = df[cols]
    means = df.mean()
    data = dict()
    data['hero'] = [hero]
    data['matches'] = [len(df)]
    data['winrate'] = [round(df.victory.value_counts()[1]/len(df), 3)]
    for i in range(len(means)):
        if means.index[i] in cols:
            data[means.index[i]] = [round(means[i], 2)]
    return data

def update_table(attr, old, new, source=source, source_table=source_table, select_hero=select_hero, select_faction=select_faction):
    faction = select_faction.value
    hero = select_hero.value
    data = source.data
    df = pd.DataFrame(data)
    data2 = table_data(df, hero, faction)
    data2 = ColumnDataSource(data2)
    source_table.data = data2.data


update_table('value', 'w/e', 'All Heroes', source=source, source_table=source_table, select_hero=select_hero, select_faction=select_faction)
updatehero('value', 'w/e', 'All Heroes', source=source, source1=source1, select_hero=select_hero, select_faction=select_faction)


selectx.on_change('value', print_stuff)
selecty.on_change('value', print_stuff)
select_hero.on_change('value', print_stuff)
select_hero.on_change('value', update_table)
select_hero.on_change('value', updatehero)
select_faction.on_change('value', update_table)
select_faction.on_change('value', updatehero)

selections = column(selecty, selectx, select_hero, select_faction)

col1 = row(selections, plot)

layout = column(col1, table)

curdoc().add_root(layout)