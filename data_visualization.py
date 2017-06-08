import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column, row
from bokeh.models import CustomJS, ColumnDataSource, Slider, RangeSlider
from bokeh.plotting import Figure, curdoc
from bokeh.models.widgets import Select, Button
from bokeh.layouts import widgetbox
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from sklearn import preprocessing
from bokeh.models import HoverTool, BoxSelectTool, FuncTickFormatter, Range1d, LabelSet
from datetime import datetime as dt
from bokeh.charts import Donut, Bar, Scatter, color
output_file("callback.html")


accID = 3342844
#accID = 326779
#accID = 55266300


try:
    engine = create_engine("postgresql://postgres:dota2db@localhost/dota")
    pstats = pd.read_sql_table(str(accID)+'_pStats_full', engine, schema='dota_data')
    mstats = pd.read_sql_table(str(accID)+'_mStats_full', engine, schema='dota_data')
except:
    pass
    pstats = pd.read_excel('pstats.xlsx')
    mstats = pd.read_excel('mstats.xlsx')



pstats.index = pstats['index']
mstats.index = mstats['index']

output_file("lines.html")


pcols = ['gold_per_min', 'xp_per_min', 'kills', 'deaths', 'assists', 'hero_name', 'player_slot']
mcols = ['radiant_win', 'duration', 'start_time']

tommy = pd.concat([pstats[pcols], mstats[mcols]], axis=1)
tommy.dropna(inplace=True)
tommy['start_time'] = tommy['start_time'].apply(dt.fromtimestamp)

weekdays = {0:'monday', 1:'tuesday', 2:'wednesday', 3:'thursday', 4:'friday',5:'saturday',6:'sunday'}
tommy['weekday'] = [weekdays[d.weekday()] for d in tommy['start_time']]



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
tommy['duration'] = [d/60 for d in tommy['duration']]
tommy = tommy[(tommy['duration'] > 15) & (tommy.gold_per_min < 1300)]
tommy['date_str'] = [str(tommy['start_time'].iloc[i]) for i in range(len(tommy))]


h_dict = dict()
num = 1
for i in range(24):
    if i%num == 0:
        hstr = str(i)+'-'+str(i+num)
    if len(str(i)) > 1:
        h_dict[str(i)] = hstr
    else:
        h_dict["0"+str(i)] = "0"+hstr
        
h_time = []
days = list(weekdays.values())
for t in tommy['start_time']:
    hour = t.strftime("%H")    
    h_time.append(h_dict[hour])

tommy['h_time'] = h_time

t_wr = []
hours = []
c_bar = []
t_mat = []
t_int = sorted(list(tommy['h_time'].unique()))
for t in t_int:
    hours.append(t)
    df = tommy.copy()
    df = df[df['h_time']==t]
#    wr = df['victory'].value_counts()[1]/len(df)
    try:
        wr = df['victory'].value_counts()[1]/len(df)
    except:
        wr = 0
    t_mat.append(len(df))
    t_wr.append(wr)
    val = (wr-0.5)
    green = 1/(1+np.exp(-val*30))
    red = 1-green
    c_bar.append("#%02x%02x%02x" % ((int(red*255)), int(green*255), int(50)))
    
    
df_wr = pd.DataFrame()
df_wr['hours'] = hours
df_wr['t_wr'] = t_wr
df_wr['t_mat'] = t_mat
df_wr['t_wr_str'] = [str(round(df_wr['t_wr'].iloc[i], 2))+' %' for i in range(len(df_wr))]
df_wr['x'] = df_wr.index
df_wr.sort_values('hours', inplace=True)

source_bar = ColumnDataSource(df_wr)

# Bar plot 1
bar = figure(plot_width=1000, plot_height=400)
bar.vbar(x=range(len(df_wr)), width=0.8, bottom=0,
       top=df_wr['t_wr'], color=c_bar)

bar.xaxis.formatter = FuncTickFormatter(code="""
    return tick + "-" + Math.floor(tick+1)
""")
bar.grid[0].ticker.desired_num_ticks = 24

# Bar plot 2

labels = LabelSet(x='x', y='t_mat', text='t_wr_str', level='glyph',
              x_offset=-15, y_offset=0, source=source_bar, render_mode='canvas', text_font_size="10pt", text_font_style="bold")

hover_bar = HoverTool()
hover_bar.tooltips = [
        ("matches", "$y"),
        ("time", "$x"),]

bar2 = figure(plot_width=1000, plot_height=400, tools=[hover_bar])
bar2.vbar(x=range(len(df_wr)), width=0.8, bottom=0,
       top=df_wr['t_mat'], color=c_bar)

bar2.xaxis.formatter = FuncTickFormatter(code="""
    return tick + "-" + Math.floor(tick+1)
""")
bar2.grid[0].ticker.desired_num_ticks = 24
bar2.add_layout(labels)

# OTher stuff



#p = figure(title='Dist. of 10th Grade Students at Lee High',
#           x_range=Range1d(0, 24))
#p.scatter(x=list(range(24)), y='t_mat', size=8, source=source_bar)
#p.xaxis[0].axis_label = 'Weight (lbs)'
#p.yaxis[0].axis_label = 'Height (in)'
#
#
#
#p.add_layout(labels)


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

#wincols = ['x', 'y', 'victory', 'heroes', 'start_time', 'weekday']
tommy2 = tommy.copy()
tommy2['x'] = tommy2['duration']
#tommy2['y'] = tommy2['gold_p']

source = ColumnDataSource(data=tommy)
source1 = ColumnDataSource(data=tommy)
sourcewin = ColumnDataSource(data=tommy2)

selecty = Select(title="Y-axis:", value=cols[0], options=cols)
selectx = Select(title="X-axis:", value=cols[0], options=cols)

table_cols = ['hero', 'gold_per_min', 'xpm_per_min', 'kills', "deaths", "assists", "duration", "winrate", "matches"]
data_table = dict()
for key in table_cols:
    data_table[key] = []

source_table = ColumnDataSource(data=data_table)
select_hero = Select(title="Hero:", value=unique_heroes[0], options=unique_heroes)

factions = ['both', 'radiant', 'dire']
select_faction = Select(title="Faction:", value=factions[0], options=factions)

range_slider = RangeSlider(start=0, end=len(tommy), range=(1,len(tommy)), step=1, title="matches")

def calc_winrate(attr, old, new, select_hero=select_hero, sourcewin=sourcewin, source=source, range_slider=range_slider, select_faction=select_faction):
    df = pd.DataFrame(source.data)
    faction = select_faction.value
    start = range_slider.range[0]
    end = range_slider.range[1]
    df = df.iloc[start:end, :]
    hero = select_hero.value
    if hero != 'All Heroes':
        df = df[df['heroes'] == hero]
    if faction != 'both':
        df = df[df['faction'] == faction]
    df['x'] = df['start_time']
    winrate = []
    for i in range(1, len(df)+1):
        try:
            winrate.append(df['victory'][0:i].value_counts()[1]/i)
        except:
            winrate.append(0)
    df['y'] = winrate
    sourcewin.data  = ColumnDataSource(df).data

range_slider.on_change('range', calc_winrate)

def updatey(attr, old, new, source1=source1):
    data = source1.data
    data['y'] = data[new]
    source1.data = data

def updatex(attr, old, new, source1=source1):
    data = source1.data
    data['x'] = data[new]
    source1.data = data
  
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

hover = HoverTool()
hover.tooltips = [
    ("(x,y)", "($x, $y)"),
    ("hero", "@heroes"),
    ("kills", "@kills"),
    ("deaths", "@deaths"),
    ("assists", "@assists"),
]

hover_wr = HoverTool()
hover_wr.tooltips = [
        ("winrate", "$y"),
        ("date", "@date_str"),]

TOOLS = [BoxSelectTool(), hover]

plot = Figure(plot_width=400, plot_height=400, tools=TOOLS)
plot.circle('x', 'y', source=source1, line_alpha=0.6, color='colors')

plot2 = Figure(plot_width=1000, plot_height=400,tools=[hover_wr], x_axis_type='datetime')
plot2.line('x', 'y', source=sourcewin, line_alpha=0.6)
bottom, top = 0.45, 0.6
plot2.y_range=Range1d(bottom, top)

plot2.x_range=Range1d(tommy['start_time'].iloc[400], tommy['start_time'].iloc[-1])


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




update_table('value', 'w/e', 'All Heroes', source=source, source_table=source_table, select_hero=select_hero, select_faction=select_faction)
updatehero('value', 'w/e', 'All Heroes', source=source, source1=source1, select_hero=select_hero, select_faction=select_faction)
calc_winrate('range', 'w/e', range(1, len(tommy)), select_hero=select_hero, sourcewin=sourcewin, source=source, range_slider=range_slider)

selecty.on_change('value', updatey)
selectx.on_change('value', updatex)
selectx.on_change('value', print_stuff)
selecty.on_change('value', print_stuff)
select_hero.on_change('value', print_stuff)
select_hero.on_change('value', update_table)
select_hero.on_change('value', updatehero)
select_hero.on_change('value', calc_winrate)
select_faction.on_change('value', update_table)
select_faction.on_change('value', updatehero)
select_faction.on_change('value', calc_winrate)

selections = column(selecty, selectx, select_hero, select_faction)

col1 = row(selections, plot, plot2)

table_slider = row(table, range_slider)

bars = row(bar, bar2)


layout = column(col1, table_slider, bars)
#show(bar)
curdoc().add_root(layout)

#%%
#
#import matplotlib.pyplot as plt
#
##plt.hist(list(tommy['xp_per_min']), bins=100)
##plt.hist(list(pstats['level']), bins=25)
#xp_gained = [tommy['xp_per_min'].iloc[i]*tommy['duration'].iloc[i] for i in range(len(tommy))]
##xp_gained_sorted = sorted(xp_gained)
##plt.hist(xp_gained, bins=100)
#p = plt.scatter(list(tommy['duration']), tommy['xp_per_min'])
#x = np.arange(min(tommy['duration']), max(tommy['duration']), 0.01)
#maxxp = 32.4e3
#y = maxxp/x
#
#maxxp2 = 26.9e3
#y2 = maxxp2/x
#
#plt.plot(x, y)
#
#plt.plot(x, y2)
#%%

