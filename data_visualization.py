import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from bokeh.plotting import figure, output_file, show



engine = create_engine("postgresql://postgres:dota2db@localhost/dota")

pstats = pd.read_sql_table('3342844_pStats_full', engine, schema='dota_data')
mstats = pd.read_sql_table('3342844_mStats_full', engine, schema='dota_data')

pstats.index = pstats['index']
mstats.index = mstats['index']

output_file("lines.html")

gpm = list(pstats['gold_per_min'])
xpm = list(pstats['xp_per_min'])

pcols = ['gold_per_min', 'xp_per_min', 'kills', 'deaths', 'assists', 'hero_name', 'player_slot']
mcols = ['radiant_win', 'duration']

tommy = pd.concat([pstats[pcols], mstats[mcols]], axis=1)

victory = []
faction = []
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

tommy['victory'] = victory
tommy['faction'] = faction

wins = tommy.victory.value_counts()[1]
losses = tommy.victory.value_counts()[0]

print('winrate = ', wins/len(tommy))

tgpm_w = sum(tommy["gold_per_min"][tommy['victory']])
tgpm_l = sum(tommy["gold_per_min"][~tommy['victory']])


print('avg gpm when wining: ', tgpm_w/wins, '\navg gpm when losing: ', tgpm_l/losses)
#p = figure(title='xpm vs gpm', x_axis_label='xpm', y_axis_label='xpm')
#
#p.scatter(xpm, gpm, legend="Temp.", line_width=2)
#
#show(p)