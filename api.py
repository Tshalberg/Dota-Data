
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import dota2api
from sqlalchemy import create_engine, MetaData
from sqlalchemy.engine import reflection

engine = create_engine('mysql+pymysql://dota2:dota2db@localhost/dota2')
metadata = MetaData()

insp = reflection.Inspector.from_engine(engine)
print(insp.get_table_names())

key = '6355122600AFA6460C66EBCA179B83C1'
api = dota2api.Initialise(key)
accID = 3342844

match = api.get_match_details(match_id=2934893085)
hist1 = api.get_match_history(account_id=accID, hero_id=74)
oldest = hist1['matches'][-1]['match_id']
#hist2 = hist1 = api.get_match_history(account_id=accID, hero_id=74, start_at_match_id =oldest)

#%%
def player_to_db(accID, api, engine):
    hist = api.get_match_history(account_id=accID)
    
    dfPlayer = pd.DataFrame()
    dfMatch = pd.DataFrame()
    for i in range(len(hist['matches'])):
        matchID = hist['matches'][i]['match_id']
        match = api.get_match_details(match_id=matchID)
        players = match['players']
        for player in players:
            if player['account_id'] == accID:
                chosenPlayer = player
                break
                
        del chosenPlayer['ability_upgrades']
        dfPlayerTemp = pd.DataFrame(chosenPlayer, index=[matchID])
        dfPlayer = dfPlayer.append(dfPlayerTemp)
        del match['players']
        dfMatchTemp = pd.DataFrame(match, index=[matchID])
        dfMatch = dfMatch.append(dfMatchTemp)

    try:
        dfPlayer.drop(['additional_units'], axis=1, inplace=True)
    except:
        pass
    dfPlayer.to_sql(str(accID)+'_pStats', engine, if_exists='replace')
    dfMatch.to_sql(str(accID)+'_mStats', engine, if_exists='replace')

def create_herodict():
    heroes = api.get_heroes()
    heroDict = {}
    for hero in heroes['heroes']:
        heroDict[hero['id']] = hero['localized_name']
    return heroDict

accID = 3342844
#accID = 326779
#player_to_db(accID, api, engine)
#%%
heroDict = create_herodict()
accID = 117417755

dfP = pd.read_sql(str(accID)+'_pstats_full', engine, index_col=['index'])
dfM = pd.read_sql(str(accID)+'_mstats_full', engine, index_col=['index'])

cols = []
for col in dfP.columns:
    if 'backpack' in col:
        cols.append(str(col))
    if 'item' in col:
        cols.append(str(col))
        
cols = cols+['account_id']
dfP.drop(cols, axis=1, inplace=True)

#%%
print(dfP.dtypes)
dfP['hero_id'] = dfP['hero_id'].astype('category')
print(dfP.dtypes)

#%%
nameliste = [str(heroDict[key])+str(key) for key in heroDict]
nameliste.sort()

countliste = dfP.hero_id.value_counts()
countliste = countliste[countliste > 25]
#%%
#%%
df = dfP.copy()
#liste = [11, 74, 9, 26, 22, 21, 37, 67, 32, 62]
#liste = [98, 32, 67, 49, 50, 20, 11, 74, 26]
#liste = list(df.hero_id.unique())
liste = list(countliste.index)
df = df[df.hero_id.isin(liste)]
#df['hero_id'] = df['hero_id'].astype('category')
df.drop(['hero_name','leaver_status_description','leaver_status_name', 'leaver_status', 'player_slot', 
         'scaled_hero_damage', 'scaled_hero_healing', 'scaled_tower_damage',
         'gold', 'gold_spent', 'hero_healing', 'hero_damage', 'tower_damage', 'denies', 'level', 'deaths'], 
         axis=1, inplace=True)




names = [heroDict[i] for i in liste]
hero_id = sorted(list(df.hero_id.unique()))
norm = plt.Normalize()
hero_id2 = list(range(len(hero_id)))
colors = plt.cm.spectral(norm(hero_id2))
hero_color = dict()

for i in range(len(hero_id)):
    hero_color[hero_id[i]] = colors[i]

hero_color_plot = []
for i in df.hero_id:
    hero_color_plot.append(hero_color[i])


allPlot = []
for num in liste:
    allPlot.append([i for i in range(len(df.index)) if df.hero_id.iloc[i] == num])
  
df.drop('hero_id', axis=1, inplace=True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()  
df = scaler.fit_transform(df)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(df)

T = pca.transform(df)
fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
for i in range(len(liste)):
    if i < -1+len(liste)/2:
        plt.scatter(T[allPlot[i],0], T[allPlot[i],1], s=50, marker='o', color=hero_color[liste[i]])
    else:
        plt.scatter(T[allPlot[i],0], T[allPlot[i],1], s=100, marker='*', color=hero_color[liste[i]])
#    ax.scatter(T[allPlot[i],0], T[allPlot[i],1], T[allPlot[i],2], s=50, marker='o', c=hero_color[liste[i]])
names = [heroDict[i] for i in liste]
plt.legend(names)
#%%
#dfcorr = df.corr()
#plt.imshow(df.corr(), cmap=plt.cm.Blues, interpolation='nearest')
#plt.colorbar()
#tick_marks = [i for i in range(len(df.columns))]
#plt.xticks(tick_marks, df.columns, rotation='vertical')
#plt.yticks(tick_marks, df.columns)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(df)

T = pca.transform(df)
fig = plt.figure()
plt.scatter(T[:,0], T[:,1], s=25, marker='o', color=hero_color_plot )
#plt.legend([])
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(T[:,0], T[:,1], T[:,2], s=25, marker='o', color=hero_color_plot)




#%%

dfPavg = []

for i in dfP.hero_id.unique():
    dfPhero = dfP[dfP.hero_id == i]
    dfPhero_avg = dfPhero.mean()
    dfPhero_avg.name = i
    dfPavg.append(dfPhero_avg)

dfPavg_all = pd.DataFrame(dfPavg)

avg_gpm = dfPavg_all.gold_per_min
avg_gpm = pd.DataFrame(avg_gpm)
avg_gpm['heroname'] = 0

for ind in avg_gpm.index:
    avg_gpm.loc[ind, 'heroname'] = heroDict[ind]


#plt.scatter(dfPavg_all.hero_id, dfPavg_all.last_hits)
#dfPhero.mean()

#%%

plt.close(ax1)
hero_id = sorted(list(dfP.hero_id.unique()))
corr =  dfP.corr()
norm = plt.Normalize()
colors = plt.cm.jet(norm(hero_id))
hero_color = dict()

x = 'hero_id'
y = 'gold_per_min'

for i in range(len(hero_id)):
    hero_color[hero_id[i]] = colors[i]

hero_color_plot = []
for i in dfP.hero_id:
    hero_color_plot.append(hero_color[i])

ax1 = plt.figure()
plt.scatter(dfP[x], dfP[y], color=hero_color_plot, s=10)
plt.xlabel(x)
plt.ylabel(y)



#plt.hist(dfP['gold_per_min'], bins=25)
#matchID = 2090547829

#%%

#hist1 = api.get_match_history(account_id=accID, hero_id=74)
#oldest = hist1['matches'][-1]['match_id']
#hist2 = hist1 = api.get_match_history(account_id=accID, hero_id=74, start_at_match_id =oldest)


#accID = 3342844

def player_to_db_full(accID, api, engine):
    matches = []
    for hero_id in range(1, 114):
        cond = True
        k = 0
        while cond:
            if k == 0:
                hist = api.get_match_history(account_id=accID, hero_id=hero_id)
                matches = matches + hist['matches']
                k+=1
            if len(hist['matches']) == 100:
                oldest = hist['matches'][-1]['match_id']
                hist = api.get_match_history(account_id=accID, hero_id=hero_id, start_at_match_id =oldest)
                matches = matches+hist['matches']
            else:
                cond = False
                
    matchIDs = []
    for i in range(len(matches)):
        matchID = matches[i]['match_id']
        matchIDs.append(matchID)
    matchIDs = list(set(matchIDs))
    
    dfPlayer = pd.DataFrame()
    dfMatch = pd.DataFrame()
    count = 1
    for matchID in matchIDs:
        if count%100 == 0:
            print(count)
            
        match = api.get_match_details(match_id=matchID)
        players = match['players']
        for player in players:
            if player['account_id'] == accID:
                chosenPlayer = player
                break
        try:
            del chosenPlayer['ability_upgrades']
        except:
            pass
        
        dfPlayerTemp = pd.DataFrame(chosenPlayer, index=[matchID])
        dfPlayer = dfPlayer.append(dfPlayerTemp)
        
        del match['players']
        try:
            del match['picks_bans']
        except:
            pass
    
        dfMatchTemp = pd.DataFrame(match, index=[matchID])
        dfMatch = dfMatch.append(dfMatchTemp)
        count += 1
    
    
    dfPlayer.drop(['additional_units'], axis=1, inplace=True, errors='ignore')
    
    dfPlayer.to_sql(str(accID)+'_pStats_full', engine, if_exists='replace')
    dfMatch.to_sql(str(accID)+'_mStats_full', engine, if_exists='replace')
    print('Extraction finished. ',count,' matches were extracted.')

accID = 117417755
#accID = 3342844
#player_to_db_full(accID, api, engine)

#%%

def all_players_to_db(accID, api, engine):
    matches = []
    for hero_id in range(1,114):
        cond = True
        k = 0
        while cond:
            if k == 0:
                hist = api.get_match_history(account_id=accID, hero_id=hero_id)
                matches = matches + hist['matches']
                k+=1
            if len(hist['matches']) == 100:
                oldest = hist['matches'][-1]['match_id']
                hist = api.get_match_history(account_id=accID, hero_id=hero_id, start_at_match_id =oldest)
                matches = matches+hist['matches']
            else:
                cond = False
                
    matchIDs = []
    for i in range(len(matches)):
        matchID = matches[i]['match_id']
        matchIDs.append(matchID)
    matchIDs = list(set(matchIDs))
    
    dfPlayer = pd.DataFrame()
    dfMatch = pd.DataFrame()
    count = 1
    for matchID in matchIDs:
        if count%100 == 0:
            print(count)
            
        match = api.get_match_details(match_id=matchID)
        players = match['players']
        chosenPlayer = []
        for player in players:
            chosenPlayer.append(player)
    #        if player['account_id'] == accID:
    #            chosenPlayer = player
    #            break
        try:
            del chosenPlayer['ability_upgrades']
        except:
            pass
        index = [matchID  for i in range(len(chosenPlayer))]
        dfPlayerTemp = pd.DataFrame(chosenPlayer, index=index)
        dfPlayer = dfPlayer.append(dfPlayerTemp)
        
        del match['players']
        try:
            del match['picks_bans']
        except:
            pass
    
        dfMatchTemp = pd.DataFrame(match, index=[matchID])
        dfMatch = dfMatch.append(dfMatchTemp)
        count += 1
    
    
    dfPlayer.drop(['additional_units'], axis=1, inplace=True, errors='ignore')
    dfPlayer.drop(['ability_upgrades'], axis=1, inplace=True, errors='ignore')
    dfPlayer.drop(['custom_game'], axis=1, inplace=True, errors='ignore')
    dfPlayer.to_sql(str(accID)+'_pstats_full', engine, if_exists='replace')
    dfMatch.to_sql(str(accID)+'_mstats_full', engine, if_exists='replace')
    print('Extraction finished. ',count,' matches were extracted.')






