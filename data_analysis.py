import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import dota2api
from sqlalchemy import create_engine, MetaData
from sqlalchemy.engine import reflection

def getHeroesList(api):
    heroes = api.get_heroes()
    heroDict = {}
    for hero in heroes['heroes']:
        heroDict[hero['id']] = hero['localized_name']
    
    nameliste = [str(heroDict[key])+'_'+str(key) for key in heroDict]
    nameliste.sort()
    return heroDict, nameliste

engine = create_engine('mysql+pymysql://dota2:dota2db@localhost/dota2')
metadata = MetaData()

insp = reflection.Inspector.from_engine(engine)
current_tables = insp.get_table_names()
print(current_tables)

key = '6355122600AFA6460C66EBCA179B83C1'
api = dota2api.Initialise(key)

tablename = '3342844_pstats_full'


heroDict, nameliste = getHeroesList(api)





dfFull = pd.read_sql(tablename, engine)
#%%
#liste = [11, 74, 9, 26, 22, 21, 37, 67, 32, 62]
chosen_columns = ['assists', 'gold_per_min',  'kills',  'last_hits',  'xp_per_min',  'hero_id']
liste = [11, 3, 1, 26, 27]
df = dfFull[chosen_columns].copy()

df.dropna(inplace=True)

names = [heroDict[i] for i in liste]
hero_id = sorted(list(df.hero_id.unique()))
norm = plt.Normalize()
hero_id2 = list(range(len(hero_id)))
colors = plt.cm.spectral(norm(hero_id2))
#othercolors = plt.cm.spectral(norm(range(len(liste)*10)))
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
pca = PCA(n_components=3)
pca.fit(df)

T = pca.transform(df)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(liste)):
#    if i < -1+len(liste)/2:
#        plt.scatter(T[allPlot[i],0], T[allPlot[i],1], s=50, marker='o', color=othercolors[i*10])
#    else:
#        plt.scatter(T[allPlot[i],0], T[allPlot[i],1], s=100, marker='*', color=othercolors[i*10])
    ax.scatter(T[allPlot[i],0], T[allPlot[i],1], T[allPlot[i],2], s=50, marker='o', c=colors[1-(1+i)*10])
#    print(othercolors[1-(1+i)*10])
names = [heroDict[i] for i in liste]
plt.legend(names)



#df.index = df['index']