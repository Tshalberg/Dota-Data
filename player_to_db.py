import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import dota2api
from sqlalchemy import create_engine, MetaData
from sqlalchemy.engine import reflection

engine = create_engine("postgresql://postgres:dota2db@localhost/dota")
metadata = MetaData()

insp = reflection.Inspector.from_engine(engine)
print(insp.get_table_names())

key = '6355122600AFA6460C66EBCA179B83C1'
api = dota2api.Initialise(key)

accID = 3342844

#def player_to_db_full(accID, api, engine):
matches = []
for hero_id in range(1, 115):
    print('hero ID ',hero_id)
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
#%%
#matchIDs2 = matchIDs[0:530]
for matchID in matchIDs:
    if count%100 == 0:
        print(count, ' out of ', len(matchIDs))
        
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

dfPlayer.to_sql(str(accID)+'_pStats_full', engine, if_exists='replace', schema='dota_data')
dfMatch.to_sql(str(accID)+'_mStats_full', engine, if_exists='replace', schema='dota_data')
print('Extraction finished. ',count,' matches were extracted.')

#accID = 117417755
#accID = 3342844
#engine = create_engine("postgresql://postgres:dota2db@localhost/dota")
#player_to_db_full(accID, api, engine)