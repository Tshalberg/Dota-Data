
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
    
    
#accID = 3342844
#all_players_to_db(accID, api, engine)