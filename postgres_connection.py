
import pandas as pd
from sqlalchemy import Table, MetaData, create_engine
engine = create_engine("postgresql://postgres:dota2db@localhost/dota")

df = pd.DataFrame({'hej':2,'esfs':234,'234':'SFD:D'}, index=['TÆST'])

df.to_sql('test', engine, schema='dota_data')