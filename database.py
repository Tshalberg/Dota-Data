from sqlalchemy import create_engine, MetaData

engine = create_engine('mysql+pymysql://root:the080893@localhost/sakila')
metadata = MetaData()
from sqlalchemy.engine import reflection

insp = reflection.Inspector.from_engine(engine)
print(insp.get_table_names())