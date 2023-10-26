'''
For docs refer to 
- https://docs.llamaindex.ai/en/stable/examples/query_engine/SQLRouterQueryEngine.html
- https://colab.research.google.com/drive/1luN8PY8K86XRnlvc1RzprczznZ_rcmox#scrollTo=rPawBPL0SwFj
'''
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    column,
    insert
)
import os
if __name__=="__main__":
    rows = [
    {"doctor_name": "Avinash", "doctors_fees": 500, "time": "5PM"},
    {"doctor_name": "Rohit", "doctors_fees": 100, "time": "6PM"},
    {"doctor_name": "Rittika", "doctors_fees": 1000, "time": "8PM"},
]
    
    database_path = os.path.join(os.getcwd(),"Data","test.db")
    engine = create_engine("sqlite:///"+database_path, echo=True)
    metadata_obj = MetaData()
    table_name = "doctors_records"
    city_stats_table = Table(
        table_name,
        metadata_obj,
        Column("doctor_name", String(16), primary_key=True),
        Column("doctors_fees", Integer),
        Column("time", String(16), nullable=False),
    )

    metadata_obj.create_all(engine)
    for row in rows:
        stmt = insert(city_stats_table).values(**row)
        with engine.begin() as connection:
            cursor = connection.execute(stmt)
