import sqlite3
import os

table_name1 = 'happy_results'
sqlite_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'db.sqlite')
conn = sqlite3.connect(sqlite_file)
c = conn.cursor()

q = """
    CREATE TABLE IF NOT EXISTS {tn} (
      id integer PRIMARY KEY,
      img text NOT NULL,
      result integer
    )
    """.format(tn=table_name1)

c.execute(q)

