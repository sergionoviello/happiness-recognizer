
import sqlite3
import os

class DBManager:
  def __init__(self, sqlite_file):
    import sqlite3
    self.conn = sqlite3.connect(sqlite_file)
    self.c = self.conn.cursor()

  def save_result(self, img, is_happy):
    check = """
      SELECT id FROM happy_results WHERE img = '{}' LIMIT 1
    """.format(img)

    self.c.execute(check)
    found = self.c.fetchone()

    if found == None:
      q = """
        INSERT INTO happy_results (img, result) VALUES ('{}', {})
      """.format(img, is_happy)
      self.c.execute(q)
      self.conn.commit()

  def load_results(self, is_happy = 1):
    q = """
      SELECT * FROM happy_results WHERE result = {} ORDER BY id desc
    """.format(is_happy)
    self.c.execute(q)
    all_rows = self.c.fetchall()
    return all_rows