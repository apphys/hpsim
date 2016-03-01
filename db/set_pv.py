import sqlite3 as lite
from subprocess import call
import os

dbfiles = ['online-201.db']

for df in dbfiles:
  con = lite.connect(df)
  with con:
    cur = con.cursor()
    cur.execute("select lcs_name, value from epics_channel")
    values = cur.fetchall()
    for v in values:
      print '--caput ' + str(v[0]) + ' ' + str(v[1])
      os.system('caput ' + str(v[0]) + ' ' + str(v[1]))
  con.close()
