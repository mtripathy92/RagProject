import sqlite3
import pandas as pd

db_path = "telecom_support.db"
conn = sqlite3.connect(db_path)

tables_df = pd.read_sql("SELECT * FROM support_tickets limit 5;", conn)
print(tables_df)