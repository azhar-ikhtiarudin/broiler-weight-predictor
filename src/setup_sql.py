import sqlite3


def setup_sql():
    connection = sqlite3.connect("soka.db")
    if(connection):
        print("Connected to SQL Database")
    return connection.cursor()


cursor = setup_sql() #setup sql connection
# cursor.execute("CREATE TABLE ayam (tanggal DATETIME, umur INTEGER, berat INTEGER)")