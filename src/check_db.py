import sys
sys.path.insert(0, r'C:\projects\vispay\src\transactions')
import upi_logic
import sqlite3
con = sqlite3.connect(upi_logic.CFG['db_path'])
row = con.execute('SELECT user_id, pin_hash FROM accounts WHERE user_id=?', ('user_001',)).fetchone()
print('DB:', upi_logic.CFG['db_path'])
print('Row:', row)
