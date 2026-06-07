# fix_pin.py  —  run once from src/transactions/
import sqlite3, hashlib

DB   = "vispay_transactions.db"
USER = "user_001"
PIN  = "1234"

def sha_hash(pin):
    return hashlib.sha256(f"vispay_{pin}_salt".encode()).hexdigest()

con = sqlite3.connect(DB)

# Check current state
row = con.execute("SELECT user_id, name, pin_hash FROM accounts WHERE user_id=?", (USER,)).fetchone()
print(f"Before: {row}")

if row is None:
    # User doesn't exist at all — insert them
    con.execute(
        "INSERT INTO accounts VALUES (?,?,?,?,?)",
        (USER, "Demo User", sha_hash(PIN), 10000.0, "2025-01-01 00:00:00")
    )
    print(f"Inserted {USER} with PIN {PIN}")
else:
    # User exists but hash is None or wrong — update it
    con.execute(
        "UPDATE accounts SET pin_hash=? WHERE user_id=?",
        (sha_hash(PIN), USER)
    )
    print(f"Updated pin_hash for {USER}")

con.commit()

# Verify
row = con.execute("SELECT user_id, name, pin_hash FROM accounts WHERE user_id=?", (USER,)).fetchone()
print(f"After : {row}")

# Confirm it matches
stored  = row[2]
computed = sha_hash(PIN)
print(f"Stored  : {stored}")
print(f"Computed: {computed}")
print(f"Match   : {stored == computed}")

con.close()