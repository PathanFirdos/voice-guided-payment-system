"""
razorpay_simulation.py  —  VisPay Razorpay Test Mode Integration
================================================================
Simulates real Razorpay payment gateway using Test API keys.
NO real money is transferred — fully safe for demo/project use.

Features:
  • Creates real Razorpay test orders via API
  • Simulates payment capture / failure / refund responses
  • Stores Razorpay order_id + payment_id in SQLite
  • Full voice-guided flow integrated with upi_logic.py
  • Webhook-style status simulation (no server needed)

Setup:
  pip install razorpay requests

  Add your TEST keys to CFG below or set env vars:
    RAZORPAY_KEY_ID     = rzp_test_XXXXXXXXXXXXXXXX
    RAZORPAY_KEY_SECRET = <your_test_secret>

  Get free test keys at: https://dashboard.razorpay.com (Test Mode)
"""

import hashlib
import hmac
import json
import os
import random
import sqlite3
import time
import uuid
from datetime import datetime

# ── Optional Razorpay SDK ─────────────────────────────────────────────────────
try:
    import razorpay
    _RZP_SDK = True
except ImportError:
    _RZP_SDK = False

# ── Optional requests fallback ────────────────────────────────────────────────
try:
    import requests as _requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  —  Fill in your Razorpay TEST keys here
# ══════════════════════════════════════════════════════════════════════════════
RZP_CFG = {
    # ── Paste your Razorpay TEST keys (from dashboard.razorpay.com) ──────────
    "key_id":     os.environ.get("RAZORPAY_KEY_ID",     "rzp_test_SxqlKvCUNl2WnR"),
    "key_secret": os.environ.get("RAZORPAY_KEY_SECRET", "OHKwdmUfpL8CqSpK9wZKm53R"),

    # ── Simulation settings ───────────────────────────────────────────────────
    "simulate_offline": False,    # True = fully offline mock (no API calls)
    "success_rate":     0.90,     # 90 % payments succeed in simulation
    "capture_timeout":  3,        # seconds to "wait" while simulating capture

    # ── DB path (must match upi_logic.py CFG["db_path"]) ─────────────────────
    "db_path": "vispay_transactions.db",
}

# ── Test card / UPI details Razorpay provides for sandbox testing ─────────────
TEST_CARDS = {
    "success": {"number": "4111111111111111", "cvv": "123", "exp": "12/27", "name": "Test User"},
    "failure": {"number": "4000000000000002", "cvv": "123", "exp": "12/27", "name": "Fail Card"},
}
TEST_UPI_IDS = {
    "success": "success@razorpay",
    "failure": "failure@razorpay",
}


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE — extends upi_logic tables
# ══════════════════════════════════════════════════════════════════════════════
def _db():
    """
    Open the shared VisPay SQLite DB with WAL mode + 30-second busy timeout.
    This prevents 'database is locked' when upi_logic and razorpay_simulation
    write concurrently (e.g. razorpay_pay → _finalize → post_auth_menu).
    """
    con = sqlite3.connect(RZP_CFG["db_path"], timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA busy_timeout=30000;")   # 30 000 ms at SQLite C layer
    return con


def init_razorpay_tables():
    """Add Razorpay columns/tables if not already present."""
    con = _db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS razorpay_orders (
            rzp_order_id    TEXT PRIMARY KEY,
            vispay_txn_id   TEXT NOT NULL,
            amount_paise    INTEGER NOT NULL,
            currency        TEXT NOT NULL DEFAULT 'INR',
            receipt         TEXT,
            status          TEXT NOT NULL DEFAULT 'created',
            created_at      TEXT NOT NULL,
            rzp_payment_id  TEXT,
            rzp_signature   TEXT,
            error_code      TEXT,
            error_desc      TEXT,
            captured_at     TEXT
        );

        CREATE TABLE IF NOT EXISTS razorpay_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type  TEXT NOT NULL,
            order_id    TEXT NOT NULL,
            payload     TEXT NOT NULL,
            recorded_at TEXT NOT NULL
        );
    """)
    con.commit()
    con.close()


# ══════════════════════════════════════════════════════════════════════════════
# RAZORPAY CLIENT  (SDK or requests fallback or offline mock)
# ══════════════════════════════════════════════════════════════════════════════
class RazorpayClient:
    """
    Thin wrapper — tries SDK → requests → offline mock in that order.
    All methods return a consistent dict so callers don't care which path ran.
    """

    def __init__(self):
        self.key_id     = RZP_CFG["key_id"]
        self.key_secret = RZP_CFG["key_secret"]
        self.offline    = RZP_CFG["simulate_offline"]
        self._sdk_client = None

        if not self.offline and _RZP_SDK:
            try:
                self._sdk_client = razorpay.Client(
                    auth=(self.key_id, self.key_secret)
                )
            except Exception:
                self._sdk_client = None

    # ── Create Order ──────────────────────────────────────────────────────────
    def create_order(self, amount_inr: float, receipt: str, notes: dict = None):
        """
        amount_inr : payment amount in rupees (e.g. 500.0)
        receipt    : your internal txn_id or reference
        Returns    : {"id": "order_XXXX", "amount": <paise>, "status": "created", ...}
        """
        amount_paise = int(amount_inr * 100)
        payload = {
            "amount":   amount_paise,
            "currency": "INR",
            "receipt":  receipt,
            "notes":    notes or {"app": "VisPay", "mode": "test"},
        }

        if self.offline:
            return self._mock_order(payload)

        # ── SDK path ──────────────────────────────────────────────────────────
        if self._sdk_client:
            try:
                order = self._sdk_client.order.create(data=payload)
                return dict(order)
            except Exception as e:
                print(f"  [RZP SDK] order.create failed: {e} — falling back to mock")

        # ── requests fallback ─────────────────────────────────────────────────
        if _REQUESTS:
            try:
                resp = _requests.post(
                    "https://api.razorpay.com/v1/orders",
                    json=payload,
                    auth=(self.key_id, self.key_secret),
                    timeout=10,
                )
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                print(f"  [RZP HTTP] order create failed: {e} — using mock")

        return self._mock_order(payload)

    # ── Fetch Order ───────────────────────────────────────────────────────────
    def fetch_order(self, order_id: str):
        if self.offline:
            return self._mock_fetch_order(order_id)

        if self._sdk_client:
            try:
                return dict(self._sdk_client.order.fetch(order_id))
            except Exception:
                pass

        if _REQUESTS:
            try:
                resp = _requests.get(
                    f"https://api.razorpay.com/v1/orders/{order_id}",
                    auth=(self.key_id, self.key_secret),
                    timeout=10,
                )
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                pass

        return self._mock_fetch_order(order_id)

    # ── Simulate Payment Capture (test mode only) ─────────────────────────────
    def simulate_payment(self, order_id: str, amount_paise: int, speak_fn=print):
        """
        In Razorpay TEST mode you can trigger a payment using their test UPI IDs.
        This method simulates the payment result locally to avoid needing a browser.
        Returns {"payment_id": ..., "status": "captured"|"failed", ...}
        """
        # Progressive feedback instead of a silent sleep — important for non-literate users
        speak_fn("Connecting to payment network...")
        time.sleep(1)
        speak_fn("Verifying account details...")
        time.sleep(1)
        speak_fn("Processing payment. Please wait.")
        time.sleep(max(0, RZP_CFG["capture_timeout"] - 2))

        success = random.random() < RZP_CFG["success_rate"]
        payment_id = "pay_" + uuid.uuid4().hex[:14].upper()

        if success:
            return {
                "payment_id": payment_id,
                "order_id":   order_id,
                "amount":     amount_paise,
                "status":     "captured",
                "method":     "upi",
                "vpa":        TEST_UPI_IDS["success"],
                "captured":   True,
                "created_at": int(time.time()),
            }
        else:
            return {
                "payment_id": payment_id,
                "order_id":   order_id,
                "amount":     amount_paise,
                "status":     "failed",
                "error_code": "BAD_REQUEST_ERROR",
                "error_desc": "Payment failed due to insufficient funds",
            }

    # ── Verify Signature ──────────────────────────────────────────────────────
    def verify_signature(self, order_id: str, payment_id: str, signature: str) -> bool:
        """Standard Razorpay HMAC-SHA256 signature verification."""
        message = f"{order_id}|{payment_id}"
        expected = hmac.new(
            self.key_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    # ── Refund (test) ─────────────────────────────────────────────────────────
    def create_refund(self, payment_id: str, amount_paise: int):
        if self.offline or not self._sdk_client:
            return {
                "id":         "rfnd_" + uuid.uuid4().hex[:14].upper(),
                "payment_id": payment_id,
                "amount":     amount_paise,
                "status":     "processed",
            }
        try:
            return dict(self._sdk_client.payment.refund(payment_id, {"amount": amount_paise}))
        except Exception as e:
            return {"error": str(e)}

    # ── Mock helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _mock_order(payload):
        return {
            "id":         "order_" + uuid.uuid4().hex[:14].upper(),
            "entity":     "order",
            "amount":     payload["amount"],
            "amount_paid": 0,
            "amount_due": payload["amount"],
            "currency":   payload.get("currency", "INR"),
            "receipt":    payload.get("receipt", ""),
            "status":     "created",
            "attempts":   0,
            "notes":      payload.get("notes", {}),
            "created_at": int(time.time()),
            "_mock":      True,
        }

    @staticmethod
    def _mock_fetch_order(order_id):
        return {"id": order_id, "status": "created", "_mock": True}


# ══════════════════════════════════════════════════════════════════════════════
# CORE PAYMENT FUNCTION  —  call this from upi_logic.py
# ══════════════════════════════════════════════════════════════════════════════
def razorpay_pay(
    vispay_txn_id: str,
    sender_id:     str,
    payee_name:    str,
    upi_id:        str,
    amount_inr:    float,
    speak_fn=print,
) -> dict:
    """
    Full Razorpay test payment flow.

    Parameters
    ----------
    vispay_txn_id : your existing txn_id from upi_logic.initiate_payment()
    sender_id     : user_id from accounts table
    payee_name    : display name of payee
    upi_id        : receiver UPI ID
    amount_inr    : amount in rupees
    speak_fn      : pass upi_logic.speak so voice guidance works

    Returns
    -------
    {
        "success":      bool,
        "rzp_order_id": str,
        "rzp_payment_id": str | None,
        "status":       "captured" | "failed" | "error",
        "message":      str,
    }
    """
    init_razorpay_tables()
    client = RazorpayClient()

    # ── STEP 1: Create Razorpay Order ─────────────────────────────────────────
    speak_fn("Creating secure payment order with Razorpay...")
    receipt = vispay_txn_id[:40]   # Razorpay receipt max 40 chars

    order = client.create_order(
        amount_inr=amount_inr,
        receipt=receipt,
        notes={"sender": sender_id, "upi_id": upi_id, "app": "VisPay"},
    )

    if "error" in str(order.get("id", "")).lower() or not order.get("id"):
        _record_rzp_event("order_create_failed", "none", order)
        return {
            "success": False,
            "rzp_order_id": None,
            "rzp_payment_id": None,
            "status": "error",
            "message": "Failed to create Razorpay order.",
        }

    rzp_order_id  = order["id"]
    amount_paise  = order["amount"]
    is_mock       = order.get("_mock", False)

    # ── Display order info ────────────────────────────────────────────────────
    mode_label = "OFFLINE MOCK" if is_mock else "RAZORPAY TEST"
    print("\n  ┌───────────────────────────────────────────────────┐")
    print(f"  │  💳  RAZORPAY TEST ORDER  [{mode_label}]")
    print(f"  │  Order ID   : {rzp_order_id}")
    print(f"  │  Amount     : ₹{amount_inr:.2f}  ({amount_paise} paise)")
    print(f"  │  Payee      : {payee_name}  ({upi_id})")
    print(f"  │  Receipt    : {receipt}")
    print("  └───────────────────────────────────────────────────┘")

    speak_fn(
        f"Razorpay order created. Order ID {rzp_order_id[:12]}. "
        f"Processing payment of ₹{amount_inr:.0f} to {payee_name}."
    )

    # ── STEP 2: Simulate Payment ──────────────────────────────────────────────
    # simulate_payment is pure logic (no DB) — safe to call before opening con.
    speak_fn("Simulating UPI payment via Razorpay test gateway. Please wait...")
    result = client.simulate_payment(rzp_order_id, amount_paise, speak_fn=speak_fn)

    rzp_payment_id = result.get("payment_id")
    pay_status     = result.get("status")   # "captured" or "failed"

    # ── STEP 3: Persist order + result in ONE connection ──────────────────────
    # Windows SQLite holds an OS-level file lock briefly after con.close().
    # If the next con = _db() arrives before that lock clears, it gets
    # "database is locked". Fix: do all writes in a single open/commit/close.
    con = _db()
    try:
        con.execute(
            """INSERT OR REPLACE INTO razorpay_orders
               (rzp_order_id, vispay_txn_id, amount_paise, receipt, status, created_at)
               VALUES (?,?,?,?,?,?)""",
            (rzp_order_id, vispay_txn_id, amount_paise, receipt, "created",
             datetime.now().isoformat())
        )
        if pay_status == "captured":
            con.execute(
                """UPDATE razorpay_orders
                   SET status='paid', rzp_payment_id=?, captured_at=?
                   WHERE rzp_order_id=?""",
                (rzp_payment_id, datetime.now().isoformat(), rzp_order_id)
            )
        else:
            con.execute(
                """UPDATE razorpay_orders
                   SET status='failed', rzp_payment_id=?, error_code=?, error_desc=?
                   WHERE rzp_order_id=?""",
                (rzp_payment_id,
                 result.get("error_code", "UNKNOWN"),
                 result.get("error_desc", "Payment failed"),
                 rzp_order_id)
            )
        now = datetime.now().isoformat()
        con.execute(
            "INSERT INTO razorpay_events (event_type, order_id, payload, recorded_at) VALUES (?,?,?,?)",
            ("order_created", rzp_order_id, json.dumps(order), now)
        )
        con.execute(
            "INSERT INTO razorpay_events (event_type, order_id, payload, recorded_at) VALUES (?,?,?,?)",
            (f"payment_{pay_status}", rzp_order_id, json.dumps(result), now)
        )
        con.commit()
    finally:
        con.close()

    # ── STEP 4: Return result ─────────────────────────────────────────────────
    if pay_status == "captured":
        print(f"\n  ✅  RAZORPAY PAYMENT CAPTURED")
        print(f"      Payment ID : {rzp_payment_id}")
        print(f"      Method     : {result.get('method','upi').upper()}")
        print(f"      VPA        : {result.get('vpa', upi_id)}\n")
        speak_fn(
            f"Payment successful! Razorpay payment ID {rzp_payment_id[:12]}. "
            f"₹{amount_inr:.0f} sent to {payee_name}."
        )
        return {
            "success":        True,
            "rzp_order_id":   rzp_order_id,
            "rzp_payment_id": rzp_payment_id,
            "status":         "captured",
            "message":        f"Payment of ₹{amount_inr:.0f} to {payee_name} captured.",
        }
    else:
        print(f"\n  ❌  RAZORPAY PAYMENT FAILED")
        print(f"      Payment ID : {rzp_payment_id}")
        print(f"      Error      : {result.get('error_desc','Unknown error')}\n")
        speak_fn(
            f"Payment failed. {result.get('error_desc', 'Please try again.')}."
        )
        return {
            "success":        False,
            "rzp_order_id":   rzp_order_id,
            "rzp_payment_id": rzp_payment_id,
            "status":         "failed",
            "message":        result.get("error_desc", "Payment failed"),
        }


# ══════════════════════════════════════════════════════════════════════════════
# REFUND HELPER
# ══════════════════════════════════════════════════════════════════════════════
def razorpay_refund(rzp_payment_id: str, amount_inr: float, speak_fn=print) -> dict:
    """Initiate a test refund for a captured payment."""
    client = RazorpayClient()
    amount_paise = int(amount_inr * 100)
    speak_fn(f"Initiating refund of ₹{amount_inr:.0f}...")
    result = client.create_refund(rzp_payment_id, amount_paise)

    if result.get("status") == "processed":
        speak_fn(f"Refund successful. Refund ID: {result['id'][:12]}.")
        _record_rzp_event("refund_processed", rzp_payment_id, result)
        return {"success": True, "refund_id": result["id"], "message": "Refund processed."}
    else:
        speak_fn("Refund failed. Please contact support.")
        return {"success": False, "message": result.get("error", "Refund failed")}


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY / RECEIPT DISPLAY
# ══════════════════════════════════════════════════════════════════════════════
def get_razorpay_history(limit=10) -> list:
    """Return recent Razorpay orders with their status."""
    init_razorpay_tables()
    con = _db()
    rows = con.execute(
        "SELECT * FROM razorpay_orders ORDER BY created_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def print_razorpay_receipt(rzp_order_id: str):
    """Print a formatted receipt for a Razorpay order."""
    init_razorpay_tables()
    con = _db()
    row = con.execute(
        "SELECT * FROM razorpay_orders WHERE rzp_order_id=?", (rzp_order_id,)
    ).fetchone()
    con.close()
    if not row:
        print("  Receipt not found."); return

    r = dict(row)
    status_icon = "✅" if r["status"] == "paid" else "❌"
    print(f"""
  ┌──────────────────────────────────────────────────┐
  │  {status_icon}  VISPAY RAZORPAY RECEIPT
  ├──────────────────────────────────────────────────┤
  │  Razorpay Order ID  : {r['rzp_order_id']}
  │  VisPay Txn ID      : {r['vispay_txn_id'][:20]}...
  │  Amount             : ₹{r['amount_paise']/100:.2f}
  │  Status             : {r['status'].upper()}
  │  Payment ID         : {r.get('rzp_payment_id') or '—'}
  │  Created            : {r['created_at'][:19]}
  │  Captured           : {r.get('captured_at') or '—'}
  └──────────────────────────────────────────────────┘""")


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _record_rzp_event(event_type: str, order_id: str, payload: dict):
    for attempt in range(1, 4):
        try:
            init_razorpay_tables()
            con = _db()
            con.execute(
                "INSERT INTO razorpay_events (event_type, order_id, payload, recorded_at) VALUES (?,?,?,?)",
                (event_type, order_id, json.dumps(payload), datetime.now().isoformat())
            )
            con.commit()
            con.close()
            return
        except sqlite3.OperationalError:
            if attempt == 3:
                return   # non-critical — don't crash payment on event log failure
            time.sleep(0.3 * attempt)


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST  —  python razorpay_simulation.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "═"*55)
    print("   VisPay  —  Razorpay Test Simulation  (standalone)")
    print("═"*55)

    # Check keys
    if "PASTE_YOUR" in RZP_CFG["key_id"]:
        print("\n  ⚠  No API keys set — running in OFFLINE MOCK mode.")
        print("     Set RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET env vars")
        print("     or edit RZP_CFG in this file.\n")
        RZP_CFG["simulate_offline"] = True
    else:
        print(f"\n  Key ID : {RZP_CFG['key_id']}")
        print(  f"  Mode   : {'OFFLINE MOCK' if RZP_CFG['simulate_offline'] else 'RAZORPAY TEST API'}\n")

    # Run a test payment
    test_txn_id = "TXN-" + uuid.uuid4().hex[:12].upper()
    result = razorpay_pay(
        vispay_txn_id = test_txn_id,
        sender_id     = "user_001",
        payee_name    = "Shopkeeper",
        upi_id        = "kirana@okhdfcbank",
        amount_inr    = 250.00,
    )

    print("\n  Final result:", json.dumps(result, indent=4))

    # Show receipt
    if result.get("rzp_order_id"):
        print_razorpay_receipt(result["rzp_order_id"])

    # Show history
    print("\n  Recent Razorpay Orders:")
    for row in get_razorpay_history(limit=5):
        amt = row["amount_paise"] / 100
        print(f"    {row['rzp_order_id'][:20]}  ₹{amt:.0f}  {row['status'].upper()}  {row['created_at'][:19]}")