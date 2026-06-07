# 🔊 VisPay — Voice-Guided Digital Payment System for Non-Literate Users

> A final-year project that makes digital payments accessible to non-literate and visually challenged users through voice guidance, face authentication, and currency recognition.

---

## 📌 Problem Statement

Over 300 million adults in India are non-literate or have very low literacy levels. Existing UPI payment apps (PhonePe, GPay, Paytm) require users to read text, type amounts, and navigate complex UI — making them inaccessible to a large population. **VisPay solves this** by replacing text-based interaction with voice guidance at every step.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎙️ Voice-Guided Payments | Every screen speaks aloud — no reading required |
| 👤 Face Authentication | Login using face recognition (FaceNet model) |
| 🔊 Voice Authentication | Speaker verification using SpeechBrain ECAPA-TDNN |
| 🔢 PIN Verification | SHA-256 hashed 4-digit PIN with voice input support |
| 🔑 Forgot PIN Recovery | Face + voice verification flow to reset PIN |
| 📇 Trusted Contacts | Pay to saved contacts by speaking their name (e.g. "son", "mother") |
| 📷 QR Code Payments | Generate and display Razorpay UPI QR codes |
| 💵 Currency Recognition | Camera-based Indian currency note detection (₹10 to ₹2000) |
| 💳 Razorpay Integration | Real-world UPI payment gateway in test mode |
| 📊 Transaction History | Voice-read history with balance tracking |
| 🌐 Offline Fallback | Works without internet using local Vosk STT model |

---

## 🏗️ System Architecture

```
VisPay/
├── src/
│   ├── main.py                          # Entry point / FSM controller
│   ├── tts.py                           # Text-to-Speech (Edge TTS + gTTS fallback)
│   ├── check_db.py                      # Database inspection utility
│   │
│   ├── transactions/
│   │   ├── upi_logic.py                 # Core payment FSM, PIN, balance, history
│   │   ├── razorpay_simulation.py       # Razorpay Test Mode integration
│   │   └── trusted_contacts.json        # Saved payee names → UPI IDs
│   │
│   ├── face_auth/
│   │   ├── verify.py                    # FaceNet-based face verification
│   │   ├── enroll.py                    # New user face enrollment
│   │   ├── user_manager.py              # User profile management
│   │   └── thresholds.py               # Similarity thresholds
│   │
│   ├── voice_auth/
│   │   ├── verify.py                    # SpeechBrain speaker verification
│   │   ├── embed.py                     # Voice embedding generation
│   │   ├── record.py                    # Microphone recording utility
│   │   └── user_manager.py             # Voice profile management
│   │
│   ├── currency_recognition/
│   │   ├── preprocess.py               # Camera loop + currency FSM
│   │   ├── train_currency_model.py     # CNN model training script
│   │   └── diagnose.py                 # Model diagnosis utility
│   │
│   ├── assistant/
│   │   ├── stt.py                       # Speech-to-Text (Vosk + SpeechRecognition)
│   │   └── prompts.py                   # Voice prompt templates
│   │
│   └── decision_engine/
│       └── flow.py                      # High-level payment flow logic
│
├── requirements.txt
└── .env                                 # API keys (not committed)
```

---

## 🔄 Payment Flow

```
App Start
   │
   ▼
Face Authentication (FaceNet)
   │
   ▼
PIN Verification (voice/keyboard input)
   │
   ▼
Main Menu (voice: PAY / BALANCE / HISTORY / CONTACTS / CURRENCY / FORGOT / EXIT)
   │
   ├── PAY ──► Choose Contact ──► Enter Amount ──► Razorpay Order ──► Confirm ──► SUCCESS
   │
   ├── BALANCE ──► Speaks current balance aloud
   │
   ├── HISTORY ──► Reads last 5 transactions aloud
   │
   ├── CURRENCY ──► Opens camera ──► Detects note denomination ──► Speaks result
   │
   └── FORGOT PIN ──► Face verify ──► Reset PIN
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Face Auth | FaceNet (keras-facenet), OpenCV |
| Voice Auth | SpeechBrain ECAPA-TDNN, torchaudio |
| Speech-to-Text | Vosk (offline), Google SpeechRecognition (online) |
| Text-to-Speech | Microsoft Edge TTS (online), gTTS (fallback) |
| Currency Recognition | TensorFlow/Keras CNN |
| Payment Gateway | Razorpay Test Mode SDK |
| Database | SQLite (WAL mode) |
| QR Code | `qrcode` library |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- Webcam
- Microphone
- Windows 10/11 (tested) or Linux

### 1. Clone the repository
```bash
git clone https://github.com/PathanFirdos/voice-guided-payment-system.git
cd voice-guided-payment-system
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:
```
RAZORPAY_KEY_ID=rzp_test_your_key_here
RAZORPAY_KEY_SECRET=your_secret_here
```

> Get free test keys at [dashboard.razorpay.com](https://dashboard.razorpay.com)

### 5. Download Vosk model (offline STT)
Download [vosk-model-en-us-0.22](https://alphacephei.com/vosk/models) and place it at:
```
src/voice_auth/vosk-model-en-us-0.22/
```

### 6. Run the app
```bash
cd src
python main.py
```

---

## 🗄️ Database Schema

VisPay uses a local SQLite database (`vispay_transactions.db`) with WAL mode for concurrency:

- **`accounts`** — user_id, balance, daily_limit
- **`transactions`** — txn_id, sender, receiver, amount, status, timestamps
- **`rate_log`** — transaction rate limiting per user
- **`razorpay_orders`** — order_id, payment_id, amount, status
- **`razorpay_events`** — full audit trail of gateway events

---

## 🔐 Security Features

- Face embeddings stored as numpy arrays (never raw images)
- PIN stored as SHA-256 hash (never plaintext)
- API keys loaded from `.env` (never hardcoded)
- SQLite WAL mode prevents database corruption
- Daily transaction limit enforced per user
- Full audit log (`vispay_audit.log`) for all events

---

## 🧪 Testing

Run the Razorpay connection test:
```bash
cd src
python test_razorpay.py
```

Inspect the database:
```bash
python check_db.py
```

---

## 📸 Target Users

- Non-literate rural population
- Elderly users unfamiliar with smartphones
- Visually impaired users
- First-time smartphone users in semi-urban areas

---

## 🚧 Limitations (Scope of Project)

- Razorpay runs in **Test Mode** — no real money is transferred
- Currency recognition trained on limited dataset — may need retraining for production
- Face authentication requires good lighting conditions
- Voice recognition accuracy depends on microphone quality and accent

---

## 👨‍💻 Author

**Pathan Firdos**
Final Year Project — Computer Science Engineering (Artificial Intelligence)
GitHub: [@PathanFirdos](https://github.com/PathanFirdos)

---

## 📄 License

This project is developed for academic purposes as a final-year project.

---

> *"Technology should work for everyone — not just those who can read."*
