import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VisPay — Voice-Guided Payment System",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero h1 {
        color: #e94560;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -1px;
    }
    .hero p {
        color: #a8b2c1;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    .hero .badge {
        display: inline-block;
        background: #e94560;
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 1rem;
        letter-spacing: 1px;
    }

    /* Metric cards */
    .metric-card {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value {
        color: #e94560;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
    }
    .metric-label {
        color: #a8b2c1;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    /* Section headers */
    .section-title {
        color: #e94560;
        font-size: 1.3rem;
        font-weight: 700;
        border-left: 4px solid #e94560;
        padding-left: 12px;
        margin: 1.5rem 0 1rem 0;
    }

    /* Feature tags */
    .feature-tag {
        display: inline-block;
        background: #0f3460;
        color: #e2e8f0;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        margin: 3px;
        border: 1px solid #1a4a7a;
    }

    /* Payment flow steps */
    .flow-step {
        background: #16213e;
        border: 1px solid #0f3460;
        border-left: 4px solid #e94560;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        color: #e2e8f0;
        font-size: 0.9rem;
    }

    /* Result box */
    .result-success {
        background: #0d2b1e;
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-warning {
        background: #2b1d0d;
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

    /* Sidebar-style info */
    .info-box {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #a8b2c1;
        font-size: 0.88rem;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Override streamlit default backgrounds */
    .stApp {
        background: #0d1117;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #16213e;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #a8b2c1;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: #e94560 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero Section ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔊 VisPay</h1>
    <p>Voice-Guided Digital Payment System for Non-Literate Users</p>
    <span class="badge">FINAL YEAR PROJECT • VIIT PUNE • 2026</span>
</div>
""", unsafe_allow_html=True)

# ── Stats Row ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">89%</div>
        <div class="metric-label">Currency Model Accuracy</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">8</div>
        <div class="metric-label">Currency Classes</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">2</div>
        <div class="metric-label">Biometric Factors</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">100%</div>
        <div class="metric-label">Voice-Guided Interaction</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "💵  Currency Recognition",
    "💳  Payment Simulator",
    "📊  Model Performance",
    "ℹ️  About Project"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CURRENCY RECOGNITION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Live Currency Detection</div>', unsafe_allow_html=True)
    st.markdown("Upload a photo of any Indian currency note — the model will identify the denomination.")

    # Load model
    @st.cache_resource
    def load_currency_model():
        try:
            model_path = os.path.join(
                "src", "currency_recognition", "currency_recognition",
                "models", "currency", "VisPay_currency_model.h5"
            )
            model = tf.keras.models.load_model(model_path)
            class_map_path = os.path.join(
                "src", "currency_recognition", "currency_recognition",
                "models", "currency", "class_map.json"
            )
            with open(class_map_path) as f:
                class_map = json.load(f)
            return model, class_map, None
        except Exception as e:
            return None, None, str(e)

    model, class_map, load_error = load_currency_model()

    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Choose currency note image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of an Indian currency note"
        )

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=True)

    with col_result:
        if uploaded and model is not None:
            with st.spinner("Analysing note..."):
                time.sleep(0.5)  # brief pause for UX

                # Preprocess
                img_resized = img.resize((224, 224))
                arr = np.array(img_resized, dtype=np.float32) / 255.0
                arr = np.expand_dims(arr, axis=0)

                # Predict
                preds = model.predict(arr, verbose=0)
                class_idx = str(np.argmax(preds))
                confidence = float(np.max(preds)) * 100
                denomination = class_map.get(class_idx, "Unknown")

                # Display result
                if denomination == "Background":
                    st.markdown("""
                    <div class="result-warning">
                        <div style="font-size:2.5rem">⚠️</div>
                        <div style="color:#f59e0b; font-size:1.3rem; font-weight:700">No Note Detected</div>
                        <div style="color:#a8b2c1; font-size:0.9rem; margin-top:0.5rem">
                            Please upload a clear photo of a currency note
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-success">
                        <div style="font-size:2.5rem">✅</div>
                        <div style="color:#22c55e; font-size:2rem; font-weight:700">₹{denomination}</div>
                        <div style="color:#a8b2c1; font-size:0.9rem; margin-top:0.3rem">
                            Indian Rupee Note Detected
                        </div>
                        <div style="color:#e2e8f0; font-size:1rem; margin-top:0.8rem; 
                                    background:#0a1f12; border-radius:8px; padding:0.5rem">
                            Confidence: <strong style="color:#22c55e">{confidence:.1f}%</strong>
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Show all class probabilities
                st.markdown("**Prediction breakdown:**")
                denominations = [class_map.get(str(i), f"Class {i}") for i in range(len(preds[0]))]
                prob_list = [
                    (f"₹{d}" if d != "Background" else "Background", float(preds[0][i]))
                    for i, d in enumerate(denominations)
                ]
                # Sort by probability descending
                prob_list.sort(key=lambda x: x[1], reverse=True)
                # Draw bars using pure HTML — no pandas needed
                bars_html = ""
                for label, prob in prob_list:
                    pct = prob * 100
                    color = "#e94560" if label == f"₹{denomination}" else "#0f3460"
                    bars_html += f"""
                    <div style="margin:4px 0">
                        <div style="display:flex; align-items:center; gap:8px">
                            <span style="color:#a8b2c1; width:80px; font-size:0.82rem">{label}</span>
                            <div style="flex:1; background:#1a2a3a; border-radius:4px; height:18px">
                                <div style="width:{pct:.1f}%; background:{color}; 
                                            height:18px; border-radius:4px;
                                            transition:width 0.5s"></div>
                            </div>
                            <span style="color:#e2e8f0; font-size:0.82rem; width:45px">{pct:.1f}%</span>
                        </div>
                    </div>"""
                st.markdown(bars_html, unsafe_allow_html=True)

        elif load_error:
            st.markdown(f"""
            <div class="info-box">
                ℹ️ <strong>Running in Demo Mode</strong><br>
                Model file not found in deployment environment.<br>
                The full system runs locally with complete camera-based detection.<br>
                <small style="color:#666">Error: {load_error[:100]}</small>
            </div>""", unsafe_allow_html=True)

            # Show demo result
            st.markdown("**Demo output (what the real system produces):**")
            st.markdown("""
            <div class="result-success">
                <div style="font-size:2.5rem">✅</div>
                <div style="color:#22c55e; font-size:2rem; font-weight:700">₹500</div>
                <div style="color:#a8b2c1; font-size:0.9rem">Indian Rupee Note Detected</div>
                <div style="color:#e2e8f0; background:#0a1f12; border-radius:8px; 
                            padding:0.5rem; margin-top:0.8rem">
                    Confidence: <strong style="color:#22c55e">94.3%</strong>
                </div>
            </div>""", unsafe_allow_html=True)

        elif not uploaded:
            st.markdown("""
            <div class="info-box" style="text-align:center; padding:2rem">
                <div style="font-size:3rem">📷</div>
                <div style="color:#e2e8f0; font-weight:600; margin-top:0.5rem">
                    Upload a currency note image
                </div>
                <div style="margin-top:1rem">
                    Supported: ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000
                </div>
            </div>""", unsafe_allow_html=True)

    # Voice output simulation
    st.markdown("---")
    st.markdown('<div class="section-title">🔊 What VisPay Says Aloud</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="flow-step">🎙️ &nbsp; <em>"Please hold your currency note in front of the camera..."</em></div>
    <div class="flow-step">🔍 &nbsp; <em>"Scanning... Please hold steady..."</em></div>
    <div class="flow-step">✅ &nbsp; <em>"Five hundred rupee note detected. Confidence: 94 percent."</em></div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PAYMENT SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">UPI Payment Flow Simulator</div>', unsafe_allow_html=True)
    st.markdown("Simulate the complete VisPay payment flow — exactly as a non-literate user experiences it.")

    # Load contacts
    contacts = {"mother": "mother@okaxis", "shopkeeper": "kirana@okhdfcbank", "son": "son@oksbi"}

    col_form, col_flow = st.columns([1, 1])

    with col_form:
        st.markdown("**Payment Details**")

        contact_name = st.selectbox(
            "👤 Select Contact (user speaks this name)",
            options=list(contacts.keys()),
            format_func=lambda x: f"{x.title()} → {contacts[x]}"
        )

        amount = st.number_input(
            "💰 Amount (₹)",
            min_value=1,
            max_value=10000,
            value=100,
            step=10
        )

        balance = st.number_input(
            "🏦 Current Balance (₹)",
            min_value=0,
            max_value=100000,
            value=10000,
            step=100
        )

        simulate_btn = st.button("▶️  Simulate Payment", use_container_width=True, type="primary")

    with col_flow:
        st.markdown("**Live Payment Flow**")

        if simulate_btn:
            upi_id = contacts[contact_name]
            steps = [
                ("🔊", f"Paying ₹{amount} to {contact_name.title()}. UPI ID: {upi_id}"),
                ("📡", "Connecting to Razorpay test gateway..."),
                ("🧾", f"Order created. Amount: ₹{amount} | Receipt: VPY-{int(time.time())%10000}"),
                ("🔐", "Verifying transaction signature..."),
                ("✅", f"Payment of ₹{amount} to {contact_name.title()} recorded as SUCCESS"),
                ("💰", f"Remaining balance: ₹{balance - amount:,.2f}"),
            ]

            if amount > balance:
                st.markdown("""
                <div class="result-warning">
                    <div style="font-size:2rem">⚠️</div>
                    <div style="color:#f59e0b; font-weight:700">Insufficient Balance</div>
                    <div style="color:#a8b2c1; font-size:0.9rem">
                        VisPay says: "Sorry, you do not have enough balance for this payment."
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                progress = st.progress(0)
                status_box = st.empty()

                for i, (icon, msg) in enumerate(steps):
                    progress.progress((i + 1) / len(steps))
                    status_box.markdown(f"""
                    <div class="flow-step">{icon} &nbsp; {msg}</div>
                    """, unsafe_allow_html=True)
                    time.sleep(0.7)

                progress.empty()
                st.markdown(f"""
                <div class="result-success">
                    <div style="font-size:2.5rem">✅</div>
                    <div style="color:#22c55e; font-size:1.3rem; font-weight:700">
                        Payment Successful
                    </div>
                    <div style="color:#a8b2c1; margin-top:0.5rem">
                        ₹{amount} → {contact_name.title()} ({upi_id})
                    </div>
                    <div style="color:#e2e8f0; background:#0a1f12; border-radius:8px; 
                                padding:0.5rem; margin-top:0.8rem; font-size:0.9rem">
                        New Balance: <strong style="color:#22c55e">₹{balance - amount:,.2f}</strong>
                    </div>
                    <div style="color:#666; font-size:0.75rem; margin-top:0.5rem">
                        ⚠️ Razorpay Test Mode — No real money transferred
                    </div>
                </div>""", unsafe_allow_html=True)

                # Voice output
                st.markdown("<br>", unsafe_allow_html=True)
                st.info(f'🔊 VisPay says: *"Payment of ₹{amount} to {contact_name.title()} successful. '
                        f'Remaining balance: ₹{balance - amount:,.2f}"*')
        else:
            st.markdown("""
            <div class="info-box" style="text-align:center; padding:2rem">
                <div style="font-size:3rem">💳</div>
                <div style="color:#e2e8f0; font-weight:600; margin-top:0.5rem">
                    Fill in payment details and click Simulate
                </div>
                <div style="margin-top:0.5rem">
                    Watch the real payment flow step by step
                </div>
            </div>""", unsafe_allow_html=True)

    # Trusted contacts display
    st.markdown("---")
    st.markdown('<div class="section-title">📇 Trusted Contacts</div>', unsafe_allow_html=True)
    st.markdown("User speaks the contact name — VisPay looks up the UPI ID automatically.")

    c1, c2, c3 = st.columns(3)
    for col, (name, upi) in zip([c1, c2, c3], contacts.items()):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2rem">{'👩' if name=='mother' else '🧑‍💼' if name=='shopkeeper' else '👦'}</div>
                <div style="color:#e2e8f0; font-weight:600; margin-top:0.5rem">{name.title()}</div>
                <div style="color:#a8b2c1; font-size:0.8rem; margin-top:0.3rem">{upi}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Currency Recognition Model — Results</div>', unsafe_allow_html=True)

    col_stats, col_misc = st.columns([1, 1])

    with col_stats:
        st.markdown("**Model Architecture**")
        stats = {
            "Total Parameters": "2,587,978",
            "Trainable Parameters": "2,185,096",
            "Non-trainable Parameters": "402,880",
            "Model Size": "9.87 MB",
            "Training Epochs": "30",
            "Fine-tune Start": "Epoch 20",
            "Final Train Accuracy": "~89%",
            "Final Val Accuracy": "~89%",
            "Total Misclassified": "6 samples",
            "Overfitting": "None (train ≈ val)",
            "Export Formats": ".h5 / .tflite / INT8",
        }
        for k, v in stats.items():
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; 
                        padding:6px 0; border-bottom:1px solid #1a2a3a; font-size:0.88rem">
                <span style="color:#a8b2c1">{k}</span>
                <span style="color:#e2e8f0; font-weight:600">{v}</span>
            </div>""", unsafe_allow_html=True)

    with col_misc:
        st.markdown("**Misclassification Analysis** (6 total)")
        misclassified = [
            ("₹100", "₹50",   "Similar color/size"),
            ("₹100", "₹10",   "Similar color"),
            ("₹10",  "₹20",   "Similar size"),
            ("₹2000","₹20",   "Poor lighting"),
            ("₹500", "₹2000", "Similar purple tone"),
            ("₹50",  "₹100",  "Similar color/size"),
        ]
        for true, pred, reason in misclassified:
            st.markdown(f"""
            <div class="flow-step" style="display:flex; justify-content:space-between; align-items:center">
                <span>True: <strong style="color:#22c55e">{true}</strong> → 
                      Pred: <strong style="color:#e94560">{pred}</strong></span>
                <span style="color:#666; font-size:0.8rem">{reason}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong style="color:#e2e8f0">Key Insight:</strong><br>
            All 6 errors occur between visually similar notes (same color family or size). 
            This is expected for CNN models and can be improved with more training data 
            and lighting augmentation.
        </div>""", unsafe_allow_html=True)

    # Training curves image
    st.markdown("---")
    st.markdown('<div class="section-title">Training Curves</div>', unsafe_allow_html=True)

    curves_path = os.path.join(
        "src", "currency_recognition", "currency_recognition",
        "train_results", "training_curves.png"
    )
    if os.path.exists(curves_path):
        st.image(curves_path, caption="Accuracy and Loss over 30 Epochs — fine-tune starts at epoch 20",
                 use_container_width=True)
    else:
        st.markdown("""
        <div class="info-box">
            📊 Training curves image available locally at:<br>
            <code>src/currency_recognition/currency_recognition/train_results/training_curves.png</code>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT PROJECT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Problem Statement</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box" style="font-size:1rem; line-height:1.7; color:#e2e8f0">
        Over <strong style="color:#e94560">300 million adults in India</strong> are non-literate 
        or have very low literacy levels. Existing UPI payment apps — PhonePe, GPay, Paytm — 
        require users to read text, type amounts, and navigate complex interfaces.<br><br>
        <strong>VisPay solves this</strong> by replacing every text-based interaction with 
        voice guidance. The user never needs to read a single word.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">System Features</div>', unsafe_allow_html=True)
    features = [
        ("🎙️", "Voice-Guided Payments", "Every screen speaks aloud — no reading required"),
        ("👤", "Face Authentication", "FaceNet 128-dim embeddings with cosine similarity"),
        ("🔊", "Voice Authentication", "SpeechBrain ECAPA-TDNN speaker verification"),
        ("🔢", "PIN Verification", "SHA-256 hashed 4-digit PIN with voice input"),
        ("🔑", "Forgot PIN Recovery", "Face + voice verification flow to reset PIN"),
        ("📇", "Trusted Contacts", "Pay by speaking contact name — no typing UPI IDs"),
        ("📷", "QR Payments", "Razorpay UPI QR code generation and display"),
        ("💵", "Currency Recognition", "CNN model — 89% accuracy across 8 denominations"),
        ("💳", "Razorpay Integration", "Real gateway in test mode — real orders, no real money"),
        ("🌐", "Offline Capability", "Vosk STT runs locally — works without internet"),
    ]

    col1, col2 = st.columns(2)
    for i, (icon, title, desc) in enumerate(features):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"""
            <div class="flow-step" style="margin:4px 0">
                <span style="font-size:1.1rem">{icon}</span>
                <strong style="color:#e2e8f0; margin-left:6px">{title}</strong><br>
                <span style="color:#a8b2c1; font-size:0.82rem; margin-left:2rem">{desc}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Payment Flow</div>', unsafe_allow_html=True)
    flow = [
        "1️⃣  App starts → Face Authentication (FaceNet)",
        "2️⃣  Voice-guided PIN verification",
        "3️⃣  Main menu — user speaks: PAY / BALANCE / HISTORY / CURRENCY / EXIT",
        "4️⃣  PAY → speak contact name → speak amount → Razorpay order created",
        "5️⃣  Confirm → transaction finalized → balance deducted → result spoken aloud",
    ]
    for step in flow:
        st.markdown(f'<div class="flow-step">{step}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Tech Stack</div>', unsafe_allow_html=True)
    tags = [
        "Python 3.10", "TensorFlow / Keras", "FaceNet", "SpeechBrain ECAPA-TDNN",
        "Vosk (Offline STT)", "Edge TTS", "Razorpay SDK", "SQLite WAL Mode",
        "OpenCV", "Transfer Learning", "INT8 Quantization", "TFLite"
    ]
    tags_html = "".join([f'<span class="feature-tag">{t}</span>' for t in tags])
    st.markdown(tags_html, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Author</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card" style="text-align:left; padding:1.5rem">
        <div style="color:#e2e8f0; font-size:1.1rem; font-weight:700">Pathan Firdos</div>
        <div style="color:#a8b2c1; margin-top:0.3rem">
            Final Year B.E. CSE (AI) — VIIT Pune | Graduating 2027
        </div>
        <div style="margin-top:0.8rem">
            <a href="https://github.com/PathanFirdos/voice-guided-payment-system" 
               style="color:#e94560; text-decoration:none; font-size:0.9rem">
                🔗 GitHub Repository
            </a>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#444; font-size:0.85rem; padding:1rem">
        ⚠️ This demo runs in Streamlit Cloud. Face auth, voice auth, and microphone features 
        require the full desktop application.<br>
        <strong style="color:#666">Razorpay Test Mode — No real money is transferred in any simulation.</strong>
    </div>""", unsafe_allow_html=True)