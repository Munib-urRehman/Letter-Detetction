import streamlit as st
import numpy as np
import math
import time
from PIL import Image, ImageDraw
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model   # type: ignore

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SignBridge · ASL Detector",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0A0E1A; color: #E8EAF0; }
section[data-testid="stSidebar"] { background: #0F1422 !important; border-right: 1px solid #1E2640; }
#MainMenu, footer, header { visibility: hidden; }
.hero-header {
    background: linear-gradient(135deg, #0F1422 0%, #141B2E 100%);
    border: 1px solid #1E2640; border-radius: 16px;
    padding: 28px 32px; margin-bottom: 20px;
    display: flex; align-items: center; gap: 20px;
}
.hero-title { font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; color:#fff; margin:0; }
.hero-subtitle { font-size:0.9rem; color:#6B7A9E; margin:4px 0 0 0; }
.hero-badge {
    background:rgba(0,212,255,0.12); border:1px solid rgba(0,212,255,0.3);
    color:#00D4FF; border-radius:8px; padding:4px 12px;
    font-size:0.75rem; font-weight:600; letter-spacing:0.5px;
    display:inline-block; margin-top:8px;
}
.panel-label {
    font-family:'Space Grotesk',sans-serif; font-size:0.7rem; font-weight:600;
    letter-spacing:1.2px; text-transform:uppercase; color:#3D4F72; margin-bottom:10px;
}
.letter-display {
    background:linear-gradient(135deg,#0D1528 0%,#111C35 100%);
    border:1px solid #1E3A5F; border-radius:14px; padding:20px;
    text-align:center; margin-bottom:14px; min-height:120px;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
}
.letter-big {
    font-family:'Space Grotesk',sans-serif; font-size:5rem; font-weight:700;
    color:#00D4FF; line-height:1; text-shadow:0 0 40px rgba(0,212,255,0.3);
}
.letter-label { font-size:0.75rem; color:#3D4F72; letter-spacing:1px; text-transform:uppercase; margin-top:6px; }
.sentence-box {
    background:#0B1020; border:1px solid #1E2640; border-radius:12px;
    padding:18px 20px; font-family:'Space Grotesk',sans-serif;
    font-size:1.3rem; font-weight:500; color:#fff;
    min-height:64px; word-break:break-all; letter-spacing:0.5px;
    margin-bottom:14px; line-height:1.5;
}
.sentence-cursor {
    display:inline-block; width:2px; height:1.2em;
    background:#00D4FF; vertical-align:text-bottom;
    animation:blink 1s step-end infinite;
}
@keyframes blink { 50% { opacity:0; } }
.conf-bar-bg { background:#1A2035; border-radius:6px; height:8px; width:100%; overflow:hidden; margin-top:6px; }
.conf-bar-fill { height:8px; border-radius:6px; background:linear-gradient(90deg,#0066AA,#00D4FF); }
.status-pill { display:inline-flex; align-items:center; gap:6px; padding:5px 12px; border-radius:20px; font-size:0.78rem; font-weight:500; }
.status-active { background:rgba(0,212,255,0.1); border:1px solid rgba(0,212,255,0.25); color:#00D4FF; }
.status-idle   { background:rgba(255,190,50,0.1);  border:1px solid rgba(255,190,50,0.25);  color:#FFBE32; }
.status-dot { width:7px; height:7px; border-radius:50%; animation:pulse-dot 1.4s ease-in-out infinite; }
.dot-active { background:#00D4FF; }
.dot-idle   { background:#FFBE32; animation:none; }
@keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.7)} }
.history-strip { display:flex; flex-wrap:wrap; gap:6px; padding:12px; background:#0B1020; border:1px solid #1A2035; border-radius:10px; }
.hist-chip { background:#141B2E; border:1px solid #1E2640; border-radius:6px; padding:4px 10px; font-family:'Space Grotesk',sans-serif; font-size:0.85rem; font-weight:600; color:#8899BB; }
.hist-chip-latest { border-color:rgba(0,212,255,0.35); color:#00D4FF; background:rgba(0,212,255,0.08); }
hr { border-color:#1E2640 !important; }
.stButton > button {
    background:rgba(0,212,255,0.1) !important; border:1px solid rgba(0,212,255,0.35) !important;
    color:#00D4FF !important; border-radius:10px !important;
    font-family:'Space Grotesk',sans-serif !important; font-weight:600 !important;
    font-size:0.85rem !important; padding:8px 20px !important;
}
.stButton > button:hover { background:rgba(0,212,255,0.2) !important; border-color:#00D4FF !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [("text",""),("prev_index",-1),("start_time",0.0),
              ("locked",False),("history",[]),("confidence",0.0),
              ("cur_letter","—"),("hand_found",False)]:
    if k not in st.session_state:
        st.session_state[k] = v

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [" "]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 18px 0'>
        <p style='font-family:Space Grotesk;font-size:1.05rem;font-weight:700;color:#fff;margin:0'>SignBridge</p>
        <p style='font-size:0.75rem;color:#3D4F72;margin:2px 0 0 0'>American Sign Language · Real-time</p>
    </div>""", unsafe_allow_html=True)

    detection_delay   = st.slider("Hold delay (s)", 0.5, 3.0, 1.0, 0.1)
    confidence_thresh = st.slider("Min confidence (%)", 0, 100, 0, 5)
    show_crop         = st.checkbox("Show hand crop preview", value=True)
    auto_space        = st.checkbox("Auto-space between letters", value=False)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem;color:#3D4F72;line-height:1.6;padding:0 2px'>
        <b style='color:#4A5978'>How to use</b><br>
        Click <b>📸 Take photo</b> below the camera, hold a sign steady, then hit it again.
        The app will detect your hand sign from each snapshot.
    </div>""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div style="font-size:2.8rem;line-height:1">🤟</div>
  <div>
    <p class="hero-title">SignBridge</p>
    <p class="hero-subtitle">Real-time American Sign Language → Text</p>
    <span class="hero-badge">ASL · 27 SIGNS</span>
  </div>
</div>""", unsafe_allow_html=True)

# ── Load models (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    # Keras model
    model = load_model("Models/keras_model.h5", compile=False)

    # Labels
    with open("Models/labels.txt", "r") as f:
        lbls = [line.strip().split(" ", 1)[-1] if " " in line.strip()
                else line.strip() for line in f.readlines()]

    # MediaPipe hands
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands_sol  = mp_hands.Hands(
        static_image_mode=True,      # ← snapshot mode, perfect for camera_input
        max_num_hands=1,
        min_detection_confidence=0.5,
    )
    return model, lbls, hands_sol, mp_hands, mp_drawing

try:
    model, file_labels, hands_sol, mp_hands, mp_drawing = load_assets()
    # Use file labels if they match expected count, else fall back to A-Z + space
    if len(file_labels) == len(labels):
        labels = file_labels
except Exception as e:
    st.error(f"❌ Could not load models: {e}")
    st.info("Make sure `Models/keras_model.h5` and `Models/labels.txt` exist in the repo root.")
    st.stop()

# ── Layout ─────────────────────────────────────────────────────────────────────
col_cam, col_panel = st.columns([3, 2], gap="medium")

with col_cam:
    st.markdown('<div class="panel-label">Camera snapshot</div>', unsafe_allow_html=True)
    # st.camera_input works on ALL cloud platforms — it uses the browser's webcam API
    camera_image = st.camera_input("📸 Take photo to detect sign")
    crop_ph = st.empty()

with col_panel:
    status_ph   = st.empty()
    st.markdown('<div class="panel-label">Detected sign</div>', unsafe_allow_html=True)
    letter_ph   = st.empty()
    st.markdown('<div class="panel-label">Confidence</div>', unsafe_allow_html=True)
    conf_ph     = st.empty()
    st.markdown('<div class="panel-label">Composed text</div>', unsafe_allow_html=True)
    sentence_ph = st.empty()
    st.markdown('<div class="panel-label">Recent letters</div>', unsafe_allow_html=True)
    history_ph  = st.empty()

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button("⌫  Backspace"):
            st.session_state.text = st.session_state.text[:-1]
    with btn_col2:
        if st.button("␣  Space"):
            st.session_state.text += " "
    with btn_col3:
        if st.button("✕  Clear"):
            st.session_state.text    = ""
            st.session_state.history = []

    st.markdown("---")
    copy_ph = st.empty()

# ── Renderers ──────────────────────────────────────────────────────────────────
def render_status(found):
    if found:
        status_ph.markdown('<div class="status-pill status-active"><span class="status-dot dot-active"></span>Hand detected</div>', unsafe_allow_html=True)
    else:
        status_ph.markdown('<div class="status-pill status-idle"><span class="status-dot dot-idle"></span>No hand in frame</div>', unsafe_allow_html=True)

def render_letter(letter):
    display = "·" if letter == " " else letter
    letter_ph.markdown(f'<div class="letter-display"><div class="letter-big">{display}</div><div class="letter-label">current sign</div></div>', unsafe_allow_html=True)

def render_confidence(pct):
    pct_int = int(pct * 100)
    color   = "#00D4FF" if pct_int >= 60 else "#FFBE32" if pct_int >= 35 else "#FF5555"
    conf_ph.markdown(f"""
    <div style="margin-bottom:14px">
        <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#6B7A9E;margin-bottom:4px">
            <span>Match strength</span><span style="color:{color};font-weight:600">{pct_int}%</span>
        </div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{pct_int}%;background:linear-gradient(90deg,#0066AA,{color})"></div>
        </div>
    </div>""", unsafe_allow_html=True)

def render_sentence(text):
    sentence_ph.markdown(f"""
    <div class="sentence-box">{text if text else '<span style="color:#2A3550">Start signing…</span>'}
        <span class="sentence-cursor"></span>
    </div>""", unsafe_allow_html=True)

def render_history(hist):
    if not hist:
        history_ph.markdown('<div class="history-strip" style="color:#2A3550;font-size:0.8rem">No letters yet</div>', unsafe_allow_html=True)
        return
    chips = "".join(
        f'<span class="{"hist-chip-latest" if i==len(hist[-18:])-1 else "hist-chip"} hist-chip">{"·" if c==" " else c}</span>'
        for i, c in enumerate(hist[-18:])
    )
    history_ph.markdown(f'<div class="history-strip">{chips}</div>', unsafe_allow_html=True)

# ── Initial render ─────────────────────────────────────────────────────────────
render_status(st.session_state.hand_found)
render_letter(st.session_state.cur_letter)
render_confidence(st.session_state.confidence)
render_sentence(st.session_state.text)
render_history(st.session_state.history)
copy_ph.code(st.session_state.text or "(nothing yet)", language=None)

# ── Process snapshot ───────────────────────────────────────────────────────────
IMG_SIZE = 300
OFFSET   = 20

def process_frame(pil_img: Image.Image):
    """Run hand detection + classification on a PIL image. Returns annotated PIL + metadata."""
    img_np  = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h_img, w_img = img_bgr.shape[:2]

    results = hands_sol.process(img_np)   # mediapipe expects RGB

    if not results.multi_hand_landmarks:
        return pil_img, False, "—", 0.0, None

    # ── Bounding box from landmarks ──
    lm   = results.multi_hand_landmarks[0].landmark
    xs   = [l.x * w_img for l in lm]
    ys   = [l.y * h_img for l in lm]
    x1   = max(0, int(min(xs)) - OFFSET)
    y1   = max(0, int(min(ys)) - OFFSET)
    x2   = min(w_img, int(max(xs)) + OFFSET)
    y2   = min(h_img, int(max(ys)) + OFFSET)

    # ── Draw landmarks ──
    annotated = img_bgr.copy()
    mp.solutions.drawing_utils.draw_landmarks(
        annotated,
        results.multi_hand_landmarks[0],
        mp_hands.HAND_CONNECTIONS,
        mp.solutions.drawing_utils.DrawingSpec(color=(0,212,255), thickness=2, circle_radius=3),
        mp.solutions.drawing_utils.DrawingSpec(color=(0,150,200), thickness=2),
    )
    cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,212,255), 2)

    # ── Crop + pad to square white canvas ──
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return pil_img, False, "—", 0.0, None

    white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
    cw, ch = x2-x1, y2-y1
    ratio  = ch / max(cw, 1)

    if ratio > 1:
        k      = IMG_SIZE / ch
        new_w  = math.ceil(cw * k)
        resized = cv2.resize(crop, (new_w, IMG_SIZE))
        gap    = math.ceil((IMG_SIZE - new_w) / 2)
        white[:, gap:gap+new_w] = resized
    else:
        k      = IMG_SIZE / max(cw, 1)
        new_h  = math.ceil(ch * k)
        resized = cv2.resize(crop, (IMG_SIZE, new_h))
        gap    = math.ceil((IMG_SIZE - new_h) / 2)
        white[gap:gap+new_h, :] = resized

    # ── Model inference ──
    inp  = cv2.resize(white, (224, 224))               # Teachable Machine default input
    inp  = inp.astype("float32") / 127.5 - 1.0        # same normalisation as TM export
    inp  = np.expand_dims(inp, axis=0)
    preds = model.predict(inp, verbose=0)[0]
    idx   = int(np.argmax(preds))
    conf  = float(preds[idx])
    letter = labels[idx] if idx < len(labels) else "?"

    ann_rgb  = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.cvtColor(white,     cv2.COLOR_BGR2RGB)
    return Image.fromarray(ann_rgb), True, letter, conf, Image.fromarray(crop_rgb)


if camera_image is not None:
    pil_in = Image.open(camera_image)
    ann_img, hand_found, letter, confidence, crop_img = process_frame(pil_in)

    # Update session
    st.session_state.hand_found  = hand_found
    st.session_state.cur_letter  = letter if hand_found else "—"
    st.session_state.confidence  = confidence if hand_found else 0.0

    # Append letter if above threshold
    if hand_found and confidence >= confidence_thresh / 100:
        st.session_state.text    += letter
        st.session_state.history.append(letter)

    # Display annotated frame back in camera column
    with col_cam:
        st.image(ann_img, use_container_width=True, caption="Detected landmarks")
        if show_crop and crop_img:
            crop_ph.image(crop_img, caption="Hand crop (model input)", width=160)

    # Re-render right panel
    render_status(hand_found)
    render_letter(st.session_state.cur_letter)
    render_confidence(st.session_state.confidence)
    render_sentence(st.session_state.text)
    render_history(st.session_state.history)
    copy_ph.code(st.session_state.text or "(nothing yet)", language=None)
else:
    with col_cam:
        st.info("👆 Click the camera button above to capture a sign.")