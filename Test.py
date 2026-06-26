import cv2
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
from PIL import Image

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

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Backgrounds ── */
.stApp {
    background: #0A0E1A;
    color: #E8EAF0;
}

section[data-testid="stSidebar"] {
    background: #0F1422 !important;
    border-right: 1px solid #1E2640;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, #0F1422 0%, #141B2E 100%);
    border: 1px solid #1E2640;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 0.9rem;
    color: #6B7A9E;
    margin: 4px 0 0 0;
}
.hero-badge {
    background: rgba(0, 212, 255, 0.12);
    border: 1px solid rgba(0, 212, 255, 0.3);
    color: #00D4FF;
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    display: inline-block;
    margin-top: 8px;
}

/* ── Cards ── */
.panel-card {
    background: #0F1422;
    border: 1px solid #1E2640;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 14px;
}
.panel-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #3D4F72;
    margin-bottom: 10px;
}

/* ── Detected letter display ── */
.letter-display {
    background: linear-gradient(135deg, #0D1528 0%, #111C35 100%);
    border: 1px solid #1E3A5F;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    margin-bottom: 14px;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.letter-big {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 5rem;
    font-weight: 700;
    color: #00D4FF;
    line-height: 1;
    text-shadow: 0 0 40px rgba(0,212,255,0.3);
}
.letter-label {
    font-size: 0.75rem;
    color: #3D4F72;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 6px;
}

/* ── Sentence output box ── */
.sentence-box {
    background: #0B1020;
    border: 1px solid #1E2640;
    border-radius: 12px;
    padding: 18px 20px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 500;
    color: #FFFFFF;
    min-height: 64px;
    word-break: break-all;
    letter-spacing: 0.5px;
    margin-bottom: 14px;
    line-height: 1.5;
}
.sentence-cursor {
    display: inline-block;
    width: 2px;
    height: 1.2em;
    background: #00D4FF;
    vertical-align: text-bottom;
    animation: blink 1s step-end infinite;
}
@keyframes blink { 50% { opacity: 0; } }

/* ── Confidence bar ── */
.conf-bar-bg {
    background: #1A2035;
    border-radius: 6px;
    height: 8px;
    width: 100%;
    overflow: hidden;
    margin-top: 6px;
}
.conf-bar-fill {
    height: 8px;
    border-radius: 6px;
    background: linear-gradient(90deg, #0066AA, #00D4FF);
    transition: width 0.3s ease;
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
}
.status-active {
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.25);
    color: #00D4FF;
}
.status-idle {
    background: rgba(255,190,50,0.1);
    border: 1px solid rgba(255,190,50,0.25);
    color: #FFBE32;
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    animation: pulse-dot 1.4s ease-in-out infinite;
}
.dot-active { background: #00D4FF; }
.dot-idle   { background: #FFBE32; animation: none; }
@keyframes pulse-dot {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.4; transform: scale(0.7); }
}

/* ── Sidebar controls ── */
.sidebar-section {
    background: #141B2E;
    border: 1px solid #1E2640;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}
.sidebar-section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #3D4F72;
    margin-bottom: 12px;
}

/* ── Streamlit widget overrides ── */
.stButton > button {
    background: rgba(0,212,255,0.1) !important;
    border: 1px solid rgba(0,212,255,0.35) !important;
    color: #00D4FF !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: rgba(0,212,255,0.2) !important;
    border-color: #00D4FF !important;
}
div[data-testid="stSlider"] label,
div[data-testid="stCheckbox"] label,
div[data-testid="stRadio"] label {
    color: #8899BB !important;
    font-size: 0.85rem !important;
}
div[data-testid="stSlider"] .stSlider > div > div {
    background: #1E2640 !important;
}

/* ── History chips ── */
.history-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding: 12px;
    background: #0B1020;
    border: 1px solid #1A2035;
    border-radius: 10px;
}
.hist-chip {
    background: #141B2E;
    border: 1px solid #1E2640;
    border-radius: 6px;
    padding: 4px 10px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: #8899BB;
}
.hist-chip-latest {
    border-color: rgba(0,212,255,0.35);
    color: #00D4FF;
    background: rgba(0,212,255,0.08);
}

/* ── Dividers ── */
hr { border-color: #1E2640 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
if "text"        not in st.session_state: st.session_state.text        = ""
if "prev_index"  not in st.session_state: st.session_state.prev_index  = -1
if "start_time"  not in st.session_state: st.session_state.start_time  = 0.0
if "locked"      not in st.session_state: st.session_state.locked      = False
if "history"     not in st.session_state: st.session_state.history     = []
if "confidence"  not in st.session_state: st.session_state.confidence  = 0.0
if "cur_letter"  not in st.session_state: st.session_state.cur_letter  = "—"
if "hand_found"  not in st.session_state: st.session_state.hand_found  = False

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [" "]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 18px 0'>
        <p style='font-family:Space Grotesk;font-size:1.05rem;font-weight:700;
                  color:#FFFFFF;margin:0'>SignBridge</p>
        <p style='font-size:0.75rem;color:#3D4F72;margin:2px 0 0 0'>
            American Sign Language · Real-time</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Detection settings</div>',
                unsafe_allow_html=True)
    detection_delay = st.slider("Hold delay (s)", 0.5, 3.0, 1.0, 0.1)
    confidence_thresh = st.slider("Min confidence (%)", 0, 100, 0, 5)
    show_crop = st.checkbox("Show hand crop preview", value=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">Output</div>',
                unsafe_allow_html=True)
    auto_space = st.checkbox("Auto-space between letters", value=False)

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">Camera</div>',
                unsafe_allow_html=True)
    cam_index = st.selectbox("Camera source", [0, 1, 2], index=0)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem;color:#3D4F72;line-height:1.6;padding:0 2px'>
        <b style='color:#4A5978'>How to use</b><br>
        Show a letter to the camera and hold it steady.
        The detector confirms after your set delay.
        Use the space gesture to add spaces.
    </div>
    """, unsafe_allow_html=True)

# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div style="font-size:2.8rem;line-height:1">🤟</div>
  <div>
    <p class="hero-title">SignBridge</p>
    <p class="hero-subtitle">Real-time American Sign Language → Text</p>
    <span class="hero-badge">ASL · 27 SIGNS</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────────────────────────────
col_cam, col_panel = st.columns([3, 2], gap="medium")

with col_cam:
    st.markdown('<div class="panel-label">Live camera feed</div>',
                unsafe_allow_html=True)
    cam_placeholder   = st.empty()
    crop_placeholder  = st.empty()

with col_panel:
    # ── Status row ──
    status_ph = st.empty()

    # ── Detected letter ──
    st.markdown('<div class="panel-label">Detected sign</div>',
                unsafe_allow_html=True)
    letter_ph = st.empty()

    # ── Confidence ──
    st.markdown('<div class="panel-label">Confidence</div>',
                unsafe_allow_html=True)
    conf_ph = st.empty()

    # ── Sentence ──
    st.markdown('<div class="panel-label">Composed text</div>',
                unsafe_allow_html=True)
    sentence_ph = st.empty()

    # ── Recent letters ──
    st.markdown('<div class="panel-label">Recent letters</div>',
                unsafe_allow_html=True)
    history_ph = st.empty()

    # ── Controls ──
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
    copy_ph.code(st.session_state.text or "(nothing yet)", language=None)


# ── Helper renderers ──────────────────────────────────────────────────────────
def render_status(hand_found: bool):
    if hand_found:
        status_ph.markdown("""
        <div class="status-pill status-active">
            <span class="status-dot dot-active"></span>Hand detected
        </div>""", unsafe_allow_html=True)
    else:
        status_ph.markdown("""
        <div class="status-pill status-idle">
            <span class="status-dot dot-idle"></span>No hand in frame
        </div>""", unsafe_allow_html=True)


def render_letter(letter: str):
    display = "·" if letter == " " else letter
    letter_ph.markdown(f"""
    <div class="letter-display">
        <div class="letter-big">{display}</div>
        <div class="letter-label">current sign</div>
    </div>""", unsafe_allow_html=True)


def render_confidence(pct: float):
    pct_int = int(pct * 100)
    color   = "#00D4FF" if pct_int >= 60 else "#FFBE32" if pct_int >= 35 else "#FF5555"
    conf_ph.markdown(f"""
    <div style="margin-bottom:14px">
        <div style="display:flex;justify-content:space-between;
                    font-size:0.82rem;color:#6B7A9E;margin-bottom:4px">
            <span>Match strength</span>
            <span style="color:{color};font-weight:600">{pct_int}%</span>
        </div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{pct_int}%;
                 background:linear-gradient(90deg,#0066AA,{color})"></div>
        </div>
    </div>""", unsafe_allow_html=True)


def render_sentence(text: str):
    sentence_ph.markdown(f"""
    <div class="sentence-box">{text if text else
        '<span style="color:#2A3550">Start signing…</span>'}
        <span class="sentence-cursor"></span>
    </div>""", unsafe_allow_html=True)


def render_history(hist: list):
    if not hist:
        history_ph.markdown(
            '<div class="history-strip" style="color:#2A3550;font-size:0.8rem">'
            'No letters yet</div>', unsafe_allow_html=True)
        return
    chips = ""
    for i, ch in enumerate(hist[-18:]):
        cls = "hist-chip-latest" if i == len(hist[-18:]) - 1 else "hist-chip"
        label = "·" if ch == " " else ch
        chips += f'<span class="{cls} hist-chip">{label}</span>'
    history_ph.markdown(f'<div class="history-strip">{chips}</div>',
                        unsafe_allow_html=True)


# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_models(cam_idx):
    detector   = HandDetector(maxHands=1)
    classifier = Classifier("Models/keras_model.h5", "Models/labels.txt")
    cap        = cv2.VideoCapture(cam_idx)
    return detector, classifier, cap


# ── Initial UI state ──────────────────────────────────────────────────────────
render_status(False)
render_letter("—")
render_confidence(0.0)
render_sentence(st.session_state.text)
render_history(st.session_state.history)

# ── Main loop ─────────────────────────────────────────────────────────────────
try:
    detector, classifier, cap = load_models(cam_index)
except Exception as e:
    st.error(f"Could not load models or camera: {e}")
    st.info("Make sure `Models/keras_model.h5` and `Models/labels.txt` are present "
            "and your camera is accessible.")
    st.stop()

offset  = 20
imgSize = 300

while True:
    success, img = cap.read()
    if not success:
        cam_placeholder.warning("Camera feed unavailable.")
        time.sleep(0.05)
        continue

    img        = cv2.flip(img, 1)
    outputImg  = img.copy()
    hands, img = detector.findHands(img)

    hand_found  = bool(hands)
    cur_letter  = st.session_state.cur_letter
    confidence  = st.session_state.confidence

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # ── safe crop ──
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            hand_found = False
        else:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            if len(imgCrop.shape) == 2:
                imgCrop = cv2.cvtColor(imgCrop, cv2.COLOR_GRAY2BGR)

            ratio = h / max(w, 1)
            if ratio > 1:
                k      = imgSize / h
                widnew = math.ceil(w * k)
                imgRsz = cv2.resize(imgCrop, (widnew, imgSize))
                wGap   = math.ceil((imgSize - widnew) / 2)
                imgWhite[:, wGap:wGap + widnew] = imgRsz
            else:
                kk     = imgSize / max(w, 1)
                hgtnew = math.ceil(h * kk)
                imgRsz = cv2.resize(imgCrop, (imgSize, hgtnew))
                hGap   = math.ceil((imgSize - hgtnew) / 2)
                imgWhite[hGap:hGap + hgtnew, :] = imgRsz

            prediction, index = classifier.getPrediction(imgWhite)
            confidence   = float(max(prediction))
            cur_letter   = labels[index]

            # ── detection timing ──
            if index != st.session_state.prev_index:
                st.session_state.start_time  = time.time()
                st.session_state.prev_index  = index
                st.session_state.locked      = False

            elapsed = time.time() - st.session_state.start_time
            # ── progress ring on bounding box ──
            progress = min(elapsed / detection_delay, 1.0)
            cx, cy   = x + w // 2, y + h // 2
            radius   = max(w, h) // 2 + offset
            # base box
            cv2.rectangle(outputImg, (x1, y1), (x2, y2),
                          (30, 50, 80), 1)
            # arc overlay (approximated with ellipse for OpenCV)
            axes    = (radius, radius)
            angle   = -90
            end_ang = int(-90 + 360 * progress)
            color   = (0, 212, 255) if progress < 1.0 else (0, 255, 160)
            cv2.ellipse(outputImg, (cx, cy), axes, 0, angle, end_ang, color, 2)

            # ── label overlay ──
            label_txt = f"{cur_letter}  {int(confidence*100)}%"
            cv2.putText(outputImg, label_txt, (x1, max(y1 - 10, 14)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 212, 255), 2,
                        cv2.LINE_AA)

            # ── commit letter ──
            if (not st.session_state.locked
                    and elapsed >= detection_delay
                    and confidence >= confidence_thresh / 100):
                letter_to_add = cur_letter
                if auto_space and st.session_state.text:
                    letter_to_add = cur_letter
                st.session_state.text    += letter_to_add
                st.session_state.history.append(cur_letter)
                st.session_state.locked   = True

            # ── show crop ──
            if show_crop:
                crop_rgb = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
                crop_placeholder.image(
                    Image.fromarray(crop_rgb),
                    caption="Hand crop (model input)",
                    use_container_width=False,
                    width=160,
                )
            else:
                crop_placeholder.empty()

    # ── render camera frame ──
    frame_rgb = cv2.cvtColor(outputImg, cv2.COLOR_BGR2RGB)
    cam_placeholder.image(
        Image.fromarray(frame_rgb),
        channels="RGB",
        use_container_width=True,
    )

    # ── update right panel ──
    st.session_state.cur_letter = cur_letter
    st.session_state.confidence = confidence
    st.session_state.hand_found = hand_found

    render_status(hand_found)
    render_letter(cur_letter if hand_found else "—")
    render_confidence(confidence if hand_found else 0.0)
    render_sentence(st.session_state.text)
    render_history(st.session_state.history)
    copy_ph.code(st.session_state.text or "(nothing yet)", language=None)

    time.sleep(0.03)  # ~30 fps cap