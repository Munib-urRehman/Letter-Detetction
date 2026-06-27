import gradio as gr
import numpy as np
import math
import cv2
from PIL import Image
from tensorflow.keras.models import load_model   # type: ignore
import mediapipe as mp

# ── Labels ────────────────────────────────────────────────────────────────────
DEFAULT_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE"]

# ── Load assets ───────────────────────────────────────────────────────────────
def load_assets():
    model = load_model("Models/keras_model.h5", compile=False)
    try:
        with open("Models/labels.txt") as f:
            lbls = [l.strip().split(" ", 1)[-1] if " " in l.strip() else l.strip()
                    for l in f if l.strip()]
    except Exception:
        lbls = DEFAULT_LABELS
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                           min_detection_confidence=0.5)
    return model, lbls, hands, mp_hands

try:
    model, LABELS, hands_sol, mp_hands = load_assets()
except Exception as e:
    raise RuntimeError(f"Could not load Models/keras_model.h5 or labels.txt: {e}")

IMG_SIZE = 300
OFFSET   = 20

# ── Core inference ────────────────────────────────────────────────────────────
def detect_sign(pil_img):
    """Returns (annotated_pil, crop_pil | None, letter, confidence, hand_found)"""
    img_np  = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h_img, w_img = img_bgr.shape[:2]

    results = hands_sol.process(img_np)
    if not results.multi_hand_landmarks:
        return pil_img, None, "—", 0.0, False

    lm = results.multi_hand_landmarks[0].landmark
    xs = [l.x * w_img for l in lm];  ys = [l.y * h_img for l in lm]
    x1 = max(0, int(min(xs)) - OFFSET);  y1 = max(0, int(min(ys)) - OFFSET)
    x2 = min(w_img, int(max(xs)) + OFFSET);  y2 = min(h_img, int(max(ys)) + OFFSET)

    annotated = img_bgr.copy()
    mp.solutions.drawing_utils.draw_landmarks(
        annotated, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
        mp.solutions.drawing_utils.DrawingSpec(color=(0,212,255), thickness=2, circle_radius=3),
        mp.solutions.drawing_utils.DrawingSpec(color=(0,150,200), thickness=2),
    )
    cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,212,255), 2)

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return pil_img, None, "—", 0.0, False

    white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
    cw, ch = x2-x1, y2-y1
    if (ch / max(cw,1)) > 1:
        k = IMG_SIZE / ch;  nw = math.ceil(cw*k)
        rsz = cv2.resize(crop, (nw, IMG_SIZE));  gap = math.ceil((IMG_SIZE-nw)/2)
        white[:, gap:gap+nw] = rsz
    else:
        k = IMG_SIZE / max(cw,1);  nh = math.ceil(ch*k)
        rsz = cv2.resize(crop, (IMG_SIZE, nh));  gap = math.ceil((IMG_SIZE-nh)/2)
        white[gap:gap+nh, :] = rsz

    inp = cv2.resize(white, (224, 224)).astype("float32") / 127.5 - 1.0
    preds = model.predict(np.expand_dims(inp, 0), verbose=0)[0]
    idx   = int(np.argmax(preds));  conf = float(preds[idx])
    letter = LABELS[idx] if idx < len(LABELS) else "?"

    ann_rgb  = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.cvtColor(white,     cv2.COLOR_BGR2RGB)
    return Image.fromarray(ann_rgb), Image.fromarray(crop_rgb), letter, conf, True


# ── Gradio handler ────────────────────────────────────────────────────────────
def on_snapshot(image, composed_text, history_state, min_conf):
    if image is None:
        return (
            None, None, "—", 0,
            composed_text or "", history_state or [],
            composed_text or "(nothing yet)",
            _history_html(history_state or []),
        )

    pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    ann, crop, letter, conf, found = detect_sign(pil)

    conf_pct = int(conf * 100)
    status   = f"✅ Hand detected — **{letter}** ({conf_pct}%)" if found else "🟡 No hand detected"

    new_text    = composed_text or ""
    new_history = list(history_state or [])

    if found and conf_pct >= min_conf:
        char = " " if letter in ("SPACE", " ") else letter
        new_text    += char
        new_history.append(char)

    return (
        ann,
        crop,
        letter if found else "—",
        conf_pct if found else 0,
        new_text,
        new_history,
        new_text or "(nothing yet)",
        _history_html(new_history),
    )

def on_backspace(text, history):
    t = (text or "")[:-1]
    h = list(history or [])
    if h: h.pop()
    return t, h, t or "(nothing yet)", _history_html(h)

def on_space(text, history):
    t = (text or "") + " ";  h = list(history or []) + [" "]
    return t, h, t, _history_html(h)

def on_clear():
    return "", [], "(nothing yet)", _history_html([])

def _history_html(hist):
    if not hist:
        return "<span style='color:#3D4F72;font-size:0.85rem'>No letters yet</span>"
    chips = ""
    for i, c in enumerate(hist[-20:]):
        label = "·" if c == " " else c
        style = ("background:rgba(0,212,255,0.12);border:1px solid rgba(0,212,255,0.4);"
                 "color:#00D4FF;" if i == len(hist[-20:])-1
                 else "background:#141B2E;border:1px solid #1E2640;color:#8899BB;")
        chips += (f"<span style='{style}border-radius:6px;padding:4px 10px;"
                  f"font-family:Space Grotesk,sans-serif;font-weight:600;"
                  f"font-size:0.9rem;display:inline-block'>{label}</span> ")
    return f"<div style='display:flex;flex-wrap:wrap;gap:6px'>{chips}</div>"


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;500&display=swap');

body, .gradio-container { background:#0A0E1A !important; color:#E8EAF0 !important; font-family:'Inter',sans-serif !important; }

/* hero */
#hero { background:linear-gradient(135deg,#0F1422,#141B2E); border:1px solid #1E2640;
        border-radius:16px; padding:28px 32px; margin-bottom:8px; }
#hero h1 { font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700;
           color:#fff; margin:0 0 4px 0; }
#hero p  { color:#6B7A9E; font-size:0.9rem; margin:0; }
#hero .badge { background:rgba(0,212,255,0.12); border:1px solid rgba(0,212,255,0.3);
               color:#00D4FF; border-radius:8px; padding:4px 12px; font-size:0.75rem;
               font-weight:600; display:inline-block; margin-top:8px; }

/* panels */
.panel { background:#0F1422; border:1px solid #1E2640; border-radius:14px; padding:20px; }
.panel-label { font-family:'Space Grotesk',sans-serif; font-size:0.68rem; font-weight:700;
               letter-spacing:1.2px; text-transform:uppercase; color:#3D4F72; margin-bottom:8px; }

/* letter big */
#letter-box { background:linear-gradient(135deg,#0D1528,#111C35);
              border:1px solid #1E3A5F; border-radius:14px; padding:24px;
              text-align:center; min-height:120px;
              display:flex; flex-direction:column; align-items:center; justify-content:center; }
#letter-big { font-family:'Space Grotesk',sans-serif; font-size:5rem; font-weight:700;
              color:#00D4FF; line-height:1; text-shadow:0 0 40px rgba(0,212,255,0.25); }

/* sentence */
#sentence { background:#0B1020; border:1px solid #1E2640; border-radius:12px;
            padding:18px 20px; font-family:'Space Grotesk',sans-serif;
            font-size:1.3rem; font-weight:500; color:#fff;
            min-height:64px; word-break:break-all; line-height:1.5; }

/* buttons */
.gr-button { background:rgba(0,212,255,0.08) !important; border:1px solid rgba(0,212,255,0.3) !important;
             color:#00D4FF !important; border-radius:10px !important;
             font-family:'Space Grotesk',sans-serif !important; font-weight:600 !important; }
.gr-button:hover { background:rgba(0,212,255,0.18) !important; border-color:#00D4FF !important; }

/* sliders / labels */
label span, .gr-form label { color:#8899BB !important; font-size:0.85rem !important; }
input[type=range] { accent-color:#00D4FF; }

/* image panels */
.gr-image img { border-radius:12px; border:1px solid #1E2640; }

/* tab / accordion chrome */
.gr-tab-nav { background:#0F1422 !important; border-color:#1E2640 !important; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="SignBridge · ASL Detector") as demo:

    # State
    composed_state = gr.State("")
    history_state  = gr.State([])

    # Hero
    gr.HTML("""
    <div id="hero">
      <h1>🤟 SignBridge</h1>
      <p>Real-time American Sign Language → Text</p>
      <span class="badge">ASL · 27 SIGNS</span>
    </div>
    """)

    with gr.Row():
        # ── Left: camera ──
        with gr.Column(scale=3):
            gr.HTML('<div class="panel-label">Camera input</div>')
            cam_in = gr.Image(sources=["webcam"], streaming=False,
                              label="", mirror_webcam=True,
                              elem_classes=["panel"])
            gr.HTML('<div class="panel-label" style="margin-top:12px">Annotated frame</div>')
            ann_out  = gr.Image(label="", interactive=False, elem_classes=["panel"])
            gr.HTML('<div class="panel-label" style="margin-top:12px">Hand crop (model input)</div>')
            crop_out = gr.Image(label="", interactive=False, width=180)

        # ── Right: panel ──
        with gr.Column(scale=2):
            status_md = gr.Markdown("🟡 Waiting for snapshot…")

            gr.HTML('<div class="panel-label">Detected sign</div>')
            letter_out = gr.HTML('<div id="letter-box"><div id="letter-big">—</div></div>')

            gr.HTML('<div class="panel-label" style="margin-top:12px">Confidence</div>')
            conf_bar = gr.Slider(minimum=0, maximum=100, value=0, interactive=False,
                                 label="Match strength %")

            gr.HTML('<div class="panel-label" style="margin-top:12px">Min confidence filter</div>')
            min_conf_sl = gr.Slider(minimum=0, maximum=100, value=0, step=5,
                                    label="Only append if confidence ≥ (%)")

            gr.HTML('<div class="panel-label" style="margin-top:12px">Composed text</div>')
            sentence_out = gr.HTML('<div id="sentence"><span style="color:#2A3550">Start signing…</span></div>')

            gr.HTML('<div class="panel-label" style="margin-top:12px">Recent letters</div>')
            history_out = gr.HTML(_history_html([]))

            with gr.Row():
                btn_bs    = gr.Button("⌫ Backspace")
                btn_space = gr.Button("␣ Space")
                btn_clear = gr.Button("✕ Clear")

            gr.HTML('<div class="panel-label" style="margin-top:12px">Copy text</div>')
            copy_box = gr.Textbox(value="(nothing yet)", interactive=False,
                                  label="", show_copy_button=True)

    # ── Wire up ───────────────────────────────────────────────────────────────
    snap_outputs = [ann_out, crop_out, letter_out, conf_bar,
                    composed_state, history_state, copy_box, history_out]

    # Fires when user clicks the snapshot button inside the webcam widget
    cam_in.change(
        fn=lambda img, text, hist, mc: on_snapshot(img, text, hist, mc),
        inputs=[cam_in, composed_state, history_state, min_conf_sl],
        outputs=snap_outputs,
    )

    btn_bs.click(on_backspace,
                 inputs=[composed_state, history_state],
                 outputs=[composed_state, history_state, copy_box, history_out])

    btn_space.click(on_space,
                    inputs=[composed_state, history_state],
                    outputs=[composed_state, history_state, copy_box, history_out])

    btn_clear.click(on_clear,
                    outputs=[composed_state, history_state, copy_box, history_out])

    # Keep sentence_out in sync with copy_box value via JS trick — update both on snap
    cam_in.change(
        fn=lambda t: f'<div id="sentence">{t if t else "<span style=\'color:#2A3550\'>Start signing…</span>"}</div>',
        inputs=[composed_state],
        outputs=[sentence_out],
    )

    # Letter box HTML
    cam_in.change(
        fn=lambda img, text, hist, mc: (
            f'<div id="letter-box"><div id="letter-big">'
            f'{on_snapshot(img, text, hist, mc)[2]}'
            f'</div><div style="font-size:0.75rem;color:#3D4F72;letter-spacing:1px;text-transform:uppercase;margin-top:6px">current sign</div></div>'
        ),
        inputs=[cam_in, composed_state, history_state, min_conf_sl],
        outputs=[letter_out],
    )

if __name__ == "__main__":
    demo.launch()