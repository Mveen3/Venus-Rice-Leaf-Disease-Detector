"""
app.py — Rice Leaf Disease Prediction  ·  Streamlit Web App
Upload a rice leaf photo → disease prediction → confidence bar → farmer-friendly advice.
"""

import os, json, base64
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import numpy as np

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "rice_disease_model.pth")
CLASS_PATH     = os.path.join(BASE_DIR, "class_names.json")
DISEASE_INFO   = os.path.join(BASE_DIR, "disease_info.json")
FARMER_IMGS    = os.path.join(BASE_DIR, "Farmer Images")
IMG_SIZE       = 224

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Rice Leaf Disease Detector",
    page_icon="🌾",
    layout="centered",
)

# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────
def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

farmer_images = sorted(
    [os.path.join(FARMER_IMGS, f) for f in os.listdir(FARMER_IMGS)
     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
)

# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
# Build carousel slide CSS for N images
n_imgs = len(farmer_images)
total_dur = n_imgs * 5  # 5 seconds per image
fade_pct = 8  # percent of total for fade in/out
show_pct = int(100 / n_imgs)

carousel_nth = ""
for i in range(n_imgs):
    delay = i * 5
    carousel_nth += f".carousel-slide:nth-child({i+1}) {{ animation-delay: {delay}s; }}\n"

dots_nth = ""
for i in range(n_imgs):
    delay = i * 5
    dots_nth += f".carousel-dot:nth-child({i+1}) {{ animation-delay: {delay}s; }}\n"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*, html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
}}

/* Hide Streamlit defaults */
#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}
[data-testid="stToolbar"] {{ display: none; }}

/* ══════ SOFT PALE GREEN PALETTE ══════
   Background:     #eef5e8  (soft sage)
   Card/Surface:   #f7faf4  (lightest green-white)
   Card border:    #c8dcc0  (muted sage border)
   Heading text:   #2d3b2d  (dark forest green)
   Body text:      #4a5e4a  (medium green-grey)
   Muted text:     #7a937a  (dusty sage)
   Accent:         #2e7d32  (rich green)
   Accent light:   #43a047  (medium green)
   Accent bg:      #d5ecd5  (pale green tint)
   Hover bg:       #e3f0de  (soft hover)
   ════════════════════════════════════ */

/* Page */
.stApp {{
    background: #eef5e8 !important;
}}

/* Force all default Streamlit text to be dark */
.stApp, .stApp p, .stApp span, .stApp label,
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] span {{
    color: #2d3b2d !important;
}}
.stApp hr {{
    border-color: #c8dcc0 !important;
}}

/* ── Carousel ── */
.carousel-wrap {{
    position: relative;
    width: 100%;
    height: 320px;
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 2rem;
    box-shadow: 0 4px 20px rgba(46,125,50,0.15), 0 1px 4px rgba(0,0,0,0.08);
    border: 1px solid #c8dcc0;
}}
.carousel-slide {{
    position: absolute;
    inset: 0;
    opacity: 0;
    animation: cfade {total_dur}s ease-in-out infinite;
}}
{carousel_nth}
@keyframes cfade {{
    0%   {{ opacity: 0; }}
    {fade_pct}%  {{ opacity: 1; }}
    {show_pct}%  {{ opacity: 1; }}
    {show_pct + fade_pct}% {{ opacity: 0; }}
    100% {{ opacity: 0; }}
}}
.carousel-slide img {{
    width: 100%; height: 100%;
    object-fit: cover;
    display: block;
}}
.carousel-overlay {{
    position: absolute; inset: 0;
    background: linear-gradient(to top,
        rgba(20,40,20,0.75) 0%,
        rgba(20,40,20,0.2) 50%,
        rgba(20,40,20,0.05) 100%
    );
    display: flex; flex-direction: column;
    justify-content: flex-end;
    padding: 2rem 2.5rem;
    z-index: 2;
}}
.carousel-title {{
    font-size: 1.8rem; font-weight: 800;
    color: #ffffff !important; letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
}}
.carousel-desc {{
    font-size: 0.92rem; color: rgba(255,255,255,0.8) !important;
    font-weight: 400; max-width: 500px;
}}
.carousel-dots {{
    position: absolute;
    bottom: 14px; right: 20px;
    display: flex; gap: 7px;
    z-index: 3;
}}
.carousel-dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    background: rgba(255,255,255,0.35);
    animation: dotfade {total_dur}s ease-in-out infinite;
}}
{dots_nth}
@keyframes dotfade {{
    0%   {{ background: rgba(255,255,255,0.35); }}
    {fade_pct}%  {{ background: rgba(255,255,255,0.95); }}
    {show_pct}%  {{ background: rgba(255,255,255,0.95); }}
    {show_pct + fade_pct}% {{ background: rgba(255,255,255,0.35); }}
    100% {{ background: rgba(255,255,255,0.35); }}
}}

/* ── Section headers ── */
.sec-label {{
    font-size: 0.72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.5px;
    color: #2e7d32 !important; margin-bottom: 0.6rem;
}}

/* ── Upload placeholder ── */
.upload-placeholder {{
    border: 2px dashed #b5ccae;
    border-radius: 14px;
    padding: 2.5rem;
    text-align: center;
    background: #f7faf4;
    transition: all 0.25s ease;
}}
.upload-placeholder:hover {{
    border-color: #2e7d32;
    background: #e3f0de;
}}

/* ── Streamlit file uploader ── */
[data-testid="stFileUploader"] {{
    margin-top: 0px;
}}

/* Dropzone wrapper — position relative so we can overlay the file input */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploadDropzone"] {{
    border: 2px dashed #b5ccae !important;
    border-radius: 14px !important;
    padding: 2.5rem 1rem 2rem !important;
    background-color: #f7faf4 !important;
    transition: all 0.25s ease !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    position: relative !important;
    min-height: 160px !important;
}}
[data-testid="stFileUploader"] section:hover,
[data-testid="stFileUploadDropzone"]:hover {{
    border-color: #2e7d32 !important;
    background-color: #e3f0de !important;
}}

/* ── Hide every native child of the inner dropzone div EXCEPT the file pill ── */
[data-testid="stFileUploadDropzone"] > div > *:not([data-testid="stFileUploaderFile"]) {{
    display: none !important;
}}

/* Stretch the hidden <input type="file"> over the whole dropzone so any click opens the picker */
[data-testid="stFileUploadDropzone"] input[type="file"],
[data-testid="stFileUploader"] section input[type="file"] {{
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    opacity: 0 !important;
    cursor: pointer !important;
    z-index: 10 !important;
}}

/* ── Uploaded file pill ── */
[data-testid="stFileUploaderFile"] {{
    display: flex !important;
    align-items: center !important;
    gap: 0.6rem !important;
    background: #e3f0de !important;
    border: 1px solid #b5ccae !important;
    border-radius: 10px !important;
    padding: 0.55rem 1rem !important;
    width: 85% !important;
    margin-top: 0.5rem !important;
    position: relative !important;
    z-index: 20 !important;
    pointer-events: auto !important;
}}
[data-testid="stFileUploaderFile"] * {{
    display: revert !important;
    color: #2d3b2d !important;
}}

/* Progress bar inside the pill */
[data-testid="stFileUploaderFile"] [data-testid="stProgressBar"],
[data-testid="stFileUploaderFile"] .stProgress,
[data-testid="stFileUploaderFile"] .stProgress > div,
[data-testid="stFileUploaderFile"] .stProgress > div > div {{
    display: block !important;
    width: 100% !important;
}}
[data-testid="stFileUploaderFile"] .stProgress > div > div > div {{
    background: linear-gradient(90deg, #2e7d32, #43a047) !important;
}}

/* ── Custom instruction text via pseudo-elements ── */
[data-testid="stFileUploader"] section::before,
[data-testid="stFileUploadDropzone"]::before {{
    content: "📸\\A Drag & drop or click here to browse files";
    white-space: pre-wrap;
    font-size: 1.15rem;
    font-weight: 700;
    color: #2d3b2d;
    display: block;
    text-align: center;
    width: 100%;
    line-height: 1.7;
    margin-bottom: 0.4rem;
    pointer-events: none;
    z-index: 1;
    position: relative;
}}
[data-testid="stFileUploader"] section::after,
[data-testid="stFileUploadDropzone"]::after {{
    content: "Limit 200MB per file • JPG, JPEG, PNG, BMP";
    font-size: 0.85rem;
    font-weight: 500;
    color: #7a937a;
    display: block;
    text-align: center;
    width: 100%;
    pointer-events: none;
    z-index: 1;
    position: relative;
}}

/* ── Result badge ── */
.result-badge {{
    display: inline-flex; align-items: center; gap: 10px;
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    padding: 12px 24px;
    border-radius: 12px;
    color: #ffffff !important; font-weight: 700;
    font-size: 1.25rem;
    box-shadow: 0 4px 15px rgba(46,125,50,0.3);
    margin: 0.5rem 0 1rem 0;
}}

/* ── Info cards ── */
.info-card {{
    background: #f7faf4;
    border: 1px solid #c8dcc0;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 6px rgba(46,125,50,0.06);
}}
.info-card-hdr {{
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 0.7rem;
}}
.info-card-icon {{
    width: 36px; height: 36px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}}
.bg-amber {{ background: #f0e6c8; }}
.bg-blue  {{ background: #c8dae6; }}
.info-card-title {{
    font-size: 0.95rem; font-weight: 700; color: #2d3b2d !important;
}}
.info-card-body {{
    font-size: 0.92rem; color: #4a5e4a !important; line-height: 1.75;
}}

/* ── Disease ref cards ── */
.ref-card {{
    background: #f7faf4;
    border: 1px solid #c8dcc0;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    min-height: 180px;
    box-shadow: 0 2px 6px rgba(46,125,50,0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
.ref-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(46,125,50,0.12);
}}

/* ── Model details strip ── */
.model-strip {{
    display: flex;
    justify-content: space-around;
    background: #f7faf4;
    border: 1px solid #c8dcc0;
    border-radius: 14px;
    padding: 1.2rem;
    margin: 1.5rem 0;
    box-shadow: 0 2px 6px rgba(46,125,50,0.06);
}}
.model-stat {{ text-align: center; }}
.model-stat-val {{
    font-size: 1.4rem; font-weight: 800; color: #1b5e20 !important;
}}
.model-stat-lbl {{
    font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px;
    color: #7a937a !important; margin-top: 2px;
}}

/* Progress bar */
[data-testid="stProgress"] > div > div > div {{
    background: linear-gradient(90deg, #2e7d32, #43a047) !important;
}}
[data-testid="stProgress"] > div > div {{
    background: #d5ecd5 !important;
}}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(CLASS_PATH) as f:
        class_names = json.load(f)
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(class_names))
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, class_names

@st.cache_data
def load_disease_info():
    with open(DISEASE_INFO) as f:
        return json.load(f)

inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

@torch.no_grad()
def predict(image: Image.Image, model, class_names):
    tensor = inference_transform(image).unsqueeze(0)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).squeeze().numpy()
    pred_idx = int(np.argmax(probs))
    return class_names[pred_idx], {c: float(probs[i]) for i, c in enumerate(class_names)}


# ══════════════════════════════════════════════
#  CAROUSEL — farmer images
# ══════════════════════════════════════════════
slides = ""
for img_path in farmer_images:
    b64 = img_to_b64(img_path)
    slides += f'<div class="carousel-slide"><img src="data:image/jpeg;base64,{b64}" /></div>'
dots = "".join(f'<div class="carousel-dot"></div>' for _ in farmer_images)

st.markdown(f"""
<div class="carousel-wrap">
    {slides}
    <div class="carousel-overlay">
        <div class="carousel-title">🌾 Rice Leaf Disease Detector</div>
        <div class="carousel-desc">Upload a photo of a rice leaf to identify the disease and get treatment advice.</div>
    </div>
    <div class="carousel-dots">{dots}</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  MODEL STATS STRIP
# ══════════════════════════════════════════════
st.markdown("""
<div class="model-strip">
    <div class="model-stat">
        <div class="model-stat-val">EfficientNet-B0</div>
        <div class="model-stat-lbl">Architecture</div>
    </div>
    <div class="model-stat">
        <div class="model-stat-val">4,804</div>
        <div class="model-stat-lbl">Training Images</div>
    </div>
    <div class="model-stat">
        <div class="model-stat-val">99.17%</div>
        <div class="model-stat-lbl">Test Accuracy</div>
    </div>
    <div class="model-stat">
        <div class="model-stat-val">3</div>
        <div class="model-stat-lbl">Classes</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  UPLOAD SECTION
# ══════════════════════════════════════════════
st.markdown('<div class="sec-label">Upload Leaf Image</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Take a clear picture of the affected leaf",
    type=["jpg", "jpeg", "png", "bmp"],
    label_visibility="collapsed",
)

# ══════════════════════════════════════════════
#  RESULTS  (if uploaded)
# ══════════════════════════════════════════════
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

    col_img, col_res = st.columns([2, 3], gap="large")

    with col_img:
        st.markdown('<div class="sec-label">Uploaded Image</div>', unsafe_allow_html=True)
        st.image(image, width="stretch")

    with col_res:
        with st.spinner("Analysing leaf…"):
            model, class_names = load_model()
            disease_info = load_disease_info()
            pred_name, confidences = predict(image, model, class_names)

        st.markdown('<div class="sec-label">Prediction</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-badge">🔬 {pred_name}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-label" style="margin-top:1rem;">Confidence Scores</div>', unsafe_allow_html=True)
        sorted_conf = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
        for cls, prob in sorted_conf:
            pct = prob * 100
            st.markdown(f"<p style='color:#2d3b2d; font-weight:600; font-size:0.92rem; margin-bottom:0;'>{cls} — {pct:.1f}%</p>", unsafe_allow_html=True)
            st.progress(prob)

    # ── Explanation & Action ──
    st.markdown("---")
    st.markdown('<div class="sec-label">Explanation &amp; Recommended Action</div>', unsafe_allow_html=True)

    info = disease_info.get(pred_name, {})
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-hdr">
                <div class="info-card-icon bg-amber">🦠</div>
                <div class="info-card-title">What is {pred_name}?</div>
            </div>
            <div class="info-card-body">
                {info.get("description", "No information available.")}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-hdr">
                <div class="info-card-icon bg-blue">💊</div>
                <div class="info-card-title">Recommended Action</div>
            </div>
            <div class="info-card-body">
                {info.get("action", "Consult your local agricultural extension officer.")}
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Empty state — show disease reference ──
    st.markdown("---")
    st.markdown('<div class="sec-label">Diseases We Detect</div>', unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    diseases_ref = [
        ("🟡", "Bacterial Leaf Blight", "Yellowing from leaf edges inward, caused by Xanthomonas oryzae. Common in humid conditions."),
        ("🟤", "Brown Spot", "Oval brown lesions with grey centres from Bipolaris oryzae. Linked to nutrient-poor soil."),
        ("⚫", "Leaf Smut", "Angular dark spots from Entyloma oryzae. Appears late in season, reduces grain quality."),
    ]
    for col, (emoji, name, desc) in zip([d1, d2, d3], diseases_ref):
        with col:
            st.markdown(f"""
            <div class="ref-card">
                <div style="font-size:2rem; margin-bottom:0.6rem;">{emoji}</div>
                <div style="font-weight:700; color:#2d3b2d; margin-bottom:0.4rem;">{name}</div>
                <div style="font-size:0.85rem; color:#4a5e4a; line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#7a937a; font-size:0.8rem; padding:0.5rem 0; line-height:1.8;'>"
    "Rice Leaf Disease Detector · EfficientNet-B0 · Trained on 4,804 images<br>"
    "Made with ❤️ by <strong style='color:#4a5e4a;'>Team Venus</strong> : "
    "<a href='https://www.linkedin.com/in/chprvaibhav/' target='_blank' style='color:#2e7d32; text-decoration:none; font-weight:600;'>Vaibhav</a>, "
    "<a href='https://www.linkedin.com/in/mveen3' target='_blank' style='color:#2e7d32; text-decoration:none; font-weight:600;'>Naveen</a>, "
    "<a href='https://www.linkedin.com/in/sriram-paruchuri-8b3160250/' target='_blank' style='color:#2e7d32; text-decoration:none; font-weight:600;'>Sri Ram</a>"
    "</div>",
    unsafe_allow_html=True,
)
