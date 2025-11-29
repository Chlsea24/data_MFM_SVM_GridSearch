import streamlit as st
import joblib
import tempfile
from PIL import Image
import numpy as np
from feature_extract import extract_numeric_vector


# Streamlit Page Config

st.set_page_config(
    page_title="Identifikasi BP-ST",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# Custom CSS

st.markdown("""
    <style>
    .stButton>button {
        background-color: #54C488 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
    }
    .stButton>button:hover { background-color: #45A874 !important; }
    </style>
""", unsafe_allow_html=True)



# Load Model & Scaler

MODEL_PATH = "saved_model/SVM_GD_91_RBF_best_model.joblib"
SCALER_PATH = "saved_model/SVM_GD_91_scaler.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)



# Header

st.title("Identifikasi Wajah Gejala Stroke vs Bell’s Palsy")
#st.write("Silahkan unggah foto wajah yang ingin diklasifikasi. (Format: JPG / JPEG / PNG)")
#st.write("Saat ini, model hanya dapat mengklasifikasikan wajah gejala ***Bell's Palsy*** dan ***Stroke*** berdasarkan citra yang diinput.")

catatan = '''Saat ini, model hanya dapat mengklasifikasikan wajah gejala ***Bell's Palsy*** dan ***Stroke*** berdasarkan citra yang diinput.
Silahkan unggah foto wajah yang ingin diklasifikasi. (Format: JPG / JPEG / PNG)
'''
st.markdown(catatan)

# Input Mode (Upload atau Kamera)

mode = st.radio("Pilih metode input:", ["📁 Upload", "📷 Kamera"])

img_path = None

if mode == "📁 Upload":
    uploaded = st.file_uploader("Upload JPG/PNG", type=["jpg", "jpeg", "png"], key="file_upload")
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded.read())
            img_path = tmp.name

        st.image(img_path, use_container_width=True)

elif mode == "📷 Kamera":
    camera_image = st.camera_input("Ambil Foto Wajah", key="camera_input")
    if camera_image:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(camera_image.getvalue())
            img_path = tmp.name

        st.image(img_path, use_container_width=True)



# Klasifikasi

if img_path is not None:

    with st.spinner("⏳ Memproses gambar..."):
        fitur = extract_numeric_vector(img_path)

    if fitur is None:
        st.error("❌ Wajah tidak terdeteksi. Silakan gunakan foto yang lebih jelas.")
        st.markdown(
            "<span style='color:#ff6b6b;'>Tips:</span> "
            "Gunakan gambar dengan pencahayaan cukup dan posisi wajah menghadap kamera.",
            unsafe_allow_html=True
        )
    else:
        fitur_scaled = scaler.transform(fitur)
        pred = model.predict(fitur_scaled)[0]

        label = "Stroke" if pred == 0 else "Bell’s Palsy"

        st.success(f"**✅ Hasil Klasifikasi: {label}**")



# Footer

st.markdown(
    """
    <div style='text-align:center; color:#FF4B4B; font-weight:600; margin-top:25px;'>
    PERINGATAN! <br>
    Hasil **belum dapat** dijadikan acuan utama dalam diagnosis. <br>
    Silakan konsultasikan lebih lanjut ke tenaga medis profesional.
    <br><br>
    <span style='font-weight:400; font-size:14px; color:#888;'>Universitas Bunda Mulia 2025 @Chelsea Effendi</span>
    </div>
    """,
    unsafe_allow_html=True
)