# app.py

import streamlit as st
from agents.predicting_dmg import classify_damage
from agents.chains import get_damage_feedback
from agents.obd_agent import get_obd_feedback  # You already built this

st.set_page_config(page_title="AutoFix-GPT", layout="centered")
st.title("🔧 AutoFix-GPT")

# --- User Input Type ---
mode = st.radio("Select input type:", ["🔍 OBD Fault Code", "📷 Upload Damage Image"])

if mode == "🔍 OBD Fault Code":
    dtc_input = st.text_input("Enter Fault Code or Description:")
    if st.button("Diagnose"):
        with st.spinner("Diagnosing..."):
            result = get_obd_feedback(dtc_input)
        st.markdown("### 🧠 Diagnosis Result:")
        st.write(result)

elif mode == "📷 Upload Damage Image":
    image_file = st.file_uploader("Upload an image of the damaged car area", type=["jpg", "png", "jpeg"])
    if image_file and st.button("Analyze Damage"):
        with st.spinner("Analyzing..."):
            # Save temp image
            with open("temp_image.jpg", "wb") as f:
                f.write(image_file.getbuffer())

            predicted_damage = classify_damage("temp_image.jpg")
            feedback = get_damage_feedback(predicted_damage)

        st.markdown("### 🔍 Detected Damage:")
        st.success(f"🛠️ {predicted_damage}")

        st.markdown("### 💬 Repair Feedback:")
        st.write(feedback)
