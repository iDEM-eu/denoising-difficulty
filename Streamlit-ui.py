import streamlit as st
import requests

st.title("Text Classification & Denoising")

model_name = st.selectbox("Choose a Model", ["baseline", "ct", "labs", "ntm", "st", "gmm-sbert", "gmm-mbert"])
data_path = st.text_input("Path to Training/Test Data")
output_dir = st.text_input("Output Directory")
mode = st.radio("Mode", ["Train", "Test"])

if st.button("Run Model"):
    url = "http://127.0.0.1:8000/train" if mode == "Train" else "http://127.0.0.1:8000/test"
    payload = {"model_name": model_name, "data_path": data_path, "output_dir": output_dir}
    response = requests.post(url, json=payload)
    st.write(response.json())
