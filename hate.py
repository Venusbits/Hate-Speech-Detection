import streamlit as st
from .HateDetection import svm, cv, clean_data

# Custom CSS
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6;
        padding: 20px;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    h1 {
        color: #ff6347;
    }
    .stTextInput, .stButton {
        border: 2px solid #ff6347;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton > button {
        background-color: #ff6347;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton > button:hover {
        background-color: #ff4500;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Hate Speech Detection")

sample = st.text_input("Enter text to analyze:")

if st.button("Analyze"):
    sample = clean_data(sample)
    data1 = cv.transform([sample]).toarray()
    svm_prediction = svm.predict(data1)
    st.write("Predicted label:", svm_prediction[0])
