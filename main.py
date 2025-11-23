import streamlit as st
import joblib
import numpy as np

model = joblib.load("spam_detect_model.pkl")
vectorizer = joblib.load("transform.pkl")

st.title("📧 Mail Spam Detection System")
st.write("---")

mail = st.text_input(" ✉️ Enter the mail or the message whatever you are try to check...", key="mail")





if st.button("Predict"):
    if mail.strip() == "":
        st.warning("⚠ Please enter a message before predicting.")
    else:
        # Convert message into vector format
        mail_vector = vectorizer.transform([mail])

        # Make prediction
        pred = model.predict(mail_vector)[0]
        proba = model.predict_proba(mail_vector)[0][1] * 100

        # Display Result in styled format
        if pred == 1:
            st.error("🚨 **Spam Alert! This message appears to be spam.**")
        else:
            st.success("✅ **This message is safe.**")
        
        # Show probability score
        st.info(f"📊 Spam Confidence: **{proba:.2f}%**")

st.markdown("""
<style>
    .stTextInput>div>div>input {
        border-radius: 10px;
        padding: 10px;
        font-size: 18px;
    }
    .prediction-box {
        font-size: 22px;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


st.write("---")
if "mail" not in st.session_state:
    st.session_state.mail = ""

# --- Define functions (callback style) ---
def set_spam():
    st.session_state.mail = "Congratulations! You've won a FREE vacation. Click the link to claim now!!!"

def set_ham():
    st.session_state.mail = "Hello, just checking if you are available tomorrow."

# --- Buttons that update the textbox BEFORE rendering ---
with st.expander("Try Sample Emails"):
    col1, col2 = st.columns(2)

    col1.button("Example Spam", on_click=set_spam)
    col2.button("Example Safe", on_click=set_ham)
