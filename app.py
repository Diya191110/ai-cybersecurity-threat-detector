import streamlit as st
import numpy as np
import joblib
import os

# Load the trained model
model_path = os.path.join("models", "best_model.pkl")
model = joblib.load(model_path)

# Feature names used in training (31 features)
feature_names = [
    'UsingIP', 'LongURL', 'ShortURL', 'Symbol@', 'Redirecting//',
    'PrefixSuffix-', 'SubDomains', 'HTTPS', 'DomainRegLen', 'Favicon',
    'NonStdPort', 'HTTPSDomainURL', 'RequestURL', 'AnchorURL', 'LinksInScriptTags',
    'ServerFormHandler', 'InfoEmail', 'AbnormalURL', 'WebsiteForwarding',
    'StatusBarCust', 'DisableRightClick', 'UsingPopupWindow', 'IframeRedirection',
    'AgeofDomain', 'DNSRecording', 'WebsiteTraffic', 'PageRank', 'GoogleIndex',
    'LinksPointingToPage', 'StatsReport', 'Index'
]

# Streamlit UI
st.set_page_config(page_title="AI Cybersecurity Threat Detector", page_icon="🛡️")
st.title("🛡️ AI-Powered Cybersecurity Threat Detector")
st.markdown("Enter the website features below to detect if it's **Phishing or Legitimate**.")

# Collect input from user
input_values = []
for feature in feature_names:
    value = st.selectbox(f"{feature}", options=[-1, 0, 1], index=1, format_func=lambda x: {
        -1: "Negative / Bad",
         0: "Neutral / Unknown",
         1: "Positive / Good"
    }[x])
    input_values.append(value)

# Prediction
if st.button("🚀 Predict"):
    try:
        input_array = np.array(input_values).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        result = "🟢 Legitimate Website" if prediction == 1 else "🔴 Phishing Website"

        st.subheader("🔍 Prediction Result:")
        if prediction == 1:
            st.success(result)
        else:
            st.error(result)
    except Exception as e:
        st.error(f"❌ Error: {e}")
