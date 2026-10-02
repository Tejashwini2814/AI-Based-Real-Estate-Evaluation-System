import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from streamlit_mic_recorder import speech_to_text
import random

# ---------- GLASSMORPHISM MAIN & TABS STYLING ----------
st.set_page_config(page_title="🏡 AI BASED REAL ESTATE EVALUATION SYSTEM", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
body, html, * { font-family: 'Inter', 'Poppins', sans-serif !important;}
body { 
    background: radial-gradient(ellipse 100% 100% at 60% 10%, #e3e9ff 0%, #e9f0fa 60%, #eef4ff 100%) !important; 
}

/* --- Tabs --- */
.stTabs [role="tablist"] { 
    background: rgba(255,255,255,0.21); 
    backdrop-filter: blur(13px); 
    border-radius: 30px; 
    box-shadow: 0 4px 30px #b3baff3a, 0 1px 12px #6659b312; 
    margin-bottom: 26px; 
    padding: 11px 0; 
    gap: 22px; 
    justify-content: center;
}
.stTabs [role="tab"] { 
    background: linear-gradient(92deg, #eaf8ff 70%, #ebeaff 100%); 
    font-size: 22px; 
    font-weight: 800; 
    border-radius:19px; 
    color: #6659b3 !important; 
    border: none; 
    padding: 16px 47px 13px 47px; 
    margin: 1px 9px 3px 9px; 
    box-shadow: 0px 2px 12px #b3baff19; 
    transition: background .18s, box-shadow .19s; 
    letter-spacing: 0.5px;
}
.stTabs [role="tab"]:hover { 
    background: linear-gradient(90deg,#eaf0ff 85%, #ceeaff 99%); 
    color: #805ad5 !important; 
    box-shadow: 0 8px 34px #767cf279;
}
.stTabs [aria-selected="true"] { 
    background: linear-gradient(90deg,#76aaff 11%, #97c3ff 80%, #eae6fd 100%); 
    color: #2f367d !important; 
    box-shadow: 0 12px 38px 0 #805ad52b; 
    border-bottom: 4px solid #805ad5;
}

/* --- Glass Cards --- */
.glass-card { 
    background: rgba(255,255,255,0.82); 
    border-radius: 22px; 
    box-shadow: 0 8px 38px 1px #805ad531,0 3px 14px #b3baff18; 
    backdrop-filter: blur(13px); 
    -webkit-backdrop-filter: blur(13px); 
    padding: 28px 24px; 
    margin-bottom: 15px; 
    border: 1.2px solid #e8e6fa58; 
    transition: box-shadow .2s;
}
.glass-card:hover { 
    box-shadow:0 28px 52px #6b46c144,0 8px 32px #805ad536;
}

/* --- Glass Prediction --- */
.prediction-box { 
    background: linear-gradient(135deg,#805ad5 4%,#6b46c1 99%); 
    color: #fff; 
    border-radius: 23px; 
    box-shadow: 0 10px 44px #805ad54a; 
    text-align: center; 
    font-size: 30px; 
    font-weight: 800; 
    margin: 32px 10px 18px 10px; 
    padding: 23px 15px; 
    animation:pulse 2.2s infinite; 
    letter-spacing: .5px;
}
@keyframes pulse {
  0%{box-shadow:0 0 24px #805ad58d;}
  50%{box-shadow:0 0 36px #6b46c19a;}
  100%{box-shadow:0 0 24px #805ad58d;}
}
.cma-summary { 
    background: linear-gradient(124deg,#afaaffb3 8%,#785ad5e6 45%, #6b46c1c8 99%); 
    color: white; 
    padding: 22px 22px; 
    border-radius: 18px; 
    box-shadow: 0 7px 38px #805ad598; 
    font-size: 18px; 
    font-weight: 400; 
    line-height: 1.7;
    margin:20px 0 14px 0; 
    border: 1.2px solid #e9dbff56;
}
.stDataFrame { 
    border-radius: 17px; 
    box-shadow: 0 6px 24px #7d68c9cf;
}
div[data-testid="stVerticalBlock"] {padding-bottom:12px;}

/* --- Button ONLY (dark) --- */
.stButton > button {
    font-size: 18px;
    padding: 17px 48px;
    border-radius: 18px;
    font-weight: 800;
    color: #fff !important;
    background: linear-gradient(90deg, #221a4d 0%, #382d6c 60%, #17458f 100%);
    box-shadow: 0 8px 38px #15182a, 0 3px 14px #382d6c44;
    border: none;
    transition: background .16s, box-shadow .16s;
    letter-spacing: 1px;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #150c3b 0%, #42344e 80%, #2358a2 100%);
    color: #bbeaff !important;
    box-shadow: 0 16px 48px #17458f60;
}

/* --- Heading glass effect --- */
.main-title { 
    text-align: center; 
    font-size: 44px; 
    font-weight: 900; 
    background: linear-gradient(90deg, #007CF0 7%, #00DFD8 97%); 
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent; 
    letter-spacing: 2px; 
    margin-bottom: 3px;
}
.subtitle { 
    text-align: center; 
    font-size: 23px; 
    color: #575c99; 
    margin-bottom: 32px;
    font-weight:700;
}

/* --- GLASS CHAT BUBBLES --- */
.stChatMessageContainer {
    margin-top: 12px !important;
    margin-bottom: 6px !important;
    border-radius: 18px !important;
    background: rgba(255,255,255,0.88) !important;
    box-shadow: 0 4px 26px #805ad534, 0 1px 10px #b3baff18;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1.1px solid #e8e6fa38;
    padding: 13px 18px !important;
    transition: box-shadow .18s;
}

.stChatMessageContainer[data-author="user"] {
    background: linear-gradient(97deg, #eaf8ff 65%, #ceeaff 100%) !important;
    border: 1.3px solid #c5e7ff8c !important;
    color: #2745a5 !important;
    text-align: right;
}

.stChatMessageContainer[data-author="assistant"] {
    background: linear-gradient(120deg, #f9f9ff 0%, #fef9f6 100%) !important;
    border: 1px solid rgba(200, 200, 200, 0.3) !important;
    color: #2d3748 !important;
    text-align: left;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    font-size: 15px !important;
    line-height: 1.5 !important;
}

/* --- Chat Input Box (blue themed) --- */
div[data-testid="stChatInputContainer"] {
    background: rgba(230, 245, 255, 0.95) !important;  /* soft blue */
    border: 1px solid rgba(120, 170, 220, 0.5) !important;
    border-radius: 12px !important;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08) !important;
    margin-top: 14px !important;
    padding: 8px 12px !important;
}
div[data-testid="stChatInputContainer"] textarea {
    border: none !important;
    outline: none !important;
    background: transparent !important;
    font-size: 15px !important;
    color: #2d3748 !important;
    resize: none !important;
}

/* Disable blur for status/notifications */
[data-testid="stStatusWidget"], 
[data-testid="stNotification"], 
[class*="blockContainer"] {
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}
</style>

            
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<h1 class='main-title'>🏠 AI BASED REAL ESTATE EVALUATION SYSTEM</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>✨ Accurate valuations, market insights, and property intelligence in one platform</p>", unsafe_allow_html=True)

# ---------- CACHED DATA & MODEL ----------
@st.cache_data
def load_historical_data():
    return pd.read_csv('india_housing_prices.csv')

@st.cache_resource
def load_model_and_encoders():
    with open('model_xgb.pkl', 'rb') as f_model, \
         open('encoder.pkl', 'rb') as f_enc, \
         open('dv.pkl', 'rb') as f_dv:
        model = pickle.load(f_model)
        model.set_params(device='cpu')
        ordinal_encoder = pickle.load(f_enc)
        dv = pickle.load(f_dv)
    return model, ordinal_encoder, dv

df_historical = load_historical_data()
states_list = sorted(df_historical['State'].dropna().unique())
model, ordinal_encoder, dv = load_model_and_encoders()

ordinal_columns = ['Property_Type', 'Furnished_Status', 'Public_Transport_Accessibility', 'Facing', 'Security']

def preprocess_input(input_df):
    # Fill missing/None values in ordinal columns to avoid encoding error
    for col in ordinal_columns:
        if col in input_df.columns:
            input_df[col] = input_df[col].fillna('unknown').astype(str).str.lower()
    # Fill missing in object dtype columns as well
    for col in input_df.select_dtypes(include='object').columns:
        input_df[col] = input_df[col].fillna('unknown').str.lower()
    input_dict = input_df.to_dict(orient='records')
    X = dv.transform(input_dict)
    return X

def predict_house_price(input_data):
    furnished_status = str(input_data["Furnished_Status"].iloc[0]).lower()
    bhk = int(input_data["BHK"].iloc[0])
    age_of_property = int(input_data["Age_of_Property"].iloc[0])
    X = preprocess_input(input_data.copy())
    preds = model.predict(X)
    if furnished_status == "furnished":
        preds = preds * 1.10
    elif furnished_status == "semi-furnished":
        preds = preds * 1.08
    if bhk > 2:
        preds = preds * (1 + 0.05 * (bhk - 2))
    age_discount = min(0.005 * age_of_property, 0.30)
    preds = preds * (1 - age_discount)
    return preds

# ---------- CHATBOT/LLM ----------
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=HF_API_TOKEN)
def query_mistral(messages):
    chat_input = [{"role": m["role"], "content": m["content"]} for m in messages]
    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=chat_input,
        max_tokens=512,
        temperature=0.7
    )
    try:
        if response and hasattr(response, "choices") and len(response.choices) > 0:
            msg = response.choices[0].message
            return msg["content"] if isinstance(msg, dict) else msg.content
    except Exception:
        pass
    return "❌ Sorry, I couldn't generate a response."

def get_comparables(input_data, df, size_tolerance=200, year_tolerance=5):
    city = input_data['City'].iloc[0].lower()
    property_type = input_data['Property_Type'].iloc[0].lower()
    size = input_data['Size_in_SqFt'].iloc[0]
    year = input_data['Year_Built'].iloc[0]
    comparables = df[
        (df['City'].str.lower() == city) &
        (df['Property_Type'].str.lower() == property_type) &
        (df['Size_in_SqFt'].between(size - size_tolerance, size + size_tolerance)) &
        (df['Year_Built'].between(year - year_tolerance, year + year_tolerance))
    ].copy()
    return comparables

# ---------- TABS ----------
tab_chat, tab1, tab2, tab3, tab_trend, tab_3d = st.tabs([
    "💬 Chatbot",
    "📍 Location",
    "🏢 Property",
    "✨ Amenities",
    "📈 Price Trend",
    "🏠 3D Model"
])


# ---------- CHATBOT PAGE (glass bubbles) ----------
with tab_chat:
    st.markdown("<div class='glass-card' style='padding:22px 18px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:27px;font-weight:800;letter-spacing:1px; \
                background:linear-gradient(92deg,#805ad5 13%,#00DFD8 85%); \
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>\
                💬 Real Estate Chatbot With Voice Assistance</div>", unsafe_allow_html=True)
    st.markdown("<hr style='opacity:.11;margin-top:3px;'>", unsafe_allow_html=True)
    st.subheader("Ask anything about house prices, property, or CMA analysis!")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "👋 Hello! Ask me anything about properties, prices, or market trends."}]
    
    chat_container = st.container()
    
    # Voice input: capture once when button pressed
        # Voice input: capture once when button pressed
    voice_query = speech_to_text(language='en', start_prompt="🎤 Speak your Query", stop_prompt="⏹",
                                just_once=True, use_container_width=True, key='voice_query')
    if voice_query:
        # st.success(f"Transcribed: {voice_query}")  # Comment this line out to hide transcription
        st.session_state.chat_history.append({"role": "user", "content": voice_query})
        bot_response = query_mistral(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

    
    # Typed input
    user_input = st.chat_input("Type your message here…")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        bot_response = query_mistral(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
    
    # Render full chat history
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                st.markdown(msg["content"])
    
    st.markdown("</div>", unsafe_allow_html=True)
# ---------- LOCATION TAB ----------
# ---- LOCATION TAB (with blank initial values) ----
with tab1:
    st.markdown("""
    <div class='glass-card' style="padding:28px 14px 26px 14px;margin-bottom:10px;">
        <div style="font-size:28px;font-weight:800;letter-spacing:1px;
                    background:linear-gradient(95deg,#805ad5 10%,#00DFD8 80%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            📍 Location Details
        </div>
        <hr style="opacity:.10;margin-top:3px;">
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        # Add placeholder
        states_options = ["Select State"] + states_list
        state = st.selectbox("Select State", states_options, index=0, key="state")
        st.markdown(f"<div style='font-size:15px;margin-top:7px;color:#666;background:rgba(220,220,255,0.11);padding:4px 11px;border-radius:8px;display:inline-block;'>Selected: <b>{state if state != 'Select State' else 'None'}</b></div></div>", unsafe_allow_html=True)

    with c2:
        # Only show cities if state is selected
        if state != "Select State":
            cities_list = sorted(df_historical[df_historical['State'] == state]['City'].dropna().unique())
            cities_options = ["Select City"] + list(cities_list)
        else:
            cities_options = ["Select City"]
        city = st.selectbox("Select City", cities_options, index=0, key="city")
        st.markdown(f"<div style='font-size:15px;margin-top:7px;color:#666;background:rgba(197,237,255,0.13);padding:4px 11px;border-radius:8px;display:inline-block;'>Selected: <b>{city if city != 'Select City' else 'None'}</b></div></div>", unsafe_allow_html=True)

    with c3:
        property_options = ["Select Type", 'Apartment', 'Villa', 'Independent House']
        property_type = st.selectbox("Type", property_options, index=0, key="ptype")
        facing = st.selectbox("Facing", ['East', 'West', 'North', 'South', 'Other'], key="facing")
        owner_type = st.radio("Owner", ['Owner', 'Builder', 'Broker'], horizontal=True, key="owner")
        st.markdown(f"<div style='font-size:15px;margin-top:7px;color:#666;background:rgba(220,220,255,0.11);padding:4px 11px;border-radius:8px;display:inline-block;'>{property_type if property_type != 'Select Type' else 'Not selected'} | {facing} | {owner_type}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)



# ---------- PROPERTY TAB ----------
with tab2:
    st.markdown("""
    <div class='glass-card' style="padding:22px 8px 18px 8px;margin-bottom:5px;">
        <div style="font-size:27px;font-weight:800;letter-spacing:1px;
                    background:linear-gradient(92deg,#805ad5 9%,#00DFD8 85%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🏢 Property Specifications
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    # BHK, Size, Year (Compact glass card)
    with c1:
        st.markdown("""
        <div style="background:rgba(154,134,255,0.11);border-radius:14px;
                    box-shadow:0 2px 12px #805ad521;padding:10px 10px 8px 12px;margin-bottom:2px;">
            <span style="font-size:20px;font-weight:700;color:#825bdb;">🛏 BHK & Size</span>
        """, unsafe_allow_html=True)
        bhk = st.slider("Bedrooms (BHK)", 1, 6, 3, key="bhk")
        size_in_sqft = st.number_input("Size (SqFt)", min_value=500, max_value=5000, value=1200, key="size")
        year_built = st.number_input("Year Built", 1990, 2025, 2015, key="year_built")
        st.markdown(f"<div style='margin-top:7px;font-size:14px;color:#555;background:rgba(220,220,255,0.13);padding:4px 11px;border-radius:8px;display:inline-block;'>BHK: <b>{bhk}</b> | Size: <b>{size_in_sqft}</b> | Year: <b>{year_built}</b></div></div>", unsafe_allow_html=True)

    # Price/SqFt, Floor No, Total Floors (Glass card)
    with c2:
        st.markdown("""
        <div style="background:rgba(23,164,255,0.09);border-radius:14px;
                    box-shadow:0 2px 10px #5ad5bb22;padding:10px 10px 8px 12px;margin-bottom:2px;">
            <span style="font-size:20px;font-weight:700;color:#197cb6;">💰 Price/Floor</span>
        """, unsafe_allow_html=True)
        price_per_sqft = st.number_input("Price per SqFt (Lakhs)", min_value=0.01, max_value=2.0, value=0.05, format="%.3f", key="pps")
        floor_no = st.number_input("Floor Number", 0, 30, 1, key="floor_no")
        total_floors = st.number_input("Total Floors", 1, 50, 5, key="total_floors")
        st.markdown(f"<div style='margin-top:7px;font-size:14px;color:#555;background:rgba(197,237,255,0.13);padding:4px 11px;border-radius:8px;display:inline-block;'>📈 Price/SqFt: <b>{price_per_sqft}</b> | Floor: <b>{floor_no}/{total_floors}</b></div></div>", unsafe_allow_html=True)

    # Age/Availability (Glass card)
    with c3:
        st.markdown("""
        <div style="background:rgba(128,90,213,0.09);border-radius:14px;
                    box-shadow:0 2px 10px #805ad519;padding:10px 10px 8px 12px;margin-bottom:2px;">
            <span style="font-size:20px;font-weight:700;color:#705ad5;">⏳ Availability</span>
        """, unsafe_allow_html=True)
        age_of_property = st.number_input("Age of Property (Years)", 0, 100, 5, key="age")
        availability_status = st.selectbox("Availability", ['Ready_to_Move', 'Under_Construction'], key="availability")
        st.markdown(f"<div style='margin-top:7px;font-size:14px;color:#555;background:rgba(220,220,255,0.11);padding:4px 11px;border-radius:8px;display:inline-block;'>Age: <b>{age_of_property}</b> yrs | Status: <b>{availability_status}</b></div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    


# ---------- AMENITIES TAB ----------
with tab3:
    st.markdown("""
    <div class='glass-card' style="padding:22px 8px 18px 8px;margin-bottom:5px;">
        <div style="font-size:27px;font-weight:800;letter-spacing:1px;
                    background:linear-gradient(92deg,#805ad5 9%,#00DFD8 85%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
             Amenities & Nearby Facilities
        </div>
    """, unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    # Nearby Schools/Hospitals
    with a1:
        st.markdown("""
        <div style="background:rgba(154,134,255,0.11);border-radius:14px;
                    box-shadow:0 2px 12px #805ad521;padding:10px 10px 8px 12px;margin-bottom:2px;">
            <span style="font-size:19px;font-weight:700;color:#825bdb;">🏫 Schools & Hospitals</span>
        """, unsafe_allow_html=True)
        nearby_schools = st.number_input("Nearby Schools", 0, 10, 3, key="schools")
        nearby_hospitals = st.number_input("Nearby Hospitals", 0, 10, 2, key="hospitals")
        st.markdown(f"<div style='margin-top:7px;font-size:14px;color:#555;background:rgba(220,220,255,0.13);padding:4px 11px;border-radius:8px;display:inline-block;'>Schools: <b>{nearby_schools}</b> | Hospitals: <b>{nearby_hospitals}</b></div></div>", unsafe_allow_html=True)

    # Transport & Parking
    with a2:
        st.markdown("""
        <div style="background:rgba(23,164,255,0.09);border-radius:14px;
                    box-shadow:0 2px 10px #5ad5bb22;padding:10px 10px 8px 12px;margin-bottom:2px;">
            <span style="font-size:19px;font-weight:700;color:#197cb6;">🚗 Transport & Parking</span>
        """, unsafe_allow_html=True)
        public_transport = st.selectbox("Public Transport Access", ['High', 'Medium', 'Low'], key="pta")
        parking_space = st.radio("Parking Available?", ['Yes', 'No'], horizontal=True, key="parking")
        st.markdown(f"<div style='margin-top:7px;font-size:14px;color:#555;background:rgba(197,237,255,0.13);padding:4px 11px;border-radius:8px;display:inline-block;'>Transit: <b>{public_transport}</b> | Parking: <b>{parking_space}</b></div></div>", unsafe_allow_html=True)

    # Security & Furnishing
    with a3:
        st.markdown("""
        <div style="background:rgba(128,90,213,0.09);border-radius:14px;
                    box-shadow:0 2px 10px #805ad519;padding:10px 10px 8px 12px;margin-bottom:2px;">
            <span style="font-size:19px;font-weight:700;color:#705ad5;">🔒 Security & Furnishing</span>
        """, unsafe_allow_html=True)
        security = st.radio("Security Available?", ['Yes', 'No'], horizontal=True, key="security")
        furnished_status = st.selectbox("Furnished Status", ['Unfurnished', 'Semi-furnished', 'Furnished'], key="furnished")
        st.markdown(f"<div style='margin-top:7px;font-size:14px;color:#555;background:rgba(220,220,255,0.11);padding:4px 11px;border-radius:8px;display:inline-block;'>Security: <b>{security}</b> | Status: <b>{furnished_status}</b></div></div>", unsafe_allow_html=True)

    # Other amenities (full card)
    st.markdown("""
    <div style="margin-top:12px;background:rgba(208,222,255,0.19);border-radius:14px;
                box-shadow:0 1px 7px #aad5ff21;padding:10px 10px 7px 14px;">
        <span style="font-size:18px;font-weight:700;color:#3b519c;">🏊 Other Amenities</span>
    """, unsafe_allow_html=True)
    amenities = st.text_area("Amenities", placeholder="e.g., Gym, Swimming Pool, Clubhouse", key="amenities")
    st.markdown(f"<div style='margin-top:4px;font-size:14px;color:#555;background:rgba(242,247,255,0.13);padding:4px 11px;border-radius:8px;display:inline-block;'>{amenities if amenities else 'None listed.'}</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- PRICE TREND TAB (ARIMA Visualization) ----------

# ARIMA price trend section for Streamlit


def auto_diff_order(series, max_diff=3, significance_level=0.05):
    diff_series = series.copy()
    for d in range(max_diff + 1):
        adf_result = adfuller(diff_series.dropna())
        p_value = adf_result[1]
        st.write(f"Differencing order = {d}, ADF p-value = {p_value:.4f}")
        if p_value < significance_level:
            return d
        diff_series = diff_series.diff().dropna()
    return max_diff

with tab_trend:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:27px;font-weight:800;letter-spacing:1px; "
        "background:linear-gradient(92deg,#805ad5 13%,#00DFD8 85%); "
        "-webkit-background-clip: text; -webkit-text-fill-color: transparent;'>"
        "📈 Price Trend Visualization (ARIMA)</div>", unsafe_allow_html=True)
    st.markdown("<hr style='opacity:.13;margin-top:3px;'>", unsafe_allow_html=True)
    st.subheader(f"Historical price & ARIMA forecast for {city}, {property_type}")

    area_df = df_historical[
        (df_historical["City"] == city) &
        (df_historical["Property_Type"] == property_type)
    ]
    if area_df.empty:
        st.warning("No historical price data is available for this selection.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        trend_df = area_df.groupby("Year_Built")["Price_in_Lakhs"].mean().reset_index().sort_values("Year_Built")
        trend_df = trend_df.set_index("Year_Built")

        median_dataset_price_per_sqft = df_historical['Price_per_SqFt'].median()
        scale_factor = price_per_sqft / median_dataset_price_per_sqft if median_dataset_price_per_sqft else 1.0
        trend_df['Price_in_Lakhs'] = trend_df['Price_in_Lakhs'] * scale_factor
        prices = trend_df['Price_in_Lakhs']

        # Determine optimal differencing order
        optimal_d = auto_diff_order(prices)
        st.write(f"Optimal differencing order (d): {optimal_d}")

        # Fit ARIMA with optimal differencing order
        Arima_model = ARIMA(prices, order=(1, optimal_d, 1))
        results = Arima_model.fit()
        last_year = trend_df.index.max()
        future_years = [last_year + i for i in range(1, 5)]  # e.g. [2024, 2025, 2026, 2027]
        forecast = results.forecast(steps=4)

        # Show ONLY 2026 and 2027 (last 2 forecast years)
        display_years = future_years[-2:]       # [2026, 2027]
        display_values = forecast[-2:]

        # Plot historical and forecast (show full lines/points)
        trace_actual = go.Scatter(
            x=trend_df.index, y=prices, mode="lines+markers",
            name="Actual Price", line=dict(color="#6b46c1", width=4)
        )
        trace_forecast = go.Scatter(
            x=future_years, y=forecast,
            mode="lines+markers", name="Forecast (ARIMA)",
            marker=dict(symbol="diamond", size=13, color="#00dfd8"),
            line=dict(color="#00dfd8", dash='dot')
        )
        fig = go.Figure([trace_actual, trace_forecast])
        fig.update_layout(
            title=f"Price Trend & Forecast ({city}, {property_type})",
            xaxis_title="Year", yaxis_title="Avg Price in Lakhs",
            plot_bgcolor='rgba(240,246,255,0.98)',
            paper_bgcolor='rgba(240,246,255,0.0)',
            font=dict(family="Inter, Poppins", size=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display ONLY 2026, 2027 forecasts
        st.markdown(f"""
<div style="margin-top:10px; font-size:17px;">
<b>Forecast (scaled):</b><br>
• <span style="color:#00dfd8;">{display_years[0]}</span>: ₹{display_values.iloc[0]:.2f} Lakhs<br>
• <span style="color:#00dfd8;">{display_years[1]}</span>: ₹{display_values.iloc[1]:.2f} Lakhs
</div>
""", unsafe_allow_html=True)

        # Save ARIMA data for summary on demand
        st.session_state['arima_display_years'] = display_years
        st.session_state['arima_display_values'] = display_values
        st.session_state['arima_city'] = city
        st.session_state['arima_property_type'] = property_type

        # Button to show ARIMA summary (calls API only when pressed)
        if st.button("📈 Show ARIMA Insight"):
            display_years = st.session_state['arima_display_years']
            display_values = st.session_state['arima_display_values']
            city = st.session_state['arima_city']
            property_type = st.session_state['arima_property_type']

            forecast_summary = ", ".join([
                f"{yr}: ₹{display_values.iloc[idx]:.2f}"
                for idx, yr in enumerate(display_years)
            ])
            arima_prompt = (
                f"You are a real estate analyst. Summarize the ARIMA model results: "
                f"For {city}, {property_type}, the price trend is forecast at {forecast_summary} (Lakhs). "
                f"Say if the price trend looks stable, rising, or falling, "
                f"and mention one key factor that could affect future trends (e.g., demand, supply, policy)."
            )
            with st.spinner("📝 Generating ARIMA summary..."):
                arima_summary = query_mistral([{"role": "user", "content": arima_prompt}])
                if not arima_summary or arima_summary.strip().startswith("❌"):
                    arima_summary = "ℹ The ARIMA summary couldn't be generated right now. Try again later."

            st.markdown(f"""
<div class="cma-summary" style="margin-top:14px;">
    <span style='font-size:22px;font-weight:700;'>📈 ARIMA Price Trend Insight</span><br>
    • <b>Forecast years:</b> {', '.join(map(str, display_years))}<br>
    • <b>Predicted prices (scaled):</b> {', '.join([f"₹{display_values.iloc[idx]:.2f}" for idx in range(len(display_years))])} Lakhs<br><br>
    <b>Summary:</b> {str(arima_summary)}<br>
    <span style="color:#f3d;">⚠</span> <i>This trend is based on statistical forecasting and may not account for sudden market changes.</i>
</div>
""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

if st.button("🔮 Predict Price"):
    # Input validation and price prediction same as before
    if state == "Select State" or city == "Select City" or property_type == "Select Type":
        st.warning("⚠ Please select valid State, City, and Property Type before predicting.")
    else:
        input_data = pd.DataFrame([{
            "State": state,
            "City": city,
            "Property_Type": property_type,
            "BHK": bhk,
            "Size_in_SqFt": size_in_sqft,
            "Price_per_SqFt": price_per_sqft,
            "Year_Built": year_built,
            "Floor_No": floor_no,
            "Total_Floors": total_floors,
            "Age_of_Property": age_of_property,
            "Nearby_Schools": nearby_schools,
            "Nearby_Hospitals": nearby_hospitals,
            "Public_Transport_Accessibility": public_transport,
            "Parking_Space": parking_space,
            "Security": security,
            "Amenities": amenities,
            "Facing": facing,
            "Owner_Type": owner_type,
            "Availability_Status": availability_status,
            "Furnished_Status": furnished_status
        }])

        price_prediction = predict_house_price(input_data)[0]
        st.markdown(f"<div class='prediction-box'>💰 Estimated Price: ₹{price_prediction:.2f} Lakhs</div>", unsafe_allow_html=True)
        st.markdown("<h2 style='font-size:2.3rem;margin:14px 0 10px 0;color:#805ad5;font-weight:900;letter-spacing:1px;'>📊 Comparative Market Analysis</h2>", unsafe_allow_html=True)

        comparables = get_comparables(input_data, df_historical)
        if not comparables.empty:
            median_dataset_price_per_sqft = df_historical['Price_per_SqFt'].median()
            scale_factor = price_per_sqft / median_dataset_price_per_sqft if median_dataset_price_per_sqft else 1.0
            if 'Price_in_Lakhs' in comparables.columns:
                comparables['Price_in_Lakhs'] = comparables['Price_in_Lakhs'] * scale_factor
            else:
                if 'Price_per_SqFt' in comparables.columns and 'Size_in_SqFt' in comparables.columns:
                    comparables['Price_in_Lakhs'] = (comparables['Price_per_SqFt'] * comparables['Size_in_SqFt']) * scale_factor

            avg_price = float(comparables['Price_in_Lakhs'].mean())
            median_price = float(comparables['Price_in_Lakhs'].median())

            # Show comparables and charts as before (without summary)
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Similar Properties", len(comparables))
            m2.metric("Avg Price (scaled)", f"₹{avg_price:.2f} Lakhs")
            m3.metric("Median Price (scaled)", f"₹{median_price:.2f} Lakhs")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.dataframe(
                comparables[['City', 'Property_Type', 'BHK', 'Size_in_SqFt', 'Year_Built', 'Price_in_Lakhs']]
                .sort_values('Price_in_Lakhs'),
                use_container_width=True,
                height=330
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # Plots (histogram and scatter) as before...

            # Save CMA data for summary button
            st.session_state['cma_comparables'] = comparables
            st.session_state['cma_price_prediction'] = price_prediction
            st.session_state['cma_avg_price'] = avg_price
            st.session_state['cma_median_price'] = median_price
            st.session_state['cma_city'] = city
        else:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.warning("⚠ No comparable properties found for CMA. Try adjusting size/year tolerances or select another city.")
            st.markdown("</div>", unsafe_allow_html=True)

# New button to show CMA Summary, hits the API only on demand
if 'cma_comparables' in st.session_state and st.button("📊 Show CMA Summary"):
    comparables = st.session_state['cma_comparables']
    price_prediction = st.session_state['cma_price_prediction']
    avg_price = st.session_state['cma_avg_price']
    median_price = st.session_state['cma_median_price']
    city = st.session_state['cma_city']

    prompt = (
        f"You are a real estate analyst. Summarize the CMA results: "
        f"{len(comparables)} comparables in {city}. "
        f"Average price: ₹{avg_price:.2f} Lakhs. Median price: ₹{median_price:.2f} Lakhs. "
        f"Predicted price: ₹{price_prediction:.2f} Lakhs. "
        f"Say if the property looks overpriced, underpriced, or fair, "
        f"and mention one main factor (size/year/furnishing)."
    )
    with st.spinner("📝 Generating CMA summary..."):
        cma_summary = query_mistral([{"role": "user", "content": prompt}])
        if not cma_summary or cma_summary.strip().startswith("❌"):
            cma_summary = "ℹ The CMA summary couldn't be generated right now. Try again later."

    st.markdown(f"""
        <div class="cma-summary">
            <span style='font-size:22px;font-weight:700;'>📊 CMA Market Insight</span><br>
            • <b>Comparables:</b> {len(comparables)} in <b>{city.title()}</b><br>
            • <b>Average price:</b> ₹{avg_price:.2f} Lakhs &nbsp;|&nbsp; <b>Median price:</b> ₹{median_price:.2f} Lakhs<br>
            • <b>Model predicted price:</b> ₹{price_prediction:.2f} Lakhs<br><br>
            <b>Summary:</b> {cma_summary}<br>
            <span style="color:#f3d;">⚠</span> <i>This insight is based on price only. For full valuation, consult a certified professional.</i>
        </div>
    """, unsafe_allow_html=True)

elif 'cma_comparables' not in st.session_state:
    # Optionally, inform user to predict price first to get CMA summary access
    st.info("ℹ Please predict the price first to enable the CMA summary.")

            

# ---------- 3D MODEL TAB ONLY ----------
def create_3d_floor_plan(bhk, sqft):
    avg_room_sqft = sqft / (bhk + 2)  # +2 for living + kitchen
    base_dim = max(int(avg_room_sqft ** 0.5), 5)
    room_dims = []
    for _ in range(bhk + 2):
        length = base_dim + random.randint(-2, 3)
        width = base_dim + random.randint(-2, 3)
        room_dims.append((max(length, 3), max(width, 3)))
    height = 10  # Fixed ceiling height
    x_offsets = [0]
    y_offsets = [0]
    half = (bhk + 2) // 2
    for i in range(1, bhk + 2):
        if i <= half:
            x_offsets.append(x_offsets[-1] + room_dims[i - 1][0] + 1)
            y_offsets.append(0)
        else:
            x_offsets.append(0)
            y_offsets.append(y_offsets[-1] + room_dims[i - 1][1] + 1)
    fig = go.Figure()
    for i in range(bhk + 2):
        x0, y0 = x_offsets[i], y_offsets[i]
        length, width = room_dims[i]
        vertices = [
            (x0, y0, 0),
            (x0 + length, y0, 0),
            (x0 + length, y0 + width, 0),
            (x0, y0 + width, 0),
            (x0, y0, height),
            (x0 + length, y0, height),
            (x0 + length, y0 + width, height),
            (x0, y0 + width, height),
        ]
        # Add room mesh
        fig.add_trace(
            go.Mesh3d(
                x=[v[0] for v in vertices],
                y=[v[1] for v in vertices],
                z=[v[2] for v in vertices],
                color="lightblue" if i == 0 else "lightgreen",
                opacity=0.55,
                alphahull=0,
            )
        )
        # Add room label
        label = "Living Area" if i == 0 else ("Kitchen" if i == bhk + 1 else f"Bedroom {i}")
        fig.add_trace(
            go.Scatter3d(
                x=[x0 + length / 2],
                y=[y0 + width / 2],
                z=[height + 1],
                mode="text",
                text=[label],
                textposition="top center",
                textfont=dict(size=14, color="navy"),
            )
        )
    fig.update_layout(
        width=800,
        height=600,
        scene=dict(
            xaxis=dict(title="Feet", backgroundcolor="rgb(230, 230,230)"),
            yaxis=dict(title="Feet", backgroundcolor="rgb(230, 230,230)"),
            zaxis=dict(title="Height (ft)", backgroundcolor="rgb(230, 230,230)"),
            aspectratio=dict(x=2, y=1.5, z=0.6),
            camera=dict(eye=dict(x=1.5, y=2, z=0.8)),
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


with tab_3d:
    st.title("Advanced 3D Block Model by BHK & SqFt")
    bhk_3d = st.slider("Bedrooms (BHK)", 1, 6, 3, key="bhk_3d")
    sqft_3d = st.number_input("Total Size (SqFt)", min_value=300, max_value=8000, value=1500, key="sqft_3d")
    if st.button("Generate 3D Model"):
        fig_3d = create_3d_floor_plan(bhk_3d, sqft_3d)
        st.plotly_chart(fig_3d, width='stretch')
    else:
        st.info("Adjust parameters and click 'Generate 3D Model' to view.")