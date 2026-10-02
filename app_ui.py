import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random
import time

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(
    page_title="PropAI - AI-Powered Real Estate Evaluation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------
# Initialize session state
# ----------------------
if "predicted_price" not in st.session_state:
    st.session_state.predicted_price = None
if "property_data" not in st.session_state:
    st.session_state.property_data = {}
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Hello! I'm your AI real estate assistant. I can help you with property valuations, market analysis, investment advice, and answer any questions about real estate. How can I assist you today?"}
    ]

# ----------------------
# Custom CSS
# ----------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp { font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%); }
#MainMenu, footer, header {visibility: hidden;}
.glass-card {background: rgba(255,255,255,0.03); backdrop-filter: blur(16px); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 2rem; margin:1rem 0; box-shadow:0 8px 32px rgba(0,0,0,0.3);}
.glass-card:hover {background: rgba(255,255,255,0.05); border-color: rgba(255,255,255,0.12); transform: translateY(-2px); transition: all 0.3s ease;}
.gradient-text {background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;}
.nav-container {background: rgba(0,0,0,0.85); backdrop-filter: blur(20px); border-bottom: 1px solid rgba(255,255,255,0.08); padding:1rem 2rem; margin:-1rem -1rem 2rem -1rem; box-shadow:0 4px 16px rgba(0,0,0,0.2);}
.nav-content {display:flex; justify-content:space-between; align-items:center; max-width:1200px; margin:0 auto;}
.logo {display:flex; align-items:center; gap:0.5rem; font-size:1.5rem; font-weight:700; background:linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;}
.chat-container {max-height:400px; overflow-y:auto; padding:1rem; background: rgba(255,255,255,0.02); border-radius:8px; margin:1rem 0;}
.chat-message {margin:1rem 0; padding:1rem; border-radius:8px;}
.chat-user {background: linear-gradient(135deg, #3b82f6 0%, rgba(59,130,246,0.8) 100%); margin-left:2rem; color:white;}
.chat-assistant {background: rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); margin-right:2rem; color:#e2e8f0;}
.prediction-result {background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%); border-radius:12px; padding:2rem; text-align:center; color:white; font-size:1.5rem; font-weight:600; box-shadow:0 8px 32px rgba(59,130,246,0.3);}
</style>
""", unsafe_allow_html=True)

# ----------------------
# Navigation
# ----------------------
def render_navigation():
    st.markdown("""
    <div class="nav-container">
        <div class="nav-content">
            <div class="logo">🏠 PropAI</div>
            <div style="color: #94a3b8; font-size: 0.9rem;">AI-Powered Real Estate Evaluation Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# Hero Section
# ----------------------
def render_hero():
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">The most powerful platform for <span class="gradient-text">AI-driven real estate</span> evaluation</h1>
        <p class="hero-subtitle">Get accurate property valuations, market insights, and investment analysis powered by advanced machine learning models.</p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# Property Form
# ----------------------
def render_property_form():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🏠 Property Details")
    
    col1, col2 = st.columns(2)
    with col1:
        state = st.selectbox("State", ["California", "New York", "Texas", "Florida", "Illinois"])
        city = st.selectbox("City", ["Los Angeles", "San Francisco", "New York", "Miami", "Chicago"])
        property_type = st.selectbox("Property Type", ["Apartment", "House", "Condo", "Townhouse"])
        bhk = st.number_input("BHK", min_value=1, max_value=10, value=3)
        size_sqft = st.number_input("Size (sq ft)", min_value=100, max_value=10000, value=1200)
    with col2:
        year_built = st.number_input("Year Built", min_value=1900, max_value=2024, value=2015)
        floor_no = st.number_input("Floor Number", min_value=1, max_value=50, value=1)
        total_floors = st.number_input("Total Floors", min_value=1, max_value=50, value=5)
        parking = st.selectbox("Parking Space", ["Yes", "No"])
        furnished = st.selectbox("Furnished Status", ["Furnished", "Semi-Furnished", "Unfurnished"])
    
    st.markdown("### 🌍 Location & Amenities")
    col3, col4 = st.columns(2)
    with col3:
        schools = st.number_input("Nearby Schools", min_value=0, max_value=20, value=3)
        hospitals = st.number_input("Nearby Hospitals", min_value=0, max_value=20, value=2)
        transport = st.selectbox("Public Transport", ["High", "Medium", "Low"])
    with col4:
        security = st.selectbox("Security", ["Yes", "No"])
        facing = st.selectbox("Facing Direction", ["North", "South", "East", "West"])
        owner_type = st.selectbox("Owner Type", ["Owner", "Dealer", "Builder"])
    
    amenities = st.text_area("Additional Amenities", placeholder="Swimming pool, gym, garden...")

    if st.button("🔮 Predict Price"):
        with st.spinner("🤖 AI is analyzing your property..."):
            time.sleep(1.5)
            base_price = size_sqft * random.uniform(150, 300)
            location_multiplier = {"California":1.5, "New York":1.4, "Texas":1.0, "Florida":1.2, "Illinois":1.1}[state]
            bhk_multiplier = 1 + (bhk-1)*0.15
            age_factor = max(0.7, 1 - (2024 - year_built)*0.01)
            predicted_price = base_price * location_multiplier * bhk_multiplier * age_factor
            st.session_state.predicted_price = predicted_price
            st.session_state.property_data = {'state':state,'city':city,'property_type':property_type,'bhk':bhk,'size_sqft':size_sqft,'year_built':year_built}

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------
# Price Prediction
# ----------------------
def render_price_prediction():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 💰 AI Price Prediction")
    if st.session_state.predicted_price:
        st.markdown(f"""
        <div class="prediction-result">
            <div style="font-size:2rem; margin-bottom:0.5rem;">💎</div>
            <div>Estimated Property Value</div>
            <div style="font-size:2.5rem; font-weight:700; margin:1rem 0;">${st.session_state.predicted_price:,.0f}</div>
            <div style="font-size:1rem; opacity:0.9;">Based on AI analysis of market data</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("🔮 Fill out the property form and click 'Predict Price' to see AI-powered valuation")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------
# Chat Interface
# ----------------------
def render_chat_interface():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI Real Estate Assistant")

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_messages:
            if msg["role"]=="user":
                st.markdown(f'<div class="chat-message chat-user"><strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message chat-assistant"><strong>🤖 AI Assistant:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input using form (avoids rerun issues)
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask me anything about real estate...")
        submit = st.form_submit_button("Send")
        if submit and user_input:
            st.session_state.chat_messages.append({"role":"user","content":user_input})
            ai_responses = [
                "Based on current market trends, I'd recommend considering the location's growth potential.",
                "The property valuation looks reasonable. Have you considered long-term appreciation?",
                "I'd suggest analyzing the rental yield and comparing it with similar properties.",
                "This area has strong fundamentals. Proximity to schools and transport adds value.",
                "Consider upcoming infrastructure projects, zoning changes, and demographic trends."
            ]
            st.session_state.chat_messages.append({"role":"assistant","content":random.choice(ai_responses)})
            st.experimental_rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------
# Main
# ----------------------
def main():
    render_navigation()
    render_hero()
    
    tab1, tab2 = st.tabs(["🏠 Property Evaluation","🤖 AI Assistant"])
    with tab1:
        col1, col2 = st.columns([2,1])
        with col1: render_property_form()
        with col2: render_price_prediction()
    
    with tab2:
        render_chat_interface()

if __name__=="__main__":
    main()
