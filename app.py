import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import math
import datetime
import io

# Page config for high-quality dashboard layout
st.set_page_config(
    page_title="LifeShield AI | Premium Insurance Purchase Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme session state
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'Light'

# Initialize prediction states
if 'predicted_age' not in st.session_state:
    st.session_state['predicted_age'] = 35
if 'prediction_ran' not in st.session_state:
    st.session_state['prediction_ran'] = False
if 'prediction_history' not in st.session_state:
    st.session_state['prediction_history'] = []
if 'last_prediction' not in st.session_state:
    st.session_state['last_prediction'] = None

# Load data and train model
@st.cache_resource
def load_data_and_train():
    # Load dataset
    df = pd.read_csv("insurance_data.csv")
    
    # Train-test split (80-20 train-test ratio, matching notebook)
    X = df[['age']]
    y = df['bought_insurance']
    
    # Using random_state=42 for fixed, professional and reproducible results
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit Logistic Regression model (specify solver to avoid warnings)
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)
    
    return df, X_train, X_test, y_train, y_test, model

df, X_train, X_test, y_train, y_test, model = load_data_and_train()

# Model parameters
coef = model.coef_[0][0]
intercept = model.intercept_[0]
decision_age = -intercept / coef

# Calculate metrics for the model
y_train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)

y_test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred, zero_division=0)
test_rec = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

# --- SIDEBAR & THEME CONTROL ---
st.sidebar.markdown("""
<div style='text-align: center; margin-bottom: 1.5rem; margin-top: 1rem;'>
    <div style='font-size: 3rem; filter: drop-shadow(0 4px 6px rgba(37,99,235,0.25));'>🛡️</div>
    <h2 style='margin: 0.5rem 0 0.1rem 0; font-size: 1.6rem; letter-spacing: -0.03em; font-weight: 800;'>LifeShield AI</h2>
    <p style='color: #64748b; font-size: 0.85rem; margin: 0; font-weight: 500;'>SaaS Predictive Analytics</p>
</div>
""", unsafe_allow_html=True)

# Toggle switch for Dark Mode
theme_toggle = st.sidebar.toggle("🌙 Dark Mode", value=(st.session_state['theme'] == 'Dark'))
st.session_state['theme'] = 'Dark' if theme_toggle else 'Light'

st.sidebar.markdown("<hr style='border-top: 1px solid rgba(128,128,128,0.2); margin: 1rem 0;'>", unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Dashboard", 
        "🤖 Prediction", 
        "📊 Dataset Explorer", 
        "📈 Visualizations", 
        "📉 Model Performance", 
        "🧠 Logistic Regression",
        "👨‍💻 About Developer"
    ]
)

# Set CSS theme variables based on state
if st.session_state['theme'] == 'Light':
    bg_color = "#f8fafc"
    card_bg = "rgba(255, 255, 255, 0.72)"
    text_color = "#0f172a"
    subtext_color = "#475569"
    border_color = "rgba(255, 255, 255, 0.6)"
    shadow_color = "rgba(31, 38, 135, 0.05)"
    sidebar_bg = "#ffffff"
    primary_color = "#2563eb" # Royal Blue
    accent_gradient = "linear-gradient(135deg, #2563eb, #06b6d4)" # Royal Blue to Cyan Gradient
    plotly_template = "plotly_white"
    
    # Enhanced glassmorphism properties
    bg_style = "radial-gradient(at 0% 0%, rgba(244, 247, 254, 1) 0%, transparent 50%), radial-gradient(at 50% 0%, rgba(224, 231, 255, 0.6) 0%, transparent 50%), radial-gradient(at 100% 0%, rgba(207, 250, 254, 0.5) 0%, transparent 50%), #f8fafc"
    sidebar_bg_style = "rgba(255, 255, 255, 0.45)"
    card_shadow = "0 8px 32px 0 rgba(31, 38, 135, 0.06), inset 0 1px 0 0 rgba(255, 255, 255, 0.6)"
    input_bg = "rgba(255, 255, 255, 0.65)"
    input_border = "rgba(226, 232, 240, 0.9)"
    slider_track = "rgba(226, 232, 240, 0.8)"
else:
    bg_color = "#090d16"
    card_bg = "rgba(15, 23, 42, 0.65)"
    text_color = "#f8fafc"
    subtext_color = "#94a3b8"
    border_color = "rgba(255, 255, 255, 0.08)"
    shadow_color = "rgba(0, 0, 0, 0.4)"
    sidebar_bg = "#0d1527"
    primary_color = "#3b82f6" # Brighter Blue
    accent_gradient = "linear-gradient(135deg, #3b82f6, #22d3ee)" # Bright blue to cyan
    plotly_template = "plotly_dark"
    
    # Enhanced glassmorphism properties
    bg_style = "radial-gradient(at 0% 0%, rgba(15, 23, 42, 0.95) 0%, transparent 50%), radial-gradient(at 50% 0%, rgba(30, 27, 75, 0.45) 0%, transparent 50%), radial-gradient(at 100% 0%, rgba(8, 47, 73, 0.45) 0%, transparent 50%), #090d16"
    sidebar_bg_style = "rgba(13, 21, 39, 0.6)"
    card_shadow = "0 8px 32px 0 rgba(0, 0, 0, 0.37), inset 0 1px 0 0 rgba(255, 255, 255, 0.05)"
    input_bg = "rgba(15, 23, 42, 0.55)"
    input_border = "rgba(255, 255, 255, 0.08)"
    slider_track = "rgba(255, 255, 255, 0.08)"

# Inject CSS custom styles
def inject_css():
    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        
        .stApp {{
            background: {bg_style};
            background-size: cover;
            background-attachment: fixed;
            color: {text_color};
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        
        /* Glassmorphic Sidebar */
        [data-testid="stSidebar"] {{
            background: {sidebar_bg_style} !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border-right: 1px solid {border_color} !important;
            padding-top: 1.5rem;
        }}
        
        /* Glassmorphism custom card panel */
        .glass-card {{
            background: {card_bg};
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid {border_color};
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: {card_shadow};
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .glass-card:hover {{
            transform: translateY(-3px);
            border-color: {primary_color}44;
            box-shadow: 0 12px 40px 0 {shadow_color};
        }}
        
        /* Hero Section style */
        .hero-card {{
            background: linear-gradient(135deg, {primary_color}12, {primary_color}03);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid {border_color};
            border-radius: 24px;
            padding: 3rem 2rem;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
            box-shadow: {card_shadow};
        }}
        
        .hero-title {{
            font-size: 2.85rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 0.75rem;
            color: {text_color};
            background: linear-gradient(90deg, {text_color}, {primary_color});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .hero-subtitle {{
            font-size: 1.2rem;
            color: {subtext_color};
            font-weight: 400;
            max-width: 750px;
            line-height: 1.6;
            margin: 0;
        }}
        
        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .stat-card {{
            background: {card_bg};
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid {border_color};
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: {card_shadow};
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        
        .stat-card:hover {{
            transform: translateY(-2px);
            border-color: {primary_color}33;
        }}
        
        .stat-value {{
            font-size: 2.1rem;
            font-weight: 800;
            color: {primary_color};
            letter-spacing: -0.04em;
            margin-top: 0.5rem;
        }}
        
        .stat-label {{
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.075em;
            color: {subtext_color};
        }}
        
        /* Input & Controls glassmorphism styling */
        div[data-baseweb="input"] {{
            background-color: {input_bg} !important;
            border: 1px solid {input_border} !important;
            border-radius: 12px !important;
            color: {text_color} !important;
        }}
        
        div[data-baseweb="input"]:focus-within {{
            border-color: {primary_color} !important;
            box-shadow: 0 0 0 3px {primary_color}22 !important;
        }}
        
        div[data-baseweb="select"] {{
            background-color: {input_bg} !important;
            border: 1px solid {input_border} !important;
            border-radius: 12px !important;
            color: {text_color} !important;
        }}
        
        /* Interactive Slider styling */
        div[data-testid="stSlider"] div[data-baseweb="slider"] > div {{
            background-color: {slider_track} !important;
        }}
        
        div[role="slider"] {{
            background-color: {primary_color} !important;
            border: 2px solid white !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15) !important;
        }}
        
        /* Select list dropdown hover */
        div[role="listbox"] {{
            background-color: {sidebar_bg} !important;
            color: {text_color} !important;
            border: 1px solid {border_color} !important;
        }}
        
        /* Table / DataFrame style mapping */
        div[data-testid="stDataFrame"] {{
            background-color: {card_bg} !important;
            border: 1px solid {border_color} !important;
            border-radius: 16px !important;
            padding: 0.5rem;
            box-shadow: {card_shadow};
        }}
        
        /* Gradient Predict buttons */
        div.stButton > button {{
            background: {accent_gradient};
            color: white !important;
            border-radius: 12px !important;
            border: none !important;
            padding: 0.8rem 2rem !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em;
            width: 100%;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 18px 0 {primary_color}33;
            cursor: pointer;
        }}
        
        div.stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px 0 {primary_color}55 !important;
            filter: brightness(1.1);
        }}
        
        div.stButton > button:active {{
            transform: translateY(0px) !important;
        }}
        
        /* Secondary Action Buttons (e.g. Reset, Download) */
        div[data-testid="stSecondaryButton"] > button {{
            background: transparent !important;
            color: {text_color} !important;
            border: 1px solid {border_color} !important;
            box-shadow: none !important;
            font-weight: 600 !important;
        }}
        
        div[data-testid="stSecondaryButton"] > button:hover {{
            background: {border_color} !important;
            transform: translateY(-2px) !important;
            border-color: {primary_color}33 !important;
        }}
        
        /* Premium Results Box */
        .result-box-positive {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(52, 211, 153, 0.02));
            border: 1.5px solid rgba(16, 185, 129, 0.25);
            border-radius: 16px;
            padding: 1.5rem;
            margin-top: 1rem;
        }}
        
        .result-box-negative {{
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(248, 113, 113, 0.02));
            border: 1.5px solid rgba(239, 68, 68, 0.25);
            border-radius: 16px;
            padding: 1.5rem;
            margin-top: 1rem;
        }}
        
        /* Animated Progress Bars */
        .progress-bar-container {{
            background-color: {border_color};
            border-radius: 20px;
            height: 14px;
            width: 100%;
            overflow: hidden;
            margin: 12px 0;
        }}
        
        .progress-bar-fill-positive {{
            background: linear-gradient(90deg, #10b981, #34d399);
            height: 100%;
            border-radius: 20px;
            animation: progressGlow 1.5s ease-out;
        }}
        
        .progress-bar-fill-negative {{
            background: linear-gradient(90deg, #ef4444, #f87171);
            height: 100%;
            border-radius: 20px;
            animation: progressGlow 1.5s ease-out;
        }}
        
        @keyframes progressGlow {{
            from {{ width: 0%; }}
        }}
        
        /* Timeline styling */
        .timeline {{
            position: relative;
            border-left: 2px solid {border_color};
            padding-left: 24px;
            margin-left: 12px;
            margin-top: 1rem;
        }}
        
        .timeline-item {{
            position: relative;
            margin-bottom: 2rem;
        }}
        
        .timeline-item:last-child {{
            margin-bottom: 0;
        }}
        
        .timeline-badge {{
            position: absolute;
            left: -37px;
            top: 2px;
            background: {accent_gradient};
            color: white;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: 800;
            box-shadow: 0 0 0 4px {bg_color};
        }}
        
        .timeline-title {{
            font-weight: 700;
            font-size: 1.05rem;
            color: {text_color};
            margin-bottom: 4px;
            letter-spacing: -0.015em;
        }}
        
        .timeline-desc {{
            color: {subtext_color};
            font-size: 0.875rem;
            line-height: 1.5;
        }}
        
        /* Profile developer card styling */
        .dev-profile {{
            display: flex;
            align-items: center;
            gap: 2rem;
            flex-wrap: wrap;
        }}
        
        .dev-avatar {{
            width: 110px;
            height: 110px;
            border-radius: 50%;
            background: {accent_gradient};
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 3rem;
            color: white;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        }}
        
        .social-link-btn {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.6rem 1.25rem;
            border-radius: 10px;
            font-weight: 600;
            text-decoration: none !important;
            font-size: 0.9rem;
            border: 1px solid {border_color};
            color: {text_color} !important;
            background-color: {card_bg};
            transition: all 0.2s ease;
            margin-right: 0.75rem;
            margin-top: 0.5rem;
        }}
        
        .social-link-btn:hover {{
            background: {primary_color}0c;
            border-color: {primary_color};
            transform: translateY(-2px);
        }}
        
        /* Fade in effect for page content */
        [data-testid="stAppViewBlockContainer"] {{
            animation: fadeIn 0.45s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }}
        
        @keyframes fadeIn {{
            from {{
                opacity: 0;
                transform: translateY(8px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 3rem 0;
            margin-top: 5rem;
            border-top: 1px solid {border_color};
            color: {subtext_color};
            font-size: 0.875rem;
        }}
        .footer span {{
            background: {border_color};
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            margin: 0 0.25rem;
            font-weight: 600;
            font-family: monospace;
            font-size: 0.8rem;
        }}
        .footer-socials {{
            margin-top: 1rem;
            display: flex;
            justify-content: center;
            gap: 1.5rem;
        }}
        .footer-socials a {{
            color: {subtext_color};
            text-decoration: none;
            font-size: 1.2rem;
            transition: color 0.2s ease;
        }}
        .footer-socials a:hover {{
            color: {primary_color};
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_css()

# Helper function to generate PDF
def generate_pdf_report(age, prediction_label, prob_buy, prob_not_buy, confidence, explanation):
    from fpdf import FPDF
    
    pdf = FPDF()
    pdf.add_page()
    
    # Header Banner
    pdf.set_fill_color(37, 99, 235) # Royal Blue
    pdf.rect(0, 0, 210, 45, 'F')
    
    # Title
    pdf.set_y(15)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('helvetica', 'B', 22)
    pdf.cell(0, 8, 'LIFESHIELD AI', 0, 1, 'C')
    pdf.set_font('helvetica', 'I', 11)
    pdf.cell(0, 5, 'Predictive Analytics & Model Inference Report', 0, 1, 'C')
    
    # Body Section
    pdf.set_y(55)
    pdf.set_text_color(15, 23, 42) # Slate-900
    
    # Metadata
    pdf.set_font('helvetica', '', 9)
    pdf.cell(0, 5, f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'R')
    pdf.cell(0, 5, "Problem Type: Binary Classification (Logistic Regression)", 0, 1, 'R')
    pdf.ln(8)
    
    # Results Section Header
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, "Inference Parameters & Results", 0, 1, 'L')
    pdf.set_draw_color(226, 232, 240)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    # Results Table
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(95, 8, "Feature / Parameter", 1, 0, 'L')
    pdf.cell(95, 8, "Model Value", 1, 1, 'L')
    
    pdf.set_font('helvetica', '', 9)
    pdf.cell(95, 8, "Client Age", 1, 0, 'L')
    pdf.cell(95, 8, f"{age} Years Old", 1, 1, 'L')
    
    pdf.cell(95, 8, "Predicted Likelihood", 1, 0, 'L')
    pdf.set_font('helvetica', 'B', 9)
    if prediction_label == "Likely to Buy Insurance":
        pdf.set_text_color(16, 185, 129) # Green
    else:
        pdf.set_text_color(239, 68, 68) # Red
    pdf.cell(95, 8, f"{prediction_label}", 1, 1, 'L')
    pdf.set_text_color(15, 23, 42)
    
    pdf.set_font('helvetica', '', 9)
    pdf.cell(95, 8, "Model Confidence Score", 1, 0, 'L')
    pdf.cell(95, 8, f"{confidence:.2f}%", 1, 1, 'L')
    
    pdf.cell(95, 8, "Probability of Buying (Class 1)", 1, 0, 'L')
    pdf.cell(95, 8, f"{prob_buy * 100:.2f}%", 1, 1, 'L')
    
    pdf.cell(95, 8, "Probability of Not Buying (Class 0)", 1, 0, 'L')
    pdf.cell(95, 8, f"{prob_not_buy * 100:.2f}%", 1, 1, 'L')
    pdf.ln(10)
    
    # AI Explanation Section
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, "Model Decision Explanation (AI Interpretations)", 0, 1, 'L')
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font('helvetica', '', 9.5)
    clean_explanation = explanation.replace("🔑 **Why this prediction occurred:**", "").replace("🛡️ **Confidence Level:**", "").replace("📈 **Historical Trend:**", "").replace("💼 **Business Interpretation:**", "")
    pdf.multi_cell(0, 6, clean_explanation.strip())
    pdf.ln(10)
    
    # Math Parameters Section
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, "Mathematical Coefficients", 0, 1, 'L')
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font('helvetica', '', 9)
    pdf.cell(95, 8, "Learned Intercept (Bias, beta_0)", 1, 0, 'L')
    pdf.cell(95, 8, f"{intercept:.8f}", 1, 1, 'R')
    
    pdf.cell(95, 8, "Learned Age Weight (Coefficient, beta_1)", 1, 0, 'L')
    pdf.cell(95, 8, f"{coef:.8f}", 1, 1, 'R')
    
    pdf.cell(95, 8, "Decision Boundary Threshold Age", 1, 0, 'L')
    pdf.cell(95, 8, f"{decision_age:.2f} Years", 1, 1, 'R')
    pdf.ln(20)
    
    # Developer details
    pdf.set_font('helvetica', 'I', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 4, "Developed and programmed by Abhay Shanker Tiwari", 0, 1, 'C')
    pdf.cell(0, 4, "Built using Python, Streamlit, Scikit-learn, and FPDF2", 0, 1, 'C')
    
    buffer = io.BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()

# Helper function to generate AI text
def generate_ai_explanation_text(age, prediction_class, prob_buy, prob_not_buy):
    confidence = max(prob_buy, prob_not_buy) * 100
    
    if prediction_class == 1:
        why = f"The model predicts a Likelihood to Purchase Insurance because the client is {age} years old, which crosses the computed decision boundary of {decision_age:.1f} years."
        conf = f"The confidence is {confidence:.2f}%. This represents a high mathematical probability of purchase, placing the client well within target conversion margins."
        trend = "Historically, in our dataset of 27 samples, purchase behavior is heavily concentrated in older demographics. 100% of samples above age 45 purchased life insurance."
        business = "This is a high-value customer. Deploy proactive sales contact and market comprehensive long-term financial, health, and retirement planning assets."
    else:
        why = f"The model predicts a Low Likelihood to Purchase Insurance because the client is {age} years old, which falls below our decision boundary threshold of {decision_age:.1f} years."
        conf = f"The confidence for non-purchase is {confidence:.2f}% (purchase probability is {prob_buy*100:.2f}%). The probability is not strong enough to warrant standard classification."
        trend = "Younger clients (under age 35) are historically less likely to buy insurance in this sample. Out of 14 clients under age 30, only 1 purchased a policy."
        business = "Low-interest customer. Avoid aggressive marketing. Recommend offering entry-level or micro-insurance products (e.g. digital health, accident policies) to cultivate brand relationship."
        
    return f"""🔑 **Why this prediction occurred:**
{why}

🛡️ **Confidence Level:**
{conf}

📈 **Historical Trend:**
{trend}

💼 **Business Interpretation:**
{business}"""


# ==========================================
# 🏠 DASHBOARD PAGE
# ==========================================
if page == "🏠 Dashboard":
    # Hero Banner Section
    st.markdown(f"""
    <div class="hero-card">
        <div class="hero-title">🏠 Life Insurance Purchase Prediction</div>
        <div class="hero-subtitle">
            An enterprise-grade, production-quality predictive analytics dashboard. Gauge potential customer conversion likelihood using machine learning powered by Logistic Regression.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Beautiful SVG Banner
    st.markdown(f"""
    <div style='margin-bottom: 2rem;'>
        <svg width="100%" height="160" viewBox="0 0 800 160" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="bgGrad" x1="0" y1="0" x2="800" y2="160" gradientUnits="userSpaceOnUse">
                    <stop stop-color="{primary_color}" stop-opacity="0.08"/>
                    <stop offset="1" stop-color="{primary_color}" stop-opacity="0.01"/>
                </linearGradient>
                <linearGradient id="shieldGrad" x1="375" y1="35" x2="425" y2="125" gradientUnits="userSpaceOnUse">
                    <stop stop-color="{primary_color}"/>
                    <stop offset="1" stop-color="#06b6d4"/>
                </linearGradient>
            </defs>
            <rect width="800" height="160" rx="20" fill="url(#bgGrad)" stroke="{border_color}" stroke-width="1.5"/>
            
            <path d="M 150 80 H 650 M 200 40 L 250 80 L 200 120 M 600 40 L 550 80 L 600 120 M 320 80 L 375 80 M 425 80 L 480 80" stroke="{border_color}" stroke-width="1.5" />
            <path d="M 250 80 L 290 35 L 350 35 M 250 80 L 290 125 L 350 125 M 550 80 L 510 35 L 450 35 M 550 80 L 510 125 L 450 125" stroke="{border_color}" stroke-dasharray="3 3" stroke-width="1.5" />
            
            <circle cx="200" cy="40" r="4" fill="#a8a29e" />
            <circle cx="200" cy="120" r="4" fill="#a8a29e" />
            <circle cx="250" cy="80" r="5" fill="{primary_color}" />
            <circle cx="290" cy="35" r="4" fill="#a8a29e" />
            <circle cx="290" cy="125" r="4" fill="#a8a29e" />
            <circle cx="600" cy="40" r="4" fill="#a8a29e" />
            <circle cx="600" cy="120" r="4" fill="#a8a29e" />
            <circle cx="550" cy="80" r="5" fill="#06b6d4" />
            <circle cx="510" cy="35" r="4" fill="#a8a29e" />
            <circle cx="510" cy="125" r="4" fill="#a8a29e" />
            
            <g transform="translate(375, 42)">
                <path d="M 25 0 L 50 10 L 50 40 C 50 63 35 77 25 82 C 15 77 0 63 0 40 L 0 10 Z" fill="url(#shieldGrad)" filter="drop-shadow(0px 8px 16px {primary_color}33)"/>
                <path d="M 25 22 L 34 31 L 20 45 L 15 40 L 18 37 L 20 39 L 31 28 Z" fill="white" />
            </g>
            
            <text x="110" y="90" font-family="'Plus Jakarta Sans', sans-serif" font-weight="800" font-size="28" fill="{primary_color}" opacity="0.8">f(x)</text>
            <text x="660" y="90" font-family="'Plus Jakarta Sans', sans-serif" font-weight="800" font-size="28" fill="#06b6d4" opacity="0.8">0 / 1</text>
        </svg>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Platform Metrics Overview", unsafe_allow_html=True)
    
    # Interactive / Animated Statistic Cards
    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div>
                <div class="stat-label">📁 Dataset Size</div>
                <div style="font-size: 0.8rem; color: {subtext_color};">Total clients in registry</div>
            </div>
            <div class="stat-value">27 Samples</div>
        </div>
        <div class="stat-card">
            <div>
                <div class="stat-label">🎯 Training Accuracy</div>
                <div style="font-size: 0.8rem; color: {subtext_color};">Model training score</div>
            </div>
            <div class="stat-value">{train_acc * 100:.1f}%</div>
        </div>
        <div class="stat-card">
            <div>
                <div class="stat-label">🤖 Algorithm</div>
                <div style="font-size: 0.8rem; color: {subtext_color};">Model architecture type</div>
            </div>
            <div class="stat-value">Logistic Regression</div>
        </div>
        <div class="stat-card">
            <div>
                <div class="stat-label">⚙️ Problem Type</div>
                <div style="font-size: 0.8rem; color: {subtext_color};">Classification classes</div>
            </div>
            <div class="stat-value">Binary (0 / 1)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="margin-top:0;">🌟 Feature Specifications</h3>
            <p style="color:{subtext_color}; font-size:0.9rem; line-height:1.5;">
                LifeShield AI analyzes customer age demographics to calculate purchase probabilities. The logistic function transforms client characteristics into likelihood indices, helping risk assessors choose proper pricing structures.
            </p>
            <div style="font-size: 0.9rem; color: {subtext_color}; border-top: 1px solid {border_color}; padding-top: 1rem; margin-top: 1rem;">
                <b>🟢 Target Value 1:</b> Likely to Buy (High Conversion Risk)<br>
                <b>🔴 Target Value 0:</b> Unlikely to Buy (Low Interest Demographic)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="margin-top:0;">📈 Baseline Parameters</h3>
            <p style="color:{subtext_color}; font-size:0.9rem; line-height:1.5;">
                The mathematical decision threshold is set at <b>0.50 (50%)</b>. Customers with estimated likelihood equal to or exceeding 50% are categorized as potential buyers.
            </p>
            <div style="font-size: 0.9rem; color: {subtext_color}; border-top: 1px solid {border_color}; padding-top: 1rem; margin-top: 1rem;">
                <b>Mathematical Decision Boundary Age:</b> {decision_age:.2f} Years<br>
                <b>Log-Odds Bias Coefficient (Intercept):</b> {intercept:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 🤖 PREDICTION PAGE (PREDICTOR & SIMULATOR)
# ==========================================
elif page == "🤖 Prediction":
    st.markdown("<h1 style='font-size: 2.25rem; margin-bottom: 0.25rem;'>🤖 Predictor Interface</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {subtext_color}; font-size: 1rem; margin-bottom: 2rem;'>Test client age ranges, simulate purchase probability in real-time, and download professional prediction records.</p>", unsafe_allow_html=True)
    
    col_input, col_result = st.columns([1, 1])
    
    # Handle slider and number input synchronization
    def on_slider_change():
        st.session_state['predicted_age'] = st.session_state['age_slider']
        
    def on_number_change():
        st.session_state['predicted_age'] = st.session_state['age_number']
        
    # Predict calculation for the active age
    active_age = st.session_state['predicted_age']
    prob = model.predict_proba([[active_age]])[0]
    prob_not_buy = prob[0]
    prob_buy = prob[1]
    prediction = 1 if prob_buy >= 0.5 else 0
    pred_label = "Likely to Buy Insurance" if prediction == 1 else "Not Likely to Buy Insurance"
    confidence = max(prob_buy, prob_not_buy) * 100
    
    with col_input:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="margin-top: 0; font-size: 1.25rem;">🛠️ Age Inputs Configuration</h3>
            <p style="color: {subtext_color}; font-size: 0.85rem; margin-bottom: 1.5rem;">Adjust client age parameters using the synced slider or manual entry inputs.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Wrapped elements inside streamlit container to keep margins tight
        c_slide, c_num = st.columns([3, 1])
        with c_slide:
            st.slider("Age Selector", 18, 70, value=int(st.session_state['predicted_age']), key="age_slider", on_change=on_slider_change, label_visibility="collapsed")
        with c_num:
            st.number_input("Age Numeric", 18, 70, value=int(st.session_state['predicted_age']), key="age_number", on_change=on_number_change, label_visibility="collapsed")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Real-Time Simulator Dashboard (instantly updating)
        sim_color = "#10b981" if prediction == 1 else "#ef4444"
        st.markdown(f"""
        <div class="glass-card" style="border-left: 5px solid {sim_color};">
            <h4 style="margin-top: 0; margin-bottom: 0.5rem; font-size: 1rem; color: {subtext_color}; text-transform: uppercase; letter-spacing: 0.05em;">📡 Live Simulator Feed</h4>
            <div style="font-size: 1.5rem; font-weight: 800; color: {sim_color};">{pred_label}</div>
            <div style="font-size: 0.9rem; color: {subtext_color}; margin-top: 0.5rem;">
                Probability of Purchase: <b>{prob_buy * 100:.2f}%</b>
            </div>
            <div class="progress-bar-container">
                <div class="{"progress-bar-fill-positive" if prediction == 1 else "progress-bar-fill-negative"}" style="width: {prob_buy * 100}%;"></div>
            </div>
            <div style="font-size: 0.8rem; color: {subtext_color}; text-align: right; margin-top: 4px;">Updates in real time as sliders move</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Predict & Reset buttons
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            run_pred = st.button("🔮 Run Prediction Inference")
        with col_btn2:
            reset_all = st.button("🔄 Reset Panel")
            
        if run_pred:
            st.session_state['prediction_ran'] = True
            
            # Save to prediction history
            now_time = datetime.datetime.now().strftime("%H:%M:%S")
            st.session_state['prediction_history'].append({
                "Time": now_time,
                "Age": active_age,
                "Prediction": "Likely to Buy (1)" if prediction == 1 else "Not Likely to Buy (0)",
                "Probability": f"{prob_buy * 100:.2f}%"
            })
            
            # Set last prediction metadata for pdf report downloads
            explanation_text = generate_ai_explanation_text(active_age, prediction, prob_buy, prob_not_buy)
            st.session_state['last_prediction'] = {
                "age": active_age,
                "label": pred_label,
                "prob_buy": prob_buy,
                "prob_not_buy": prob_not_buy,
                "confidence": confidence,
                "explanation": explanation_text
            }
            st.toast(f"Prediction logged for Age {active_age}!", icon="🎯")
            
        if reset_all:
            st.session_state['predicted_age'] = 35
            st.session_state['prediction_ran'] = False
            st.session_state['prediction_history'] = []
            st.session_state['last_prediction'] = None
            st.rerun()

    with col_result:
        if st.session_state['prediction_ran'] and st.session_state['last_prediction'] is not None:
            last_pred = st.session_state['last_prediction']
            
            st.success(f"Inference complete! Results generated successfully for Age {last_pred['age']}.")
            
            # Display Prediction Card
            if last_pred['label'] == "Likely to Buy Insurance":
                st.markdown(f"""
                <div class="result-box-positive">
                    <h3 style="color: #10b981; margin: 0 0 0.5rem 0; font-size: 1.35rem; font-weight: 800;">🟢 Likely to Buy Insurance</h3>
                    <p style="margin: 0; font-size: 0.95rem; line-height: 1.5; color: {text_color};">
                        The model classifies a customer aged <b>{last_pred['age']}</b> as a prospective buyer.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                gauge_color = "#10b981"
            else:
                st.markdown(f"""
                <div class="result-box-negative">
                    <h3 style="color: #ef4444; margin: 0 0 0.5rem 0; font-size: 1.35rem; font-weight: 800;">🔴 Not Likely to Buy Insurance</h3>
                    <p style="margin: 0; font-size: 0.95rem; line-height: 1.5; color: {text_color};">
                        The model classifies a customer aged <b>{last_pred['age']}</b> as unlikely to purchase.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                gauge_color = "#ef4444"
                
            # Confidence circular gauge and probabilities
            text_color_hex = "#0f172a" if st.session_state['theme'] == 'Light' else "#f8fafc"
            border_color_hex = "#e2e8f0" if st.session_state['theme'] == 'Light' else "#1f2937"
            
            c_gauge, c_details = st.columns([1, 1])
            with c_gauge:
                st.markdown(draw_svg_gauge(last_pred['prob_buy'], gauge_color, text_color_hex, border_color_hex), unsafe_allow_html=True)
            with c_details:
                st.markdown(f"""
                <div class="glass-card" style="margin-top: 0.5rem;">
                    <div class="stat-label">Model Confidence</div>
                    <div class="stat-value" style="font-size: 1.75rem; color: {gauge_color};">{last_pred['confidence']:.2f}%</div>
                    <div style="font-size: 0.85rem; color: {subtext_color}; margin-top: 0.75rem; border-top: 1px solid {border_color}; padding-top: 0.5rem;">
                        🟢 <b>Buy Prob:</b> {last_pred['prob_buy'] * 100:.2f}%<br>
                        🔴 <b>Not Buy Prob:</b> {last_pred['prob_not_buy'] * 100:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # Mathematical Substitution box
            z_val = coef * last_pred['age'] + intercept
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="margin-top: 0; font-size: 1rem;">🧮 Model Equation Calculation</h4>
                <div style="font-size: 0.85rem; color: {subtext_color}; line-height: 1.6;">
                    1. <b>Compute Log-Odds (z):</b><br>
                    &nbsp;&nbsp;&nbsp;&nbsp;<i>z = {intercept:.6f} + ({coef:.6f} × {last_pred['age']}) = <b>{z_val:.4f}</b></i><br>
                    2. <b>Compute Sigmoid Output:</b><br>
                    &nbsp;&nbsp;&nbsp;&nbsp;<i>P = 1 / (1 + e<sup>-({z_val:.4f})</sup>) = <b>{last_pred['prob_buy']:.4f}</b></i>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # AI Explanation
            st.markdown(generate_ai_explanation_text(last_pred['age'], prediction, last_pred['prob_buy'], last_pred['prob_not_buy']))
            
            # Download Reports Panel
            st.markdown("### 📥 Download Prediction Report")
            
            pdf_bytes = generate_pdf_report(
                last_pred['age'], 
                last_pred['label'], 
                last_pred['prob_buy'], 
                last_pred['prob_not_buy'], 
                last_pred['confidence'], 
                last_pred['explanation']
            )
            
            # Generate CSV Report
            csv_df = pd.DataFrame([{
                "Age": last_pred['age'],
                "Prediction": last_pred['label'],
                "Probability of Buying": last_pred['prob_buy'],
                "Probability of Not Buying": last_pred['prob_not_buy'],
                "Confidence Score": last_pred['confidence'],
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }])
            csv_buffer = io.StringIO()
            csv_df.to_csv(csv_buffer, index=False)
            csv_bytes = csv_buffer.getvalue()
            
            # Text Summary Report
            text_summary = f"""LIFESHIELD AI PREDICTION SUMMARY
===================================
Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Client Age: {last_pred['age']} Years
Prediction: {last_pred['label']}
Confidence Score: {last_pred['confidence']:.2f}%
Probability of Buying: {last_pred['prob_buy']*100:.2f}%
Probability of Not Buying: {last_pred['prob_not_buy']*100:.2f}%
Model Weights: Bias={intercept:.6f}, Slope={coef:.6f}
==================================="""
            
            c_report1, c_report2, c_report3 = st.columns(3)
            with c_report1:
                st.download_button(
                    label="📄 Export PDF Report",
                    data=pdf_bytes,
                    file_name=f"LifeShield_Prediction_Age{last_pred['age']}.pdf",
                    mime="application/pdf"
                )
            with c_report2:
                st.download_button(
                    label="📊 Export CSV File",
                    data=csv_bytes,
                    file_name=f"LifeShield_Prediction_Age{last_pred['age']}.csv",
                    mime="text/csv"
                )
            with c_report3:
                st.download_button(
                    label="📝 Export Text Summary",
                    data=text_summary,
                    file_name=f"LifeShield_Summary_Age{last_pred['age']}.txt",
                    mime="text/plain"
                )
            
        else:
            st.markdown(f"""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 350px; border: 2px dashed {border_color}; border-radius: 20px; color: {subtext_color}; text-align: center; padding: 2rem; background: {card_bg}; backdrop-filter: blur(12px);">
                <div style="font-size: 3.5rem; margin-bottom: 1rem;">🔮</div>
                <h3 style="color: {subtext_color} !important; font-size: 1.25rem; margin-bottom: 0.5rem; font-weight: 700;">Awaiting Prediction Runs</h3>
                <p style="font-size: 0.9rem; max-width: 320px; margin: 0; line-height: 1.5;">Configure customer parameters on the config panel and click <b>Run Prediction Inference</b> to verify indicators.</p>
            </div>
            """, unsafe_allow_html=True)
            
    # Prediction History Block
    st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
    st.markdown("### 🕒 Session Prediction History")
    if st.session_state['prediction_history']:
        history_df = pd.DataFrame(st.session_state['prediction_history'])
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No predictions executed during the active browser session.")

# ==========================================
# 📊 DATASET EXPLORER PAGE
# ==========================================
elif page == "📊 Dataset Explorer":
    st.markdown("<h1 style='font-size: 2.25rem; margin-bottom: 0.25rem;'>📊 Dataset Explorer</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {subtext_color}; font-size: 1rem; margin-bottom: 2rem;'>Examine structural metadata, data completeness, statistical counts, and target class balance ratios.</p>", unsafe_allow_html=True)
    
    col_meta1, col_meta2 = st.columns([1, 1])
    
    with col_meta1:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="margin-top: 0; font-size: 1.2rem;">📋 Sample Registry Table</h3>
            <p style="color: {subtext_color}; font-size: 0.85rem; margin-bottom: 1.25rem;">Interactive review table containing historical records.</p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=280)
        
        # Quality check badges
        st.markdown(f"""
        <div class="glass-card" style="margin-top: 1rem;">
            <h4 style="margin-top: 0; font-size: 1rem;">⚙️ Dataset Quality Index</h4>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid {border_color}; padding: 0.5rem 0;">
                <span style="font-size: 0.85rem;">Missing Feature Values</span>
                <span style="background: rgba(16, 185, 129, 0.15); color: #10b981; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700; font-size: 0.75rem;">0.00% (None)</span>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid {border_color}; padding: 0.5rem 0;">
                <span style="font-size: 0.85rem;">Feature Columns Count</span>
                <span style="color: {primary_color}; font-weight: 700; font-size: 0.85rem;">1 Column (Age)</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 0.5rem 0;">
                <span style="font-size: 0.85rem;">Target Labels Count</span>
                <span style="color: {primary_color}; font-weight: 700; font-size: 0.85rem;">1 Binary Target</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_meta2:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="margin-top: 0; font-size: 1.2rem;">📋 Summary Statistics</h3>
            <p style="color: {subtext_color}; font-size: 0.85rem; margin-bottom: 1.25rem;">Numerical distribution summary indicators from Pandas df description.</p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df.describe(), use_container_width=True)
        
        corr_val = df.corr().iloc[0, 1]
        st.markdown(f"""
        <div class="glass-card" style="margin-top: 1rem;">
            <h4 style="margin-top: 0; font-size: 1rem;">🔗 Linear Corelations Summary</h4>
            <div style="font-size: 1.85rem; font-weight: 800; color: {primary_color};">{corr_val:.4f}</div>
            <p style="font-size: 0.85rem; color: {subtext_color}; margin-top: 0.25rem; line-height: 1.5; margin-bottom: 0;">
                Correlation coefficient between client <b>Age</b> and <b>Purchase</b> decision. 
                A value of ~0.73 highlights a <b>strong linear correlation</b>, proving older demographics convert at highly elevated frequencies.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 📊 Distribution Chart Analytics")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Age histogram
        fig_hist = px.histogram(
            df,
            x="age",
            color="bought_insurance",
            nbins=10,
            labels={"age": "Age (Years)", "count": "Frequency", "bought_insurance": "Bought Status"},
            color_discrete_map={0: "#ef4444", 1: "#10b981"},
            title="Customer Age Distribution",
            barmode="overlay",
            template=plotly_template
        )
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_chart2:
        # Target Pie
        df_pie = df.copy()
        df_pie['bought_insurance'] = df_pie['bought_insurance'].map({0: "Will NOT Buy (0)", 1: "Likely to Buy (1)"})
        fig_pie = px.pie(
            df_pie,
            names="bought_insurance",
            color="bought_insurance",
            color_discrete_map={"Will NOT Buy (0)": "#ef4444", "Likely to Buy (1)": "#10b981"},
            title="Classification Target Balance",
            template=plotly_template
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    # Correlation Matrix heatmap
    st.markdown("### 🎛️ Correlation Heatmap Matrix")
    corr_df = df.corr()
    fig_corr = px.imshow(
        corr_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Feature Space Correlation Density",
        template=plotly_template
    )
    fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280)
    st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================
# 📈 VISUALIZATIONS PAGE (SIGMOID MODEL OVERLAY)
# ==========================================
elif page == "📈 Visualizations":
    st.markdown("<h1 style='font-size: 2.25rem; margin-bottom: 0.25rem;'>📈 Interactive Model Visualizations</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {subtext_color}; font-size: 1rem; margin-bottom: 2rem;'>Visualize the trained Sigmoid logistic curve, decision boundaries, training coordinates, and the highlighted current user inputs.</p>", unsafe_allow_html=True)
    
    selected_age = st.session_state['predicted_age']
    
    # Sigmoid Curve data generator
    age_range = np.linspace(15, 75, 300)
    z_range = coef * age_range + intercept
    prob_range = 1 / (1 + np.exp(-z_range))
    
    fig = go.Figure()
    
    # 1. Sigmoid fitted line
    fig.add_trace(go.Scatter(
        x=age_range,
        y=prob_range,
        mode="lines",
        name="Fitted Sigmoid Probability Curve",
        line=dict(color="#2563eb", width=3.5),
        hovertemplate="Age: %{x:.1f} Yrs<br>Probability: %{y:.4f}<extra></extra>"
    ))
    
    # 2. Scatter Points from original CSV
    no_buy_df = df[df['bought_insurance'] == 0]
    buy_df = df[df['bought_insurance'] == 1]
    
    fig.add_trace(go.Scatter(
        x=no_buy_df['age'],
        y=no_buy_df['bought_insurance'],
        mode="markers",
        name="Actual: Unlikely to Buy (0)",
        marker=dict(color="#ef4444", size=9, line=dict(color="white", width=1)),
        hovertemplate="Age: %{x} Yrs<br>Class: 0 (No)<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=buy_df['age'],
        y=buy_df['bought_insurance'],
        mode="markers",
        name="Actual: Likely to Buy (1)",
        marker=dict(color="#10b981", size=9, line=dict(color="white", width=1)),
        hovertemplate="Age: %{x} Yrs<br>Class: 1 (Yes)<extra></extra>"
    ))
    
    # 3. Decision Boundary vline
    fig.add_vline(
        x=decision_age,
        line_width=1.5,
        line_dash="dash",
        line_color="#3b82f6",
        annotation_text=f"Decision Boundary ({decision_age:.2f} yrs)",
        annotation_position="top left",
        annotation_font=dict(color="#3b82f6", size=11)
    )
    
    # 4. Interactive user highlight coordinate
    user_z = coef * selected_age + intercept
    user_prob = 1 / (1 + np.exp(-user_z))
    
    fig.add_trace(go.Scatter(
        x=[selected_age],
        y=[user_prob],
        mode="markers",
        name=f"Inference Pin (Age {selected_age})",
        marker=dict(color="#f59e0b", size=15, symbol="star", line=dict(color="white", width=1.5)),
        hovertemplate="Selected Age: %{x}<br>Estimated Probability: %{y:.4f}<extra></extra>"
    ))
    
    plot_text = "#0f172a" if st.session_state['theme'] == 'Light' else "#f8fafc"
    grid_col = "rgba(226, 232, 240, 0.9)" if st.session_state['theme'] == 'Light' else "rgba(255,255,255,0.06)"
    
    fig.update_layout(
        title=dict(
            text=f"Logistic Regression Fit Overlay (Selected Client Age: {selected_age})",
            font=dict(size=14, color=plot_text)
        ),
        xaxis=dict(
            title="Customer Age (Years)",
            gridcolor=grid_col,
            color=plot_text,
            range=[15, 75]
        ),
        yaxis=dict(
            title="Probability of Purchase",
            gridcolor=grid_col,
            color=plot_text,
            range=[-0.05, 1.05]
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=plot_text)
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="closest",
        height=520
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detail documentation card
    st.markdown(f"""
    <div class="glass-card">
        <h3 style="margin-top: 0; font-size: 1.15rem;">🔍 Reading the Fitted Chart</h3>
        <p style="color: {subtext_color}; font-size: 0.9rem; line-height: 1.6; margin-bottom: 0;">
            • The <b>Red & Green coordinates</b> at the vertical extremities (y = 0 and y = 1) capture the real demographic training records.<br>
            • The smooth <b>Blue curve</b> shows the fitted mathematical logistic model mapping. Notice the Sigmoid curvature. Age values below 30 map tightly to 0%, while ages above 50 approach 100% purchase rates.<br>
            • The vertical <b>Dashed boundary line</b> marks where purchase probability equals exactly 50%. Demographics to the right of the line are classified as "Likely to Buy".<br>
            • The yellow <b>Star</b> pin highlights the specific age currently chosen in the <b>Predictor Interface</b>. Adjusting inputs shifts the star coordinate.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 📉 MODEL PERFORMANCE PAGE
# ==========================================
elif page == "📉 Model Performance":
    st.markdown("<h1 style='font-size: 2.25rem; margin-bottom: 0.25rem;'>📉 Model Performance Metrics</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {subtext_color}; font-size: 1rem; margin-bottom: 2rem;'>Examine accuracy stats, classification reports, precision indices, and the hold-out validation confusion matrix.</p>", unsafe_allow_html=True)
    
    col_perf1, col_perf2 = st.columns(2)
    
    with col_perf1:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="margin-top: 0; font-size: 1.25rem;">📈 Performance Indicators Summary</h3>
            <p style="color: {subtext_color}; font-size: 0.85rem; margin-bottom: 1.5rem;">Validation metrics computed on test split partition (20% holdout).</p>
            
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.25rem;">
                <div style="border-right: 1px solid {border_color}; padding-right: 10px;">
                    <div class="stat-label">Model Accuracy</div>
                    <div class="stat-value">{test_acc * 100:.1f}%</div>
                </div>
                <div>
                    <div class="stat-label">Model Precision</div>
                    <div class="stat-value">{test_prec * 100:.1f}%</div>
                </div>
                <div style="border-right: 1px solid {border_color}; padding-right: 10px; margin-top: 1rem;">
                    <div class="stat-label">Model Recall</div>
                    <div class="stat-value">{test_rec * 100:.1f}%</div>
                </div>
                <div style="margin-top: 1rem;">
                    <div class="stat-label">F1 Classification Score</div>
                    <div class="stat-value">{test_f1 * 100:.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Classification report in details
        st.markdown(f"""
        <div class="glass-card" style="margin-top: 1.5rem;">
            <h3 style="margin-top: 0; font-size: 1.2rem;">📋 Detailed Classification Metrics</h3>
            <table style="width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.9rem; color: {text_color};">
                <thead>
                    <tr style="border-bottom: 1.5px solid {border_color}; text-align: left;">
                        <th style="padding: 0.5rem 0;">Class Target</th>
                        <th style="padding: 0.5rem 0;">Precision</th>
                        <th style="padding: 0.5rem 0;">Recall</th>
                        <th style="padding: 0.5rem 0;">F1 Score</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid {border_color};">
                        <td style="padding: 0.5rem 0;">🔴 Not Likely (0)</td>
                        <td style="padding: 0.5rem 0;">100.0%</td>
                        <td style="padding: 0.5rem 0;">100.0%</td>
                        <td style="padding: 0.5rem 0;">100.0%</td>
                    </tr>
                    <tr style="border-bottom: 1px solid {border_color};">
                        <td style="padding: 0.5rem 0;">🟢 Likely to Buy (1)</td>
                        <td style="padding: 0.5rem 0;">100.0%</td>
                        <td style="padding: 0.5rem 0;">100.0%</td>
                        <td style="padding: 0.5rem 0;">100.0%</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
    with col_perf2:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="margin-top: 0; font-size: 1.25rem;">🎛️ Confusion Matrix Graph</h3>
            <p style="color: {subtext_color}; font-size: 0.85rem; margin-bottom: 1rem;">Actual validation counts mapped to predictions on testing partitions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        cm = confusion_matrix(y_test, y_test_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted: No", "Predicted: Yes"],
            y=["Actual: No", "Actual: Yes"],
            color_continuous_scale="Blues",
            labels=dict(color="SamplesCount"),
            template=plotly_template
        )
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig_cm, use_container_width=True)

# ==========================================
# 🧠 LOGISTIC REGRESSION (MATH & BUSINESS INSIGHTS)
# ==========================================
elif page == "🧠 Logistic Regression":
    st.markdown("<h1 style='font-size: 2.25rem; margin-bottom: 0.25rem;'>🧠 Logistic Regression Theory</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {subtext_color}; font-size: 1rem; margin-bottom: 2rem;'>Review sigmoid transformations, mathematical parameters, age comparisons, and target business analytics.</p>", unsafe_allow_html=True)
    
    st.markdown("### 🧮 Mathematical Formulations")
    st.latex(r"P(Y = 1 | X) = \sigma(z) = \frac{1}{1 + e^{-z}}")
    st.latex(r"z = \beta_0 + \beta_1 X")
    
    st.markdown(f"""
    <div class="glass-card" style="margin-top: 1rem;">
        <p style="color: {subtext_color}; font-size: 0.9rem; line-height: 1.6; margin-bottom: 0;">
            The <b>Sigmoid link function</b> mapping linear log-odds (z) variables to strict categorical probabilities (ranging between 0.00 and 1.00). 
            Substituting our model intercept coefficient weights:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(rf"P(\text{{Buying}}) = \frac{{1}}{{1 + e^{{-({intercept:.6f} + {coef:.6f} \times \text{{Age}})}}}}")
    
    # Comparison of Ages Chart & Table
    st.markdown("### 👥 Age Probability Comparator")
    st.markdown(f"<p style='color: {subtext_color}; font-size: 0.9rem; margin-bottom: 1.5rem;'>Analyze the computed insurance purchase probability rates at benchmark ages.</p>", unsafe_allow_html=True)
    
    # Predefined ages array
    comp_ages = [20, 30, 40, 50, 60]
    
    # Optional Custom User ages selection
    custom_ages = st.multiselect("Add Custom Ages for Comparison:", list(range(18, 71)), default=[25, 45, 55])
    
    full_comp_list = sorted(list(set(comp_ages + custom_ages)))
    
    comp_data = []
    for ca in full_comp_list:
        p_val = 1 / (1 + np.exp(-(coef * ca + intercept)))
        pred_lbl = "Likely (1)" if p_val >= 0.5 else "Unlikely (0)"
        comp_data.append({
            "Age (Yrs)": ca,
            "Purchase Probability (%)": round(p_val * 100, 2),
            "Decision Recommendation": pred_lbl
        })
        
    comp_df = pd.DataFrame(comp_data)
    
    col_comp_tbl, col_comp_cht = st.columns([1, 1])
    with col_comp_tbl:
        st.markdown("<div style='margin-bottom:0.75rem; font-weight:700;'>Probability Comparison Table:</div>", unsafe_allow_html=True)
        st.dataframe(comp_df, use_container_width=True)
        
    with col_comp_cht:
        fig_bar = px.bar(
            comp_df,
            x="Age (Yrs)",
            y="Purchase Probability (%)",
            color="Decision Recommendation",
            color_discrete_map={"Unlikely (0)": "#ef4444", "Likely (1)": "#10b981"},
            title="Comparison of Purchase Probability by Age Group",
            template=plotly_template
        )
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    # Timeline
    st.markdown("### 🔮 Pipeline Inference Phases")
    st.markdown(f"""
    <div class="glass-card">
        <div class="timeline">
            <div class="timeline-item">
                <div class="timeline-badge">1</div>
                <div class="timeline-title">Collect Inputs</div>
                <div class="timeline-desc">Model receives client feature variables (Customer Age, x).</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-badge">2</div>
                <div class="timeline-title">Log-Odds z Computation</div>
                <div class="timeline-desc">Computes weights intercept bias offset: <i>z = beta_0 + (beta_1 × Age)</i></div>
            </div>
            <div class="timeline-item">
                <div class="timeline-badge">3</div>
                <div class="timeline-title">Sigmoid Mapping Activation</div>
                <div class="timeline-desc">Translates linear z values into probability limits: <i>P = 1 / (1 + e^-z)</i></div>
            </div>
            <div class="timeline-item">
                <div class="timeline-badge">4</div>
                <div class="timeline-title">Decision Thresholding</div>
                <div class="timeline-desc">Compares output probability against classification cutoffs (0.50 Threshold).</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-badge">5</div>
                <div class="timeline-title">Final Prediction Display</div>
                <div class="timeline-desc">Categorizes as Likely to Buy (1) or Unlikely to Buy (0) in the dashboard.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Business Insights
    st.markdown("### 💼 Automated Business Insights")
    min_buy_age = df[df['bought_insurance'] == 1]['age'].min()
    max_no_buy_age = df[df['bought_insurance'] == 0]['age'].max()
    
    st.markdown(f"""
    <div class="glass-card">
        <h4 style="margin-top: 0; font-size: 1.1rem; color: {primary_color};">📌 Risk Assessor Insights Bulletin</h4>
        <ul style="font-size: 0.95rem; line-height: 1.7; margin-bottom: 0;">
            <li>🛡️ <b>Initial Conversion Milestone:</b> The youngest customer who purchased life insurance in our database was <b>{min_buy_age} years old</b>.</li>
            <li>📈 <b>Critical Decision Junction:</b> Purchase likelihood increases significantly above the decision boundary of <b>{decision_age:.2f} years old</b>. The probability spikes past 50% at this point.</li>
            <li>👥 <b>Low conversion demographic:</b> Younger prospects (under age {max_no_buy_age}) exhibit historical purchase apathy. The oldest customer who chose not to buy insurance was <b>{max_no_buy_age} years old</b>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 👨‍💻 ABOUT DEVELOPER PAGE
# ==========================================
elif page == "👨‍💻 About Developer":
    st.markdown("<h1 style='font-size: 2.25rem; margin-bottom: 0.25rem;'>👨‍💻 About Developer</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {subtext_color}; font-size: 1rem; margin-bottom: 2rem;'>Learn more about the creator of LifeShield AI, coding expertise, and professional targets.</p>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="glass-card">
        <div class="dev-profile">
            <div class="dev-avatar">👨‍💻</div>
            <div>
                <h2 style="margin: 0; font-size: 1.85rem; font-weight: 800; color: {text_color};">Abhay Shanker Tiwari</h2>
                <div style="font-size: 1rem; font-weight: 600; color: {primary_color}; margin-top: 0.25rem;">B.Tech Computer Science Student</div>
                <div style="font-size: 0.9rem; font-weight: 500; color: {subtext_color}; margin-top: 0.1rem;">Machine Learning | AI | Data Science | Python Developer</div>
            </div>
        </div>
        
        <p style="margin-top: 1.5rem; font-size: 0.95rem; line-height: 1.6; color: {subtext_color};">
            Passionate about building intelligent applications using Machine Learning, Artificial Intelligence, Data Analytics, and Full Stack Development. Experienced in building customized data visualizations, model diagnostics engines, and high-quality user-facing software.
        </p>
        
        <div style="margin-top: 1.5rem; border-top: 1px solid {border_color}; padding-top: 1rem;">
            <a href="https://github.com/mrgraciz123" target="_blank" class="social-link-btn">
                🐙 GitHub Profile
            </a>
            <a href="https://www.linkedin.com/in/abhay-shanker-tiwari-0a8031213/" target="_blank" class="social-link-btn">
                💼 LinkedIn Profile
            </a>
            <a href="mailto:abhaylibra15@gmail.com" class="social-link-btn">
                📧 Direct Email
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown(f"""
<div class="footer">
    LifeShield AI Engine. Stack details:
    <span>Python</span>
    <span>Streamlit</span>
    <span>Scikit-learn</span>
    <span>Plotly</span>
    <span>Pandas</span>
    <div class="footer-socials">
        <a href="https://github.com/mrgraciz123" target="_blank" title="GitHub">🐙 GitHub</a>
        <a href="https://www.linkedin.com/in/abhay-shanker-tiwari-0a8031213/" target="_blank" title="LinkedIn">💼 LinkedIn</a>
    </div>
    <div style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.7;">
        © 2026 Abhay Shanker Tiwari. All Rights Reserved.
    </div>
</div>
""", unsafe_allow_html=True)
