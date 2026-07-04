import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import math

# Page config for high-quality dashboard layout
st.set_page_config(
    page_title="Life Insurance Purchase Prediction | Predictive Analytics Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme session state
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'Light'

# Initialize prediction state
if 'prediction_ran' not in st.session_state:
    st.session_state['prediction_ran'] = False
if 'predicted_age' not in st.session_state:
    st.session_state['predicted_age'] = 35

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

# --- THEME MANAGEMENT ---
st.sidebar.markdown("""
<div style='text-align: center; margin-bottom: 1.5rem; margin-top: 1rem;'>
    <div style='font-size: 2.5rem; filter: drop-shadow(0 4px 6px rgba(37,99,235,0.2));'>🛡️</div>
    <h2 style='margin: 0.5rem 0 0.1rem 0; font-size: 1.35rem; letter-spacing: -0.025em;'>LifeShield AI</h2>
    <p style='color: #64748b; font-size: 0.8rem; margin: 0;'>SaaS Predictive Platform</p>
</div>
""", unsafe_allow_html=True)

# Interactive theme selector
theme_btn = st.sidebar.selectbox(
    "🎨 UI Appearance",
    ["Light Theme", "Dark Theme"],
    index=0 if st.session_state['theme'] == 'Light' else 1,
    label_visibility="visible"
)
st.session_state['theme'] = 'Light' if theme_btn == "Light Theme" else 'Dark'

# Navigation Links
st.sidebar.markdown("<br>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Dashboard", 
        "📊 Dataset", 
        "🤖 Prediction Panel", 
        "📈 Visualizations", 
        "📚 Model Technical Details", 
        "ℹ️ About Logistic Regression"
    ],
    index=0
)

# Colors and CSS definitions depending on the selected theme
if st.session_state['theme'] == 'Light':
    primary_color = "#2563eb"  # Deep Blue
    accent_color = "#3b82f6"   # Electric Blue
    background_color = "#f8fafc"
    card_bg = "#ffffff"
    text_color = "#0f172a"
    subtext_color = "#475569"
    border_color = "#e2e8f0"
    shadow_color = "rgba(0, 0, 0, 0.04)"
    sidebar_bg = "#ffffff"
    gauge_bg = "#e2e8f0"
    table_hdr = "#f1f5f9"
else:
    primary_color = "#3b82f6"  # Brighter Blue for dark contrast
    accent_color = "#60a5fa"   # Light Blue
    background_color = "#0b0f19"
    card_bg = "#111827"        # Dark charcoal
    text_color = "#f9fafb"
    subtext_color = "#9ca3af"
    border_color = "#1f2937"
    shadow_color = "rgba(0, 0, 0, 0.4)"
    sidebar_bg = "#111827"
    gauge_bg = "#1f2937"
    table_hdr = "#1f2937"

# Custom CSS Injector
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Main body background & custom fonts */
    .stApp {{
        background-color: {background_color};
        color: {text_color};
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    
    /* Clean up Streamlit header & margins */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Sidebar custom branding & padding */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border_color};
        padding-top: 1rem;
    }}
    
    /* Sidebar radio selections */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
        font-weight: 600;
        color: {text_color} !important;
        margin-bottom: 0.5rem;
    }}
    
    /* Modern Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {text_color} !important;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin-bottom: 1rem;
    }}
    
    /* Glassmorphism custom card panel */
    .card {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px -5px {shadow_color};
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    .card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 20px 40px -10px {shadow_color};
        border-color: {primary_color}33;
    }}
    
    .metric-value {{
        font-size: 2.25rem;
        font-weight: 800;
        color: {primary_color};
        line-height: 1.1;
        letter-spacing: -0.04em;
    }}
    
    .metric-label {{
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.075em;
        color: {subtext_color};
        margin-bottom: 0.5rem;
    }}
    
    /* Custom inputs decoration */
    .stNumberInput input, .stTextInput input {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        font-size: 1rem !important;
    }}
    
    .stNumberInput div[data-baseweb="input"] {{
        background-color: transparent !important;
        border: none !important;
    }}
    
    /* Beautiful gradients for Predict buttons */
    div.stButton > button {{
        background: linear-gradient(135deg, {primary_color}, {accent_color});
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 0.8rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 18px 0 {primary_color}33;
        cursor: pointer;
    }}
    
    div.stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px 0 {primary_color}66 !important;
        background: linear-gradient(135deg, {accent_color}, {primary_color});
    }}
    
    div.stButton > button:active {{
        transform: translateY(0px) !important;
    }}

    /* Premium result block styling */
    .result-box-positive {{
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(52, 211, 153, 0.03));
        border: 1.5px solid rgba(16, 185, 129, 0.25);
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 1rem;
    }}
    
    .result-box-negative {{
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(248, 113, 113, 0.03));
        border: 1.5px solid rgba(239, 68, 68, 0.25);
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 1rem;
    }}

    /* Modern gauge animated progress bars */
    .progress-bar-container {{
        background-color: {gauge_bg};
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

    /* Workflow Timeline styling */
    .timeline {{
        position: relative;
        border-left: 2px solid {border_color};
        padding-left: 24px;
        margin-left: 12px;
        margin-top: 1rem;
    }}
    
    .timeline-item {{
        position: relative;
        margin-bottom: 1.75rem;
    }}
    
    .timeline-item:last-child {{
        margin-bottom: 0;
    }}
    
    .timeline-badge {{
        position: absolute;
        left: -37px;
        top: 2px;
        background: linear-gradient(135deg, {primary_color}, {accent_color});
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 800;
        box-shadow: 0 0 0 4px {background_color};
    }}
    
    .timeline-title {{
        font-weight: 700;
        font-size: 1rem;
        color: {text_color};
        margin-bottom: 4px;
        letter-spacing: -0.015em;
    }}
    
    .timeline-desc {{
        color: {subtext_color};
        font-size: 0.875rem;
        line-height: 1.45;
    }}
    
    /* Footer styles */
    .footer {{
        text-align: center;
        padding: 2rem 0;
        margin-top: 4rem;
        border-top: 1px solid {border_color};
        color: {subtext_color};
        font-size: 0.85rem;
    }}
    .footer span {{
        background: {border_color};
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        margin: 0 0.25rem;
        font-weight: 500;
        font-family: monospace;
        font-size: 0.75rem;
    }}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 🏠 DASHBOARD PAGE
# ==========================================
if page == "🏠 Dashboard":
    st.markdown("<h1 style='font-size: 2.5rem; margin-bottom: 0.25rem;'>Life Insurance Purchase Prediction</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {subtext_color}; font-size: 1.1rem; margin-bottom: 2rem;'>Predict whether a customer is likely to purchase life insurance using Logistic Regression trained on historical data.</p>", unsafe_allow_html=True)
    
    # Beautiful AI + Insurance SVG Illustration Banner
    st.markdown(f"""
    <div style='margin-bottom: 2rem;'>
        <svg width="100%" height="160" viewBox="0 0 800 160" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="bgGrad" x1="0" y1="0" x2="800" y2="160" gradientUnits="userSpaceOnUse">
                    <stop stop-color="{primary_color}" stop-opacity="0.08"/>
                    <stop offset="1" stop-color="{accent_color}" stop-opacity="0.01"/>
                </linearGradient>
                <linearGradient id="shieldGrad" x1="375" y1="35" x2="425" y2="125" gradientUnits="userSpaceOnUse">
                    <stop stop-color="{primary_color}"/>
                    <stop offset="1" stop-color="{accent_color}"/>
                </linearGradient>
            </defs>
            <rect width="800" height="160" rx="16" fill="url(#bgGrad)" stroke="{border_color}" stroke-width="1"/>
            
            <!-- Tech grids and visual connections -->
            <path d="M 150 80 H 650 M 200 40 L 250 80 L 200 120 M 600 40 L 550 80 L 600 120 M 320 80 L 375 80 M 425 80 L 480 80" stroke="{border_color}" stroke-width="1.5" />
            <path d="M 250 80 L 290 35 L 350 35 M 250 80 L 290 125 L 350 125 M 550 80 L 510 35 L 450 35 M 550 80 L 510 125 L 450 125" stroke="{border_color}" stroke-dasharray="3 3" stroke-width="1.5" />
            
            <!-- Nodes -->
            <circle cx="200" cy="40" r="4" fill="#a8a29e" />
            <circle cx="200" cy="120" r="4" fill="#a8a29e" />
            <circle cx="250" cy="80" r="5" fill="{primary_color}" />
            <circle cx="290" cy="35" r="4" fill="#a8a29e" />
            <circle cx="290" cy="125" r="4" fill="#a8a29e" />
            <circle cx="600" cy="40" r="4" fill="#a8a29e" />
            <circle cx="600" cy="120" r="4" fill="#a8a29e" />
            <circle cx="550" cy="80" r="5" fill="{accent_color}" />
            <circle cx="510" cy="35" r="4" fill="#a8a29e" />
            <circle cx="510" cy="125" r="4" fill="#a8a29e" />
            
            <!-- Center Shield Icon representing Protection and AI -->
            <g transform="translate(375, 42)">
                <path d="M 25 0 L 50 10 L 50 40 C 50 63 35 77 25 82 C 15 77 0 63 0 40 L 0 10 Z" fill="url(#shieldGrad)" filter="drop-shadow(0px 8px 16px {primary_color}33)"/>
                <path d="M 25 22 L 34 31 L 20 45 L 15 40 L 18 37 L 20 39 L 31 28 Z" fill="white" />
            </g>
            
            <!-- Floating icons/math hints -->
            <text x="110" y="90" font-family="'Plus Jakarta Sans', sans-serif" font-weight="800" font-size="28" fill="{primary_color}" opacity="0.8">f(x)</text>
            <text x="660" y="90" font-family="'Plus Jakarta Sans', sans-serif" font-weight="800" font-size="28" fill="{accent_color}" opacity="0.8">0 / 1</text>
        </svg>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Model Metadata & Problem Configuration", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Model Architecture</div>
            <div class="metric-value">Logistic Regression</div>
            <div style="font-size: 0.9rem; color: {subtext_color}; margin-top: 0.75rem; border-top: 1px solid {border_color}; padding-top: 0.75rem;">
                <b>Algorithm:</b> Sigmoid Logistic Regression<br>
                <b>Optimization Solver:</b> L-BFGS (Limited-memory BFGS)<br>
                <b>Loss Function:</b> Binary Cross-Entropy
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Target Variables</div>
            <div class="metric-value">2 Decision Classes</div>
            <div style="font-size: 0.9rem; color: {subtext_color}; margin-top: 0.75rem; border-top: 1px solid {border_color}; padding-top: 0.75rem;">
                🔴 <b style="color:#ef4444">0 → Will NOT Buy Insurance</b> (Age groups leaning younger)<br>
                🟢 <b style="color:#10b981">1 → Likely to Buy Insurance</b> (Age groups leaning older)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Model Input Dimensions</div>
            <div class="metric-value">1 Feature (Age)</div>
            <div style="font-size: 0.9rem; color: {subtext_color}; margin-top: 0.75rem; border-top: 1px solid {border_color}; padding-top: 0.75rem;">
                <b>Feature Name:</b> Client Age<br>
                <b>Minimum Limit:</b> 18 Years<br>
                <b>Maximum Limit:</b> 70 Years
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Training Set Size</div>
            <div class="metric-value">27 Sample Records</div>
            <div style="font-size: 0.9rem; color: {subtext_color}; margin-top: 0.75rem; border-top: 1px solid {border_color}; padding-top: 0.75rem;">
                <b>Dataset Origin:</b> `insurance_data.csv`<br>
                <b>Data Integrity:</b> 100% complete, 0 missing features<br>
                <b>Class Distribution:</b> 15 Will Not Buy, 12 Likely to Buy
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Timeline
    st.markdown(f"""
    <div class="card">
        <h3 style="margin-top: 0;">🔮 Real-Time Machine Learning Pipeline</h3>
        <p style="color: {subtext_color}; font-size: 0.9rem; margin-bottom: 1.5rem;">How the application computes the buy probability step-by-step:</p>
        <div class="timeline">
            <div class="timeline-item">
                <div class="timeline-badge">1</div>
                <div class="timeline-title">Collect Inputs</div>
                <div class="timeline-desc">The dashboard collects the customer's age (e.g. 35 years) from the sidebar or prediction page.</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-badge">2</div>
                <div class="timeline-title">Log-Odds Computation</div>
                <div class="timeline-desc">The algorithm computes a linear combination: <i>z = Intercept + (Coefficient × Age)</i>, translating age into logs of odds.</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-badge">3</div>
                <div class="timeline-title">Sigmoid Activation Function</div>
                <div class="timeline-desc">The logistic curve transforms <i>z</i> into a probability bound: <i>p = 1 / (1 + e⁻ᶻ)</i> representing values between 0 and 1.</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-badge">4</div>
                <div class="timeline-title">Binary Thresholding</div>
                <div class="timeline-desc">A strict 0.5 decision threshold classifies probability <i>p ≥ 50%</i> as a positive buy outcome and <i>p < 50%</i> as a reject/not-buying outcome.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 📊 DATASET PAGE
# ==========================================
elif page == "📊 Dataset":
    st.markdown("<h1 style='font-size: 2.25rem;'>📊 Dataset Exploratory Analysis</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {subtext_color}; font-size: 1rem; margin-bottom: 2rem;'>Explore the statistical details and distributions of customer data in <code>insurance_data.csv</code>.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="card">
            <h3 style="margin-top: 0; font-size: 1.15rem;">📋 Interactive Dataset Table</h3>
            <p style="color: {subtext_color}; font-size: 0.85rem; margin-bottom: 1rem;">Full historical records containing client ages and their decision.</p>
        </div>
        """, unsafe_allow_html=True)
        # Display dataset table
        st.dataframe(df, use_container_width=True, height=350)
        
        # Missing values check
        st.markdown(f"""
        <div class="card" style="margin-top: 1rem;">
            <h3 style="margin-top: 0; font-size: 1.15rem;">🛡️ Data Quality Check</h3>
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid {border_color}; padding: 0.5rem 0;">
                <span style="font-size: 0.9rem;">Missing Age Entries</span>
                <span style="background: rgba(16, 185, 129, 0.15); color: #10b981; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700; font-size: 0.75rem;">0.00% (None)</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0;">
                <span style="font-size: 0.9rem;">Missing Purchase Labels</span>
                <span style="background: rgba(16, 185, 129, 0.15); color: #10b981; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700; font-size: 0.75rem;">0.00% (None)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="card">
            <h3 style="margin-top: 0; font-size: 1.15rem;">📈 Dataset Descriptive Summary</h3>
            <p style="color: {subtext_color}; font-size: 0.85rem; margin-bottom: 1rem;">Summary statistics calculated from pandas <code>describe()</code> function.</p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df.describe(), use_container_width=True)
        
        # Correlation
        correlation = df.corr().iloc[0, 1]
        st.markdown(f"""
        <div class="card" style="margin-top: 1rem;">
            <h3 style="margin-top: 0; font-size: 1.15rem;">🔗 Statistical Correlation</h3>
            <div style="font-size: 1.8rem; font-weight: 800; color: {primary_color};">{correlation:.4f}</div>
            <div style="font-size: 0.85rem; color: {subtext_color}; margin-top: 0.25rem;">
                Correlation coefficient between <b>Age</b> and <b>Bought Insurance</b>. 
                A value of ~0.73 indicates a <b>strong positive correlation</b>, demonstrating that older customers are significantly more likely to purchase insurance.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Dataset Distributions", unsafe_allow_html=True)
    
    # Histogram of Age vs Purchase Status
    fig_hist = px.histogram(
        df, 
        x="age", 
        color="bought_insurance", 
        nbins=12,
        labels={"age": "Age", "bought_insurance": "Bought Insurance"},
        color_discrete_map={0: "#ef4444", 1: "#10b981"},
        barmode="overlay",
        title="Distribution of Age by Purchase Behavior"
    )
    grid_col = "#e2e8f0" if st.session_state['theme'] == 'Light' else "#1f2937"
    fig_hist.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=text_color,
        legend_title_text="Bought Insurance",
        margin=dict(l=40, r=40, t=50, b=40),
        bargap=0.08
    )
    fig_hist.update_xaxes(showgrid=True, gridcolor=grid_col)
    fig_hist.update_yaxes(showgrid=True, gridcolor=grid_col)
    st.plotly_chart(fig_hist, use_container_width=True)


# ==========================================
# 🤖 PREDICTION PANEL PAGE
# ==========================================
elif page == "🤖 Prediction Panel":
    st.markdown("<h1 style='font-size: 2.25rem;'>🤖 Predictor Interface</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {subtext_color}; font-size: 1rem; margin-bottom: 2rem;'>Test custom customer ages against the trained Logistic Regression model and generate confidence metrics.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="card">
            <h3 style="margin-top: 0; font-size: 1.15rem;">🎛️ Input Customer Age</h3>
            <p style="color: {subtext_color}; font-size: 0.85rem; margin-bottom: 1.5rem;">Adjust the values below. The fields are synchronized dynamically.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Synchronized slider and input widgets using session state
        def update_slider():
            st.session_state.age_slider_widget = st.session_state.age_num_widget
            st.session_state['predicted_age'] = st.session_state.age_num_widget
            
        def update_num():
            st.session_state.age_num_widget = st.session_state.age_slider_widget
            st.session_state['predicted_age'] = st.session_state.age_slider_widget
            
        if 'age_slider_widget' not in st.session_state:
            st.session_state.age_slider_widget = st.session_state['predicted_age']
            st.session_state.age_num_widget = st.session_state['predicted_age']

        # Interactive controls layout inside column
        age_col1, age_col2 = st.columns([3, 1])
        with age_col1:
            st.slider("Age (Slider Selector)", 18, 70, key="age_slider_widget", on_change=update_num)
        with age_col2:
            st.number_input("Age (Numeric)", 18, 70, key="age_num_widget", on_change=update_slider)
            
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Run Predictive Inference")
        
        if predict_btn:
            st.session_state['prediction_ran'] = True
            
    with col2:
        if st.session_state['prediction_ran']:
            selected_age = st.session_state['predicted_age']
            
            # Predict
            prob = model.predict_proba([[selected_age]])[0]
            prob_not_buy = prob[0]
            prob_buy = prob[1]
            prediction = 1 if prob_buy >= 0.5 else 0
            confidence = max(prob_buy, prob_not_buy) * 100
            
            # Toast and success banner
            st.toast(f"Prediction computed for Age {selected_age}!", icon="🎯")
            
            if prediction == 1:
                st.markdown(f"""
                <div class="result-box-positive">
                    <h3 style="color: #10b981; margin: 0 0 0.5rem 0; font-size: 1.25rem;">🟢 Likely to Buy Insurance</h3>
                    <div style="font-size: 0.9rem; color: {text_color};">
                        Model predicts that a customer aged <b>{selected_age}</b> will buy life insurance.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box-negative">
                    <h3 style="color: #ef4444; margin: 0 0 0.5rem 0; font-size: 1.25rem;">🔴 Not Likely to Buy Insurance</h3>
                    <div style="font-size: 0.9rem; color: {text_color};">
                        Model predicts that a customer aged <b>{selected_age}</b> will NOT buy life insurance.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # Confidence Card
            st.markdown(f"""
            <div class="card" style="margin-top: 1rem;">
                <div class="metric-label">Model Prediction Confidence</div>
                <div class="metric-value">{confidence:.2f}%</div>
                <div class="progress-bar-container">
                    <div class="{"progress-bar-fill-positive" if prediction == 1 else "progress-bar-fill-negative"}" style="width: {confidence}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem; border-top: 1px solid {border_color}; padding-top: 0.75rem; font-size: 0.9rem;">
                    <span>🟢 Buying Probability:</span>
                    <span style="font-weight: 700;">{prob_buy * 100:.2f}%</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem; font-size: 0.9rem;">
                    <span>🔴 Not Buying Probability:</span>
                    <span style="font-weight: 700;">{prob_not_buy * 100:.2f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive prediction flow computation steps
            z = coef * selected_age + intercept
            st.markdown(f"""
            <div class="card" style="margin-top: 1rem;">
                <h3 style="margin-top: 0; font-size: 1.1rem;">📝 Mathematical Validation</h3>
                <div style="font-size: 0.85rem; color: {subtext_color}; line-height: 1.5;">
                    1. <b>Compute Log-Odds (z):</b><br>
                    &nbsp;&nbsp;&nbsp;&nbsp;<i>z = Intercept + (Coeff × Age)</i><br>
                    &nbsp;&nbsp;&nbsp;&nbsp;<i>z = {intercept:.6f} + ({coef:.6f} × {selected_age}) = <b>{z:.4f}</b></i><br><br>
                    2. <b>Apply Sigmoid Link Function:</b><br>
                    &nbsp;&nbsp;&nbsp;&nbsp;<i>p = 1 / (1 + e⁻ᶻ)</i><br>
                    &nbsp;&nbsp;&nbsp;&nbsp;<i>p = 1 / (1 + e^({-z:.4f})) = <b>{prob_buy:.4f}</b></i>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown(f"""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 320px; border: 2px dashed {border_color}; border-radius: 16px; color: {subtext_color}; text-align: center; padding: 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔮</div>
                <h3 style="color: {subtext_color} !important; font-size: 1.2rem; margin-bottom: 0.5rem;">Awaiting Inference</h3>
                <p style="font-size: 0.85rem; max-width: 300px; margin: 0;">Adjust the customer age slider on the left panel and click <b>Run Predictive Inference</b> to view statistics.</p>
            </div>
            """, unsafe_allow_html=True)


# ==========================================
# 📈 VISUALIZATIONS PAGE
# ==========================================
elif page == "📈 Visualizations":
    st.markdown("<h1 style='font-size: 2.25rem;'>📈 Interactive Model Visualizations</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {subtext_color}; font-size: 1rem; margin-bottom: 2rem;'>Analyze the fitted Sigmoid logistic function, Decision boundary, training dataset distribution, and highlight predictions.</p>", unsafe_allow_html=True)
    
    # Generate interactive Plotly sigmoid curve
    selected_age = st.session_state['predicted_age']
    
    # Generates sigmoid curve values
    age_range = np.linspace(15, 75, 300)
    z_range = coef * age_range + intercept
    prob_range = 1 / (1 + np.exp(-z_range))
    
    fig = go.Figure()
    
    # Grid colors depending on the active theme
    grid_col = "#e2e8f0" if st.session_state['theme'] == 'Light' else "#1f2937"
    plot_text_col = "#1e293b" if st.session_state['theme'] == 'Light' else "#f1f5f9"
    
    # 1. Decision Boundary line
    decision_age = -intercept / coef
    fig.add_vline(
        x=decision_age, 
        line_width=1.5, 
        line_dash="dash", 
        line_color="#3b82f6",
        annotation_text=f"Decision Boundary ({decision_age:.1f} yrs)", 
        annotation_position="top left",
        annotation_font=dict(color="#3b82f6", size=11, family="'Plus Jakarta Sans', sans-serif")
    )
    
    # 2. Sigmoid Curve line
    fig.add_trace(go.Scatter(
        x=age_range, 
        y=prob_range,
        mode='lines',
        name='Fitted Sigmoid Curve',
        line=dict(color='#2563eb', width=3),
        hovertemplate='Age: %{x:.1f}<br>Prob: %{y:.4f}<extra></extra>'
    ))
    
    # 3. Scatter Points
    bought_pts = df[df['bought_insurance'] == 1]
    not_bought_pts = df[df['bought_insurance'] == 0]
    
    fig.add_trace(go.Scatter(
        x=not_bought_pts['age'], 
        y=not_bought_pts['bought_insurance'],
        mode='markers',
        name='Bought: No (0)',
        marker=dict(color='#ef4444', size=9, line=dict(width=1, color='white')),
        hovertemplate='Age: %{x}<br>Class: 0 (No)<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=bought_pts['age'], 
        y=bought_pts['bought_insurance'],
        mode='markers',
        name='Bought: Yes (1)',
        marker=dict(color='#10b981', size=9, line=dict(width=1, color='white')),
        hovertemplate='Age: %{x}<br>Class: 1 (Yes)<extra></extra>'
    ))
    
    # 4. Highlight current prediction point if ran
    user_z = coef * selected_age + intercept
    user_prob = 1 / (1 + np.exp(-user_z))
    
    fig.add_trace(go.Scatter(
        x=[selected_age],
        y=[user_prob],
        mode='markers',
        name=f'Inference Point (Age {selected_age})',
        marker=dict(color='#f59e0b', size=16, symbol='star', line=dict(width=2, color='white')),
        hovertemplate='Age: %{x}<br>Probability: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Logistic Regression Model Probability Mapping (Highlighting Client Age {selected_age})",
            font=dict(size=15, color=plot_text_col, family="'Plus Jakarta Sans', sans-serif")
        ),
        xaxis=dict(
            title="Customer Age (Years)", 
            gridcolor=grid_col, 
            color=plot_text_col,
            range=[15, 75]
        ),
        yaxis=dict(
            title="Estimated Probability of Purchasing Insurance", 
            gridcolor=grid_col, 
            color=plot_text_col,
            range=[-0.05, 1.05]
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(color=plot_text_col)
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="closest",
        height=500
    )
    
    # Display Chart Card
    st.plotly_chart(fig, use_container_width=True)
    
    # Plot analysis description
    st.markdown(f"""
    <div class="card" style="margin-top: 1rem;">
        <h3 style="margin-top: 0; font-size: 1.15rem;">🔍 Chart Analysis Insights</h3>
        <p style="color: {subtext_color}; font-size: 0.9rem; line-height: 1.5; margin: 0;">
            • The <b>Red Dots</b> at y = 0 and <b>Green Dots</b> at y = 1 represent the actual historical customer samples.<br>
            • The <b>Blue Curve</b> is the computed Sigmoid model. At younger ages (e.g. 18-30), the curve approaches 0, indicating extremely low interest. At older ages (e.g. 50+), the curve approaches 1, indicating higher interest.<br>
            • The vertical <b>Dashed Line</b> shows the decision boundary at age <b>{decision_age:.2f}</b> where purchase probability is exactly 50%. Any client older than this boundary is classified as "Likely to Buy".<br>
            • The yellow <b>Star</b> highlights the customer prediction age configured in your Predictor Interface.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 📚 MODEL TECHNICAL DETAILS PAGE
# ==========================================
elif page == "📚 Model Technical Details":
    st.markdown("<h1 style='font-size: 2.25rem;'>📚 Model Architecture & Validation Parameters</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {subtext_color}; font-size: 1rem; margin-bottom: 2rem;'>Review training weights, optimization metrics, validation confusion matrix, and prediction scores.</p>", unsafe_allow_html=True)
    
    # Metrics columns
    val_acc = model.score(X_test, y_test)
    num_train = len(X_train)
    num_test = len(X_test)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="card">
            <h3 style="margin-top: 0; font-size: 1.15rem;">🎯 Performance Validation Statistics</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1rem;">
                <div style="border-right: 1px solid {border_color};">
                    <div class="metric-label">Model Accuracy</div>
                    <div class="metric-value">{val_acc * 100:.1f}%</div>
                </div>
                <div>
                    <div class="metric-label">Total Train / Test Samples</div>
                    <div class="metric-value">{num_train} / {num_test}</div>
                </div>
            </div>
            <div style="margin-top: 1.5rem; font-size: 0.85rem; color: {subtext_color}; line-height: 1.45; border-top: 1px solid {border_color}; padding-top: 0.75rem;">
                The validation accuracy is computed against testing data split. Because of the clear age-based purchase trend, the Logistic Regression model can easily find a clear boundary.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display weights table
        st.markdown(f"""
        <div class="card" style="margin-top: 1.5rem;">
            <h3 style="margin-top: 0; font-size: 1.15rem;">⚙️ Learned Optimization Weights</h3>
            <table style="width: 100%; border-collapse: collapse; margin-top: 0.75rem; font-size: 0.9rem; color: {text_color};">
                <thead>
                    <tr style="border-bottom: 1.5px solid {border_color}; background-color: {table_hdr};">
                        <th style="padding: 0.5rem; text-align: left;">Parameter Type</th>
                        <th style="padding: 0.5rem; text-align: left;">Symbol</th>
                        <th style="padding: 0.5rem; text-align: right;">Learned Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid {border_color};">
                        <td style="padding: 0.5rem;">Age Feature Weight (Coefficient)</td>
                        <td style="padding: 0.5rem;"><code>β₁ (Slope)</code></td>
                        <td style="padding: 0.5rem; text-align: right; font-weight: 700; color: {primary_color};">{coef:.8f}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid {border_color};">
                        <td style="padding: 0.5rem;">Model Bias (Intercept)</td>
                        <td style="padding: 0.5rem;"><code>β₀ (Bias)</code></td>
                        <td style="padding: 0.5rem; text-align: right; font-weight: 700; color: {primary_color};">{intercept:.8f}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Confusion Matrix Heatmap
        st.markdown(f"""
        <div class="card">
            <h3 style="margin-top: 0; font-size: 1.15rem;">🎛️ Confusion Matrix</h3>
            <p style="color: {subtext_color}; font-size: 0.85rem; margin-bottom: 1rem;">Comparing predictions against actual values on the test partition.</p>
        </div>
        """, unsafe_allow_html=True)
        
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted Label", y="True Label", color="Samples Count"),
            x=['Will NOT Buy (0)', 'Likely to Buy (1)'],
            y=['Will NOT Buy (0)', 'Likely to Buy (1)'],
            color_continuous_scale='Blues'
        )
        grid_col = "#e2e8f0" if st.session_state['theme'] == 'Light' else "#1f2937"
        fig_cm.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color=text_color,
            width=400,
            height=280,
            margin=dict(l=40, r=40, t=30, b=40)
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
    st.markdown("### 🧮 Mathematical Formulation", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="card">
        <p style="color: {subtext_color}; font-size: 0.9rem; margin-bottom: 1rem;">
            The logistic probability <i>P</i> is mapping linear equations onto curves using the following formula:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"P(Y = 1 | X) = \sigma(z) = \frac{1}{1 + e^{-z}}")
    st.latex(r"z = \beta_0 + \beta_1 X")
    
    st.markdown(f"""
    <div class="card" style="margin-top: 1rem;">
        <p style="color: {subtext_color}; font-size: 0.9rem; margin-bottom: 1rem;">
            Substituting our model's trained intercept and coefficient results in the exact prediction equation:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(rf"P(\text{{Buying}}) = \frac{{1}}{{1 + e^{{-({intercept:.6f} + {coef:.6f} \times \text{{Age}})}}}}")


# ==========================================
# ℹ️ ABOUT LOGISTIC REGRESSION PAGE
# ==========================================
elif page == "ℹ️ About Logistic Regression":
    st.markdown("<h1 style='font-size: 2.25rem;'>ℹ️ About Logistic Regression</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {subtext_color}; font-size: 1rem; margin-bottom: 2rem;'>Understand the theoretical background of Logistic Regression and its applications in classification.</p>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="card">
        <h3 style="margin-top: 0; font-size: 1.25rem; color: {primary_color};">📌 What is Logistic Regression?</h3>
        <p style="color: {text_color}; font-size: 0.95rem; line-height: 1.6;">
            Unlike Linear Regression which predicts continuous outputs (e.g. housing prices), <b>Logistic Regression</b> is a classification algorithm used to predict binary categorical outcomes (Yes/No, True/False, 1/0). 
            It computes the odds of an event occurring and applies the <b>Sigmoid activation function</b> to map the output onto a strict probability scale between 0.0 and 1.0.
        </p>
    </div>
    
    <div class="card">
        <h3 style="margin-top: 0; font-size: 1.25rem; color: {primary_color};">📈 The Sigmoid Link Function</h3>
        <p style="color: {text_color}; font-size: 0.95rem; line-height: 1.6;">
            A simple linear regression line matches equation <i>y = mx + b</i> and could predict probabilities below 0 or above 1, which violates probability axioms. 
            To solve this, logistic regression applies the <b>Sigmoid curve</b>:
            <br><br>
            <code style="background: {border_color}; padding: 0.25rem 0.5rem; border-radius: 4px; font-weight:700;">S(z) = 1 / (1 + e^-z)</code>
            <br><br>
            When <i>z</i> goes to infinity, the output approaches 1. When <i>z</i> goes to negative infinity, the output approaches 0.
        </p>
    </div>
    
    <div class="card">
        <h3 style="margin-top: 0; font-size: 1.25rem; color: {primary_color};">⚖️ The Decision Boundary</h3>
        <p style="color: {text_color}; font-size: 0.95rem; line-height: 1.6;">
            A <b>decision boundary</b> is a threshold value that separates classes. 
            In standard binary classification, this is set at <b>0.5</b>. 
            Any calculated customer probability exceeding 50% is classified into Class 1, whereas probabilities below 50% are categorized as Class 0.
        </p>
    </div>

    <div class="card">
        <h3 style="margin-top: 0; font-size: 1.25rem; color: {primary_color};">👥 Why Age Influences Insurance Purchasing</h3>
        <p style="color: {text_color}; font-size: 0.95rem; line-height: 1.6;">
            In demographic datasets, Age acts as a proxy for life milestones and health factors:
            <br><br>
            • <b>Dependents:</b> Younger demographics (ages 18-28) are less likely to have mortgages, spouses, or children, reducing the perceived need for financial safety nets.<br>
            • <b>Health Risk:</b> As customers age, health awareness naturally rises, and premium rates begin to increase, encouraging customers to purchase coverage before they enter higher-risk brackets.<br>
            • <b>Financial Planning:</b> Older clients typically have higher disposable income and are more engaged in long-term legacy planning.
        </p>
    </div>
    """, unsafe_allow_html=True)


# --- FOOTER ---
st.markdown(f"""
<div class="footer">
    Predictive Inference Engine built with:
    <span>Python</span>
    <span>Streamlit</span>
    <span>Scikit-learn</span>
    <span>Plotly</span>
    <span>Pandas</span>
</div>
""", unsafe_allow_html=True)
