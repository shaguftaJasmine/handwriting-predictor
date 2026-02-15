import streamlit as st
import numpy as np
import pandas as pd
import random
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import cv2
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import time

# Page config
st.set_page_config(
    page_title="🌸 Handwriting Personality Predictor by Shagufta Jasmine 🌸",
    page_icon="🖍️",
    layout="centered"
)

# ============================================
# CUTE CSS WITH SMALLER FLOATING TEXT
# ============================================

st.markdown("""
<style>
    /* Pastel background */
    .stApp {
        background: linear-gradient(135deg, #fff0f5, #f8e8ff);
    }
    
    /* Floating animation */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
        100% { transform: translateY(0px); }
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-3px); }
    }
    
    /* Main title - SMALLER and FLOATING */
    .main-title {
        font-family: 'Comic Sans MS', 'Chalkboard SE', cursive;
        text-align: center;
        font-size: 2.2em;  /* MADE SMALLER */
        font-weight: bold;
        background: linear-gradient(45deg, #ff1493, #8a5a8a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 0 #ffb6c1;
        padding: 10px;
        animation: float 3s ease-in-out infinite;
        border: 2px solid #ffb6c1;
        border-radius: 40px;
        margin: 10px 0;
        background-color: rgba(255, 255, 255, 0.5);
        width: 80%;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* YOUR NAME - SMALLER and FLOATING */
    .creator-section {
        text-align: center;
        background: white;
        padding: 10px;  /* REDUCED PADDING */
        border-radius: 40px;
        margin: 10px auto;
        border: 3px solid #ff69b4;
        box-shadow: 0 5px 10px rgba(255,105,180,0.2);
        max-width: 400px;  /* SMALLER WIDTH */
        animation: float 4s ease-in-out infinite;
    }
    
    .created-by {
        font-family: 'Comic Sans MS', cursive;
        font-size: 1.2em;  /* SMALLER */
        color: #8a5a8a;
        margin-bottom: 2px;
    }
    
    .your-name {
        font-family: 'Comic Sans MS', cursive;
        font-size: 1.8em;  /* SMALLER */
        font-weight: bold;
        background: linear-gradient(45deg, #ff1493, #9400d3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 1px 1px 2px #ffb6c1;
        margin: 5px 0;
        letter-spacing: 1px;
    }
    
    .heart-decoration {
        font-size: 1.2em;  /* SMALLER */
        animation: bounce 1s ease-in-out infinite;
        display: inline-block;
    }
    
    /* Welcome text - SMALLER */
    .welcome-text {
        text-align: center;
        font-size: 1.1em;  /* SMALLER */
        color: #8a5a8a;
        margin: 5px 0;
        font-family: 'Comic Sans MS', cursive;
    }
    
    /* Feature boxes */
    .feature-box {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 15px;
        margin: 10px 0;
        border: 2px solid #ffb6c1;
        box-shadow: 0 5px 10px rgba(255, 182, 193, 0.2);
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ffb6c1, #ff69b4) !important;
        height: 10px !important;
    }
    
    /* Button - SMALLER */
    .stButton > button {
        background: linear-gradient(45deg, #ffb6c1, #ff69b4) !important;
        color: white !important;
        font-size: 18px !important;  /* SMALLER */
        border-radius: 40px !important;
        border: 2px solid white !important;
        padding: 8px 20px !important;  /* SMALLER */
        font-family: 'Comic Sans MS', cursive !important;
        animation: bounce 2s ease-in-out infinite !important;
    }
    
    /* Footer - SMALLER */
    .footer {
        text-align: center;
        padding: 15px;
        margin-top: 30px;
        background: white;
        border-radius: 40px;
        border: 2px solid #ffb6c1;
        width: 80%;
        margin-left: auto;
        margin-right: auto;
    }
    
    .footer-name {
        font-size: 1.3em;  /* SMALLER */
        background: linear-gradient(45deg, #ff1493, #8a5a8a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    /* Make all text slightly smaller */
    h3 {
        font-size: 1.3em !important;
    }
    
    p {
        font-size: 0.9em !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# YOUR NAME - SMALLER AND FLOATING
# ============================================

st.markdown("""
<div class='main-title'>
    ✍️ Handwriting Personality Predictor 🌸
</div>

<div class='creator-section'>
    <div class='created-by'>✨ created with love by ✨</div>
    <div class='your-name'>🌸 Shagufta Jasmine 🌸</div>
    <div>
        <span class='heart-decoration'>💖</span>
        <span class='heart-decoration'>🎀</span>
        <span class='heart-decoration'>💖</span>
    </div>
</div>

<div class='welcome-text'>
    ⭐ Welcome to my magical handwriting predictor! ⭐
</div>
""", unsafe_allow_html=True)

st.info("""
**✨ How it works ✨**  
1. Draw your handwriting sample below 🎨  
2. Watch the magic happen! ✨  
3. Discover your unique personality! 💖  
""")

# ============================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================

def normalize_value(value, min_val, max_val):
    """Normalize a value to be between 0 and 1"""
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

def extract_handwriting_features(image_array):
    """Extract REAL handwriting features from an image"""
    features = {}
    
    try:
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # 1. PEN PRESSURE
        ink_pixels = gray[gray < 250]
        if len(ink_pixels) > 0:
            pressure_raw = np.mean(ink_pixels) / 255.0
            pressure_raw = 1.0 - pressure_raw
        else:
            pressure_raw = 0.5
        features['pressure'] = float(normalize_value(pressure_raw, 0.2, 0.8))
        
        # 2. LETTER SIZE
        ink_pixel_count = np.sum(gray < 200)
        total_pixels = gray.shape[0] * gray.shape[1]
        size_raw = ink_pixel_count / total_pixels
        features['size'] = float(normalize_value(size_raw, 0.05, 0.3))
        
        # 3. SLANT
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=20, maxLineGap=10)
        
        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines[:10]:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            avg_slant = np.mean(angles) if angles else 0
            slant_normalized = normalize_value(avg_slant + 45, 0, 90)
        else:
            slant_normalized = 0.5
        features['slant'] = float(slant_normalized)
        
        # 4. SPACING
        if ink_pixel_count > 0:
            coords = np.argwhere(gray < 200)
            if len(coords) > 1:
                distances = []
                sample_size = min(100, len(coords))
                indices = np.random.choice(len(coords), sample_size, replace=False)
                for i in range(sample_size-1):
                    dist = np.linalg.norm(coords[indices[i]] - coords[indices[i+1]])
                    distances.append(dist)
                spacing_raw = np.mean(distances) if distances else 50
            else:
                spacing_raw = 50
        else:
            spacing_raw = 50
        
        features['spacing'] = float(normalize_value(spacing_raw, 10, 100))
        
        # 5. ROUNDNESS
        if len(ink_pixels) > 0:
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            direction = np.arctan2(sobely, sobelx)
            
            direction_var = np.std(direction[magnitude > 50]) if np.any(magnitude > 50) else 0
            roundness_raw = normalize_value(abs(direction_var), 0, 1)
        else:
            roundness_raw = 0.5
        
        features['roundness'] = float(roundness_raw)
        
    except Exception as e:
        features = {
            'pressure': 0.5,
            'size': 0.5,
            'slant': 0.5,
            'spacing': 0.5,
            'roundness': 0.5
        }
    
    return features

def predict_personality_from_features(features):
    """Predict personality from handwriting features"""
    
    scores = {}
    
    # 1. EXTROVERSION
    extroversion = (features['size'] * 0.6 + features['pressure'] * 0.4) * 100
    scores['Extroversion'] = min(100, extroversion)
    
    # 2. EMOTIONALITY
    emotionality = features['slant'] * 100
    scores['Emotionality'] = min(100, emotionality)
    
    # 3. CONSCIENTIOUSNESS
    conscientiousness = ((1 - features['spacing']) * 0.5 + (1 - features['size']) * 0.5) * 100
    scores['Conscientiousness'] = min(100, conscientiousness)
    
    # 4. OPENNESS
    openness = (features['roundness'] * 0.7 + (1 - features['pressure']) * 0.3) * 100
    scores['Openness'] = min(100, openness)
    
    # 5. AGREEABLENESS
    agreeableness = ((1 - features['pressure']) * 0.4 + features['roundness'] * 0.3 + 
                    (1 - abs(features['slant'] - 0.5)) * 0.3) * 100
    scores['Agreeableness'] = min(100, agreeableness)
    
    return scores

def get_personality_type(scores):
    """Determine personality type based on Big Five scores"""
    
    sorted_traits = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_traits = [t[0] for t in sorted_traits[:3]]
    
    personality_map = {
        ('Extroversion', 'Openness', 'Agreeableness'): {
            'name': 'The Creative Socialite 🎨',
            'emoji': '🌟',
            'color': '#ffb6c1',
            'desc': 'You are outgoing, imaginative, and friendly!',
            'advice': 'Keep sharing your creative ideas!'
        },
        ('Conscientiousness', 'Emotionality', 'Agreeableness'): {
            'name': 'The Caring Organizer 📋',
            'emoji': '💝',
            'color': '#c1e1c1',
            'desc': 'You are reliable, caring, and detail-oriented!',
            'advice': 'People trust you with their secrets!'
        },
        ('Openness', 'Emotionality', 'Extroversion'): {
            'name': 'The Passionate Dreamer 🌈',
            'emoji': '✨',
            'color': '#fdd0f2',
            'desc': 'You feel deeply and express yourself creatively!',
            'advice': 'Your emotions are your superpower!'
        },
        ('Conscientiousness', 'Openness', 'Emotionality'): {
            'name': 'The Thoughtful Innovator 💭',
            'emoji': '🦋',
            'color': '#b5d3e7',
            'desc': 'You combine practical thinking with creative ideas!',
            'advice': 'You see possibilities others miss!'
        },
        ('Agreeableness', 'Conscientiousness', 'Emotionality'): {
            'name': 'The Gentle Guardian 🕊️',
            'emoji': '🌸',
            'color': '#d4c2f0',
            'desc': 'You are kind, reliable, and in tune with others!',
            'advice': 'Your empathy is a gift!'
        },
        ('Extroversion', 'Agreeableness', 'Emotionality'): {
            'name': 'The Warm Friend ☀️',
            'emoji': '🧸',
            'color': '#ffd9b0',
            'desc': 'You light up rooms with your presence!',
            'advice': 'Your warmth attracts genuine friendships!'
        }
    }
    
    best_match = None
    best_score = 0
    
    for trait_combo, personality in personality_map.items():
        match_score = 0
        for trait in trait_combo:
            if trait in top_traits:
                match_score += scores.get(trait, 0)
        if match_score > best_score:
            best_score = match_score
            best_match = personality
    
    if not best_match:
        best_match = {
            'name': 'The Unique Individual 🌟',
            'emoji': '💫',
            'color': '#c0d6d2',
            'desc': 'You are wonderfully unique!',
            'advice': 'Embrace what makes you, YOU!'
        }
    
    return best_match

# ============================================
# MAIN APP
# ============================================

def main():
    # Try to import canvas
    try:
        from streamlit_drawable_canvas import st_canvas
        
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🎨 Draw Your Sample")
            st.markdown("Write: **'The quick brown fox jumps over the lazy dog'**")
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=3,
                stroke_color="#8a5a8a",
                background_color="#fff0f5",
                height=180,  # SLIGHTLY SMALLER
                width=350,   # SLIGHTLY SMALLER
                drawing_mode="freedraw",
                key="canvas",
            )
        
        with col2:
            st.markdown("### 📊 Live Analysis")
            
            if canvas_result.image_data is not None and np.any(canvas_result.image_data):
                features = extract_handwriting_features(canvas_result.image_data)
                
                st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
                
                # Pressure
                pressure_val = features['pressure']
                pressure_text = "Heavy" if pressure_val > 0.7 else "Light" if pressure_val < 0.3 else "Moderate"
                st.markdown(f"**Pen Pressure:** {pressure_text}")
                st.progress(float(pressure_val))
                
                # Size
                size_val = features['size']
                size_text = "Large" if size_val > 0.7 else "Small" if size_val < 0.3 else "Medium"
                st.markdown(f"**Letter Size:** {size_text}")
                st.progress(float(size_val))
                
                # Slant
                slant_val = features['slant']
                slant_text = "Right" if slant_val > 0.6 else "Left" if slant_val < 0.4 else "Straight"
                st.markdown(f"**Slant:** {slant_text}")
                st.progress(float(slant_val))
                
                # Spacing
                space_val = features['spacing']
                space_text = "Wide" if space_val > 0.7 else "Narrow" if space_val < 0.3 else "Normal"
                st.markdown(f"**Spacing:** {space_text}")
                st.progress(float(space_val))
                
                # Roundness
                round_val = features['roundness']
                round_text = "Round" if round_val > 0.7 else "Angular" if round_val < 0.3 else "Mixed"
                st.markdown(f"**Shape:** {round_text}")
                st.progress(float(round_val))
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                if st.button("🔮 Reveal My Personality ✨", use_container_width=True):
                    with st.spinner('✨ Analyzing... ✨'):
                        time.sleep(1)
                        
                        scores = predict_personality_from_features(features)
                        personality = get_personality_type(scores)
                        
                        st.balloons()
                        
                        st.markdown("---")
                        st.markdown("## 🎀 Your Personality Profile 🎀")
                        
                        # Personality card
                        st.markdown(f"""
                        <div style='background-color: {personality["color"]}30; border-radius: 30px; padding: 20px; text-align: center; border: 3px solid {personality["color"]}; margin: 15px 0;'>
                            <span style='font-size: 3.5em;'>{personality["emoji"]}</span>
                            <h3 style='color: {personality["color"]}; margin: 5px 0;'>{personality["name"]}</h3>
                            <p style='font-size: 0.95em;'>{personality["desc"]}</p>
                            <p><b>✨ Insight:</b> {personality["advice"]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Scores
                        st.markdown("### 📊 Your Scores")
                        for trait, score in scores.items():
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                st.markdown(f"**{trait[:3]}**")
                            with col2:
                                st.progress(score/100)
            else:
                st.info("🎨 Draw something to see analysis!")
    
    except ImportError as e:
        st.error("Please install: `pip install streamlit-drawable-canvas opencv-python`")
    
    # ============================================
    # FOOTER WITH YOUR NAME - SMALLER
    # ============================================
    
    st.markdown("""
    <div class='footer'>
        <div style='font-size: 1.1em; color: #8a5a8a;'>Made with 💖 by</div>
        <div class='footer-name'>🌸 Shagufta Jasmine 🌸</div>
        <div style='font-size: 1.2em; margin: 5px;'>✨ 🎀 ✨</div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()