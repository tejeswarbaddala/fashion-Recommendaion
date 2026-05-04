
import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
from PIL import Image
import base64
from io import BytesIO

from diffusers import StableDiffusionPipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1) Page & Style Config
# -----------------------------
st.set_page_config(page_title="Fashion Assistant", layout="wide")

def load_css():
    """
    Load custom CSS styling for Streamlit app with improved text visibility
    and calmer background
    """
    st.markdown("""
    <style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container styling */
    .css-1d391kg {
        padding: 1rem;
    }
    
    /* Calmer background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e3eeff 100%);
    }
    
    /* Title and headers */
    h1 {
        color: #1a237e;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Product grid - fixed 4 cards per row */
    .product-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
        padding: 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .product-card {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        width: 180px;  /* Fixed width */
        margin: 0 auto;
        display: flex;
        flex-direction: column;
    }
    
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .product-image {
        width: 180px;
        height: 180px;
        object-fit: cover;
    }
    
    .product-info {
        padding: 0.75rem;
        background: white;
        border-top: 1px solid #eee;
    }
    
    .product-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #1a237e;
        margin-bottom: 0.5rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .product-title::before {
        content: "👗 ";
    }
    
    .product-details {
        font-size: 0.8rem;
        color: #333;
        line-height: 1.4;
    }
    
    .product-details p {
        margin: 0.25rem 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: #444;
    }
    
    .product-details strong {
        color: #1a237e;
        font-weight: 600;
    }
    
    .match-score {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: #1a237e;
        color: white;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Chat styling */
    .chat-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .message {
        padding: 0.75rem 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        max-width: 80%;
        clear: both;
        color: #333;
    }
    
    .bot-message {
        background: #f3f4f6;
        float: left;
    }
    
    .bot-message::before {
        content: "🤖 ";
    }
    
    .user-message {
        background: #e8eaf6;
        float: right;
    }
    
    .user-message::before {
        content: "👤 ";
    }
    
    /* Streamlit button styling */
    .stButton > button {
        background: #1a237e;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        background: #283593;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    /* Streamlit select box styling */
    .stSelectbox > div > div {
        background: white;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    /* Ensure content fits mobile screens */
    @media (max-width: 768px) {
        .product-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .product-card {
            width: 160px;
        }
        
        .product-image {
            width: 160px;
            height: 160px;
        }
        
        h1 {
            font-size: 2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# 2) Caching: Load CSV
# -----------------------------
@st.cache_data
def load_data(csv_path="updated_recommendation.csv"):
    df = pd.read_csv(csv_path)
    df['product_attributes'] = df['product_attributes'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else {}
    )
    df['measurement_ranges'] = df['measurement_ranges'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else {}
    )
    if 'year' in df.columns:
        df.drop(columns=['year'], inplace=True, errors='ignore')

    for col in ['gender','masterCategory','subCategory','articleType',
                'baseColour','season','usage','fit_type','pattern','material','style']:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    return df

# -----------------------------
# 3) Caching: Load Model
# -----------------------------
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "MohamedRashad/diffusion_fashion", torch_dtype=torch.float32
    )
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

# -----------------------------
# 4) Helper: Optimize images
# -----------------------------
def optimize_image(image_path, max_size=(512,512)):
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            buffer.seek(0)
            encoded = base64.b64encode(buffer.read()).decode()
            return encoded
    except Exception:
        return get_placeholder_image()

def get_placeholder_image():
    img = Image.new('RGB', (512, 512), color='#cccccc')
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode()
    return encoded

def display_product_card(product, image_path):
    base64_img = optimize_image(image_path)
    if not base64_img:
        base64_img = get_placeholder_image()

    html = f"""
    <div class="product-card">
        <img src="data:image/jpeg;base64,{base64_img}" class="product-image" />
        <div class="product-info">
            <div class="product-title">{product['productDisplayName']}</div>
            <div class="product-details">
                <p><strong>Category:</strong> {product['masterCategory']} - {product['subCategory']}</p>
                <p><strong>Color:</strong> {product['baseColour']}</p>
                <p><strong>Usage:</strong> {product['usage']}</p>
                <p><strong>Size:</strong> {product.get('recommended_size','Standard')}</p>
            </div>
            <div class="match-score">
                Match: {product['similarity_score']:.2f}
            </div>
        </div>
    </div>
    """
    return html

# -----------------------------
# 5) Main FashionChatbot Class
# -----------------------------
class FashionChatbot:
    def __init__(self):
        load_css()
        self.df = load_data()
        self.pipeline = load_model()
        self.init_session_state()

    def init_session_state(self):
        if 'current_stage' not in st.session_state:
            st.session_state.current_stage = 'welcome'
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {}
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'welcome_shown' not in st.session_state:
            st.session_state.welcome_shown = False
        if 'last_stage' not in st.session_state:
            st.session_state.last_stage = None

    def add_bot_message(self, text):
        # Only add message if it's not already the last bot message
        if not st.session_state.messages or st.session_state.messages[-1].get('text') != text:
            st.session_state.messages.append({"text": text, "is_user": False})

    def add_user_message(self, text):
        # Only add message if it's not already the last user message
        if not st.session_state.messages or st.session_state.messages[-1].get('text') != text:
            st.session_state.messages.append({"text": text, "is_user": True})

    def show_chat_messages(self):
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for msg in st.session_state.messages:
            klass = "user-message" if msg["is_user"] else "bot-message"
            st.markdown(f"<div class='message {klass}'>{msg['text']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    def collect_user_info(self):
        # Only add the welcome message once
        if not st.session_state.welcome_shown:
            self.add_bot_message("Hello! I'm your fashion assistant. Let's find the perfect item for you! 🛍️")
            st.session_state.welcome_shown = True
        
        self.show_chat_messages()
        
        possible_categories = sorted(self.df["masterCategory"].unique())
        cat_choice = st.selectbox("Which main category are you interested in? 🔎", possible_categories)
        gender_choice = st.selectbox("Select Gender:", ["Men","Women","Unisex"])

        if st.button("Next ➡️"):
            st.session_state.user_preferences["category"] = cat_choice
            st.session_state.user_preferences["gender"] = gender_choice
            st.session_state.current_stage = 'preferences'
            st.rerun()

    def collect_preferences(self):
        # Only add message if entering this stage
        if st.session_state.current_stage != st.session_state.last_stage:
            self.add_bot_message("Please enter your style preferences. 👀")
            st.session_state.last_stage = st.session_state.current_stage

        self.show_chat_messages()

        st.subheader("Measurements 📏")
        col1, col2 = st.columns(2)
        with col1:
            chest = st.number_input("Chest (inches)", min_value=28, max_value=60, value=36)
            waist = st.number_input("Waist (inches)", min_value=24, max_value=60, value=32)
        with col2:
            shoulder = st.number_input("Shoulder (inches)", min_value=14, max_value=24, value=16)
            hip = None
            if st.session_state.user_preferences["gender"] in ["Women","Unisex"]:
                hip = st.number_input("Hip (inches)", min_value=30, max_value=60, value=38)

        cat_filter = st.session_state.user_preferences["category"]
        gender_filter = st.session_state.user_preferences["gender"]
        if gender_filter == "Unisex":
            subcats = self.df[self.df["masterCategory"] == cat_filter]["subCategory"].unique()
        else:
            subcats = self.df[
                (self.df["masterCategory"] == cat_filter) &
                ((self.df["gender"] == gender_filter) | (self.df["gender"] == "Unisex"))
            ]["subCategory"].unique()

        subcats = sorted(subcats)

        st.subheader("Item Preferences 🛍️")
        chosen_subcat = st.selectbox(f"Which {cat_filter} sub-category?", subcats)
        
        usage_options = self.df[
            (self.df["masterCategory"] == cat_filter) &
            (self.df["subCategory"] == chosen_subcat) &
            ((self.df["gender"] == gender_filter) | (self.df["gender"] == "Unisex"))
        ]["usage"].unique()
        usage_options = sorted(usage_options)

        usage_choice = st.selectbox("Occasion / Usage:", usage_options)

        color_options = self.df[
            (self.df["masterCategory"] == cat_filter) &
            (self.df["subCategory"] == chosen_subcat) &
            ((self.df["gender"] == gender_filter) | (self.df["gender"] == "Unisex")) &
            (self.df["usage"] == usage_choice)
        ]["baseColour"].unique()
        color_options = sorted(color_options)

        colors = st.multiselect("Preferred color(s):", color_options, default=color_options[:1])

        if st.button("Get Recommendations"):
            st.session_state.user_preferences["measurements"] = {
                "chest": chest, 
                "waist": waist, 
                "shoulder": shoulder
            }
            if hip is not None:
                st.session_state.user_preferences["measurements"]["hip"] = hip

            st.session_state.user_preferences["subCategory"] = chosen_subcat
            st.session_state.user_preferences["usage"] = usage_choice
            st.session_state.user_preferences["preferred_colors"] = colors

            st.session_state.current_stage = 'recommendations'
            st.rerun()

    def show_recommendations(self):
        # Only add message if entering this stage
        if st.session_state.current_stage != st.session_state.last_stage:
            self.add_bot_message("Here are some recommended items for you! 💕")
            st.session_state.last_stage = st.session_state.current_stage

        self.show_chat_messages()

        recs = self.get_recommendations(st.session_state.user_preferences)
        if len(recs) == 0:
            st.warning("No exact matches found. Let's generate custom designs!")
            st.session_state.current_stage = "generate_custom"
            st.rerun()
        else:
            st.markdown("<div class='product-grid'>", unsafe_allow_html=True)
            for _, product in recs.iterrows():
                image_path = f"images/{product['image']}"
                st.markdown(display_product_card(product, image_path), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("👍 Looks Good"):
                    st.balloons()
                    st.success("Glad you liked it! 🎉")
                    st.session_state.current_stage = "done"
                    st.rerun()
            with c2:
                if st.button("🙁 Not Satisfied"):
                    st.session_state.current_stage = "generate_custom"
                    st.rerun()

    def generate_custom_design(self):
        # Only add message if entering this stage
        if st.session_state.current_stage != st.session_state.last_stage:
            self.add_bot_message("Let's create a custom design. Enter any style or details you like! 🖌️")
            st.session_state.last_stage = st.session_state.current_stage

        self.show_chat_messages()

        user_prompt = st.text_area(
            "Describe your desired fashion item and style:",
            "A stylish formal handbag with metallic accents, modern design, etc."
        )

        if st.button("Generate Design"):
            with st.spinner("Generating custom design..."):
                try:
                    image = self.pipeline(user_prompt).images[0]
                    st.image(image, caption="Your Custom Design", use_column_width=True)
                except Exception as e:
                    st.error(f"Error generating image: {e}")

        if st.button("↩️ Back to Preferences"):
            st.session_state.current_stage = "preferences"
            st.rerun()
        if st.button("Done"):
            st.session_state.current_stage = "done"
            st.rerun()

    def get_recommendations(self, user_prefs):
        df = self.df
        cat = user_prefs["category"]
        subcat = user_prefs["subCategory"]
        gender = user_prefs["gender"]
        usage = user_prefs["usage"]
        colors = user_prefs.get("preferred_colors", [])

        # Filter by category, subcategory, gender + Unisex, usage, and preferred colors
        filtered = df[df["masterCategory"] == cat]
        filtered = filtered[filtered["subCategory"] == subcat]
        if gender != "Unisex":
            filtered = filtered[(filtered["gender"] == gender) | (filtered["gender"] == "Unisex")]
        filtered = filtered[filtered["usage"] == usage]
        if colors:
            filtered = filtered[filtered["baseColour"].isin(colors)]
        if len(filtered) == 0:
            return pd.DataFrame()

        # Basic similarity approach with dummies
        cols = ["masterCategory","subCategory","articleType","baseColour","usage"]
        filtered[cols] = filtered[cols].astype(str)
        features = pd.get_dummies(filtered[cols])
        similarities = cosine_similarity(features)

        top_n = min(len(filtered), 10)  # Get up to 10 items
        top_idx = similarities.mean(axis=0).argsort()[-top_n:][::-1]
        recs = filtered.iloc[top_idx].copy()
        recs["similarity_score"] = similarities.mean(axis=0)[top_idx]

        # Add recommended size
        if "measurements" in user_prefs:
            recs["recommended_size"] = recs.apply(
                lambda x: self.recommend_size(x, user_prefs["measurements"]),
                axis=1
            )
        return recs

    def recommend_size(self, product, measurements):
        """
        Basic size recommendation logic
        """
        subcat = product.get("subCategory","").lower()
        if "topwear" in subcat:
            c = measurements["chest"]
            if c <= 36: return "S"
            elif c <= 38: return "M"
            elif c <= 42: return "L"
            else: return "XL"
        elif "bottomwear" in subcat:
            w = measurements["waist"]
            return str(round(w/2)*2)
        return "Standard"

    def run(self):
        st.title("AI Fashion Assistant 👗👜")
        if st.session_state.current_stage == 'welcome':
            self.collect_user_info()
        elif st.session_state.current_stage == 'preferences':
            self.collect_preferences()
        elif st.session_state.current_stage == 'recommendations':
            self.show_recommendations()
        elif st.session_state.current_stage == 'generate_custom':
            self.generate_custom_design()
        elif st.session_state.current_stage == 'done':
            st.success("Thank you for using the AI Fashion Assistant! 🎉")
            if st.button("Restart"):
                st.session_state.clear()
                st.rerun()
        else:
            self.collect_user_info()
# -----------------------------
# 6) Main Function
# -----------------------------
def main():
    chatbot = FashionChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()