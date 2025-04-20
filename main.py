import pandas as pd
import requests
import streamlit as st
from models import SUPPORTED_MODELS, bytes_to_array, prepare_image
import base64
from PIL import Image
import io

# Page config with custom theme
st.set_page_config(
    page_title="SnapClassifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #FF4B4B;
    }
    .sub-header {
        font-size: 1.5rem;
        padding-top: 1rem;
        padding-bottom: 0.5rem;
        color: #4B4BFF;
    }
    .success-box {
        background-color: #D1F7C4;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28A745;
    }
    .model-info-box {
        background-color: #9E9E9E;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stDataFrame {
        border: 1px solid #E0E0E0;
        border-radius: 0.5rem;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #6C757D;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# App title with icon
st.markdown("<h1 class='main-header'>📸 SnapClassifier - AI Image Recognition</h1>", unsafe_allow_html=True)

# App description
st.markdown("""
SnapClassifier uses state-of-the-art deep learning models to identify objects in your images.
Upload an image or take a photo with your webcam to see the magic happen!
""")

# Dictionary of model architectures and descriptions
MODEL_INFO = {
    "VGG-16": {
        "architecture": "16-layer CNN with 13 convolutional layers and 3 fully connected layers",
        "parameters": "138 million parameters",
        "developed_by": "Visual Geometry Group, Oxford",
        "year": "2014",
        "description": "Known for its simplicity using only 3×3 convolutional layers stacked on top of each other. It was one of the first very deep networks that showed depth improves accuracy.",
        "accuracy": "Top-5 accuracy of 92.7% on ImageNet"
    },
    "MobileNet": {
        "architecture": "Depthwise separable convolutions with 28 layers",
        "parameters": "4.2 million parameters",
        "developed_by": "Google",
        "year": "2017",
        "description": "Designed for mobile and embedded vision applications. Uses depthwise separable convolutions to build lightweight deep neural networks.",
        "accuracy": "Top-5 accuracy of 89.5% on ImageNet"
    }
}

# Sidebar with upload options
with st.sidebar:
    st.title("📤 Upload An Image")
    
    upload_type = st.radio(
        label="Choose upload method:",
        options=("From file", "From URL", "From webcam"),
        key="upload_method"
    )
    
    image_bytes = None
    
    if upload_type == "From file":
        st.info("Supports PNG, JPG, and JPEG files")
        file = st.file_uploader(
            "Upload image file", 
            type=["png", "jpg", "jpeg"], 
            accept_multiple_files=False
        )
        if file:
            image_bytes = file.getvalue()
    
    if upload_type == "From URL":
        url = st.text_input("Paste image URL")
        if url:
            if url.startswith('data:'):
                # Handle data URLs
                try:
                    # Extract the base64 part after the comma
                    header, encoded = url.split(",", 1)
                    # Decode the base64 data
                    image_bytes = base64.b64decode(encoded)
                except Exception as e:
                    st.error(f"Error processing data URL: {e}")
            else:
                # Regular HTTP/HTTPS URL
                try:
                    image_bytes = requests.get(url).content
                except Exception as e:
                    st.error(f"Error fetching image: {e}")
    
    if upload_type == "From webcam":
        camera = st.camera_input("Take a picture!")
        if camera:
            image_bytes = camera.getvalue()
    
    # Show model information in sidebar
    st.markdown("---")
    st.markdown("### 🧠 About Our Models")
    
    tab1, tab2 = st.tabs(["VGG-16", "MobileNet"])
    
    with tab1:
        st.markdown(f"""
        **{MODEL_INFO['VGG-16']['architecture']}**
        
        Developed by: {MODEL_INFO['VGG-16']['developed_by']} ({MODEL_INFO['VGG-16']['year']})
        
        Parameters: {MODEL_INFO['VGG-16']['parameters']}
        
        {MODEL_INFO['VGG-16']['description']}
        """)
    
    with tab2:
        st.markdown(f"""
        **{MODEL_INFO['MobileNet']['architecture']}**
        
        Developed by: {MODEL_INFO['MobileNet']['developed_by']} ({MODEL_INFO['MobileNet']['year']})
        
        Parameters: {MODEL_INFO['MobileNet']['parameters']}
        
        {MODEL_INFO['MobileNet']['description']}
        """)

# Main content area
if image_bytes:
    st.markdown("<h2 class='sub-header'>Uploaded Image</h2>", unsafe_allow_html=True)
    
    # Create columns for image and image info
    img_col, info_col = st.columns([1, 2])
    
    with img_col:
        st.markdown("<div class='success-box'>🎉 Image successfully uploaded!</div>", unsafe_allow_html=True)
        st.image(image_bytes, width=300)
    
    with info_col:
        # Display image info
        try:
            img = Image.open(io.BytesIO(image_bytes))
            st.markdown("#### Image Information")
            st.markdown(f"""
            - **Format**: {img.format}
            - **Size**: {img.width} x {img.height} pixels
            - **Mode**: {img.mode}
            """)
        except:
            st.warning("Could not extract detailed image information")
    
    # Display prediction results
    st.markdown("<h2 class='sub-header'>Model Predictions</h2>", unsafe_allow_html=True)
    
    # Create columns for side-by-side model results
    columns = st.columns(2)
    
    # Run predictions for each model
    for column_index, model_name in enumerate(SUPPORTED_MODELS.keys()):
        with columns[column_index]:
            with st.spinner(f"Running {model_name} prediction..."):
                # Display model info card
                st.markdown(f"<div class='model-info-box'><strong>{model_name}</strong><br>{MODEL_INFO[model_name]['description']}</div>", unsafe_allow_html=True)
                
                # Run prediction
                load_model, preprocess_input, decode_predictions = SUPPORTED_MODELS[model_name].values()
                model = load_model()
                image_array = bytes_to_array(image_bytes)
                image_array = prepare_image(image_array, _model_preprocess=preprocess_input)
                prediction = model.predict(image_array)
                prediction_df = pd.DataFrame(decode_predictions(prediction, 5)[0])
                prediction_df.columns = ["label_id", "label", "probability"]
                
                # Format probability as percentage
                prediction_df["probability"] = prediction_df["probability"].apply(lambda x: f"{x*100:.2f}%")
                
                # Highlight top prediction
                top_prediction = prediction_df.iloc[0]["label"].replace("_", " ").title()
                top_confidence = prediction_df.iloc[0]["probability"]
                
                st.markdown(f"### Top prediction: **{top_prediction}**")
                st.markdown(f"Confidence: **{top_confidence}**")
                
                # Display full results table
                st.markdown("#### Top 5 Results:")
                st.dataframe(
                    prediction_df.sort_values(by="probability", ascending=False),
                    column_config={
                        "label": "Prediction",
                        "probability": "Confidence",
                        "label_id": None  # Hide label_id column
                    },
                    hide_index=True
                )
else:
    # Show instruction message when no image is uploaded
    st.info("👈 Please upload an image using the sidebar to get started")
    
    # Show example of what the app does
    st.markdown("### How SnapClassifier Works")
    st.markdown("""
    1. **Upload an image** through the sidebar using one of the three methods
    2. **View results** from two powerful deep learning models:
       - **VGG-16**: A deep CNN model with high accuracy
       - **MobileNet**: A lightweight model optimized for speed
    3. **Compare predictions** to see how different architectures perform
    """)
    
    # Stop further execution
    st.stop()

# Add footer
st.markdown("---")
st.markdown("<p class='disclaimer'>SnapClassifier uses pre-trained models on the ImageNet dataset with 1000 common object categories. Results may vary depending on image quality and content.</p>", unsafe_allow_html=True)