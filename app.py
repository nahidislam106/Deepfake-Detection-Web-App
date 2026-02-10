import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import tempfile

# Page configuration
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .detection-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .real-box {
        background-color: transparent;
        border: 2px solid #28a745;
    }
    .fake-box {
        background-color: transparent;
        border: 2px solid #dc3545;
    }
    .metric-card {
        background-color: transparent;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

@st.cache_resource
def load_model(model_path):
    """Load the YOLO model"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def draw_boxes(image, results):
    """Draw bounding boxes on the image"""
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get class and confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names[cls]
            
            # Choose color based on class (red for fake, green for real)
            color = (0, 0, 255) if 'fake' in class_name.lower() else (0, 255, 0)
            
            # Draw rectangle
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name}: {conf:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_bgr, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Convert back to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def analyze_results(results):
    """Analyze detection results and return summary"""
    detections = []
    real_confidences = []
    fake_confidences = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names[cls]
            
            # Check if detection is RostroReal or RostroFalse
            is_fake = False
            if 'rostroreal' in class_name.lower():
                real_confidences.append(conf)
                is_fake = False
            elif 'rostrofalse' in class_name.lower() or 'rostrofalso' in class_name.lower() or 'fake' in class_name.lower() or 'deepfake' in class_name.lower():
                fake_confidences.append(conf)
                is_fake = True
            
            detections.append({
                'class': class_name,
                'confidence': conf,
                'is_fake': is_fake
            })
    
    # Determine if image is deepfake based on average confidence comparison
    avg_real = np.mean(real_confidences) if real_confidences else 0
    avg_fake = np.mean(fake_confidences) if fake_confidences else 0
    has_deepfake = avg_fake > avg_real
    
    return detections, has_deepfake

# Header
st.markdown('<div class="main-header">🔍 Deepfake Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload an image to detect potential deepfakes using YOLOv11</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model path
    model_path = st.text_input(
        "Model Path",
        value="best.pt",
        help="Path to your trained YOLO model file"
    )
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    # Load model button
    if st.button("🔄 Load Model", type="primary"):
        if os.path.exists(model_path):
            with st.spinner("Loading model..."):
                st.session_state.model = load_model(model_path)
                if st.session_state.model is not None:
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                else:
                    st.session_state.model_loaded = False
                    st.error("❌ Failed to load model")
        else:
            st.error(f"❌ Model file not found: {model_path}")
    
    # Model status
    st.markdown("---")
    st.subheader("📊 Model Status")
    if st.session_state.model_loaded:
        st.success("✅ Model Ready")
        if st.session_state.model is not None:
            st.info(f"Classes: {', '.join(st.session_state.model.names.values())}")
    else:
        st.warning("⚠️ Model Not Loaded")
    
    # Info
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown("""
    This application uses a YOLOv11 model trained on the Roboflow deepfake dataset 
    to detect potential deepfakes in images.
    
    **How to use:**
    1. Load the model using the button above
    2. Upload an image or use your camera
    3. View detection results with bounding boxes
    """)

# Main content
if not st.session_state.model_loaded:
    st.info("👈 Please load the model from the sidebar to begin")
else:
    # File upload options
    upload_tab, camera_tab, sample_tab = st.tabs(["📁 Upload Image", "📷 Camera", "🖼️ Sample Images"])
    
    uploaded_image = None
    
    with upload_tab:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            help="Upload an image to analyze for deepfakes"
        )
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
    
    with camera_tab:
        camera_photo = st.camera_input("Take a picture")
        if camera_photo is not None:
            uploaded_image = Image.open(camera_photo)
    
    with sample_tab:
        st.info("You can add sample images in a 'samples' folder for quick testing")
        samples_dir = "samples"
        if os.path.exists(samples_dir):
            sample_files = [f for f in os.listdir(samples_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if sample_files:
                selected_sample = st.selectbox("Select a sample image", sample_files)
                if st.button("Load Sample"):
                    uploaded_image = Image.open(os.path.join(samples_dir, selected_sample))
            else:
                st.warning("No sample images found in the samples folder")
        else:
            st.warning("Create a 'samples' folder and add images for quick testing")
    
    # Process image
    if uploaded_image is not None:
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Original Image")
            st.image(uploaded_image, use_container_width=True)
        
        # Run detection
        with st.spinner("🔍 Analyzing image..."):
            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                uploaded_image.save(tmp_file.name)
                temp_path = tmp_file.name
            
            # Run inference
            results = st.session_state.model.predict(
                source=temp_path,
                conf=confidence_threshold,
                verbose=False
            )
            
            # Clean up temp file
            os.unlink(temp_path)
            
            # Analyze results
            detections, has_deepfake = analyze_results(results)
            
            # Draw boxes on image
            annotated_image = draw_boxes(uploaded_image, results)
        
        with col2:
            st.subheader("🎯 Detection Results")
            st.image(annotated_image, use_container_width=True)
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Analysis Report")
        
        # Overall verdict
        if len(detections) == 0:
            st.info("ℹ️ No detections found. The image appears clean with the current confidence threshold.")
        else:
            # Count RostroReal vs RostroFalse
            real_count = sum(1 for d in detections if not d['is_fake'])
            fake_count = sum(1 for d in detections if d['is_fake'])
            
            # Calculate average confidence for each type
            real_confidences = [d['confidence'] for d in detections if not d['is_fake']]
            fake_confidences = [d['confidence'] for d in detections if d['is_fake']]
            
            avg_real_conf = np.mean(real_confidences) if real_confidences else 0
            avg_fake_conf = np.mean(fake_confidences) if fake_confidences else 0
            
            # Decision based on average confidence instead of count
            if avg_fake_conf > avg_real_conf:
                st.markdown(
                    '<div class="detection-box fake-box">'
                    '<h3>⚠️ AI GENERATED / DEEPFAKE</h3>'
                    f'<p>Average RostroFalse confidence ({avg_fake_conf:.2%}) is higher than RostroReal confidence ({avg_real_conf:.2%}).</p>'
                    f'<p><small>Detections: {fake_count} RostroFalse, {real_count} RostroReal</small></p>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="detection-box real-box">'
                    '<h3>✅ REAL IMAGE</h3>'
                    f'<p>Average RostroReal confidence ({avg_real_conf:.2%}) is higher than RostroFalse confidence ({avg_fake_conf:.2%}).</p>'
                    f'<p><small>Detections: {real_count} RostroReal, {fake_count} RostroFalse</small></p>'
                    '</div>',
                    unsafe_allow_html=True
                )
            
            # Detailed detections
            st.subheader("🔍 Detailed Detections")
            
            cols = st.columns(3)
            for idx, detection in enumerate(detections):
                col_idx = idx % 3
                with cols[col_idx]:
                    icon = "🚨" if detection['is_fake'] else "✅"
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<h4>{icon} {detection["class"]}</h4>'
                        f'<p>Confidence: <strong>{detection["confidence"]:.2%}</strong></p>'
                        '</div>',
                        unsafe_allow_html=True
                    )
            
            # Statistics
            st.markdown("---")
            st.subheader("📈 Statistics")
            stat_cols = st.columns(4)
            
            with stat_cols[0]:
                st.metric("Total Detections", len(detections))
            
            with stat_cols[1]:
                fake_count = sum(1 for d in detections if d['is_fake'])
                st.metric("Deepfake Detections", fake_count)
            
            with stat_cols[2]:
                avg_conf = np.mean([d['confidence'] for d in detections])
                st.metric("Average Confidence", f"{avg_conf:.2%}")
            
            with stat_cols[3]:
                max_conf = max([d['confidence'] for d in detections])
                st.metric("Max Confidence", f"{max_conf:.2%}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; padding: 1rem;'>"
    "Deepfake Detection System | Powered by YOLOv11 & Streamlit | 2024"
    "</div>",
    unsafe_allow_html=True
)
