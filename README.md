# 🔍 Deepfake Detection Web App

A web application built with Streamlit that uses YOLOv11 to detect deepfakes in images.

## Features

- 🖼️ **Image Upload**: Upload images from your device
- 📷 **Camera Support**: Take photos directly from your webcam
- 🎯 **Real-time Detection**: Instant deepfake detection with bounding boxes
- 📊 **Detailed Analysis**: View confidence scores and detection statistics
- ⚙️ **Adjustable Settings**: Configure confidence threshold on the fly

## Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Model File**
   Make sure your `best.pt` model file is in the same directory as `app.py`, or update the path in the sidebar.

## Usage

1. **Run the Application**
   ```bash
   streamlit run app.py
   ```

2. **Load the Model**
   - Click "🔄 Load Model" in the sidebar
   - Wait for the success message

3. **Upload an Image**
   - Use the "📁 Upload Image" tab to upload an image
   - Or use the "📷 Camera" tab to take a photo
   - Or add sample images to a `samples` folder for quick testing

4. **View Results**
   - See the original and annotated images side by side
   - Check the analysis report for deepfake detection
   - Review detailed detection statistics

## Configuration

### Confidence Threshold
Adjust the confidence threshold slider in the sidebar to control detection sensitivity:
- **Higher values** (0.5-1.0): More strict, fewer false positives
- **Lower values** (0.1-0.4): More sensitive, may catch subtle deepfakes

### Model Path
If your model is in a different location, update the model path in the sidebar.

## Project Structure

```
Deepfake/
├── app.py                # Main Streamlit application
├── best.pt              # Trained YOLOv11 model
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── Notebook/
│   └── demo.ipynb      # Training notebook
└── samples/            # (Optional) Sample images for testing
```

## Adding Sample Images

Create a `samples` folder and add test images:
```bash
mkdir samples
# Copy test images to samples folder
```

## Model Information

- **Framework**: YOLOv11
- **Dataset**: Roboflow deepfake dataset
- **Task**: Object Detection for Deepfake Identification
- **Model File**: best.pt

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)
- Webcam (optional, for camera feature)

## Troubleshooting

### Model Won't Load
- Verify the `best.pt` file exists in the specified path
- Check that the file is a valid YOLOv11 model

### Slow Inference
- The first prediction may be slower as the model initializes
- Consider using a smaller image size
- GPU acceleration will significantly improve speed

### No Detections
- Lower the confidence threshold
- Ensure the image contains faces or relevant content
- Verify the model was trained on similar data

## Tips

- Use clear, high-resolution images for best results
- Adjust the confidence threshold based on your use case
- The model works best on images similar to its training data

## License

This project uses:
- **Streamlit**: Apache License 2.0
- **Ultralytics YOLOv11**: AGPL-3.0 License

## Credits

- Model trained using [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- Dataset from [Roboflow](https://roboflow.com/)
- Web interface built with [Streamlit](https://streamlit.io/)
