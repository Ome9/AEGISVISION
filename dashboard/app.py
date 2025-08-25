import json
import cv2
import torch
import numpy as np
from pathlib import Path
import streamlit as st
import tempfile
import os
import sys
from datetime import datetime
import pandas as pd
import time
from pathlib import Path


# Add aegisvision to path
sys.path.append(str(Path(__file__).parent))

# Import your existing modules
try:
    from aegisvision.models.videomae import VideoMAEDetector
except ImportError:
    # Fallback VideoMAE implementation if your module doesn't exist
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    
    class VideoMAEDetector:
        def __init__(self, model_path, confidence_threshold=0.5):
            self.model_path = model_path
            self.confidence_threshold = confidence_threshold
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load model and processor
            self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
            self.model = VideoMAEForVideoClassification.from_pretrained(
                str(Path(model_path).resolve()),   # ensures absolute path
                local_files_only=True
            )
            self.model.to(self.device)
            self.model.eval()
            
            self.class_names = ['Normal', 'Anomaly']
        
        def preprocess_video_clip(self, frames):
            """Preprocess video frames for VideoMAE"""
            processed_frames = []
            for frame in frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                processed_frames.append(frame_resized)
            
            inputs = self.processor(processed_frames, return_tensors="pt")
            return inputs['pixel_values'].to(self.device)
        
        def predict_clip(self, frames):
            """Predict anomaly for a video clip"""
            if len(frames) < 16:
                return 0, 0.0
            
            indices = np.linspace(0, len(frames) - 1, 16, dtype=int)
            sampled_frames = [frames[i] for i in indices]
            
            inputs = self.preprocess_video_clip(sampled_frames)
            
            with torch.no_grad():
                outputs = self.model(inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence = probabilities.max().item()
                predicted_class = probabilities.argmax().item()
            
            return predicted_class, confidence

try:
    from aegisvision.analytics.anomaly import AnomalyAnalyzer
except ImportError:
    class AnomalyAnalyzer:
        def __init__(self):
            pass
        def analyze_timeline(self, alerts):
            return {"total": len(alerts), "high_confidence": len([a for a in alerts if float(a['confidence']) > 0.8])}

def extract_frames_from_video(video_path, max_frames=1000):
    """Extract frames from video file"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break
        frames.append(frame)
        frame_count += 1
    
    cap.release()
    return frames

def process_video_with_videomae(video_path, model_detector, clip_length=32, stride=16):
    """Process entire video with VideoMAE model"""
    frames = extract_frames_from_video(video_path)
    
    alerts = []
    annotated_frames = []
    
    total_clips = max(1, (len(frames) - clip_length) // stride + 1)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(frames) - clip_length + 1, stride):
        clip_frames = frames[i:i + clip_length]
        predicted_class, confidence = model_detector.predict_clip(clip_frames)
        
        progress = min(1.0, (i // stride + 1) / total_clips)
        progress_bar.progress(progress)
        status_text.text(f"Processing clip {i//stride + 1}/{total_clips}")
        
        middle_frame = clip_frames[len(clip_frames)//2].copy()
        
        if predicted_class == 1 and confidence > model_detector.confidence_threshold:
            timestamp = i / 30.0
            
            alert = {
                'timestamp': f"{timestamp:.2f}s",
                'frame_number': i + clip_length // 2,
                'confidence': f"{confidence:.3f}",
                'type': 'Anomaly Detected',
                'description': f'Suspicious activity detected with {confidence:.1%} confidence',
                'location': 'Camera_01'  # You can enhance this
            }
            alerts.append(alert)
            
            cv2.rectangle(middle_frame, (10, 10), (middle_frame.shape[1]-10, middle_frame.shape[0]-10), (0, 0, 255), 3)
            cv2.putText(middle_frame, f'ANOMALY: {confidence:.2f}', (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.rectangle(middle_frame, (10, 10), (middle_frame.shape[1]-10, middle_frame.shape[0]-10), (0, 255, 0), 2)
            cv2.putText(middle_frame, f'NORMAL: {confidence:.2f}', (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        annotated_frames.append(middle_frame)
    
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    return alerts, annotated_frames

def create_annotated_video(annotated_frames, output_path, fps=10):
    """Create annotated video from frames"""
    if not annotated_frames:
        return False
    
    height, width, _ = annotated_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame in annotated_frames:
        out.write(frame)
    
    out.release()
    return True

# Streamlit Dashboard
st.set_page_config(page_title="AegisVision Dashboard", layout="wide", page_icon="🛡️")

# Custom CSS
st.markdown("""
<style>
/* Global */
html, body, [class^="block-container"] {
    background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
}
.main-header {
    text-align: center;
    color: #e5f2ff;
    font-size: 3rem;
    margin-bottom: 1.25rem;
    letter-spacing: 0.5px;
}
.subheader-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.15);
    border: 1px solid rgba(56, 189, 248, 0.35);
    color: #7dd3fc;
    font-weight: 600;
}

/* Cards */
.card {
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(6px);
    padding: 1rem 1.25rem;
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    margin-bottom: 0.75rem;
}
.alert-box {
    background: linear-gradient(135deg, #ef4444, #b91c1c);
    color: white;
    padding: 0.9rem 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 10px 25px rgba(239, 68, 68, 0.25);
}
.normal-box {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: #02150a;
    padding: 0.9rem 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    border: 1px solid rgba(0,0,0,0.1);
    box-shadow: 0 10px 25px rgba(34, 197, 94, 0.25);
}
.metric-container {
    background: linear-gradient(135deg, rgba(30,41,59,0.7), rgba(15,23,42,0.65));
    color: #e5e7eb;
    padding: 1rem 1.25rem;
    border-radius: 12px;
    border: 1px solid rgba(148,163,184,0.2);
}

/* Status dots */
.status-indicator { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
.status-online { background-color: #22c55e; box-shadow: 0 0 0 3px rgba(34,197,94,0.25); }
.status-processing { background-color: #f59e0b; box-shadow: 0 0 0 3px rgba(245,158,11,0.25); }
.status-offline { background-color: #ef4444; box-shadow: 0 0 0 3px rgba(239,68,68,0.25); }

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #2563eb) !important;
    color: white !important;
    border: 0 !important;
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
    font-weight: 600 !important;
    box-shadow: 0 10px 25px rgba(14,165,233,0.35) !important;
}
div.stButton > button:hover { filter: brightness(1.05); }

/* Tabs */
div[data-baseweb="tab-list"] { gap: 8px; }
div[role="tab"] {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    color: #cbd5e1;
    border: 1px solid rgba(255,255,255,0.08);
}
div[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(14,165,233,0.25), rgba(37,99,235,0.25));
    color: #e5f2ff;
    border: 1px solid rgba(56,189,248,0.35);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">AegisVision — AI Surveillance System</h1>', unsafe_allow_html=True)
st.markdown('<span class="subheader-badge">Powered by VideoMAE • 83.59% Accuracy</span>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title(" System Configuration")

# Model settings
st.sidebar.markdown("####  Model Settings")
model_path = st.sidebar.text_input("VideoMAE Model Path", value="D:/AegisVision/checkpoints/videomae_finetuned_proper/checkpoint-1516")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Output settings
st.sidebar.markdown("####  Output Settings")
output_dir = st.sidebar.text_input("Output Directory", value="output")

# Video settings
st.sidebar.markdown("####  Video Processing")
clip_length = st.sidebar.slider("Clip Length (frames)", 16, 64, 32)
stride = st.sidebar.slider("Processing Stride", 8, 32, 16)

# Create output directory
out_dir = Path(output_dir)
out_dir.mkdir(exist_ok=True)

# File upload
st.sidebar.markdown("#### Video Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload Surveillance Video", 
    type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
    help="Supported formats: MP4, AVI, MOV, MKV, WMV"
)

## Synthetic handled in a dedicated tab below




# System status
st.sidebar.markdown("---")
st.sidebar.markdown("#### System Status")
device_type = "CUDA" if torch.cuda.is_available() else "CPU"
st.sidebar.markdown(f'<span class="status-indicator status-online"></span>VideoMAE Model: Ready', unsafe_allow_html=True)
st.sidebar.markdown(f'<span class="status-indicator status-online"></span>Device: {device_type}', unsafe_allow_html=True)
st.sidebar.markdown(f'<span class="status-indicator status-online"></span>Dashboard: Online', unsafe_allow_html=True)
st.sidebar.markdown(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")

# Tabbed experience
overview_tab, analyze_tab, alerts_tab, results_tab, synthetic_tab = st.tabs(["Overview", "Analyze", "Alerts", "Results", "Synthetic"])

with overview_tab:
    st.markdown("""
    <div class="card" style="text-align: center;">
        <h3 style="margin: 0 0 8px 0; color: #e5f2ff;">Welcome to AegisVision</h3>
        <p style="font-size: 1rem; color: #cbd5e1; margin: 0">Upload a surveillance video from the sidebar to begin analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card">
            <h4 style="margin: 0 0 6px 0; color: #e2e8f0;">AI-Powered Detection</h4>
            <ul style="margin: 0; padding-left: 18px; color: #94a3b8;">
                <li>VideoMAE transformer model</li>
                <li>83.59% accuracy rate</li>
                <li>Real-time processing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
            <h4 style="margin: 0 0 6px 0; color: #e2e8f0;">Smart Analytics</h4>
            <ul style="margin: 0; padding-left: 18px; color: #94a3b8;">
                <li>Confidence scoring</li>
                <li>Timestamp tracking</li>
                <li>Alert classification</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card">
            <h4 style="margin: 0 0 6px 0; color: #e2e8f0;">Video Processing</h4>
            <ul style="margin: 0; padding-left: 18px; color: #94a3b8;">
                <li>Multiple format support</li>
                <li>Frame-by-frame analysis</li>
                <li>Annotated output</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with analyze_tab:
    if uploaded_file is None:
        st.info("Upload a video from the sidebar to analyze.")
    else:
        col1, col2 = st.columns([1.3, 0.7])
        with col1:
            st.subheader("📹 Video Analysis")
            st.video(uploaded_file)
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
                "File type": uploaded_file.type
            }
            with st.expander(" Video Information"):
                for key, value in file_details.items():
                    st.write(f"**{key}:** {value}")
            if st.button(" Start Anomaly Detection", use_container_width=True):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    video_path = tmp_file.name
                try:
                    with st.spinner(" Loading VideoMAE model..."):
                        detector = VideoMAEDetector(model_path, confidence_threshold)
                    st.success(" VideoMAE model loaded successfully!")
                    with st.spinner(" Analyzing video for anomalies..."):
                        alerts, annotated_frames = process_video_with_videomae(
                            video_path, detector, clip_length, stride
                        )
                    alerts_path = out_dir / "alerts.json"
                    with open(alerts_path, 'w') as f:
                        json.dump(alerts, f, indent=2)
                    annotated_video_path = out_dir / "annotated.mp4"
                    if create_annotated_video(annotated_frames, annotated_video_path):
                        st.success(f" Analysis complete! Found {len(alerts)} anomalies")
                        if alerts:
                            avg_confidence = np.mean([float(alert['confidence']) for alert in alerts])
                            st.info(f"Average anomaly confidence: {avg_confidence:.3f}")
                    else:
                        st.error("Failed to create annotated video")
                    os.unlink(video_path)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.expander("Error Details").write(str(e))
        with col2:
            st.subheader("Alert Snapshot")
            alerts_path = out_dir / "alerts.json"
            if alerts_path.exists():
                alerts = json.loads(alerts_path.read_text(encoding="utf-8"))
                c21, c22 = st.columns(2)
                with c21:
                    st.metric("Total Alerts", len(alerts))
                with c22:
                    high_conf = len([a for a in alerts if float(a['confidence']) > 0.8]) if alerts else 0
                    st.metric("High Confidence", high_conf)
                if alerts:
                    st.markdown("#### Latest Anomalies")
                    for i, alert in enumerate(alerts[-3:], 1):
                        st.markdown(f"""
                        <div class=\"alert-box\">
                            <strong>Alert #{i}</strong><br>
                            Time: {alert['timestamp']}<br>
                            Confidence: {alert['confidence']}<br>
                            {alert['description']}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class=\"normal-box\">No anomalies detected</div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No alerts yet. Run analysis to see results.")

with alerts_tab:
    alerts_path = out_dir / "alerts.json"
    if alerts_path.exists():
        alerts = json.loads(alerts_path.read_text(encoding="utf-8"))
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Alerts", len(alerts))
        with c2:
            high_conf = len([a for a in alerts if float(a['confidence']) > 0.8]) if alerts else 0
            st.metric("High Confidence", high_conf)
        with c3:
            status = "Active" if alerts else "All Clear"
            st.metric("Status", status)
        st.markdown("#### Recent Anomalies")
        if alerts:
            for i, alert in enumerate(alerts[::-1][:10], 1):
                st.markdown(f"""
                <div class=\"card\">
                    <div style=\"display:flex;justify-content:space-between;align-items:center;\">
                        <div style=\"color:#e2e8f0; font-weight:600;\">Alert #{i}</div>
                        <div class=\"subheader-badge\">{alert['timestamp']}</div>
                    </div>
                    <div style=\"margin-top:6px;color:#cbd5e1;\">{alert['description']}</div>
                    <div style=\"margin-top:6px;color:#93c5fd;\">Confidence: {alert['confidence']}</div>
                </div>
                """, unsafe_allow_html=True)
        alerts_json = json.dumps(alerts, indent=2)
        st.download_button(
            "Download Alerts JSON",
            alerts_json,
            "aegisvision_alerts.json",
            "application/json",
            use_container_width=True
        )
    else:
        st.info("No alerts yet. Run analysis to populate this section.")

with results_tab:
    annotated_video_path = out_dir / "annotated.mp4"
    if annotated_video_path.exists():
        col_video, col_download = st.columns([3, 1])
        with col_video:
            st.video(str(annotated_video_path))
        with col_download:
            st.markdown("#### Downloads")
            with open(annotated_video_path, "rb") as file:
                st.download_button(
                    "Download Video",
                    data=file,
                    file_name="aegisvision_analysis.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
            alerts_path = out_dir / "alerts.json"
            if alerts_path.exists():
                st.download_button(
                    "Download Report",
                    data=alerts_path.read_text(),
                    file_name="aegisvision_report.json",
                    mime="application/json",
                    use_container_width=True
                )
    else:
        st.info("Processed video results will appear here after analysis.")

with synthetic_tab:
    if uploaded_file is None:
        st.info("Upload a video from the sidebar to generate synthetic variants.")
    else:
        st.markdown("Generate a synthetic variant of your uploaded video for testing.")
        if st.button(" Generate Synthetic Video", use_container_width=True):
            st.info("Generating enhanced synthetic video...")
            synthetic_out_dir = Path("synthetic_output")
            synthetic_out_dir.mkdir(exist_ok=True)
            synthetic_video_path = synthetic_out_dir / "enhanced_synthetic.mp4"
            synthetic_alerts_path = synthetic_out_dir / "alerts_synthetic.json"

            # Save uploaded video temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.read())
                uploaded_path = tmp.name

            # Generate synthetic video using RIFE
            from synthetic_generator import generate_synthetic_video
            frames, alerts = generate_synthetic_video(uploaded_path, synthetic_video_path)

            # Save alerts (empty in RIFE case)
            with open(synthetic_alerts_path, "w") as f:
                json.dump(alerts, f, indent=2)

            st.success("Synthetic video generated!")
            st.video(str(synthetic_video_path))  # view directly in app
            st.download_button(
                "Download Synthetic Video",
                data=open(synthetic_video_path, "rb"),
                file_name="enhanced_synthetic.mp4",
                use_container_width=True
            )
            st.download_button(
                "Download Alerts JSON",
                data=open(synthetic_alerts_path, "rb"),
                file_name="synthetic_alerts.json",
                use_container_width=True
            )

# Results are now presented in the Results tab above

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🛡️ <strong>AegisVision</strong> | Powered by VideoMAE | Built with Streamlit</p>
    <p>For technical support, check the logs or contact the development team</p>
</div>
""", unsafe_allow_html=True)