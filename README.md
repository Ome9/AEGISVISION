<<<<<<< HEAD
### AegisVision — AI-Powered Surveillance (Anomaly Detection)

This project detects behavioral anomalies in video feeds using a hybrid approach:
- **Object/Person detection** with YOLO
- **Video-level anomaly cues** with VideoMAE (UCF-Crime finetuned)
- **Simple tracking + heuristics** for loitering and object abandonment
- **Streamlit dashboard** to review alerts with timestamps and annotated videos

#### Quickstart
1) Create a Python 3.10+ virtual environment.
2) Install dependencies:
```bash
pip install -r requirements.txt
```
3) Run inference on a sample video:
```bash
python scripts/run_inference.py --video path/to/input.mp4 --output_dir output/sample_run
```
4) Launch dashboard and select the generated outputs:
```bash
streamlit run dashboard/app.py
```

#### What to do now (step-by-step)
1) Environment setup (once):
```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows PowerShell
pip install -r requirements.txt
```
2) End-to-end smoke test on any video:
```bash
python scripts/run_inference.py --video path\\to\\your_video.mp4 --output_dir output\\run1
```
Then open the dashboard to review alerts and the annotated video:
```bash
streamlit run dashboard/app.py
```
3) Evaluate on Avenue (requires GT in MATLAB .m):
```bash
python scripts/convert_avenue_m_to_npy.py --m_file path\\to\\avenue_gt.m --videos_dir path\\to\\avenue_videos --output avenue_map.npy
python scripts/eval_avenue.py --videos_dir path\\to\\avenue_videos --gt_map avenue_map.npy --clip_len 16 --stride 8
```
4) Evaluate on UCSD (frame folders + .m GT):
```bash
python scripts/convert_ucsd_m_to_npy.py --m_file path\\to\\Peds2\\TestFrameGT.m --test_root path\\to\\Peds2\\Test --output ucsd_peds2_map.npy
python scripts/eval_ucsd_frames.py --test_root path\\to\\Peds2\\Test --gt_map ucsd_peds2_map.npy --clip_len 16 --stride 8
```
5) Optional: Fine-tune VideoMAE on Avenue/UCSD manifests:
```bash
python scripts/finetune_videomae.py --train_manifest path\\to\\train.jsonl --val_manifest path\\to\\val.jsonl --output_dir checkpoints\\videomae_finetuned --epochs 5 --batch_size 2
```
6) Optional: Batch classify a folder with `model.py`:
```bash
# Put test videos under AegisVision\\sample_data or edit video_folder in model.py
python model.py
```
7) Prepare submission PDF (outline):
```bash
python scripts/export_outline_to_pdf.py --outline submission_outline.md --output submission_outline.pdf
```

#### Key Components
- `aegisvision/detection/yolo.py`: YOLO detector wrapper (Ultralytics)
- `aegisvision/models/videomae.py`: VideoMAE anomaly scorer (OPear/videomae-large-finetuned-UCF-Crime)
- `aegisvision/tracking/tracker.py`: Lightweight tracker maintaining IDs and dwell/stationary counts
- `aegisvision/analytics/anomaly.py`: Heuristics for loitering, abandonment, unusual movement
- `aegisvision/pipeline/process.py`: End-to-end processing and alert generation
- `dashboard/app.py`: Streamlit dashboard to review alerts and annotated videos
- `scripts/finetune_videomae.py`: Template to finetune on Avenue/UCSD datasets
- `scripts/export_outline_to_pdf.py`: Create PDF from the submission outline
 - `model.py`: Simple batch inference over a folder using VideoMAE
 - `scripts/eval_avenue.py`: Avenue evaluation sketch with frame-level ROC-AUC

#### Avenue / UCSD Evaluation
Prepare frame-level label maps:
```bash
# UCSD: convert .m to .npy map
python scripts/convert_ucsd_m_to_npy.py --m_file path/to/Peds2/TestFrameGT.m --test_root path/to/Peds2/Test --output ucsd_peds2_map.npy

# Avenue: create a .npy dict {video_stem: [0/1,...]} using your GT
```
Run evaluation (Avenue example):
```bash
python scripts/eval_avenue.py --videos_dir path/to/avenue_videos --gt_map path/to/avenue_map.npy --model_id OPear/videomae-large-finetuned-UCF-Crime --clip_len 16 --stride 8
```

#### Datasets (references)
- Avenue Dataset — reported in literature for anomaly detection
- UCSD Anomaly Detection Dataset — pedestrian scenarios

#### Notes
- GPU highly recommended for real-time or near real-time performance.
- You can adjust thresholds in `aegisvision/config.py`.


#### Avenue / UCSD Evaluation
UCSD GT conversion:
```bash
python scripts/convert_ucsd_m_to_npy.py --m_file path/to/Peds2/TestFrameGT.m --test_root path/to/Peds2/Test --output ucsd_peds2_map.npy
```
Evaluate on UCSD frame folders:
```bash
python scripts/eval_ucsd_frames.py --test_root path/to/Peds2/Test --gt_map ucsd_peds2_map.npy --clip_len 16 --stride 8
```
Evaluate on Avenue videos:
```bash
# Convert MATLAB GT to .npy
python scripts/convert_avenue_m_to_npy.py --m_file path/to/avenue_gt.m --videos_dir path/to/avenue_videos --output avenue_map.npy

# Evaluate
python scripts/eval_avenue.py --videos_dir path/to/avenue_videos --gt_map avenue_map.npy --clip_len 16 --stride 8
```

=======
# AEGISVISION
>>>>>>> 8cc3c77543351878d3e3d000886307d220a22af8
