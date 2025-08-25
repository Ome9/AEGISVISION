import argparse
from pathlib import Path

from aegisvision.config import Config
from aegisvision.pipeline.process import process_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output_dir", default=None, help="Output directory for results")
    args = parser.parse_args()

    cfg = Config()
    annotated_path = process_video(args.video, cfg, args.output_dir)
    print(f"Annotated video saved to: {annotated_path}")
    out_dir = Path(args.output_dir or cfg.app.output_dir)
    print(f"Alerts JSON: {out_dir / cfg.app.alerts_json}")


if __name__ == "__main__":
    main()


