#!/usr/bin/env python3
import argparse
import subprocess
import logging
from pathlib import Path

from configparser import ConfigParser

def main():
    # -------------------
    # CLI arguments
    # -------------------
    parser = argparse.ArgumentParser(description="Run YOLO pose prediction on all videos in a folder.")
    parser.add_argument("--video_folder", required=True, help="Path to the folder containing video files")
    parser.add_argument("--project", required=True, help="Project name for YOLO output")
    # parser.add_argument("--yolo_model", default="../model_files", help="Path to YOLO models (default: ../model_files)")
    args = parser.parse_args()

    ## Confis 

    config = ConfigParser()
    

    video_folder = Path(args.video_folder)
    project = args.project
    SCRIPT_DIR = Path(__file__).resolve().parent

    config.read(SCRIPT_DIR / '../config.ini')

    # Read config value (could be relative, e.g. "../model_files")
    yolo_model_rel = config['PATHS']['yolo_model_path']

    # Make it absolute relative to script location
    yolo_model = (SCRIPT_DIR / yolo_model_rel).resolve()

    # -------------------
    # Logging setup
    # -------------------
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO
    )

    if not video_folder.exists() or not video_folder.is_dir():
        logging.error(f"Video folder does not exist: {video_folder}")
        return

    model_path = Path(yolo_model / "yolo11x-pose.pt")
    if not model_path.exists():
        logging.error(f"YOLO model not found at {model_path}")
        return

    logging.info(f"Using YOLO model: {model_path}")
    logging.info(f"Saving results to project: {project}")

    # -------------------
    # Loop over videos
    # -------------------
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}  # extend if needed
    videos = [f for f in video_folder.iterdir() if f.suffix.lower() in video_extensions]

    if not videos:
        logging.warning("No video files found in the provided folder.")
        return

    for video in videos:
        video_id = video.stem  # filename without extension
        logging.info(f"Processing video: {video.name}")

        cmd = [
            "yolo", "pose", "predict",
            f"model={model_path}",
            f"source={video}",
            "save_txt=True",
            f"project={project}",
            f"name={video_id}"
        ]

        try:
            subprocess.run(cmd, check=True)
            logging.info(f"Finished processing {video.name}")
        except subprocess.CalledProcessError as e:
            logging.error(f"YOLO command failed for {video.name}: {e}")

if __name__ == "__main__":
    main()
