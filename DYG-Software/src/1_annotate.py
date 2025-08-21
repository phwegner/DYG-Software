#!/usr/bin/env python3
import argparse
import subprocess
import logging
from pathlib import Path
import torch

device = "0" if torch.cuda.is_available() else "cpu"
from configparser import ConfigParser

def main():
    # -------------------
    # Script directory and config setup
    # -------------------
    SCRIPT_DIR = Path(__file__).parent.resolve()
    config_path = SCRIPT_DIR / ".." / "config.ini"

    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        return

    config = ConfigParser()
    config.read(config_path)

    # -------------------
    # CLI arguments
    # -------------------
    parser = argparse.ArgumentParser(description="Run YOLO pose prediction on all videos in a folder.")
    parser.add_argument("--video_folder", required=False, default=Path(SCRIPT_DIR / config['PATHS']['default_input_path']), help="Path to the folder containing video files")
    parser.add_argument("--project", required=False, default=Path(SCRIPT_DIR / config['PATHS']['default_project']), help="Project name for YOLO output")
    parser.add_argument("--device", default=device, help="Device to use for YOLO processing (default: auto-detected GPU or CPU)")
    # parser.add_argument("--yolo_model", default="../model_files", help="Path to YOLO models (default: ../model_files)")
    args = parser.parse_args()


    video_folder = Path(args.video_folder)
    project = args.project
    

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

        # --- Check video validity with ffmpeg ---
        # check_cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"]
        # result = subprocess.run(check_cmd, capture_output=True, text=True)

        # if result.returncode != 0 or result.stderr:
        #     logging.warning(f"Video {video.name} seems corrupted, attempting repair...")
        #     fixed_video = video.with_name(f"{video.stem}_fixed{video.suffix}")
        #     repair_cmd = [
        #         "ffmpeg", "-y", "-i", str(video),
        #         "-c", "copy", "-movflags", "faststart",
        #         str(fixed_video)
        #     ]
        #     try:
        #         subprocess.run(repair_cmd, check=True)
        #         logging.info(f"Repaired video saved as {fixed_video.name}")
        #         video = fixed_video  # use repaired video for YOLO
        #     except subprocess.CalledProcessError as e:
        #         logging.error(f"Repair failed for {video.name}, skipping...")
        #         continue

        if args.device:
            device = args.device
        if device == "0":
            logging.info("Using GPU for YOLO processing")
        else:
            logging.info("Using CPU for YOLO processing")
        cmd = [
            "yolo", "pose", "predict",
            f"model={model_path}",
            f"source={video}",
            "save_txt=True",
            f"project={project}",
            f"name={video_id}",
            f"device={device}",
        ]

        try:
            subprocess.run(cmd, check=True)
            logging.info(f"Finished processing {video.name}")
        except subprocess.CalledProcessError as e:
            logging.error(f"YOLO command failed for {video.name}: {e}")

if __name__ == "__main__":
    main()
