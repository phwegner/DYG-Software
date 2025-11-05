import os
import argparse
import logging
from tqdm import tqdm
from _base import * 



data = []
sequences_raw=[]

keypoints = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16
    }

keypoints_revert = {v: k for k, v in keypoints.items()}

absolute_metrics = [
                    'x-pos_center_shoulder',
                    'x-pos_center_hip',
                    'x-pos_left_ankle',
                    'x-pos_right_ankle',
                    'x-pos_right_wrist',
                    'x-pos_left_wrist',
                    'x-pos_left_hip',
                    'x-pos_right_hip',
                    'x-pos_left_shoulder',
                    'x-pos_right_shoulder',
                    'x-pos_nose',
                    'x-pos_left_eye',
                    'x-pos_right_eye',
                    'x-pos_left_ear',
                    'x-pos_right_ear',
                    'x-pos_left_elbow',
                    'x-pos_right_elbow',
                    'x-pos_left_knee',
                    'x-pos_right_knee',
]

test_marker = 'dist_left_wrist-right_wrist'

def process_video(data, dir, video, pid=None, subsample=False):

    output = YOLOOutput(f"{dir}{video}")

    sequence_precleaned = output.coco_sequence.copy()



    if subsample:
        output.subsample_sequence(2, replace=True)
    output.clean_sequence(replace=True)
    
    output.impute_sequence_linear_interpolation(replace=True)
    output.smooth_sequence(11, 3, replace=True)
    # sequences_raw.append((output.coco_sequence, output.pid if not pid else pid))
    # Persist the raw COCO sequence to a CSV in the subject folder instead of
    # storing it in the in-memory `sequences_raw` list.
    try:
        from pathlib import Path
        video_id = Path(video).stem
        subject_dir = Path(dir) / video_id
        subject_dir.mkdir(parents=True, exist_ok=True)

        raw_path = subject_dir / "raw_coco_sequence.csv"
        try:
            df_raw = pd.DataFrame(output.coco_sequence)
            df_raw.columns = [keypoints_revert.get(k, k) for k in df_raw.columns]
        except Exception:
            # Fallback: try converting an iterable of mappings to DataFrame
            try:
                df_raw = pd.DataFrame(list(output.coco_sequence))
                
            except Exception:
                df_raw = None

        if df_raw is not None:
            df_raw.to_csv(raw_path, index=False)
            sequence_precleaned = pd.DataFrame(sequence_precleaned)
            sequence_precleaned.columns = [keypoints_revert.get(k, k) for k in sequence_precleaned.columns]
            sequence_precleaned.to_csv(subject_dir / "raw_coco_sequence_precleaned.csv", index=False)
            logging.info(f"Saved raw coco sequence to {raw_path}")
        else:
            logging.warning(f"Could not convert raw coco sequence to DataFrame for {video}")
    except Exception:
        logging.exception(f"Failed to save raw sequence for {video}")
    # output.impute_sequence(replace=True)
    output.to_pandas(replace=True)
    # output.sanity_check()
    x_pos = output.extract_x_positions()
    dist = output.extract_distances()
    angle = output.extract_angles()
    area = output.extract_areas()
    diffs = output.calc_angle_differences(angle)
    centers = output.calculate_centerpoints()

    joint_df = pd.concat([x_pos, dist, angle, area, diffs, centers], axis=1)
    
    # joint_df = output.standard_normal_scale(joint_df, std=True)

    joint_df = output.norm_to_left(joint_df, absolute_metrics)

    # Save joint dataframe to CSV inside a subject-specific folder named after the video stem
    try:
        from pathlib import Path
        video_id = Path(video).stem
        subject_dir = Path(dir) / video_id
        subject_dir.mkdir(parents=True, exist_ok=True)
        csv_path = subject_dir / "joint_df.csv"
        joint_df.to_csv(csv_path, index=False)
        logging.info(f"Saved joint dataframe to {csv_path}")

        # Sanity-check plot of the requested test marker using seaborn
        try:
            try:
                import seaborn as sns
                import matplotlib.pyplot as plt
            except Exception:
                logging.warning("seaborn not available; falling back to matplotlib for plotting")
                import matplotlib.pyplot as plt
                sns = None

            if test_marker in joint_df.columns:
                plt.figure(figsize=(10, 4))
                if sns is not None:
                    sns.lineplot(data=joint_df[test_marker])
                else:
                    plt.plot(joint_df[test_marker].values)
                plt.title(f"{video_id}: {test_marker}")
                plt.xlabel("frame")
                plt.ylabel(test_marker)
                plot_fname = subject_dir / f"{test_marker}.png"
                # sanitize only the marker portion of the filename (keep subject_dir path)
                import re
                safe_marker = re.sub(r"[^\w\-_. ]", "_", test_marker)
                plot_fname = subject_dir / f"{safe_marker}.png"
                plt.tight_layout()
                plt.savefig(plot_fname, dpi=150, bbox_inches='tight')
                plt.close()
                logging.info(f"Saved test marker plot to {plot_fname}")
            else:
                logging.warning(f"Test marker '{test_marker}' not found in joint_df columns for {video}")
        except Exception:
            logging.exception(f"Failed to create/save plot for test marker {test_marker} for {video}")

    except Exception:
        logging.exception(f"Failed to save joint dataframe for {video}")

    # Previously we created MotionData and appended to `data`. That in-memory collection is
    # no longer used for persistence; data can still be inspected if needed, so we keep
    # creating MotionData but do not require storing it persistently here.
    try:
        motion_data = MotionData(joint_df, output.y, output.pid if not pid else pid)
        data.append(motion_data)
    except Exception:
        logging.debug("MotionData creation failed or is not available; continuing")

# %%/Users/wegnerp/Desktop/Projects/donate_your_gate/pose_estimation/

# Removed hard-coded PATH processing; make this a callable script

def main():
    parser = argparse.ArgumentParser(description="Extract time series from YOLO outputs in a directory")
    parser.add_argument("--path", required=True, help="Path to directory containing YOLO tagged output")
    args = parser.parse_args()

    # Basic logging configuration
    numeric_level = getattr(logging, "INFO", logging.INFO)
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=numeric_level)

    PATH = args.path
    logging.info(f"Processing path: {PATH}")

    if not os.path.exists(PATH) or not os.path.isdir(PATH):
        logging.error(f"Provided path is not a directory: {PATH}")
        return

    # Ensure PATH ends with a separator to match previous behaviour
    if not PATH.endswith(os.sep):
        PATH = PATH + os.sep

    directory = os.listdir(PATH)
    logging.info(f"Found {len(directory)} items in {PATH}")

    with tqdm(total=len(directory)) as pbar:
        for video in directory:
            if video.startswith("."):
                logging.debug(f"Skipping hidden/system file: {video}")
                pbar.update(1)
                continue

            logging.info(f"Processing video/file: {video}")
            try:
                process_video(data, PATH, video, subsample=False)
            except Exception:
                logging.exception(f"Failed to process {video}")
            pbar.update(1)

    logging.info("Processing complete")


if __name__ == "__main__":
    main()