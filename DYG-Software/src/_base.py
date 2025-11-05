# %%
from dataclasses import dataclass
from typing import List, Tuple
from os import path, listdir
import pandas as pd
import numpy as np
from itertools import combinations, permutations
import math
from scipy.signal import savgol_filter
import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_angle(x1, y1, x2, y2, x3, y3):
    # Vector v1 from (x1, y1) to (x2, y2)
    v1x = x2 - x1
    v1y = y2 - y1
    
    # Vector v2 from (x1, y1) to (x3, y3)
    v2x = x3 - x1
    v2y = y3 - y1
    
    # Dot product of v1 and v2
    dot_product = v1x * v2x + v1y * v2y
    
    # Magnitudes of v1 and v2
    mag_v1 = math.sqrt(v1x**2 + v1y**2)
    mag_v2 = math.sqrt(v2x**2 + v2y**2)
    
    # Calculate the cosine of the angle
    cos_theta = dot_product / (mag_v1 * mag_v2)
    
    # Ensure the value is within the valid range for arccos due to numerical precision
    cos_theta = max(-1, min(1, cos_theta))
    
    # Calculate the angle in radians
    angle_radians = math.acos(cos_theta)
    
    # Convert the angle to degrees
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees

def calculate_area(x1, y1, x2, y2, x3, y3, x4, y4):
    
    area = 0.5 * abs(
        x1*y2 + x2*y3 + x3*y4 + x4*y1 -
        (y1*x2 + y2*x3 + y3*x4 + y4*x1)
    )
    
    return area

def forward_transform(df, id_col="ID"):
    """
    Transforms a DataFrame by splitting tuple columns into separate x and y columns, handling None values.

    Args:
        df (pd.DataFrame): The original DataFrame with tuple coordinates.
        id_col (str): The name of the column to exclude from splitting.

    Returns:
        pd.DataFrame: The transformed DataFrame with x and y columns.
    """
    columns_to_split = [col for col in df.columns if col != id_col]

    for col in columns_to_split:
        # Replace None with a default tuple (e.g., (NaN, NaN))
        df[col] = df[col].apply(lambda x: x if isinstance(x, (tuple, list)) and len(x) == 2 else (np.nan, np.nan))

        # Split the tuples/lists into separate columns
        df[[f"{col}_x", f"{col}_y"]] = pd.DataFrame(df[col].tolist(), index=df.index)
    
    # Drop the original columns that were split
    df = df.drop(columns=columns_to_split)
    return df

def backward_transform(df, id_col="ID"):
    """
    Transforms a DataFrame by recombining x and y columns into tuple coordinates.
    
    Args:
        df (pd.DataFrame): The transformed DataFrame with x and y columns.
        id_col (str): The name of the column to exclude from transformation.
    
    Returns:
        pd.DataFrame: The original DataFrame with tuple coordinates.
    """
    # Identify the base column names by removing `_x` and `_y`
    x_columns = [col for col in df.columns if col.endswith("_x")]
    base_columns = [col[:-2] for col in x_columns]
    
    for base in base_columns:
        df[base] = list(zip(df[f"{base}_x"], df[f"{base}_y"]))
        df = df.drop(columns=[f"{base}_x", f"{base}_y"])

    return df


class YOLOOutput:

    PERSON = 0

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

    def __init__(self, path_to_output) -> None:
        '''
        Initialize the YOLOOutput object
        '''

        self.path = path_to_output
        self.pid = "".join(self.path.split("/")[-1].split(".")[:-1])
        self.coco_sequence = []
        self.y = 1
        self.is_pandas = False
        self.is_cleaned = False
        self.is_imputed = False
        self.is_smoothed = False

        if "HC" in self.pid or "ATX_BN" in self.path:
            self.y = 0

        for label in listdir(path.join(self.path, "labels")):
            if label.endswith(".txt"):
                
                frame = label.split("_")[-1].replace(".txt", "")
                # logging.info(f"Frame: {frame}")
                with open(path.join(self.path, "labels", label), "r") as f:
                    person_0 = list(map(float, f.readlines()[self.PERSON].split()))
                    parsed = self.parse_pose_data(person_0)
                    parsed['frame'] = frame
                    extracted_keypoints = parsed["keypoints"]
                    self.coco_sequence.append(([(extracted_keypoints[keypoint]["x"], extracted_keypoints[keypoint]["y"], extracted_keypoints[keypoint]["confidence"]) for keypoint in self.keypoints], frame))
        self.coco_sequence = sorted(self.coco_sequence, key=lambda x: int(x[1]))
        # logging.info(f"Sequence: {self.coco_sequence}")
        self.coco_sequence = [x[0] for x in self.coco_sequence]

        if len(self.coco_sequence) == 0:
            raise ValueError("No data found in the provided path")
        
    ## decorator to check if the data is empty 
    def check_data(func):
        def wrapper(self, *args, **kwargs):
            if self.coco_sequence == []:
                raise ValueError(f"No data found in the provided path at function: {func.__name__}")
            return func(self, *args, **kwargs)
        return wrapper

    def parse_pose_data(self, data):
        '''
        Parse the pose data from the YOLO output

        Args:
            data (list): List of floats representing the pose data

        Returns:
            dict: Dictionary containing the class ID, bounding box, and keypoints
        '''
        # Parse class ID
        class_id = data[0]

        bbox = {
            "x_min": data[1],
            "y_min": data[2],
            "width": data[3],
            "height": data[4]
        }
        # print(f"Bounding Box: {bbox}")

        # Parse keypoints
        keypoint_data = {}
        start_idx = 5  # Keypoints start after class ID and bounding box
        for name, idx in self.keypoints.items():
            keypoint_idx = start_idx + 3 * idx
            if keypoint_idx + 2 < len(data):  # Ensure valid indexing
                x, y, conf = data[keypoint_idx:keypoint_idx + 3]
                keypoint_data[name] = {"x": x, "y": y, "confidence": conf}
            else:
                keypoint_data[name] = {"x": 0, "y": 0, "confidence": 0}  # Handle missing data

        return {"class_id": class_id, "bbox": bbox, "keypoints": keypoint_data}
    
    @check_data
    def clean_sequence(self, replace=False) -> List[List[Tuple[float, float]]]:
        '''
        Replace all (0.0, 0.0) with None

        Args:
            replace (bool): Replace the current sequence with the cleaned sequence        
        '''
        if replace:
            self.coco_sequence = [[(x, y) if x != 0.0 and y != 0.0 else None for x, y in frame] for frame in self.coco_sequence]
            self.is_cleaned = True
            return self.coco_sequence
        return [[(x, y) if x != 0.0 and y != 0.0 else None for x, y in frame] for frame in self.coco_sequence]
    
    def smooth_sequence(self, window_size=60, poly_order=2, replace=False) -> List[List[Tuple[float, float]]]:
        sequences = self.to_pandas()
        sequences = forward_transform(sequences)
        if replace:
            self.coco_sequence = backward_transform(sequences.apply(lambda x: savgol_filter(x, window_size, poly_order))).values.tolist()
            if self.is_pandas:
                self.to_pandas(replace=True)
            self.is_smoothed = True
            return self.coco_sequence
        else:
            self.is_smoothed = True
            return backward_transform(sequences.apply(lambda x: savgol_filter(x, window_size, poly_order))).values.tolist()
        
    def apply_solgol(self, df, window_size=15, poly_order=2):
        return df.apply(lambda x: savgol_filter(x, window_size, poly_order))
    
    def min_max_scale(self, df):
        """
        Scale each column of the DataFrame to the range [0, 1].
        """
        return df.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if np.ptp(x) > 0 else x)


    def standard_normal_scale(self, df, cols=None, std=False):
        """
        Perform standard normalization (z-score scaling) on the DataFrame.
        If std=False, scales only by centering the data (subtracting the mean).
        
        Parameters:
            df: DataFrame to be scaled.
            cols: Optional list of columns to scale. If None, scales all columns.
            std: Whether to scale by standard deviation (default: False).
        """
        if cols:
            for col in cols:
                mean = df[col].mean()
                std_dev = df[col].std() if std else 1
                df[col] = (df[col] - mean) / std_dev
        else:
            df = df.apply(lambda x: (x - x.mean()) / (x.std() if std else 1) if np.std(x) > 0 else x)
        return df
    
    def norm_to_left(self, df, cols):
        overall_min_value = df[cols].min().min()
        df[cols] = df[cols].apply(lambda x: x - overall_min_value)
        return df
        

    @check_data
    def impute_sequence(self, replace=False) -> List[List[Tuple[float, float]]]:
        '''
        Replace all None with forward and backward fill

        Args:
            replace (bool): Replace the current sequence with the imputed sequence        
        ''' 
        df = self.to_pandas()
        if replace:
            self.coco_sequence = df.ffill().bfill().values.tolist()
            self.is_imputed = True
            return self.coco_sequence
        return df.ffill().bfill().values.tolist()
    
    @check_data
    def impute_sequence_linear_interpolation(self, replace=False) -> List[List[Tuple[float, float]]]:
        '''
        Replace all None with linear interpolation

        Args:
            replace (bool): Replace the current sequence with the imputed sequence        
        ''' 
        df = self.to_pandas()
        # print(df)
        df = forward_transform(df)
        self.is_imputed = True
        if replace:
            self.coco_sequence = backward_transform(df.interpolate(method='linear', axis=0).ffill().bfill()).values.tolist()
            return self.coco_sequence
        return backward_transform(df.interpolate(method='linear', axis=0).ffill().bfill()).ffill().bfill().values.tolist()
    
    def trim_sequence(self, start_frame, end_frame, replace=False) -> List[List[Tuple[float, float]]]:
        '''
        Trim the sequence to the specified start and end frame

        Args:
            start_frame (int): Start frame to trim to
            end_frame (int): End frame to trim to
            replace (bool): Replace the current sequence with the trimmed sequence
        '''
        if replace:
            self.coco_sequence = self.coco_sequence[int(start_frame):int(end_frame)+1]
            return self.coco_sequence
        return self.coco_sequence[int(start_frame):int(end_frame)+1]
    

    def subsample_sequence(self, k=1, replace=False) -> List[List[Tuple[float, float]]]:
        '''
        Subsample to every kth frame
        '''
        if replace:
            self.coco_sequence = self.coco_sequence[::k]
            return self.coco_sequence
        return self.coco_sequence[::k]

    
    def to_pandas(self, replace=False) -> pd.DataFrame:
        '''
        Convert the sequence to a pandas DataFrame

        Args:
            replace (bool): Replace the current sequence with the pandas DataFrame
        '''
        if self.coco_sequence == []:
            raise ValueError("No data to convert to pandas")
        if self.is_pandas:
            return self.coco_sequence
        if replace:
            self.coco_sequence = pd.DataFrame(self.coco_sequence, columns = self.keypoints.keys())
            self.is_pandas = True
            return self.coco_sequence
        return pd.DataFrame(self.coco_sequence, columns = self.keypoints.keys())
    
    def drop_face_keypoints(self, replace=False) -> List[List[Tuple[float, float]]]:
        '''
        Drop the face keypoints from the sequence

        Args:
            replace (bool): Replace the current sequence with the face keypointless sequence
        '''
        if replace:
            if self.is_pandas:
                self.coco_sequence = self.coco_sequence.drop(columns=["nose", "left_eye", "right_eye", "left_ear", "right_ear"])
                return self.coco_sequence
            else:
                self.to_pandas(replace=True)
                return self.drop_face_keypoints(replace=True)
        else:
            if self.is_pandas:
                return self.coco_sequence.drop(columns=["nose", "left_eye", "right_eye", "left_ear", "right_ear"])
            else:
                self.to_pandas(replace=True)
                return self.drop_face_keypoints(replace=False)
        
    
    def sanity_check(self):
        assert not any([any([x == None for x in frame]) for frame in self.coco_sequence]), "Data not cleaned"


    def extract_x_positions(self) -> pd.DataFrame:
        if not self.is_pandas and not self.is_cleaned and not self.is_imputed:
            raise ValueError("Data not in pandas format or clean or impute first")

        df = self.coco_sequence.copy()
        df.columns = [f"x-pos_{key}" for key in df.columns]
        df = df.map(lambda x: x[0])
        return df
    
    def extract_distances(self) -> pd.DataFrame:
        if not self.is_pandas and not self.is_cleaned and not self.is_imputed:
            raise ValueError("Data not in pandas format or clean or impute first")


        all_subsets_of_size_2 = list(combinations(self.coco_sequence.columns, 2))
        data = {}

        for subset in all_subsets_of_size_2:
            column = f"dist_{subset[0]}-{subset[1]}"
            first_col = self.coco_sequence[subset[0]].tolist()
            second_col = self.coco_sequence[subset[1]].tolist()
            dist = [np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) for (x1, y1), (x2, y2) in zip(first_col, second_col)]
            data[column] = dist

        return pd.DataFrame(data)
    
    def extract_angles(self) -> pd.DataFrame:
        if not self.is_pandas and not self.is_cleaned and not self.is_imputed:
            raise ValueError("Data not in pandas format or clean or impute first")

        all_subsets_of_size_3 = list(permutations(self.coco_sequence.columns, 3))
        
        data = {}

        for subset in all_subsets_of_size_3:
            column = f"angle_{subset[0]}-({subset[1]}-{subset[2]})"
            first_col = self.coco_sequence[subset[0]].tolist()
            second_col = self.coco_sequence[subset[1]].tolist()
            third_col = self.coco_sequence[subset[2]].tolist()
            angle = [calculate_angle(x1, y1, x2, y2, x3, y3) for (x1, y1), (x2, y2), (x3, y3) in zip(first_col, second_col, third_col)]
            data[column] = angle

        return pd.DataFrame(data)
    
    def calc_angle_differences(self, angle_df) -> pd.DataFrame:

        data = {}
        
        right_shoulder = angle_df['angle_right_shoulder-(right_wrist-right_hip)'].tolist()
        left_shoulder = angle_df['angle_left_shoulder-(left_hip-left_wrist)'].tolist()

        right_hip = angle_df['angle_right_hip-(right_ankle-left_ankle)'].tolist()
        left_hip = angle_df['angle_left_hip-(left_ankle-right_ankle)'].tolist()

        data['diff_shoulders'] = [left-right for (left, right) in zip(left_shoulder, right_shoulder)]
        data['diff_hips'] = [left-right for (left, right) in zip(left_hip, right_hip)]

        return pd.DataFrame(data)

    
    def extract_areas(self) -> pd.DataFrame:
        '''
        Extract the areas: rectangle [shoulders (left, right) - hips (left, right)]
                           rectangle [hips (left, right) - ankles (left, right)]
        '''

        if not self.is_pandas and not self.is_cleaned and not self.is_imputed:
            raise ValueError("Data not in pandas format or clean or impute first")
        
        areas = {}

        # Rectangle 1
        left_shoulder = self.coco_sequence["left_shoulder"].tolist()
        right_shoulder = self.coco_sequence["right_shoulder"].tolist()
        left_hip = self.coco_sequence["left_hip"].tolist()
        right_hip = self.coco_sequence["right_hip"].tolist()

        area1 = [calculate_area(x1, y1, x2, y2, x3, y3, x4, y4) for (x1, y1), (x2, y2), (x3, y3), (x4, y4) in zip(left_shoulder, right_shoulder, left_hip, right_hip)]

        areas["area_shoulders-hips"] = area1

        # Rectangle 2

        left_ankle = self.coco_sequence["left_ankle"].tolist()
        right_ankle = self.coco_sequence["right_ankle"].tolist()

        area2 = [calculate_area(x1, y1, x2, y2, x3, y3, x4, y4) for (x1, y1), (x2, y2), (x3, y3), (x4, y4) in zip(left_hip, right_hip, left_ankle, right_ankle)]

        areas["area_hips-ankles"] = area2

        return pd.DataFrame(areas)
    
    def calculate_centerpoints(self) -> pd.DataFrame:
        '''
        Calculate the centerpoints:
        center_shoulder = (left_shoulder_x + right_shoulder_x) / 2
        center_hip = (left_hip_x + right_hip_x) / 2
        '''
        
        if not self.is_pandas and not self.is_cleaned and not self.is_imputed:
            raise ValueError("Data not in pandas format or clean or impute first")
        
        centerpoints = {}

        left_shoulder = self.coco_sequence["left_shoulder"].tolist()
        right_shoulder = self.coco_sequence["right_shoulder"].tolist()
        left_hip = self.coco_sequence["left_hip"].tolist()
        right_hip = self.coco_sequence["right_hip"].tolist()

        center_shoulder = [(x1 + x2) / 2 for (x1, _), (x2, _) in zip(left_shoulder, right_shoulder)]
        center_hip = [(x1 + x2) / 2 for (x1, _), (x2, _) in zip(left_hip, right_hip)]

        centerpoints["x-pos_center_shoulder"] = center_shoulder

        centerpoints["x-pos_center_hip"] = center_hip

        return pd.DataFrame(centerpoints)

@dataclass
class MotionData:
    data: pd.DataFrame
    label: int
    pid: int

    def reduce_cols(self, cols):
        self.data = self.data[cols]
        return self



