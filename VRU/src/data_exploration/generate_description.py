import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path to import reason_generate
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from reason_generate import (
    analyze_trajectory,
    infer_reason,
    add_weather_factor,
    add_ego_factor
)

class ObjectInfo:
    def __init__(self, class_id, initial_box):
        self.class_id = class_id
        self.class_name = self.get_class_name(class_id)
        self.trajectory = [initial_box]
        self.motion_analysis = None  # Will store analyze_trajectory results
        
    @staticmethod
    def get_class_name(class_id):
        """Convert class ID to human readable name"""
        class_names = {
            0: "car",
            1: "truck",
            2: "pedestrian",
            3: "motorcyclist",
            4: "bicycle"
        }
        return class_names.get(int(class_id), "vehicle")

def process_detection_data(det_data):
    """
    Process detection boxes to track objects and their movements
    Args:
        det_data: numpy array of detection boxes (frames, objects, 6)
                 where each box is [x1,y1,x2,y2,conf,cls]
    Returns:
        Dictionary mapping frame indices to object observations
    """
    tracked_objects = {}
    frame_data = {}
    
    for frame_idx, frame_dets in enumerate(det_data):
        frame_data[frame_idx] = {}
        
        for det in frame_dets:
            x1, y1, x2, y2, conf, cls = det
            if conf == 0:  # Skip invalid detections
                continue
                
            box_data = {
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls
            }
            
            # Create unique object ID based on position and class
            obj_id = f"{int(cls)}_{int(x1)}_{int(y1)}"
            
            if obj_id not in tracked_objects:
                tracked_objects[obj_id] = ObjectInfo(cls, box_data)
            else:
                tracked_objects[obj_id].trajectory.append(box_data)
                
            frame_data[frame_idx][obj_id] = box_data
            
    return tracked_objects, frame_data

def analyze_scene_dynamics(tracked_objects, frame_data):
    """
    Analyze the movement patterns and interactions of objects
    """
    scene_info = {
        "moving_objects": [],
        "interactions": [],
        "main_event": None
    }
    
    # Analyze individual object movements
    for obj_id, obj in tracked_objects.items():
        if len(obj.trajectory) > 5:  # Only analyze objects visible for multiple frames
            motion = {
                "object_type": obj.class_name,
                "frames_visible": len(obj.trajectory),
                "movement": analyze_object_movement(obj.trajectory)
            }
            scene_info["moving_objects"].append(motion)
    
    # Detect potential interactions
    if len(scene_info["moving_objects"]) > 1:
        scene_info["interactions"] = analyze_object_interactions(scene_info["moving_objects"])
        
    return scene_info

def analyze_object_movement(trajectory):
    """
    Analyze movement pattern of an object from its trajectory
    """
    if len(trajectory) < 2:
        return "briefly visible"
        
    start_box = trajectory[0]["bbox"]
    end_box = trajectory[-1]["bbox"]
    
    dx = end_box[0] - start_box[0]
    dy = end_box[1] - start_box[1]
    displacement = np.sqrt(dx*dx + dy*dy)
    
    if displacement < 50:
        return "stationary"
    elif abs(dx) > abs(dy):
        direction = "right" if dx > 0 else "left"
        return f"moving {direction}"
    else:
        direction = "down" if dy > 0 else "up"
        return f"moving {direction}"

def analyze_object_interactions(moving_objects):
    """
    Detect potential interactions between objects based on their movements
    """
    interactions = []
    objects_count = len(moving_objects)
    
    if objects_count > 1:
        movement_types = [obj["movement"] for obj in moving_objects]
        object_types = [obj["object_type"] for obj in moving_objects]
        
        # Look for opposite directions
        if any("right" in m for m in movement_types) and any("left" in m for m in movement_types):
            interactions.append("objects moving in opposite directions")
            
        # Check for specific object type combinations
        if "car" in object_types and "pedestrian" in object_types:
            interactions.append("vehicle-pedestrian interaction")
        elif "car" in object_types and "motorcyclist" in object_types:
            interactions.append("vehicle-motorcycle interaction")
            
    return interactions

def generate_text_description(scene_info):
    """
    Generate natural language description of the scene
    """
    description = []
    
    # Describe moving objects
    if scene_info["moving_objects"]:
        obj_descriptions = []
        for obj in scene_info["moving_objects"]:
            obj_descriptions.append(f"a {obj['object_type']} {obj['movement']}")
            
        if len(obj_descriptions) == 1:
            description.append(f"The scene shows {obj_descriptions[0]}.")
        else:
            description.append(f"The scene shows {', '.join(obj_descriptions[:-1])} and {obj_descriptions[-1]}.")
    
    # Describe interactions
    if scene_info["interactions"]:
        description.append(f"There are {' and '.join(scene_info['interactions'])}.")
    
    # Combine all descriptions
    return " ".join(description)

def process_video_data(npz_path):
    """
    Process video data and generate description
    Args:
        npz_path: Path to npz file containing video data
    Returns:
        str: Natural language description of the scene
    """
    try:
        # Load data
        data = np.load(npz_path, allow_pickle=True)
        features = data['data']  # VGG16 features
        detections = data['det']  # Detection boxes
        
        # Convert detections to frame dictionary format
        frame_dict = {}
        for frame_idx, frame_dets in enumerate(detections):
            frame_dict[frame_idx] = {}
            for det in frame_dets:
                x1, y1, x2, y2, conf, cls = det
                if conf == 0:  # Skip invalid detections
                    continue
                obj_id = f"{int(cls)}_{int(x1)}_{int(y1)}"
                frame_dict[frame_idx][obj_id] = [x1, y1, x2, y2]
        
        # Create objects list for reason generation
        objects = []
        tracked_objects = {}
        
        # Track unique objects and analyze their trajectories
        for frame_idx, frame in frame_dict.items():
            for obj_id, bbox in frame.items():
                cls = int(obj_id.split('_')[0])
                if obj_id not in tracked_objects:
                    tracked_objects[obj_id] = {
                        "id": obj_id,
                        "cls": ObjectInfo.get_class_name(cls),
                        "trajectory_analysis": analyze_trajectory(frame_dict, obj_id)
                    }
                    objects.append(tracked_objects[obj_id])
        
        # Generate base reason using existing functions
        trajectory_analysis = {obj["id"]: obj["trajectory_analysis"] for obj in objects}
        base_reason = infer_reason(objects, trajectory_analysis)
        
        # Add weather and ego vehicle factors (assuming default values for demo)
        weather_condition = "Rainy"  # This could be determined from video features
        ego_involved = True  # This could be determined from object positions/classes
        
        weather_reason = add_weather_factor(base_reason, weather_condition)
        full_reason = add_ego_factor(weather_reason, ego_involved)
        
        # Generate final description
        description = (
            f"The scene shows multiple vehicles in motion. "
            f"{full_reason}. "
            f"The video features indicate {len(objects)} objects involved in the incident."
        )
        
        return description
        
    except Exception as e:
        return f"Error processing video: {str(e)}"

if __name__ == "__main__":
    # Example usage
    test_file = r"F:\data\CarCrash\vgg16_features\positive\000260.npz"
    description = process_video_data(test_file)
    print(f"Generated description:\n{description}")