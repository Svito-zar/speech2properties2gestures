""" This script can read a BVH file frame by frame.
    Each frame will be converted into 3d Coordinates
    """

# @authors: Taras Kucherenko, Rajmund Nagy


import my_code.data_processing.visualization.motion_visualizer.bvh_helper as BVH

import numpy as np


def append(current_node, current_coords, main_joints):

    # check if we want this coordinate
    if current_node.name in main_joints:
        # append it to the coordinates of the current node
        curr_point = current_node.coordinates.reshape(3)
        current_coords.append(curr_point)

    for kids in current_node.children:
        append(kids, current_coords, main_joints)


def obtain_coords(root, frames, duration, main_joints):

    total_coords = []

    for fr in range(duration):

        current_coords = []

        root.load_frame(frames[fr])
        root.apply_transformation()

        # Visualize the frame
        append(root, current_coords, main_joints)

        total_coords.append(current_coords)

    return total_coords


def read_bvh_to_array(bvh_file):

    root, frames, frame_time = BVH.load(bvh_file)
    duration = len(frames)

    main_joints = [
        "Pelvis",
        "Spine_01",
        "Spine_02",
        "Spine_03",
        "Neck",

        "L_Clavice",
        "L_Shoulder",
        "L_Elbow",  
        "L_Wrist",
        
        "L_Hand_Thumb_00",
        "L_Hand_Thumb_01",
        "L_Hand_Thumb_02",

        "L_Hand_Index_00",
        "L_Hand_Index_01",
        "L_Hand_Index_02",

        "L_Hand_Middle_00",
        "L_Hand_Middle_01",
        "L_Hand_Middle_02",

        "L_Hand_Ring_00",
        "L_Hand_Ring_01",
        "L_Hand_Ring_02",

        "L_Hand_Little_00",
        "L_Hand_Little_01",
        "L_Hand_Little_02",

        "R_Clavice",
        "R_Shoulder",
        "R_Elbow",  
        "R_Wrist",
        
        "R_Hand_Thumb_00",
        "R_Hand_Thumb_01",
        "R_Hand_Thumb_02",

        "R_Hand_Index_00",
        "R_Hand_Index_01",
        "R_Hand_Index_02",

        "R_Hand_Middle_00",
        "R_Hand_Middle_01",
        "R_Hand_Middle_02",

        "R_Hand_Ring_00",
        "R_Hand_Ring_01",
        "R_Hand_Ring_02",

        "R_Hand_Little_00",
        "R_Hand_Little_01",
        "R_Hand_Little_02",
    ]

    coord = obtain_coords(root, frames, duration, main_joints)

    coords_np = np.array(coord)

    # Center to hips
    hips = coords_np[:, 0, :]
    coords_np = coords_np - hips[:, np.newaxis, :]

    return coords_np


if __name__ == "__main__":

    file_path = "/home/taras/Documents/Datasets/SpeechToMotion/Irish/raw/TestMotions/NaturalTalking_001.bvh"

    result = read_bvh_to_array(file_path)
