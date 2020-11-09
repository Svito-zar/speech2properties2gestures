import tempfile
import argparse
import numpy as np
import os

from gesticulator.visualization.motion_visualizer.model_animator import create_video


def visualize(
    input_array,
    output_path,
    start_t=0,
    end_t=10,
    fps=16
):
    assert isinstance(input_array, np.ndarray), "input_array must be a numpy array"
    create_video(input_array, output_path, start_t, end_t, fps)


if __name__ == "__main__":
    # Parse command line params
    parser = argparse.ArgumentParser(description="Create video from npy file")

    parser.add_argument(
        "input", help="Motion in 3D coordinates"
    )
    parser.add_argument("output_path", help="Output path for video")

    # Video params
    parser.add_argument("--fps", default=20, help="Video fps")
    parser.add_argument(
        "--start_t", "-st", default=0, help="Start time for the sequence"
    )
    parser.add_argument("--end_t", "-end", default=10, help="End time for the sequence")

    args = parser.parse_args()

    input_data = np.load(args.input)['body']

    if input_data.ndim == 2:
        input_data = np.expand_dims(input_data, axis=0)

    visualize(
        input_data[2],
        args.output_path,
        start_t=args.start_t,
        end_t=args.end_t,
    )
