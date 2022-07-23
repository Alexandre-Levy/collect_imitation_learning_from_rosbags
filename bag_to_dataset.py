import subprocess
import os
import sys
import argparse

def run_bash(cmd):
    subprocess.run(cmd)

def main(args=None):
    parser = argparse.ArgumentParser(description='Bag file parser')
    parser.add_argument('--bag_file', type=str, help='Path to bag file', required=True)
    parser.add_argument('--dataset_folder', type=str, help='Result_folder', required=True)
    parser.add_argument("--topic_image", type=str, help="Topic_image", required=True)
    parser.add_argument("--topic_gps", type=str, help="Topic_gps", required=True)
    args = parser.parse_args()
    bash_command_get_bag_info=[
        "python", "bag_gps_to_json.py",
        "--bag_file", args.bag_file, 
        "--result_folder", args.dataset_folder,
        "--topic_image", args.topic_image,
        "--topic_gps", args.topic_gps
        ]
    bash_command_interpolate_waypoints = [
        "python", "interpolate_waypoints.py",
        "--gps_positions", f'{args.dataset_folder}/gps_positions.json',
        "--timestamps_img", f'{args.dataset_folder}/img_timestamps.json',
        "--imgs_positions_folder", args.dataset_folder
        ]
    run_bash(bash_command_get_bag_info)
    run_bash(bash_command_interpolate_waypoints)

if __name__ == "__main__":
    main()