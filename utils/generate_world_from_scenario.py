import yaml
import json
import math
import argparse

# Function to calculate marker pose
# Marker is placed 1 meter in front of the viewpoint's orientation (yaw)
def calculate_marker_pose(viewpoint, marker_distance=1.0):
    x = viewpoint['x'] + marker_distance * math.cos(viewpoint['y'])
    y = viewpoint['y'] + marker_distance * math.sin(viewpoint['y'])
    z = viewpoint['z']  # Assuming marker is at the same height as the viewpoint
    return {'x': x, 'y': y, 'z': z}

# Read the YAML scenario file
def read_scenario(file_path):
    with open(file_path, 'r') as file:
        scenario = yaml.safe_load(file)
    return scenario

# Write the JSON world configuration
def write_world_config(scenario, model_type, world_name, output_path, marker_distance):
    world_config = {
        "world_name": world_name,
        "origin": {
            "latitude": 40.4405287,
            "longitude": -3.6898277,
            "altitude": 100.0
        },
        "drones": [
            {
                "model_type": model_type,
                "model_name": "cf0",
                "xyz": [0.0, 0.0, 0.5],
                "rpy": [0, 0, 0.0],
                "flight_time": 60,
                "payload": [
                    {
                        "model_name": "hd_camera",
                        "model_type": "hd_camera",
                        "rpy": [0.0, 1.58, 0.0]
                    }
                ]
            }
        ],
        "objects": []
    }

    # Add obstacles as objects
    for key, obstacle in scenario.get('obstacles', {}).items():
        world_config['objects'].append({
            "model_type": "obstacle",
            "model_name": f"obstacle_{key}",
            "xyz": [obstacle['x'], obstacle['y'], obstacle['z']],
            "rpy": [0, 0, 0]
        })

    # Add viewpoint markers as ArUco markers
    for key, viewpoint in scenario.get('viewpoint_poses', {}).items():
        marker_pose = calculate_marker_pose(viewpoint, marker_distance)
        world_config['objects'].append({
            "model_type": f"aruco_id{key}_marker",
            "model_name": f"id{key}",
            "xyz": [marker_pose['x'], marker_pose['y'], marker_pose['z']],
            "rpy": [0, 0, viewpoint['y']]
        })

    with open(output_path, 'w') as file:
        yaml.dump(world_config, file)
        # json.dump(world_config, file, indent=4)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Generate JSON world configuration from YAML scenario.")
    parser.add_argument('input_file', type=str, help="Path to the input YAML scenario file.")
    parser.add_argument('output_file', type=str, help="Path to the output YAML world configuration file.")
    parser.add_argument('--model_type', type=str, default="quadrotor_gripper_base", help="Model type for the drone.")
    parser.add_argument('--world_name', type=str, default="empty_world", help="Name of the world.")
    parser.add_argument('--marker_distance', type=float, default=1.0, help="Distance away from viewpoint to generate marker.")

    args = parser.parse_args()

    scenario = read_scenario(args.input_file)
    write_world_config(scenario, args.model_type, args.world_name, args.output_file, args.marker_distance)
    print(f"World configuration written to {args.output_file}")

if __name__ == "__main__":
    main()
