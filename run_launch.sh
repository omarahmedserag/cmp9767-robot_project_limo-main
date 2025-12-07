#!/usr/bin/env bash
set -euo pipefail

# Usage:
# ./run_launch.sh [world_path]
# Example with the user's command:
# ./run_launch.sh src/my_ros2_package/my_ros2_package/worlds/office2.world

ROS_DISTRO=${ROS_DISTRO:-humble}
WS_ROOT="/workspaces/cmp9767-robot_project_limo"
DEFAULT_WORLD="$WS_ROOT/src/rally_obstacle.world"

# take first arg as world path, otherwise use default
INPUT_WORLD="${1:-$DEFAULT_WORLD}"

# resolve to absolute path: if not absolute and exists under workspace, prefix workspace
if [[ "$INPUT_WORLD" != /* ]]; then
  CANDIDATE="$WS_ROOT/$INPUT_WORLD"
else
  CANDIDATE="$INPUT_WORLD"
fi

if [[ -f "$CANDIDATE" ]]; then
  WORLD_PATH="$CANDIDATE"
else
  echo "World file not found at: $CANDIDATE"
  echo "Tried input: $INPUT_WORLD and workspace candidate: $WS_ROOT/$INPUT_WORLD"
  exit 1
fi

# source ROS 2
source /opt/ros/$ROS_DISTRO/setup.bash
# source workspace overlay if present
if [[ -f "$WS_ROOT/install/setup.bash" ]]; then
  source "$WS_ROOT/install/setup.bash"
fi

echo "Launching with world: $WORLD_PATH"
ros2 launch limo_gazebosim limo_gazebo_diff.launch.py world:="$WORLD_PATH"
