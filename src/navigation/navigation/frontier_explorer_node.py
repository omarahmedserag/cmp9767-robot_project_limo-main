# navigation/frontier_explorer_node.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from navigation.frontier_detector import detect_frontiers
import numpy as np
import math
import time

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # Nav2 interface
        self.navigator = BasicNavigator()
        self.set_initial_pose()
        self.navigator.waitUntilNav2Active()
        self.navigator.clearAllCostmaps()

        # State
        self.map_received = False
        self.exploration_active = True
        self.max_goals = 20  # prevent infinite loop
        self.goal_count = 0

        # Subscribe to map
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.get_logger().info("✅ Frontier explorer node initialized.")

    def set_initial_pose(self):
        """Set initial pose to (0, 0), matching Gazebo spawn."""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.orientation.w = 1.0
        self.navigator.setInitialPose(pose)

    def get_robot_position(self) -> Tuple[float, float]:
        """Estimate robot position from TF or assume origin during exploration."""
        # For simplicity, assume robot is at last goal or origin
        # In advanced version, use TF lookup to 'base_link'
        return 0.0, 0.0

    def map_callback(self, msg: OccupancyGrid):
        if not self.exploration_active or self.goal_count >= self.max_goals:
            return

        # Convert to numpy
        grid = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))

        # Detect frontiers
        frontiers = detect_frontiers(
            grid,
            msg.info.resolution,
            msg.info.origin.position.x,
            msg.info.origin.position.y
        )

        if not frontiers:
            self.get_logger().info("🔍 No frontiers found. Exploration complete!")
            self.exploration_active = False
            return

        # Select closest frontier
        robot_x, robot_y = self.get_robot_position()
        best = min(frontiers, key=lambda p: math.hypot(p[0] - robot_x, p[1] - robot_y))

        # Create goal
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.navigator.get_clock().now().to_msg()
        goal.pose.position.x = best[0]
        goal.pose.position.y = best[1]
        goal.pose.orientation.w = 1.0

        self.get_logger().info(f"🎯 Sending goal to frontier: ({best[0]:.2f}, {best[1]:.2f})")

        # Navigate
        self.navigator.goToPose(goal)
        while not self.navigator.isTaskComplete():
            time.sleep(0.1)

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info("✅ Reached frontier.")
        else:
            self.get_logger().warn("⚠️ Failed to reach frontier. Continuing...")

        self.goal_count += 1

        # Trigger next exploration cycle (in real system, use timer or callback)
        # For demo, request new map by re-subscribing indirectly
        # In practice, SLAM continuously publishes /map

def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()

    # Use single-threaded executor (sufficient for this flow)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()