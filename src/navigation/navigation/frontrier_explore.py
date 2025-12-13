#!/usr/bin/env python3
"""
Frontier-Based Exploration (Nav2 + SLAM)
For rally_obstacle5.world / LIMO in Gazebo

Key features:
- Non-blocking Nav2 readiness (won't hang forever at startup)
- Frontiers detected on /map (OccupancyGrid from slam_toolbox)
- Frontier clustering (BFS)
- Utility-based goal selection (cluster_size / distance)
- Failure handling + goal blacklisting
- RViz visualization markers of frontier clusters
- Optional return-to-home at completion

Author: (your name)
"""

import math
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult


# ----------------------------
# Utility helpers
# ----------------------------
def yaw_to_quat_z_w(yaw: float):
    """Return quaternion z,w for planar yaw (x=y=0)."""
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def make_pose(nav: BasicNavigator, x: float, y: float, yaw: float = 0.0) -> PoseStamped:
    """Create PoseStamped in map frame."""
    qz, qw = yaw_to_quat_z_w(yaw)
    p = PoseStamped()
    p.header.frame_id = "map"
    p.header.stamp = nav.get_clock().now().to_msg()
    p.pose.position.x = float(x)
    p.pose.position.y = float(y)
    p.pose.orientation.z = qz
    p.pose.orientation.w = qw
    return p


def euclid(ax, ay, bx, by) -> float:
    return math.hypot(ax - bx, ay - by)


# ----------------------------
# Main node
# ----------------------------
class FrontierExplore(Node):
    """
    Frontier exploration on /map for SLAM environments.

    Works best when:
    - slam_toolbox is publishing /map
    - Nav2 is running with BT navigator + controller/planner active
    - robot provides odom + TF for base_link etc.
    """

    def __init__(self):
        super().__init__("frontier_explore")

        # ----------------------------
        # Parameters (tune for your world)
        # ----------------------------
        self.min_cluster_cells = 8          # ignore tiny noisy frontiers
        self.max_goals = 200                # exploration steps limit
        self.goal_timeout_sec = 60.0        # cancel goal if stuck
        self.blacklist_radius = 0.60        # meters: avoid retrying same failed area
        self.utility_alpha = 1.0            # size weight
        self.utility_beta = 1.0             # distance weight (in denominator)
        self.publish_markers = True
        self.return_home_when_done = True

        # If SLAM starts with a fully unknown map, a small nudge helps sometimes:
        self.enable_kickstart = True
        self.kickstart_distance = 0.7       # meters

        # ----------------------------
        # Nav2 and state
        # ----------------------------
        self.nav = BasicNavigator()
        self.nav_ready = False
        self.exploring = True

        self.initial_pose_set = False
        self.home_pose = None

        self.current_goal = None
        self.goal_sent_time = None
        self.goal_count = 0

        # Map storage
        self.map_msg = None
        self.map_grid = None  # numpy HxW int8
        self.map_w = 0
        self.map_h = 0

        # Failed goals blacklist (world coords)
        self.blacklist = []  # list of (x,y)

        # ----------------------------
        # Subscriptions / publishers
        # ----------------------------
        qos_map = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, "/map", self.map_cb, qos_map)

        self.marker_pub = self.create_publisher(MarkerArray, "/explore/frontiers", 10)

        # ----------------------------
        # Timers (non-blocking)
        # ----------------------------
        self.create_timer(1.0, self.check_nav2_ready)     # readiness gate
        self.create_timer(2.0, self.plan_next_goal)       # choose next frontier
        self.create_timer(0.5, self.monitor_goal)         # monitor execution

        self.get_logger().info("FrontierExplore node started. Waiting for Nav2 + /map...")

    # ----------------------------
    # Map callback
    # ----------------------------
    def map_cb(self, msg: OccupancyGrid):
        if msg.data is None or len(msg.data) == 0:
            return

        self.map_msg = msg
        w = msg.info.width
        h = msg.info.height

        if w == 0 or h == 0:
            return

        # Convert to numpy grid
        data = np.array(msg.data, dtype=np.int8)
        self.map_grid = data.reshape((h, w))
        self.map_w, self.map_h = w, h

    # ----------------------------
    # Coordinate transforms
    # ----------------------------
    def map_to_world(self, mx: float, my: float):
        """Map cell center -> world coordinates."""
        info = self.map_msg.info
        wx = info.origin.position.x + (mx + 0.5) * info.resolution
        wy = info.origin.position.y + (my + 0.5) * info.resolution
        return wx, wy

    def world_to_map(self, wx: float, wy: float):
        """World -> map cell index (int)."""
        info = self.map_msg.info
        mx = int((wx - info.origin.position.x) / info.resolution)
        my = int((wy - info.origin.position.y) / info.resolution)
        if 0 <= mx < info.width and 0 <= my < info.height:
            return mx, my
        return None

    # ----------------------------
    # Nav2 readiness (non-blocking)
    # ----------------------------
    def check_nav2_ready(self):
        if self.nav_ready:
            return

        # BasicNavigator has isNav2Active() in many versions.
        # If your version doesn't, fallback to waitUntilNav2Active() once map exists.
        try:
            if self.nav.isNav2Active():
                self.nav_ready = True
                self.get_logger().info("✅ Nav2 is ACTIVE.")
            else:
                self.get_logger().info("⏳ Waiting for Nav2 to become ACTIVE...")
                return
        except Exception:
            # fallback behavior
            self.get_logger().info("⏳ Waiting for Nav2 (fallback wait)...")
            self.nav.waitUntilNav2Active()
            self.nav_ready = True
            self.get_logger().info("✅ Nav2 is ACTIVE (fallback).")

        # Once active, set initial pose (home) once.
        if not self.initial_pose_set:
            self.home_pose = make_pose(self.nav, 0.0, 0.0, 0.0)
            self.nav.setInitialPose(self.home_pose)
            self.nav.clearAllCostmaps()
            self.initial_pose_set = True
            self.get_logger().info("✅ Initial/home pose set at (0,0). Costmaps cleared.")

    # ----------------------------
    # Frontier detection
    # ----------------------------
    def is_frontier_cell(self, x: int, y: int) -> bool:
        """
        Frontier definition:
        - cell is FREE (0)
        - at least one 4-neighbour is UNKNOWN (-1)
        """
        if self.map_grid[y, x] != 0:
            return False

        # 4-neighbours
        if self.map_grid[y, x + 1] == -1:
            return True
        if self.map_grid[y, x - 1] == -1:
            return True
        if self.map_grid[y + 1, x] == -1:
            return True
        if self.map_grid[y - 1, x] == -1:
            return True
        return False

    def bfs_cluster(self, start_x: int, start_y: int, visited: np.ndarray):
        """
        BFS cluster of connected frontier cells.
        8-connected BFS but only expands through FREE cells (0),
        and only includes cells that satisfy frontier property.
        """
        q = deque([(start_x, start_y)])
        visited[start_y, start_x] = True
        cells = []

        # 8-neighbours for clustering
        neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while q:
            x, y = q.popleft()
            if self.is_frontier_cell(x, y):
                cells.append((x, y))

            for dx, dy in neighbours:
                nx, ny = x + dx, y + dy
                if nx <= 0 or ny <= 0 or nx >= self.map_w - 1 or ny >= self.map_h - 1:
                    continue
                if visited[ny, nx]:
                    continue
                # expand only through FREE cells (0)
                if self.map_grid[ny, nx] == 0:
                    visited[ny, nx] = True
                    q.append((nx, ny))

        return cells

    def get_frontier_clusters(self):
        """Return list of clusters: each cluster is list of (mx,my) frontier cells."""
        if self.map_grid is None:
            return []

        visited = np.zeros((self.map_h, self.map_w), dtype=bool)
        clusters = []

        for y in range(1, self.map_h - 1):
            for x in range(1, self.map_w - 1):
                if visited[y, x]:
                    continue
                # only start BFS from FREE cells
                if self.map_grid[y, x] != 0:
                    continue
                # quick check if this cell is frontier-like
                if not self.is_frontier_cell(x, y):
                    visited[y, x] = True
                    continue

                cells = self.bfs_cluster(x, y, visited)
                if len(cells) >= self.min_cluster_cells:
                    clusters.append(cells)

        return clusters

    # ----------------------------
    # Blacklist helpers
    # ----------------------------
    def is_blacklisted(self, wx: float, wy: float) -> bool:
        for bx, by in self.blacklist:
            if euclid(wx, wy, bx, by) <= self.blacklist_radius:
                return True
        return False

    def blacklist_point(self, wx: float, wy: float):
        self.blacklist.append((wx, wy))
        # Keep blacklist bounded
        if len(self.blacklist) > 1000:
            self.blacklist = self.blacklist[-500:]

    # ----------------------------
    # Goal selection
    # ----------------------------
    def plan_next_goal(self):
        """Choose next frontier goal and send it to Nav2."""
        if not self.exploring:
            return
        if not self.nav_ready or not self.initial_pose_set:
            return
        if self.map_grid is None:
            return
        if self.goal_count >= self.max_goals:
            self.get_logger().warn("Max goals reached -> stopping exploration.")
            self.finish_exploration()
            return
        # Do not send new goal if one is active
        if self.current_goal is not None and not self.nav.isTaskComplete():
            return

        clusters = self.get_frontier_clusters()
        if self.publish_markers:
            self.publish_cluster_markers(clusters)

        if len(clusters) == 0:
            # If map is totally unknown initially, sometimes you need a kickstart.
            if self.enable_kickstart:
                self.enable_kickstart = False  # only once
                self.try_kickstart()
                return

            self.get_logger().info("🎉 No frontier clusters remain -> exploration complete.")
            self.finish_exploration()
            return

        # robot pose
        rp = self.nav.getPoseStamped()
        rx = rp.pose.position.x
        ry = rp.pose.position.y

        # Compute cluster candidates (centroid + size + utility)
        candidates = []
        for cells in clusters:
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            wx, wy = self.map_to_world(cx, cy)

            if self.is_blacklisted(wx, wy):
                continue

            d = euclid(rx, ry, wx, wy)
            size = len(cells)
            utility = (self.utility_alpha * size) / (self.utility_beta * (d + 1e-6))
            candidates.append((utility, wx, wy, size, d))

        if not candidates:
            self.get_logger().warn("Frontiers exist but all are blacklisted. Stopping.")
            self.finish_exploration()
            return

        # Select best utility
        candidates.sort(key=lambda t: t[0], reverse=True)
        _, gx, gy, size, distm = candidates[0]

        self.get_logger().info(
            f"🧭 Goal {self.goal_count+1}: frontier centroid "
            f"({gx:.2f},{gy:.2f})  size={size}  dist={distm:.2f}"
        )

        goal = make_pose(self.nav, gx, gy, 0.0)
        self.nav.goToPose(goal)

        self.current_goal = (gx, gy)
        self.goal_sent_time = self.get_clock().now()
        self.goal_count += 1

    # ----------------------------
    # Goal monitoring / timeout / blacklist
    # ----------------------------
    def monitor_goal(self):
        if self.current_goal is None:
            return
        if not self.nav_ready:
            return

        # Timeout handling
        if self.goal_sent_time is not None:
            elapsed = (self.get_clock().now() - self.goal_sent_time).nanoseconds * 1e-9
            if elapsed > self.goal_timeout_sec and not self.nav.isTaskComplete():
                self.get_logger().warn("⏱️ Goal timeout -> cancel + blacklist.")
                try:
                    self.nav.cancelTask()
                except Exception:
                    pass
                gx, gy = self.current_goal
                self.blacklist_point(gx, gy)
                self.current_goal = None
                self.goal_sent_time = None
                return

        if not self.nav.isTaskComplete():
            return

        result = self.nav.getResult()
        gx, gy = self.current_goal

        if result == TaskResult.SUCCEEDED:
            self.get_logger().info("✅ Goal succeeded.")
        elif result == TaskResult.CANCELED:
            self.get_logger().warn("⚠️ Goal canceled -> blacklisting.")
            self.blacklist_point(gx, gy)
        else:
            self.get_logger().warn("❌ Goal failed -> blacklisting.")
            self.blacklist_point(gx, gy)

        self.current_goal = None
        self.goal_sent_time = None

    # ----------------------------
    # Kickstart
    # ----------------------------
    def try_kickstart(self):
        """Move forward slightly to reveal some map at startup."""
        try:
            pose = self.nav.getPoseStamped()
            qz = pose.pose.orientation.z
            qw = pose.pose.orientation.w
            yaw = 2.0 * math.atan2(qz, qw)
            tx = pose.pose.position.x + self.kickstart_distance * math.cos(yaw)
            ty = pose.pose.position.y + self.kickstart_distance * math.sin(yaw)

            self.get_logger().info(f"⏩ Kickstart: moving {self.kickstart_distance:.2f} m forward")
            self.nav.goToPose(make_pose(self.nav, tx, ty, yaw))
            self.current_goal = (tx, ty)
            self.goal_sent_time = self.get_clock().now()
        except Exception as e:
            self.get_logger().warn(f"Kickstart failed: {e}")

    # ----------------------------
    # Visualization
    # ----------------------------
    def publish_cluster_markers(self, clusters):
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()

        # Clear old markers by reusing ids and sending DELETEALL is not always supported,
        # so we publish ADD markers with bounded ids.
        max_markers = min(len(clusters), 200)

        for i in range(max_markers):
            cells = clusters[i]
            cx = float(np.mean([c[0] for c in cells]))
            cy = float(np.mean([c[1] for c in cells]))
            wx, wy = self.map_to_world(cx, cy)
            size = len(cells)

            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = now
            m.ns = "frontier_clusters"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = wx
            m.pose.position.y = wy
            m.pose.position.z = 0.1
            # size scaling
            s = max(0.12, min(0.8, 0.08 * math.sqrt(size)))
            m.scale.x = s
            m.scale.y = s
            m.scale.z = 0.08
            m.color.a = 0.9
            m.color.r = 1.0
            m.color.g = 0.2
            m.color.b = 0.2
            ma.markers.append(m)

        self.marker_pub.publish(ma)

    # ----------------------------
    # Finish / return home
    # ----------------------------
    def finish_exploration(self):
        self.exploring = False

        if self.return_home_when_done and self.home_pose is not None:
            self.get_logger().info("🏁 Exploration finished -> returning home (0,0).")
            self.nav.goToPose(self.home_pose)
            # We do not block; monitor_goal will track it if you want.
        else:
            self.get_logger().info("🏁 Exploration finished. No return-home requested.")

        # You can choose to shutdown automatically:
        # rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplore()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

