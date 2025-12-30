#!/usr/bin/env python3
"""
OBJECT LOCALIZER FOR TASK 3 - MAP FRAME LOCALIZATION

Publishes detected objects to /object_location in map frame
Compatible with Nav2 waypoint navigation
"""

import rclpy
from rclpy.node import Node
import cv2 as cv
import numpy as np
import math
from rclpy.duration import Duration

# ROS
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped


class ObjectLocalizer(Node):

    def __init__(self):
        super().__init__('object_localizer')

        # Camera intrinsics (from your camera_info)
        self.fx = 448.6252424876914
        self.fy = 448.6252424876914
        self.cx = 320.5
        self.cy = 240.5
        self.camera_frame = "depth_link"
        self.map_frame = "map"  # ✅ CHANGED TO MAP FRAME FOR TASK 3

        # ROS setup
        self.bridge = CvBridge()
        self.depth_image = None

        self.create_subscription(Image, '/limo/depth_camera_link/image_raw', self.rgb_callback, 10)
        self.create_subscription(Image, '/limo/depth_camera_link/depth/image_raw', self.depth_callback, 10)
        
        # ✅ PUBLISH TO /object_location as required by Task 3
        self.detection_pub = self.create_publisher(PoseStamped, '/object_location', 10)

        # TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Parameters
        self.min_area = 300
        self.kernel = np.ones((5, 5), np.uint8)
        self.red_low1 = (0, 100, 60)
        self.red_high1 = (10, 255, 255)
        self.red_low2 = (160, 100, 60)
        self.red_high2 = (179, 255, 255)

        self.get_logger().info("✅ Object Localizer started (publishing to /object_location in map frame)")

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough").astype(np.float32)
        except:
            self.depth_image = None

    def rgb_callback(self, msg):
        if self.depth_image is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            return

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # RED OBJECTS (fire extinguishers and first aid kits)
        red_mask = (
            cv.inRange(hsv, self.red_low1, self.red_high1) |
            cv.inRange(hsv, self.red_low2, self.red_high2)
        )
        red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, self.kernel)

        # NON-RED OBJECTS (optional - remove if only detecting red)
        non_red_mask = cv.inRange(hsv, (0, 50, 30), (180, 255, 255))
        non_red_mask = cv.bitwise_and(non_red_mask, cv.bitwise_not(red_mask))
        non_red_mask = cv.morphologyEx(non_red_mask, cv.MORPH_OPEN, self.kernel)

        self.process_objects(red_mask, frame, is_red=True)
        self.process_objects(non_red_mask, frame, is_red=False)

        cv.imshow("Object Localizer", frame)
        cv.waitKey(1)

    def depth_analysis(self, x, y, w, h, cx, cy):
        H, W = self.depth_image.shape
        depth_center = -1.0

        if 0 <= cx < W and 0 <= cy < H:
            z = self.depth_image[cy, cx]
            if np.isfinite(z) and z > 0:
                depth_center = float(z)
            else:
                return False, False, depth_center, (0, 0, 0), 0.0, 0.0, False
        else:
            return False, False, depth_center, (0, 0, 0), 0.0, 0.0, False

        margin = max(5, min(w, h) // 6)

        # --- HORIZONTAL PROFILE ---
        xs_h = np.arange(x + margin, x + w - margin)
        depths_h = []
        for xi in xs_h:
            if 0 <= xi < W:
                d = self.depth_image[cy, xi]
                if np.isfinite(d) and d > 0:
                    depths_h.append(float(d))
                else:
                    depths_h.append(np.nan)
            else:
                depths_h.append(np.nan)

        depths_h = np.array(depths_h)
        valid_h = ~np.isnan(depths_h)
        if np.sum(valid_h) < 8:
            return False, False, depth_center, (0, 0, 0), 0.0, 0.0, False

        xs_h_valid = xs_h[valid_h] - cx
        xs_h_norm = xs_h_valid / np.max(np.abs(xs_h_valid)) if np.max(np.abs(xs_h_valid)) > 0 else xs_h_valid
        depths_h_valid = depths_h[valid_h]

        try:
            coeffs_h_lin = np.polyfit(xs_h_norm, depths_h_valid, 1)
            coeffs_h_quad = np.polyfit(xs_h_norm, depths_h_valid, 2)
            pred_lin = np.polyval(coeffs_h_lin, xs_h_norm)
            pred_quad = np.polyval(coeffs_h_quad, xs_h_norm)
            mse_lin = np.mean((depths_h_valid - pred_lin) ** 2)
            mse_quad = np.mean((depths_h_valid - pred_quad) ** 2)
            curvature_h = abs(coeffs_h_quad[0])
            is_quadratic_h = mse_quad < 0.8 * mse_lin
        except:
            return False, False, depth_center, (0, 0, 0), 0.0, 0.0, False

        # --- VERTICAL PROFILE ---
        ys_v = np.arange(y + margin, y + h - margin)
        depths_v = []
        for yi in ys_v:
            if 0 <= yi < H:
                d = self.depth_image[yi, cx]
                if np.isfinite(d) and d > 0:
                    depths_v.append(float(d))
                else:
                    depths_v.append(np.nan)
            else:
                depths_v.append(np.nan)

        depths_v = np.array(depths_v)
        valid_v = ~np.isnan(depths_v)
        if np.sum(valid_v) < 8:
            return False, False, depth_center, (0, 0, 0), 0.0, 0.0, False

        ys_v_valid = ys_v[valid_v] - cy
        ys_v_norm = ys_v_valid / np.max(np.abs(ys_v_valid)) if np.max(np.abs(ys_v_valid)) > 0 else ys_v_valid
        depths_v_valid = depths_v[valid_v]

        try:
            coeffs_v_lin = np.polyfit(ys_v_norm, depths_v_valid, 1)
            coeffs_v_quad = np.polyfit(ys_v_norm, depths_v_valid, 2)
            pred_lin = np.polyval(coeffs_v_lin, ys_v_norm)
            pred_quad = np.polyval(coeffs_v_quad, ys_v_norm)
            mse_lin = np.mean((depths_v_valid - pred_lin) ** 2)
            mse_quad = np.mean((depths_v_valid - pred_quad) ** 2)
            curvature_v = abs(coeffs_v_quad[0])
            is_quadratic_v = mse_quad < 0.8 * mse_lin
        except:
            return False, False, depth_center, (0, 0, 0), 0.0, 0.0, False

        # --- DIRECTIONAL CURVATURE ANALYSIS ---
        if depth_center > 0:
            base_threshold = max(0.0008, 0.0025 / depth_center)
        else:
            base_threshold = 0.0015

        threshold = base_threshold

        is_cylinder = (
            (curvature_h > threshold and is_quadratic_h and not is_quadratic_v) or
            (curvature_v > threshold and is_quadratic_v and not is_quadratic_h)
        )

        is_curved = (curvature_h > threshold and is_quadratic_h) or (curvature_v > threshold and is_quadratic_v)

        X = (cx - self.cx) * depth_center / self.fx if depth_center > 0 else 0.0
        Y = (cy - self.cy) * depth_center / self.fy if depth_center > 0 else 0.0
        Z = depth_center

        return True, is_curved, depth_center, (X, Y, Z), curvature_h, curvature_v, is_cylinder

    def process_objects(self, mask, frame, is_red):
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv.contourArea(c)
            if area < self.min_area:
                continue

            peri = cv.arcLength(c, True)
            if peri <= 0:
                continue

            x, y, w, h = cv.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2

            circularity = 4 * math.pi * area / (peri * peri)
            aspect_ratio = max(w, h) / max(min(w, h), 1)

            hull = cv.convexHull(c)
            hull_area = cv.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            epsilon = 0.02 * peri
            approx = cv.approxPolyDP(c, epsilon, True)
            corners = len(approx)

            depth_valid, is_curved, depth_center, (X_cam, Y_cam, Z_cam), curvature_h, curvature_v, is_cylinder = self.depth_analysis(x, y, w, h, cx, cy)

            label, confidence, reason = self.classify(is_red, is_curved, is_cylinder, circularity, aspect_ratio, solidity, corners)

            # Create camera-frame pose
            camera_pose = Pose(
                position=Point(x=X_cam, y=Y_cam, z=Z_cam),
                orientation=Quaternion(w=1.0)
            )

            try:
                # Transform to MAP frame (not odom!)
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,           # "map" - ✅ GLOBAL FRAME
                    self.camera_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=1.0)
                )
                map_pose = do_transform_pose(camera_pose, transform)

                # ✅ Publish to /object_location as required by Task 3
                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = self.map_frame  # "map"
                pose_stamped.header.stamp = self.get_clock().now().to_msg()
                pose_stamped.pose = map_pose
                self.detection_pub.publish(pose_stamped)

                # Visualize classification
                color = (0, 0, 255) if "fire" in label else (0, 255, 0)
                cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv.putText(frame, f"{label} ({confidence:.2f})", (x, y - 25),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Log detected position in MAP frame
                self.get_logger().info(
                    f"[MAP] {label:18} | X={map_pose.position.x:6.2f} Y={map_pose.position.y:6.2f} Z={map_pose.position.z:6.2f}"
                )

            except Exception as e:
                self.get_logger().warn(f"TF2 error: {e}")

    def classify(self, is_red, is_curved, is_cylinder, circularity, aspect_ratio, solidity, corners):
        """Classify objects for emergency detection"""
        if not is_red:
            # Only classify large non-red objects as first_aid_kit
            if aspect_ratio > 0.3 and aspect_ratio < 3.0 and solidity > 0.8:
                return "first_aid_kit", 0.85, "Non-red box-like"
            else:
                return "first_aid_kit", 0.1, "Non-red noise"

        if corners == 4 and solidity > 0.95:
            return "first_aid_kit", 0.95, "4 corners + high solidity → box"

        if not is_curved:
            return "first_aid_kit", 0.90, "Flat surface"

        if is_cylinder:
            return "fire_extinguisher", 0.90, "3D curvature → cylinder"

        if circularity >= 0.82:
            return "fire_extinguisher", 0.94, "Spherical shape"

        if solidity < 0.9:
            return "first_aid_kit", 0.80, "Low solidity → noise"

        return "first_aid_kit", 0.85, "Curved but not extinguisher"


def main():
    rclpy.init()
    node = ObjectLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
