#!/usr/bin/env python3
"""
EMERGENCY OBJECT DETECTOR – FINAL VERSION (SMALL CYLINDERS + ROTATED BOXES FIXED)

Key fixes:
  - Tall red objects (height > 1.4 × width) + any depth cue → fire_extinguisher
  - Depth symmetry analysis to detect rotated boxes
  - Cylinder protection to prevent misclassification
  - Preserves all robustness for boxes and large objects
"""

import rclpy
from rclpy.node import Node
import cv2 as cv
import numpy as np
import math
from rclpy.duration import Duration

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped


class EmergencyDetector(Node):

    def __init__(self):
        super().__init__('emergency_detector')

        # Camera intrinsics
        self.fx = 448.6252424876914
        self.fy = 448.6252424876914
        self.cx = 320.5
        self.cy = 240.5
        self.camera_frame = "depth_link"
        self.map_frame = "odom"

        self.bridge = CvBridge()
        self.depth_image = None

        self.create_subscription(Image, '/limo/depth_camera_link/image_raw', self.rgb_callback, 10)
        self.create_subscription(Image, '/limo/depth_camera_link/depth/image_raw', self.depth_callback, 10)
        self.detection_pub = self.create_publisher(PoseStamped, '/emergency_detections', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.min_area = 150
        self.kernel = np.ones((5, 5), np.uint8)
        self.red_low1 = (0, 70, 30)
        self.red_high1 = (10, 255, 255)
        self.red_low2 = (160, 70, 30)
        self.red_high2 = (179, 255, 255)

        self.ground_depth = None

        self.get_logger().info("Emergency Detector - FINAL (small cylinders + rotated boxes fixed)")

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough").astype(np.float32)
            self.depth_image[self.depth_image == 0] = np.nan
        except Exception as e:
            self.get_logger().warn(f"Depth error: {e}")
            self.depth_image = None

    def rgb_callback(self, msg):
        if self.depth_image is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"RGB error: {e}")
            return

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        red_mask = (
            cv.inRange(hsv, self.red_low1, self.red_high1) |
            cv.inRange(hsv, self.red_low2, self.red_high2)
        )
        red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, self.kernel)

        non_red_mask = cv.inRange(hsv, (0, 50, 30), (180, 255, 255))
        non_red_mask = cv.bitwise_and(non_red_mask, cv.bitwise_not(red_mask))
        non_red_mask = cv.morphologyEx(non_red_mask, cv.MORPH_OPEN, self.kernel)

        self.process_objects(red_mask, frame, is_red=True)
        self.process_objects(non_red_mask, frame, is_red=False)

        cv.imshow("Emergency Detector", frame)
        cv.waitKey(1)

    def compute_depth_cv(self, contour):
        if self.depth_image is None:
            return 0.0

        mask = np.zeros(self.depth_image.shape, dtype=np.uint8)
        cv.drawContours(mask, [contour], -1, 255, -1)

        vals = self.depth_image[mask == 255]
        vals = vals[np.isfinite(vals)]
        vals = vals[vals > 0]

        if len(vals) < 30:
            return 0.0

        q75, q25 = np.percentile(vals, [75, 25])
        iqr = q75 - q25
        filtered_vals = vals[(vals >= q25 - 1.5 * iqr) & (vals <= q75 + 1.5 * iqr)]
        
        if len(filtered_vals) < 10:
            return 0.0

        mean = float(np.mean(filtered_vals))
        std = float(np.std(filtered_vals))
        if mean <= 1e-6:
            return 0.0
        return std / mean

    def radial_depth_monotonicity(self, cx, cy, r_min=2, r_max=18, step=1, n_angles=12):
        if self.depth_image is None:
            return False

        H, W = self.depth_image.shape
        
        mask_around = np.zeros((H, W), dtype=np.uint8)
        cv.circle(mask_around, (int(cx), int(cy)), 10, 255, -1)
        object_pixels = self.depth_image[mask_around == 255]
        object_pixels = object_pixels[np.isfinite(object_pixels) & (object_pixels > 0)]
        adaptive_r_max = max(8, min(20, int(25 / np.mean(object_pixels)))) if len(object_pixels) > 0 else 12

        angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
        good_rays = 0
        valid_rays = 0

        for a in angles:
            samples = []
            for r in range(r_min, adaptive_r_max, step):
                px = int(cx + r * math.cos(a))
                py = int(cy + r * math.sin(a))
                if 0 <= px < W and 0 <= py < H:
                    z = self.depth_image[py, px]
                    if np.isfinite(z) and z > 0:
                        samples.append(float(z))
        
            if len(samples) >= 3:
                valid_rays += 1
                overall_increasing = samples[-1] >= samples[0]
                violations = sum(1 for i in range(len(samples)-1) if samples[i] > samples[i+1])
                mostly_monotonic = violations <= len(samples) * 0.3
                
                if overall_increasing and mostly_monotonic:
                    good_rays += 1

        required_ratio = 0.5 if valid_rays >= 8 else 0.6 if valid_rays >= 5 else 0.7
        return valid_rays > 0 and (good_rays / valid_rays) >= required_ratio

    def estimate_ground_plane(self, c):
        if self.depth_image is None:
            return None
            
        x, y, w, h = cv.boundingRect(c)
        buffer = max(15, min(w, h) // 3)
        x1 = max(0, x - buffer)
        y1 = max(0, y - buffer)  
        x2 = min(self.depth_image.shape[1], x + w + buffer)
        y2 = min(self.depth_image.shape[0], y + h + buffer)
        
        surrounding = self.depth_image[y1:y2, x1:x2]
        surrounding = surrounding[np.isfinite(surrounding) & (surrounding > 0)]
        
        if len(surrounding) < 80:
            return None
            
        q75, q25 = np.percentile(surrounding, [75, 25])
        iqr = q75 - q25
        filtered_surrounding = surrounding[(surrounding >= q25 - 1.5 * iqr) & (surrounding <= q75 + 1.5 * iqr)]
        
        if len(filtered_surrounding) < 30:
            return None
            
        ground_depth = np.median(filtered_surrounding)
        if self.ground_depth is None:
            self.ground_depth = ground_depth
        else:
            self.ground_depth = 0.85 * self.ground_depth + 0.15 * ground_depth
            
        return self.ground_depth

    def is_object_on_ground(self, c, ground_depth):
        if ground_depth is None:
            return False
        
        mask = np.zeros(self.depth_image.shape[:2], dtype=np.uint8)
        cv.drawContours(mask, [c], -1, 255, -1)
    
        if np.sum(mask) < 8:
            return False
        
        kernel = np.ones((3,3), np.uint8)
        eroded = cv.erode(mask, kernel, iterations=1)
        boundary_mask = mask - eroded
        
        if np.sum(boundary_mask) < 4:
            return False
        
        boundary_depths = self.depth_image[boundary_mask.astype(bool)]
        boundary_depths = boundary_depths[np.isfinite(boundary_depths) & (boundary_depths > 0)]
        
        if len(boundary_depths) < 2:
            return False
        
        avg_boundary_depth = np.mean(boundary_depths)
        adaptive_threshold = max(0.03, ground_depth * 0.1)
        return abs(avg_boundary_depth - ground_depth) < adaptive_threshold

    def get_depth_profile(self, c):
        mask = np.zeros(self.depth_image.shape, dtype=np.uint8)
        cv.drawContours(mask, [c], -1, 255, -1)
        
        depth_vals = self.depth_image[mask == 255]
        depth_vals = depth_vals[np.isfinite(depth_vals) & (depth_vals > 0)]
        
        if len(depth_vals) < 15:
            return {'is_flat': False, 'depth_cv': 0.0}
        
        depth_std = np.std(depth_vals)
        depth_mean = np.mean(depth_vals)
        depth_cv = depth_std / depth_mean if depth_mean > 0 else 0
        
        depth_range = np.max(depth_vals) - np.min(depth_vals)
        depth_range_ratio = depth_range / depth_mean if depth_mean > 0 else 1.0
        
        is_flat = (depth_cv < 0.02) and (depth_range_ratio < 0.15)
        
        return {'is_flat': is_flat, 'depth_cv': depth_cv}

    def compute_depth_symmetry(self, cx, cy, margin=10):
        """
        Compute depth symmetry score (0=asymmetric, 1=symmetric)
        Cylinders are symmetric, boxes are asymmetric even when rotated
        """
        if self.depth_image is None:
            return 0.0

        H, W = self.depth_image.shape
        
        # Sample depth in opposite directions
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        symmetry_scores = []
        
        for dx, dy in directions:
            # Sample in positive direction
            pos_samples = []
            for r in range(1, margin + 1):
                px = int(cx + r * dx)
                py = int(cy + r * dy)
                if 0 <= px < W and 0 <= py < H:
                    z = self.depth_image[py, px]
                    if np.isfinite(z) and z > 0:
                        pos_samples.append(z)
            
            # Sample in negative direction  
            neg_samples = []
            for r in range(1, margin + 1):
                px = int(cx - r * dx)
                py = int(cy - r * dy)
                if 0 <= px < W and 0 <= py < H:
                    z = self.depth_image[py, px]
                    if np.isfinite(z) and z > 0:
                        neg_samples.append(z)
            
            if len(pos_samples) > 0 and len(neg_samples) > 0:
                # Compare depth profiles in opposite directions
                min_len = min(len(pos_samples), len(neg_samples))
                if min_len >= 3:
                    pos_profile = pos_samples[:min_len]
                    neg_profile = neg_samples[:min_len]
                    
                    # Normalize profiles
                    pos_norm = np.array(pos_profile) - np.mean(pos_profile)
                    neg_norm = np.array(neg_profile) - np.mean(neg_profile)
                    
                    # Compute correlation (symmetric objects have high correlation)
                    if np.std(pos_norm) > 0 and np.std(neg_norm) > 0:
                        correlation = np.corrcoef(pos_norm, neg_norm)[0, 1]
                        symmetry_scores.append(abs(correlation))
        
        if len(symmetry_scores) == 0:
            return 0.0
        
        return np.mean(symmetry_scores)

    def depth_analysis(self, x, y, w, h, cx, cy):
        H, W = self.depth_image.shape
        depth_center = -1.0

        if 0 <= cx < W and 0 <= cy < H:
            z = self.depth_image[cy, cx]
            if np.isfinite(z) and z > 0:
                depth_center = float(z)
            else:
                return False, False, depth_center, (0.0, 0.0, 0.0), 0.0, 0.0, False
        else:
            return False, False, depth_center, (0.0, 0.0, 0.0), 0.0, 0.0, False

        object_size = max(w, h)
        if depth_center < 1.0:
            margin = max(1, min(object_size // 15, 4))
        elif depth_center < 2.0:
            margin = max(2, min(object_size // 10, 6))
        else:
            margin = max(3, min(object_size // 8, 8))

        # Horizontal curvature
        xs_h = np.arange(x + margin, x + w - margin)
        depths_h = []
        for xi in xs_h:
            d = self.depth_image[cy, xi] if 0 <= xi < W else np.nan
            depths_h.append(float(d) if np.isfinite(d) and d > 0 else np.nan)

        depths_h = np.array(depths_h)
        valid_h = ~np.isnan(depths_h)
        if np.sum(valid_h) < 6:
            return False, False, depth_center, (0.0, 0.0, 0.0), 0.0, 0.0, False

        xs_h_valid = xs_h[valid_h] - cx
        xs_h_norm = xs_h_valid / np.max(np.abs(xs_h_valid)) if np.max(np.abs(xs_h_valid)) > 0 else xs_h_valid
        depths_h_valid = depths_h[valid_h]

        if len(depths_h_valid) > 10:
            q75, q25 = np.percentile(depths_h_valid, [75, 25])
            iqr = q75 - q25
            valid_indices = (depths_h_valid >= q25 - 1.5 * iqr) & (depths_h_valid <= q75 + 1.5 * iqr)
            xs_h_norm = xs_h_norm[valid_indices]
            depths_h_valid = depths_h_valid[valid_indices]

        try:
            coeffs_h_quad = np.polyfit(xs_h_norm, depths_h_valid, 2)
            curvature_h = abs(coeffs_h_quad[0])
        except:
            return False, False, depth_center, (0.0, 0.0, 0.0), 0.0, 0.0, False

        # Vertical curvature
        ys_v = np.arange(y + margin, y + h - margin)
        depths_v = []
        for yi in ys_v:
            d = self.depth_image[yi, cx] if 0 <= yi < H else np.nan
            depths_v.append(float(d) if np.isfinite(d) and d > 0 else np.nan)

        depths_v = np.array(depths_v)
        valid_v = ~np.isnan(depths_v)
        if np.sum(valid_v) < 6:
            return False, False, depth_center, (0.0, 0.0, 0.0), 0.0, 0.0, False

        ys_v_valid = ys_v[valid_v] - cy
        ys_v_norm = ys_v_valid / np.max(np.abs(ys_v_valid)) if np.max(np.abs(ys_v_valid)) > 0 else ys_v_valid
        depths_v_valid = depths_v[valid_v]

        if len(depths_v_valid) > 10:
            q75, q25 = np.percentile(depths_v_valid, [75, 25])
            iqr = q75 - q25
            valid_indices = (depths_v_valid >= q25 - 1.5 * iqr) & (depths_v_valid <= q75 + 1.5 * iqr)
            ys_v_norm = ys_v_norm[valid_indices]
            depths_v_valid = depths_v_valid[valid_indices]

        try:
            coeffs_v_quad = np.polyfit(ys_v_norm, depths_v_valid, 2)
            curvature_v = abs(coeffs_v_quad[0])
        except:
            return False, False, depth_center, (0.0, 0.0, 0.0), 0.0, 0.0, False

        object_area = w * h
        if depth_center < 1.0:
            base_threshold = max(0.0002, 0.0008 / depth_center)
            if object_area < 400:
                base_threshold *= 0.7
        elif depth_center < 2.0:
            base_threshold = max(0.0003, 0.0010 / depth_center)
            if object_area < 600:
                base_threshold *= 0.8
        else:
            base_threshold = max(0.0005, 0.0018 / depth_center)
            if object_area < 800:
                base_threshold *= 0.85

        threshold = base_threshold

        is_cylinder = (
            (curvature_h > threshold and not curvature_v > threshold) or
            (curvature_v > threshold and not curvature_h > threshold)
        )

        is_curved = (curvature_h > threshold) or (curvature_v > threshold)

        X = float((cx - self.cx) * depth_center / self.fx) if depth_center > 0 else 0.0
        Y = float((cy - self.cy) * depth_center / self.fy) if depth_center > 0 else 0.0
        Z = float(depth_center)

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

            # NEW: Tall object check
            is_tall = h > 1.4 * w  # Height significantly greater than width

            circularity = 4 * math.pi * area / (peri * peri)
            aspect_ratio = max(w, h) / max(min(w, h), 1)

            hull = cv.convexHull(c)
            hull_area = cv.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            approx = cv.approxPolyDP(c, 0.02 * peri, True)
            corners = len(approx)

            depth_valid, is_curved, depth_center, (X_cam, Y_cam, Z_cam), curvature_h, curvature_v, is_cylinder = \
                self.depth_analysis(x, y, w, h, cx, cy)

            obj_roi = frame[y:y + h, x:x + w]
            edges = cv.Canny(obj_roi, 80, 180)
            edge_count = int(np.sum(edges > 0))

            depth_cv = self.compute_depth_cv(c)
            radial_monotonic = self.radial_depth_monotonicity(cx, cy)

            ground_depth = self.estimate_ground_plane(c)
            is_on_ground = self.is_object_on_ground(c, ground_depth)

            depth_profile = self.get_depth_profile(c)

            # NEW: Compute depth symmetry for rotated box detection
            symmetry_score = self.compute_depth_symmetry(cx, cy)

            label, confidence, reason = self.classify(
                is_red=is_red,
                is_curved=is_curved,
                is_cylinder=is_cylinder,
                circularity=circularity,
                aspect_ratio=aspect_ratio,
                solidity=solidity,
                corners=corners,
                edge_count=edge_count,
                area=area,
                depth_center=depth_center,
                depth_cv=depth_cv,
                radial_monotonic=radial_monotonic,
                is_on_ground=is_on_ground,
                depth_profile=depth_profile,
                is_tall=is_tall,
                symmetry_score=symmetry_score  # ← Passed to classifier
            )

            camera_pose = Pose(
                position=Point(x=X_cam, y=Y_cam, z=Z_cam),
                orientation=Quaternion(w=1.0)
            )

            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.camera_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=1.0)
                )
                map_pose = do_transform_pose(camera_pose, transform)

                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = label
                pose_stamped.header.stamp = self.get_clock().now().to_msg()
                pose_stamped.pose = map_pose
                self.detection_pub.publish(pose_stamped)

                color = (0, 0, 255) if label == "fire_extinguisher" else (0, 255, 0)
                cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv.putText(frame, f"{label} ({confidence:.2f})", (x, y - 25),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                pos_text = f"X:{map_pose.position.x:.2f} Y:{map_pose.position.y:.2f} Z:{map_pose.position.z:.2f}"
                cv.putText(frame, pos_text, (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                dbg = f"Z:{depth_center:.2f} cv:{depth_cv:.3f} rad:{int(radial_monotonic)} cor:{circularity:.2f} k:{corners} tall:{is_tall} sym:{symmetry_score:.2f}"
                cv.putText(frame, dbg, (x, y + h + 15),
                           cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

                self.get_logger().info(
                    f"[DET] {label:18} | X:{map_pose.position.x:6.2f} Y:{map_pose.position.y:6.2f} Z:{map_pose.position.z:6.2f} | "
                    f"CV:{depth_cv:.3f} Rad:{radial_monotonic} Circ:{circularity:.2f} Corners:{corners} Tall:{is_tall} Sym:{symmetry_score:.2f} | {reason}"
                )

            except Exception as e:
                self.get_logger().warn(f"TF2 error: {e}")

    def classify(self, *, is_red, is_curved, is_cylinder, circularity, aspect_ratio, solidity,
                 corners, edge_count, area, depth_center, depth_cv, radial_monotonic, is_on_ground, depth_profile, is_tall, symmetry_score):
        if not is_red:
            return "first_aid_kit", 0.85, "Non-red object"

        # ✅ CYLINDER PROTECTION: If strong cylinder evidence exists, skip box constraints
        strong_cylinder_evidence = (
            (radial_monotonic and circularity >= 0.6) or
            (is_cylinder and circularity >= 0.65) or
            (is_tall and (radial_monotonic or is_cylinder or depth_cv > 0.012)) or
            (symmetry_score > 0.8 and circularity >= 0.7 and depth_cv > 0.018)
        )

        # ✅ SMART BOX DETECTION: Only apply hard box constraint if NOT strong cylinder
        if not strong_cylinder_evidence:
            if corners == 4 and solidity > 0.92 and depth_cv < 0.012:
                return "first_aid_kit", 0.99, "Smart box: 4 corners + high solidity + low depth variation"

        # ✅ Rotated box detection using depth symmetry
        if circularity > 0.5 and circularity < 0.85 and symmetry_score < 0.6:
            return "first_aid_kit", 0.94, "Rotated box: moderate circularity + low symmetry"

        # ✅ Tall red object + any depth cue → fire extinguisher (fixes small/close cylinders)
        if is_tall and (radial_monotonic or is_cylinder or depth_cv > 0.012):
            return "fire_extinguisher", 0.96, "Tall red object + depth cue → fire extinguisher"

        # HARD BOX RULES (kept for robustness)
        if corners == 4 and solidity > 0.92 and edge_count > 300:
            return "first_aid_kit", 0.98, "Clear box: 4 corners + high solidity + edges"
        if depth_profile and depth_profile['is_flat']:
            return "first_aid_kit", 0.96, "Flat depth profile → box"

        # Ultra-small handling
        if area < 250:
            if radial_monotonic and circularity >= 0.25:
                return "fire_extinguisher", 0.94, "Ultra-small: radial monotonicity"
            elif is_cylinder:
                return "fire_extinguisher", 0.90, "Ultra-small: 3D curvature"
            elif depth_cv > 0.015:
                return "fire_extinguisher", 0.88, "Ultra-small: high depth variation"
            else:
                return "first_aid_kit", 0.88, "Ultra-small: no strong cue"

        # Distance regimes
        near = depth_center > 0 and depth_center < 0.8
        far = depth_center > 2.5
        if near or far:
            if radial_monotonic or is_cylinder or depth_cv > 0.008:
                return "fire_extinguisher", 0.95, "Near/Far: depth physics cue"
            return "first_aid_kit", 0.90, "Near/Far: no depth cue"

        # Primary depth cues
        if radial_monotonic:
            return "fire_extinguisher", 0.96, "Radial monotonicity → cylinder"
        if is_cylinder:
            return "fire_extinguisher", 0.94, "3D curvature → cylinder"

        # Size-aware circularity
        circularity_threshold = 0.80 if area < 400 else 0.86 if area < 800 else 0.88
        if circularity >= circularity_threshold:
            return "fire_extinguisher", 0.98, "High circularity"

        if circularity >= 0.75:
            return "fire_extinguisher", 0.90, "Moderate circularity"

        # Default safe
        return "first_aid_kit", 0.88, "No strong cylinder evidence"

def main():
    rclpy.init()
    node = EmergencyDetector()
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