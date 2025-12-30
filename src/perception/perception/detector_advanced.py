#!/usr/bin/env python3
"""
EMERGENCY OBJECT DETECTOR – ROBUST VERSION (SMALL BOX FIX + CYLINDER SAFE)

Fixes added to stop misclassifying small/rotated boxes as fire extinguishers:
  - Rotated-rectangle rectangularity (minAreaRect) for box evidence even when rotated
  - Depth planarity (plane fit RMSE) inside contour: boxes are planar, cylinders are not
  - Orthogonal line evidence from edges (HoughLinesP)
  - Cylinder decisions are gated: tiny objects need BOTH cylinder evidence AND NOT strong box evidence

Keeps your existing robustness:
  - Red/non-red segmentation
  - Depth CV, radial monotonicity, curvature-based cylinder test
  - Depth symmetry for rotated box detection
  - Tall-object heuristic for fire extinguisher
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

        # Camera intrinsics (yours)
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

        # Vision params
        self.min_area = 150
        self.kernel = np.ones((5, 5), np.uint8)

        # Red HSV range (yours)
        self.red_low1 = (0, 70, 30)
        self.red_high1 = (10, 255, 255)
        self.red_low2 = (160, 70, 30)
        self.red_high2 = (179, 255, 255)

        self.ground_depth = None

        # ---- NEW thresholds (tuned to be conservative for cylinders) ----
        self.SMALL_AREA = 550         # "small object" regime
        self.TINY_AREA = 280          # "ultra-small" regime

        # Strong box evidence thresholds
        self.RECTANGULARITY_STRONG = 0.78   # contour_area / minAreaRect_area
        self.RECTANGULARITY_GOOD = 0.70
        self.PLANAR_RMSE_STRONG = 0.006     # meters (Gazebo depth is usually stable)
        self.PLANAR_RMSE_GOOD = 0.010
        self.ORTHO_LINE_STRONG = 4          # number of near-orthogonal line votes

        # Cylinder “extra confirmation” for small objects
        self.CYL_CIRC_STRONG = 0.70
        self.CYL_SYM_STRONG = 0.80

        self.get_logger().info("Emergency Detector – ROBUST (small box fix + cylinder safe)")

    # -------------------- ROS callbacks --------------------

    def depth_callback(self, msg):
        try:
            d = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            d = d.astype(np.float32)
            d[d == 0] = np.nan
            self.depth_image = d
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

    # -------------------- Depth helpers --------------------

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
        vals = vals[(vals >= q25 - 1.5 * iqr) & (vals <= q75 + 1.5 * iqr)]

        if len(vals) < 10:
            return 0.0

        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if mean <= 1e-6:
            return 0.0
        return std / mean

    def radial_depth_monotonicity(self, cx, cy, r_min=2, r_max=18, step=1, n_angles=12):
        """
        Cylinder-like: depth increases away from center (object bulges out).
        NOTE: Small planar objects can appear monotonic due to depth noise; we gate this later.
        """
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

    def get_depth_profile(self, c):
        mask = np.zeros(self.depth_image.shape, dtype=np.uint8)
        cv.drawContours(mask, [c], -1, 255, -1)

        depth_vals = self.depth_image[mask == 255]
        depth_vals = depth_vals[np.isfinite(depth_vals) & (depth_vals > 0)]

        if len(depth_vals) < 15:
            return {'is_flat': False, 'depth_cv': 0.0, 'range_ratio': 1.0}

        depth_std = np.std(depth_vals)
        depth_mean = np.mean(depth_vals)
        depth_cv = depth_std / depth_mean if depth_mean > 0 else 0.0

        depth_range = np.max(depth_vals) - np.min(depth_vals)
        depth_range_ratio = depth_range / depth_mean if depth_mean > 0 else 1.0

        is_flat = (depth_cv < 0.02) and (depth_range_ratio < 0.12)
        return {'is_flat': is_flat, 'depth_cv': depth_cv, 'range_ratio': depth_range_ratio}

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
        surrounding = surrounding[(surrounding >= q25 - 1.5 * iqr) & (surrounding <= q75 + 1.5 * iqr)]
        if len(surrounding) < 30:
            return None

        ground_depth = np.median(surrounding)
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

        kernel = np.ones((3, 3), np.uint8)
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

    def compute_depth_symmetry(self, cx, cy, margin=10):
        """
        0 = asymmetric, 1 = symmetric.
        Cylinders tend to be symmetric.
        Boxes (esp rotated) tend to show asymmetric depth profiles due to edges/planes.
        """
        if self.depth_image is None:
            return 0.0

        H, W = self.depth_image.shape
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        symmetry_scores = []

        for dx, dy in directions:
            pos_samples, neg_samples = [], []

            for r in range(1, margin + 1):
                px, py = int(cx + r * dx), int(cy + r * dy)
                if 0 <= px < W and 0 <= py < H:
                    z = self.depth_image[py, px]
                    if np.isfinite(z) and z > 0:
                        pos_samples.append(z)

            for r in range(1, margin + 1):
                px, py = int(cx - r * dx), int(cy - r * dy)
                if 0 <= px < W and 0 <= py < H:
                    z = self.depth_image[py, px]
                    if np.isfinite(z) and z > 0:
                        neg_samples.append(z)

            if len(pos_samples) > 0 and len(neg_samples) > 0:
                min_len = min(len(pos_samples), len(neg_samples))
                if min_len >= 3:
                    pos = np.array(pos_samples[:min_len]) - np.mean(pos_samples[:min_len])
                    neg = np.array(neg_samples[:min_len]) - np.mean(neg_samples[:min_len])
                    if np.std(pos) > 0 and np.std(neg) > 0:
                        corr = np.corrcoef(pos, neg)[0, 1]
                        symmetry_scores.append(abs(corr))

        if not symmetry_scores:
            return 0.0
        return float(np.mean(symmetry_scores))

    def depth_analysis(self, x, y, w, h, cx, cy):
        """
        Uses quadratic fit along horizontal and vertical axes through center for curvature evidence.
        """
        H, W = self.depth_image.shape
        depth_center = -1.0

        if not (0 <= cx < W and 0 <= cy < H):
            return False, False, depth_center, (0.0, 0.0, 0.0), 0.0, 0.0, False

        zc = self.depth_image[cy, cx]
        if not (np.isfinite(zc) and zc > 0):
            return False, False, depth_center, (0.0, 0.0, 0.0), 0.0, 0.0, False

        depth_center = float(zc)
        object_size = max(w, h)

        if depth_center < 1.0:
            margin = max(1, min(object_size // 15, 4))
        elif depth_center < 2.0:
            margin = max(2, min(object_size // 10, 6))
        else:
            margin = max(3, min(object_size // 8, 8))

        # Horizontal samples
        xs_h = np.arange(x + margin, x + w - margin)
        depths_h = []
        for xi in xs_h:
            d = self.depth_image[cy, xi] if 0 <= xi < W else np.nan
            depths_h.append(float(d) if np.isfinite(d) and d > 0 else np.nan)

        depths_h = np.array(depths_h)
        valid_h = ~np.isnan(depths_h)
        if np.sum(valid_h) < 6:
            return True, False, depth_center, self.pixel_to_cam(cx, cy, depth_center), 0.0, 0.0, False

        xs_h_valid = xs_h[valid_h] - cx
        denom = np.max(np.abs(xs_h_valid))
        xs_h_norm = xs_h_valid / denom if denom > 0 else xs_h_valid
        depths_h_valid = depths_h[valid_h]

        if len(depths_h_valid) > 10:
            q75, q25 = np.percentile(depths_h_valid, [75, 25])
            iqr = q75 - q25
            keep = (depths_h_valid >= q25 - 1.5 * iqr) & (depths_h_valid <= q75 + 1.5 * iqr)
            xs_h_norm = xs_h_norm[keep]
            depths_h_valid = depths_h_valid[keep]

        try:
            coeffs_h = np.polyfit(xs_h_norm, depths_h_valid, 2)
            curvature_h = abs(coeffs_h[0])
        except Exception:
            curvature_h = 0.0

        # Vertical samples
        ys_v = np.arange(y + margin, y + h - margin)
        depths_v = []
        for yi in ys_v:
            d = self.depth_image[yi, cx] if 0 <= yi < H else np.nan
            depths_v.append(float(d) if np.isfinite(d) and d > 0 else np.nan)

        depths_v = np.array(depths_v)
        valid_v = ~np.isnan(depths_v)
        if np.sum(valid_v) < 6:
            return True, False, depth_center, self.pixel_to_cam(cx, cy, depth_center), curvature_h, 0.0, False

        ys_v_valid = ys_v[valid_v] - cy
        denom = np.max(np.abs(ys_v_valid))
        ys_v_norm = ys_v_valid / denom if denom > 0 else ys_v_valid
        depths_v_valid = depths_v[valid_v]

        if len(depths_v_valid) > 10:
            q75, q25 = np.percentile(depths_v_valid, [75, 25])
            iqr = q75 - q25
            keep = (depths_v_valid >= q25 - 1.5 * iqr) & (depths_v_valid <= q75 + 1.5 * iqr)
            ys_v_norm = ys_v_norm[keep]
            depths_v_valid = depths_v_valid[keep]

        try:
            coeffs_v = np.polyfit(ys_v_norm, depths_v_valid, 2)
            curvature_v = abs(coeffs_v[0])
        except Exception:
            curvature_v = 0.0

        # Threshold adapts with range/size
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

        is_cylinder = (
            (curvature_h > base_threshold and not curvature_v > base_threshold) or
            (curvature_v > base_threshold and not curvature_h > base_threshold)
        )
        is_curved = (curvature_h > base_threshold) or (curvature_v > base_threshold)

        return True, is_curved, depth_center, self.pixel_to_cam(cx, cy, depth_center), curvature_h, curvature_v, is_cylinder

    def pixel_to_cam(self, u, v, z):
        X = float((u - self.cx) * z / self.fx)
        Y = float((v - self.cy) * z / self.fy)
        Z = float(z)
        return (X, Y, Z)

    # -------------------- NEW features for robust small box detection --------------------

    def rotated_rect_features(self, contour):
        """
        minAreaRect handles rotation robustly.
        Returns:
          rectangularity = area / rect_area (clamped 0..1)
          rect_aspect = max(side)/min(side)
        """
        area = cv.contourArea(contour)
        rect = cv.minAreaRect(contour)
        (rw, rh) = rect[1]
        rect_area = float(rw * rh) if rw > 1e-6 and rh > 1e-6 else 0.0
        if rect_area <= 1e-6:
            return 0.0, 999.0
        rectangularity = float(np.clip(area / rect_area, 0.0, 1.0))
        rect_aspect = float(max(rw, rh) / max(min(rw, rh), 1e-6))
        return rectangularity, rect_aspect

    def depth_planarity_rmse(self, contour, sample_cap=900):
        """
        Fit plane z = a*u + b*v + c to depth points inside contour.
        Planar objects (boxes) -> low RMSE
        Curved objects (cylinders) -> higher RMSE

        Returns:
          rmse (meters), valid(bool)
        """
        if self.depth_image is None:
            return 1.0, False

        H, W = self.depth_image.shape
        mask = np.zeros((H, W), dtype=np.uint8)
        cv.drawContours(mask, [contour], -1, 255, -1)

        ys, xs = np.where(mask == 255)
        if len(xs) < 50:
            return 1.0, False

        # sample points to keep fast
        if len(xs) > sample_cap:
            idx = np.random.choice(len(xs), sample_cap, replace=False)
            xs = xs[idx]
            ys = ys[idx]

        zs = self.depth_image[ys, xs]
        valid = np.isfinite(zs) & (zs > 0)
        xs = xs[valid].astype(np.float32)
        ys = ys[valid].astype(np.float32)
        zs = zs[valid].astype(np.float32)

        if len(zs) < 40:
            return 1.0, False

        # robust trim
        q75, q25 = np.percentile(zs, [75, 25])
        iqr = q75 - q25
        keep = (zs >= q25 - 1.5 * iqr) & (zs <= q75 + 1.5 * iqr)
        xs, ys, zs = xs[keep], ys[keep], zs[keep]
        if len(zs) < 30:
            return 1.0, False

        # least squares plane
        A = np.stack([xs, ys, np.ones_like(xs)], axis=1)  # [u, v, 1]
        try:
            coeff, *_ = np.linalg.lstsq(A, zs, rcond=None)
            z_hat = A @ coeff
            residual = zs - z_hat
            rmse = float(np.sqrt(np.mean(residual * residual)))
            return rmse, True
        except Exception:
            return 1.0, False

    def orthogonal_line_score(self, roi_bgr):
        """
        Detect lines and count evidence of two near-orthogonal dominant directions.
        Returns a score (int). Higher means more box-like.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return 0

        gray = cv.cvtColor(roi_bgr, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 70, 170)

        lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=35,
                               minLineLength=max(12, min(roi_bgr.shape[0], roi_bgr.shape[1]) // 4),
                               maxLineGap=6)
        if lines is None:
            return 0

        angles = []
        for l in lines[:, 0]:
            x1, y1, x2, y2 = l
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) + abs(dy) < 8:
                continue
            ang = math.degrees(math.atan2(dy, dx))
            ang = (ang + 180.0) % 180.0  # 0..180
            angles.append(ang)

        if len(angles) < 3:
            return 0

        angles = np.array(angles, dtype=np.float32)

        # Bin angles into 0..180
        bins = np.linspace(0, 180, 19)  # 10-degree bins
        hist, _ = np.histogram(angles, bins=bins)
        top = np.argsort(hist)[::-1]

        # Take top 2 bins and see if ~90 degrees apart
        b1 = top[0]
        b2 = top[1] if len(top) > 1 else top[0]
        a1 = (bins[b1] + bins[b1 + 1]) * 0.5
        a2 = (bins[b2] + bins[b2 + 1]) * 0.5
        diff = abs(a1 - a2)
        diff = min(diff, 180.0 - diff)

        ortho = 80.0 <= diff <= 100.0
        score = int(hist[b1]) + int(hist[b2]) if ortho else int(hist[b1])
        return score

    # -------------------- Main detection --------------------

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

            # Tall red object check (your rule)
            is_tall = h > 1.4 * w

            circularity = 4 * math.pi * area / (peri * peri)
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
            depth_profile = self.get_depth_profile(c)

            ground_depth = self.estimate_ground_plane(c)
            is_on_ground = self.is_object_on_ground(c, ground_depth)

            symmetry_score = self.compute_depth_symmetry(cx, cy)

            # NEW features for small/rotated box protection
            rectangularity, rect_aspect = self.rotated_rect_features(c)
            planar_rmse, planar_valid = self.depth_planarity_rmse(c)
            ortho_score = self.orthogonal_line_score(obj_roi)

            label, confidence, reason = self.classify(
                is_red=is_red,
                is_curved=is_curved,
                is_cylinder=is_cylinder,
                circularity=circularity,
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
                symmetry_score=symmetry_score,
                rectangularity=rectangularity,
                rect_aspect=rect_aspect,
                planar_rmse=planar_rmse,
                planar_valid=planar_valid,
                ortho_score=ortho_score
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

                dbg = (f"Z:{depth_center:.2f} cv:{depth_cv:.3f} rad:{int(radial_monotonic)} "
                       f"circ:{circularity:.2f} k:{corners} tall:{int(is_tall)} sym:{symmetry_score:.2f} "
                       f"rect:{rectangularity:.2f} plane:{planar_rmse:.3f} ortho:{ortho_score}")
                cv.putText(frame, dbg, (x, y + h + 15),
                           cv.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)

                self.get_logger().info(
                    f"[DET] {label:18} | X:{map_pose.position.x:6.2f} Y:{map_pose.position.y:6.2f} Z:{map_pose.position.z:6.2f} | "
                    f"Circ:{circularity:.2f} Corn:{corners} Rect:{rectangularity:.2f} PlaneRMSE:{planar_rmse:.4f} "
                    f"Rad:{radial_monotonic} Cyl:{is_cylinder} Sym:{symmetry_score:.2f} | {reason}"
                )

            except Exception as e:
                self.get_logger().warn(f"TF2 error: {e}")

    # -------------------- Classification logic --------------------

    def classify(self, *,
                 is_red, is_curved, is_cylinder, circularity, solidity, corners, edge_count, area,
                 depth_center, depth_cv, radial_monotonic, is_on_ground, depth_profile, is_tall,
                 symmetry_score, rectangularity, rect_aspect, planar_rmse, planar_valid, ortho_score):

        # Non-red objects are first aid kits in your world
        if not is_red:
            return "first_aid_kit", 0.85, "Non-red object"

        # --------- Strong BOX evidence (works for small + rotated boxes) ----------
        strong_box_evidence = False
        box_reasons = []

        if rectangularity >= self.RECTANGULARITY_STRONG:
            strong_box_evidence = True
            box_reasons.append("high rectangularity")

        if planar_valid and planar_rmse <= self.PLANAR_RMSE_STRONG:
            strong_box_evidence = True
            box_reasons.append("strong planarity")

        if ortho_score >= self.ORTHO_LINE_STRONG:
            strong_box_evidence = True
            box_reasons.append("orthogonal lines")

        # Supportive (weaker) box evidence
        weak_box_evidence = (
            (rectangularity >= self.RECTANGULARITY_GOOD) or
            (planar_valid and planar_rmse <= self.PLANAR_RMSE_GOOD) or
            (corners >= 4 and solidity > 0.90) or
            (depth_profile and depth_profile.get('is_flat', False))
        )

        # --------- Cylinder evidence (but guarded for small planar boxes) ----------
        strong_cylinder_evidence = (
            (radial_monotonic and circularity >= self.CYL_CIRC_STRONG) or
            (is_cylinder and circularity >= 0.65) or
            (is_tall and (radial_monotonic or is_cylinder or depth_cv > 0.012)) or
            (symmetry_score > self.CYL_SYM_STRONG and circularity >= 0.75 and depth_cv > 0.015)
        )

        # ------------------ 1) SMALL OBJECT SPECIAL GUARD ------------------
        # If it's small and looks like a planar rectangle, force BOX.
        if area <= self.SMALL_AREA:
            # If we have strong box evidence, do NOT allow "radial_monotonic noise" to flip it.
            if strong_box_evidence or (weak_box_evidence and planar_valid and planar_rmse <= self.PLANAR_RMSE_GOOD):
                return "first_aid_kit", 0.97, f"Small object + box evidence ({', '.join(box_reasons) if box_reasons else 'planar/rect'})"

            # For small objects, require EXTRA cylinder confirmation
            if strong_cylinder_evidence:
                # if planar too flat, block cylinder
                if planar_valid and planar_rmse <= self.PLANAR_RMSE_GOOD and rectangularity >= self.RECTANGULARITY_GOOD:
                    return "first_aid_kit", 0.95, "Small object: cylinder cues but planar+rect blocks it"
                return "fire_extinguisher", 0.95, "Small object: strong cylinder evidence (guarded)"

            # If nothing strong, default box (safer for your complaint)
            return "first_aid_kit", 0.90, "Small object: no strong cylinder evidence"

        # ------------------ 2) ROTATED BOX DETECTION (global) ------------------
        # Rotated boxes often show moderate circularity but low symmetry + good rectangularity/planarity.
        if (0.50 < circularity < 0.86) and (symmetry_score < 0.60) and (rectangularity >= self.RECTANGULARITY_GOOD):
            return "first_aid_kit", 0.95, "Rotated box: moderate circularity + low symmetry + rectangularity"

        # ------------------ 3) TALL RED OBJECT (extinguisher) ------------------
        if is_tall and (radial_monotonic or is_cylinder or depth_cv > 0.012):
            # If it's tall but also extremely planar+rectangular, treat as box
            if planar_valid and planar_rmse <= self.PLANAR_RMSE_STRONG and rectangularity >= self.RECTANGULARITY_STRONG:
                return "first_aid_kit", 0.93, "Tall but planar+rectangular => box"
            return "fire_extinguisher", 0.96, "Tall red object + depth cue => fire extinguisher"

        # ------------------ 4) CLEAR BOX RULES ------------------
        if corners == 4 and solidity > 0.92 and edge_count > 300:
            return "first_aid_kit", 0.98, "Clear box: 4 corners + high solidity + edges"

        if planar_valid and planar_rmse <= self.PLANAR_RMSE_STRONG and rectangularity >= self.RECTANGULARITY_GOOD:
            return "first_aid_kit", 0.97, "Planar + rectangular => box"

        if depth_profile and depth_profile.get('is_flat', False):
            return "first_aid_kit", 0.96, "Flat depth profile => box"

        # ------------------ 5) CYLINDER RULES ------------------
        # Near/Far regime
        near = depth_center > 0 and depth_center < 0.8
        far = depth_center > 2.5
        if near or far:
            # Block cylinder if object appears planar rectangle
            if planar_valid and planar_rmse <= self.PLANAR_RMSE_GOOD and rectangularity >= self.RECTANGULARITY_GOOD:
                return "first_aid_kit", 0.93, "Near/Far: planar+rect blocks cylinder"
            if radial_monotonic or is_cylinder or depth_cv > 0.008:
                return "fire_extinguisher", 0.95, "Near/Far: depth physics cue"
            return "first_aid_kit", 0.90, "Near/Far: no depth cue"

        # Primary depth cues (guard with planarity)
        if radial_monotonic:
            if planar_valid and planar_rmse <= self.PLANAR_RMSE_GOOD and rectangularity >= self.RECTANGULARITY_GOOD:
                return "first_aid_kit", 0.92, "Radial monotonic but planar+rect => box"
            return "fire_extinguisher", 0.96, "Radial monotonicity => cylinder"

        if is_cylinder:
            if planar_valid and planar_rmse <= self.PLANAR_RMSE_GOOD and rectangularity >= self.RECTANGULARITY_GOOD:
                return "first_aid_kit", 0.92, "Curvature cue but planar+rect => box"
            return "fire_extinguisher", 0.94, "3D curvature => cylinder"

        # Size-aware circularity
        circularity_threshold = 0.80 if area < 400 else 0.86 if area < 800 else 0.88
        if circularity >= circularity_threshold:
            # Again block if planar rectangle
            if planar_valid and planar_rmse <= self.PLANAR_RMSE_GOOD and rectangularity >= self.RECTANGULARITY_GOOD:
                return "first_aid_kit", 0.91, "High circularity but planar+rect => box"
            return "fire_extinguisher", 0.98, "High circularity"

        if circularity >= 0.75:
            if planar_valid and planar_rmse <= self.PLANAR_RMSE_GOOD and rectangularity >= self.RECTANGULARITY_GOOD:
                return "first_aid_kit", 0.90, "Moderate circularity but planar+rect => box"
            return "fire_extinguisher", 0.90, "Moderate circularity"

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
