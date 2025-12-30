#!/usr/bin/env python3
"""
INVENTORY AGGREGATOR – CMP9767 COMPLIANT
Based on workshop counter_3d.py with enhanced features
"""

import rclpy
from rclpy.node import Node
from rclpy import qos
from collections import defaultdict
import math
import json
import time
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, PoseArray, Point
from visualization_msgs.msg import Marker, MarkerArray


class InventoryAggregator(Node):
    # Workshop parameter - adjust based on your object spacing
    detection_threshold = 0.5  # meters

    def __init__(self):
        super().__init__('inventory_aggregator')
        
        # Track objects separately by class
        self.detected_objects = {
            'fire_extinguisher': [],
            'first_aid_kit': []
        }
        
        # Subscribe to emergency_detector output
        self.subscriber = self.create_subscription(
            PoseStamped, 
            '/emergency_detections',  # Must match emergency_detector's publish topic
            self.counter_callback,
            qos_profile=qos.qos_profile_sensor_data
        )
        
        # Publishers
        self.pose_array_pub = self.create_publisher(PoseArray, '/object_count_array', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/inventory_markers', 10)
        
        # JSON export
        self.export_file = "/tmp/inventory_report.json"
        self.report_timer = self.create_timer(5.0, self.generate_report)

        self.get_logger().info("✅ Inventory Aggregator started")

    def counter_callback(self, msg):
        # Extract class from frame_id (published by emergency_detector)
        obj_class = msg.header.frame_id
        new_object = msg.pose
        
        if obj_class not in ['fire_extinguisher', 'first_aid_kit']:
            return
            
        # Check if this object is already counted (de-duplication)
        object_exists = False
        for existing_obj in self.detected_objects[obj_class]:
            d = self.calculate_distance(existing_obj.position, new_object.position)
            if d < self.detection_threshold:
                object_exists = True
                break
        
        # Only add new objects
        if not object_exists:
            self.detected_objects[obj_class].append(new_object)
            self.get_logger().info(f"✅ New {obj_class} detected! Total: {len(self.detected_objects[obj_class])}")
        
        # Publish for RViz visualization
        self.publish_visualizations(msg.header.frame_id)

    def calculate_distance(self, pos_a, pos_b):
        """Calculate 3D Euclidean distance"""
        return math.sqrt(
            (pos_a.x - pos_b.x) ** 2 + 
            (pos_a.y - pos_b.y) ** 2 + 
            (pos_a.z - pos_b.z) ** 2
        )

    def publish_visualizations(self, frame_id):
        """Publish both PoseArray and MarkerArray for RViz"""
        # PoseArray (from workshop)
        pose_array = PoseArray(header=Header(frame_id="map"))
        all_objects = self.detected_objects['fire_extinguisher'] + self.detected_objects['first_aid_kit']
        for obj in all_objects:
            pose_array.poses.append(obj)
        self.pose_array_pub.publish(pose_array)
        
        # MarkerArray (enhanced)
        marker_array = MarkerArray()
        marker_id = 0
        
        for obj_class, objects in self.detected_objects.items():
            for obj in objects:
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "inventory_objects"
                marker.id = marker_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose = obj
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = 0.3
                marker.color.a = 0.8
                
                if obj_class == "fire_extinguisher":
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                else:
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                
                marker_array.markers.append(marker)
                marker_id += 1
        
        self.marker_pub.publish(marker_array)

    def generate_report(self):
        """Generate terminal report and JSON export"""
        fire_ext_count = len(self.detected_objects['fire_extinguisher'])
        first_aid_count = len(self.detected_objects['first_aid_kit'])
        total_count = fire_ext_count + first_aid_count
        
        if total_count == 0:
            self.get_logger().info("📊 INVENTORY REPORT: No objects detected")
            return
            
        # Terminal report
        self.get_logger().info(
            "\n" + "="*50 + "\n"
            f"📊 INVENTORY REPORT (CMP9767 COMPLIANT)\n"
            f"Fire Extinguishers: {fire_ext_count}\n"
            f"First Aid Kits:     {first_aid_count}\n"
            f"Total Objects:      {total_count}\n"
            "="*50
        )
        
        # JSON export
        report_data = {
            'timestamp': time.time(),
            'counts': {
                'fire_extinguisher': fire_ext_count,
                'first_aid_kit': first_aid_count
            },
            'poses': {
                'fire_extinguisher': [self.pose_to_list(pose) for pose in self.detected_objects['fire_extinguisher']],
                'first_aid_kit': [self.pose_to_list(pose) for pose in self.detected_objects['first_aid_kit']]
            },
            'total_objects': total_count
        }
        
        try:
            with open(self.export_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            self.get_logger().info(f"📄 Report exported to: {self.export_file}")
        except Exception as e:
            self.get_logger().warn(f"Failed to export JSON: {e}")

    def pose_to_list(self, pose):
        """Convert Pose to [x, y, z] list for JSON"""
        return [pose.position.x, pose.position.y, pose.position.z]


def main(args=None):
    rclpy.init(args=args)
    node = InventoryAggregator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()