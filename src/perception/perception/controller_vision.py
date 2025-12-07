import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data

from geometry_msgs.msg import Twist
from geometry_msgs.msg import PolygonStamped


class VisionController(Node):
    def __init__(self):
        super().__init__('vision_controller')

        # Parameters
        self.declare_parameter('image_width', 640)
        self.declare_parameter('mode', 'bang_bang')      # 'bang_bang', 'bang_bang_hyst', 'proportional'
        self.declare_parameter('ang_speed', 0.4)         # rad/s for bang-bang
        self.declare_parameter('hyst_threshold', 30.0)   # pixels
        self.declare_parameter('k_p', 0.002)             # rad/s per pixel
        self.declare_parameter('max_ang_speed', 0.6)     # limit for proportional

        self.image_width = float(self.get_parameter('image_width').value)

        # Subscribe to detector output (sensor QoS is fine here)
        self.sub_polygon = self.create_subscription(
            PolygonStamped,
            '/object_polygon',
            self.polygon_callback,
            qos_profile_sensor_data
        )

        # Publish velocity commands with RELIABLE QoS for /cmd_vel
        qos_cmd = QoSProfile(
            depth=10,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            qos_cmd
        )

        self.get_logger().info('Vision-based controller started.')

    def polygon_callback(self, msg: PolygonStamped):
        if not msg.polygon.points:
            # No object detected -> stop turning
            self.publish_twist(0.0)
            return

        # Task 1 encoding: point[0] = (x_min, y_min), point[1] = (width, height)
        x_min = msg.polygon.points[0].x
        width = msg.polygon.points[1].x

        obj_center_x = x_min + width / 2.0
        img_center_x = self.image_width / 2.0
        error = img_center_x - obj_center_x  # +ve => object is to the left

        mode = self.get_parameter('mode').value
        if mode == '':
            mode = 'bang_bang'

        if mode == 'bang_bang':
            self.control_bang_bang(error)
        elif mode == 'bang_bang_hyst':
            self.control_bang_bang_hysteresis(error)
        elif mode == 'proportional':
            self.control_proportional(error)
        else:
            self.get_logger().warn(f'Unknown mode "{mode}", stopping.')
            self.publish_twist(0.0)

    def control_bang_bang(self, error: float):
        ang_speed = float(self.get_parameter('ang_speed').value)

        if error > 0.0:
            # Object to the left -> turn left (positive angular z)
            wz = ang_speed
        elif error < 0.0:
            # Object to the right -> turn right (negative angular z)
            wz = -ang_speed
        else:
            wz = 0.0

        self.publish_twist(wz)

    def control_bang_bang_hysteresis(self, error: float):
        ang_speed = float(self.get_parameter('ang_speed').value)
        threshold = float(self.get_parameter('hyst_threshold').value)

        if error > threshold:
            wz = ang_speed
        elif error < -threshold:
            wz = -ang_speed
        else:
            # Inside deadzone -> do not rotate
            wz = 0.0

        self.publish_twist(wz)

    def control_proportional(self, error: float):
        k_p = float(self.get_parameter('k_p').value)
        max_wz = float(self.get_parameter('max_ang_speed').value)

        wz = k_p * error
        # Saturate
        if wz > max_wz:
            wz = max_wz
        elif wz < -max_wz:
            wz = -max_wz

        self.publish_twist(wz)

    def publish_twist(self, angular_z: float):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = float(angular_z)
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = VisionController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
