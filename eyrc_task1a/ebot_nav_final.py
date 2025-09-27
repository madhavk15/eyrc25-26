import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math

class FinalBug2(Node):
    def __init__(self):
        super().__init__('final_bug2_node')

        # Publishers & Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Robot pose
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Waypoints (x, y, yaw)
        self.waypoints = [
            (-1.53, -1.95, 1.57),
            (0.13, 1.24, 0.0),
            (0.38, -3.32, -1.57)
        ]
        self.current_wp_index = 0

        # Parameters
        self.KP_LIN = 1.0
        self.KP_ANG = 1.8
        self.MAX_LIN = 0.5
        self.MAX_ANG = 1.0
        self.POS_THRESH = 0.05
        self.YAW_THRESH = 0.05
        self.SAFE_FRONT = 0.5

        # Control states
        self.state = "idle"
        self.forward_target_y = None

        # Lidar info
        self.lidar_ranges = []
        self.angle_min = -math.pi/2
        self.angle_max = math.pi/2

        self.get_logger().info("✅ FinalBug2 Node Started")

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        self.lidar_ranges = msg.ranges

        if self.current_wp_index >= len(self.waypoints):
            self.stop_robot()
            self.get_logger().info("🎯 All waypoints reached!")
            return

        x_goal, y_goal, yaw_goal = self.waypoints[self.current_wp_index]
        dist_to_goal = math.hypot(x_goal - self.x, y_goal - self.y)
        yaw_error = self.normalize_angle(yaw_goal - self.yaw)

        # Check if waypoint reached
        if dist_to_goal < self.POS_THRESH and abs(yaw_error) < self.YAW_THRESH:
            self.get_logger().info(f"🎯 Waypoint {self.current_wp_index+1} reached!")
            self.get_logger().info(f"    Actual Pose: x={self.x:.2f}, y={self.y:.2f}, yaw={self.yaw:.2f}")
            self.get_logger().info(f"    Waypoint: x={x_goal:.2f}, y={y_goal:.2f}, yaw={yaw_goal:.2f}")
            self.current_wp_index += 1
            self.state = "idle"
            return

        # Idle state: check M-line
        if self.state == "idle":
            blocked = self.is_obstacle_on_mline(x_goal, y_goal)
            if not blocked:
                self.get_logger().info("✅ M-line clear → align yaw first")
                self.state = "align_yaw_then_move"
            else:
                self.get_logger().info("🚧 M-line blocked → moving forward |dy| first")
                self.forward_target_y = y_goal
                self.state = "forward_then_turn"

        # Case: M-line clear
        if self.state == "align_yaw_then_move":
            if abs(self.normalize_angle(math.atan2(y_goal - self.y, x_goal - self.x) - self.yaw)) > self.YAW_THRESH:
                self.get_logger().info("↪️ Aligning yaw to M-line")
                self.angular_control(math.atan2(y_goal - self.y, x_goal - self.x))
            else:
                self.get_logger().info("➡️ Moving along M-line to waypoint")
                if self.is_front_blocked():
                    self.get_logger().warn("⛔ Front blocked while moving straight → stopping")
                    self.stop_robot()
                else:
                    self.proportional_move(x_goal, y_goal, yaw_goal)
        
        # Case: Obstacle on M-line
        elif self.state == "forward_then_turn":
            if self.is_front_blocked():
                self.get_logger().warn("⛔ Front blocked during forward step → stop")
                self.stop_robot()
            elif abs(self.y - self.forward_target_y) > 0.05:
                self.get_logger().info("➡️ Moving forward until |dy| reached")
                self.proportional_forward_step()
            else:
                self.get_logger().info("↪️ Forward step complete → rotate toward waypoint")
                self.angular_control(math.atan2(y_goal - self.y, x_goal - self.x))
                self.state = "align_yaw_then_move"

    # --- Movement functions ---
    def proportional_move(self, x_goal, y_goal, yaw_goal):
        dist_error = math.hypot(x_goal - self.x, y_goal - self.y)
        angle_to_goal = math.atan2(y_goal - self.y, x_goal - self.x)
        yaw_error = self.normalize_angle(yaw_goal - self.yaw)
        angle_error = self.normalize_angle(angle_to_goal - self.yaw)

        twist = Twist()
        twist.linear.x = min(self.KP_LIN * dist_error, self.MAX_LIN)
        twist.angular.z = min(self.KP_ANG * angle_error, self.MAX_ANG)
        self.cmd_pub.publish(twist)
        self.get_logger().info(f"🚀 Moving: dist_error={dist_error:.2f}, angle_error={angle_error:.2f}")

    def proportional_forward_step(self):
        dy_error = abs(self.forward_target_y - self.y)
        twist = Twist()
        twist.linear.x = min(self.KP_LIN * dy_error, self.MAX_LIN)
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        self.get_logger().info(f"➡️ Forward step: dy_error={dy_error:.2f}")

    def angular_control(self, target_yaw):
        yaw_error = self.normalize_angle(target_yaw - self.yaw)
        twist = Twist()
        twist.angular.z = min(self.KP_ANG * yaw_error, self.MAX_ANG)
        self.cmd_pub.publish(twist)
        self.get_logger().info(f"↪️ Rotating: yaw_error={yaw_error:.2f}")

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    # --- Helpers ---
    def is_obstacle_on_mline(self, x_goal, y_goal):
        angle_to_goal = math.atan2(y_goal - self.y, x_goal - self.x)
        rel_angle = self.normalize_angle(angle_to_goal - self.yaw)
        if not (-math.pi/2 <= rel_angle <= math.pi/2):
            return False
        idx = int((rel_angle - self.angle_min) / (self.angle_max - self.angle_min) * len(self.lidar_ranges))
        idx = max(0, min(idx, len(self.lidar_ranges)-1))
        obs_dist = self.lidar_ranges[idx]
        dist_to_goal = math.hypot(x_goal - self.x, y_goal - self.y)
        return (not math.isinf(obs_dist)) and obs_dist < dist_to_goal

    def is_front_blocked(self):
        if not self.lidar_ranges:
            return False
        mid_idx = len(self.lidar_ranges) // 2
        return self.lidar_ranges[mid_idx] < self.SAFE_FRONT

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = FinalBug2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
