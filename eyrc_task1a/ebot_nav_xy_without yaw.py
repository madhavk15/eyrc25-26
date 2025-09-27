import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math


class SimpleBug2(Node):
    def __init__(self):
        super().__init__('simple_bug2_node')

        # Publishers & Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Robot pose
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Waypoints
        self.waypoints = [(-1.53, -1.95), (0.13, 1.24), (0.38, -3.32)]
        self.current_wp_index = 0
        self.goal_thresh = 0.15

        # Lidar info
        self.lidar_ranges = []
        self.angle_min = -math.pi/2
        self.angle_max = math.pi/2

        # Control states
        self.state = "idle"
        self.forward_target_y = None

        self.get_logger().info("✅ SimpleBug2 Node Started")

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        # quaternion → yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        self.lidar_ranges = msg.ranges

        if self.current_wp_index >= len(self.waypoints):
            self.stop_robot()
            self.get_logger().info("🎯 All waypoints reached!")
            return

        # Current goal
        x_goal, y_goal = self.waypoints[self.current_wp_index]
        dist_to_goal = math.hypot(x_goal - self.x, y_goal - self.y)

        if dist_to_goal < self.goal_thresh:
            self.get_logger().info(f"🎯 Waypoint {self.current_wp_index+1} reached!")
            self.current_wp_index += 1
            self.state = "idle"
            return

        # Handle idle state → decide path
        if self.state == "idle":
            self.get_logger().info("🔄 Checking M-line...")
            blocked = self.is_obstacle_on_mline(x_goal, y_goal)

            if not blocked:
                self.get_logger().info("✅ M-line clear → go to waypoint directly")
                self.state = "go_to_goal"
            else:
                self.get_logger().info("🚧 M-line blocked → moving forward |dy| before turning")
                self.forward_target_y = y_goal
                self.state = "forward_then_turn"

        # Handle direct goal
        if self.state == "go_to_goal":
            if self.is_front_blocked():
                self.get_logger().warn("⛔ Front blocked while on M-line → stopping")
                self.stop_robot()
            else:
                self.move_to_goal(x_goal, y_goal)

        # Handle forward step
        elif self.state == "forward_then_turn":
            if self.is_front_blocked():
                self.get_logger().warn("⛔ Front blocked during forward step → stop")
                self.stop_robot()
                return
            if abs(self.y - self.forward_target_y) > 0.05:
                self.get_logger().info("➡️ Moving forward until |dy| reached")
                self.move_forward()
            else:
                self.get_logger().info("↪️ Forward step complete → turning right")
                self.turn_right()
                self.state = "go_to_goal"

    # --- Movement primitives ---
    def move_forward(self, speed=0.15):
        twist = Twist()
        twist.linear.x = speed
        self.cmd_pub.publish(twist)

    def move_to_goal(self, x_goal, y_goal):
        angle_to_goal = math.atan2(y_goal - self.y, x_goal - self.x)
        angle_diff = self.normalize_angle(angle_to_goal - self.yaw)

        twist = Twist()
        twist.linear.x = 0.2
        twist.angular.z = 0.8 * angle_diff
        self.cmd_pub.publish(twist)
        self.get_logger().info(f"🚀 Heading to waypoint: ({x_goal:.2f}, {y_goal:.2f})")

    def turn_right(self):
        twist = Twist()
        twist.angular.z = -0.6
        for _ in range(15):  # turn for fixed iterations (~π/2 approx)
            self.cmd_pub.publish(twist)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    # --- Helpers ---
    def is_obstacle_on_mline(self, x_goal, y_goal):
        # check straight line (robot → waypoint)
        angle_to_goal = math.atan2(y_goal - self.y, x_goal - self.x)
        rel_angle = self.normalize_angle(angle_to_goal - self.yaw)

        if not (-math.pi/2 <= rel_angle <= math.pi/2):
            return False  # not in lidar FOV

        idx = int((rel_angle - self.angle_min) / (self.angle_max - self.angle_min) * len(self.lidar_ranges))
        idx = max(0, min(idx, len(self.lidar_ranges)-1))
        obs_dist = self.lidar_ranges[idx]

        dist_to_goal = math.hypot(x_goal - self.x, y_goal - self.y)
        return (not math.isinf(obs_dist)) and obs_dist < dist_to_goal

    def is_front_blocked(self, threshold=0.4):
        if not self.lidar_ranges:
            return False
        mid_idx = len(self.lidar_ranges) // 2
        return self.lidar_ranges[mid_idx] < threshold

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = SimpleBug2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
