import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math


class Bug2(Node):
    def __init__(self):
        super().__init__('bug2_node')

        # Publishers & Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Timer for odom logging (every 5 seconds)
        self.create_timer(5.0, self.print_odom)

        # Robot pose
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Waypoints (add as many as needed)
        self.waypoints = [(-1.53, -1.95), (0.13, 1.24)]  # Example: two waypoints
        self.current_wp_index = 0
        self.goal_thresh = 0.1

        # State
        self.hit_point_distance = None
        self.following_wall = False
        self.front_clear_dist = 0.5  # meters

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def print_odom(self):
        self.get_logger().info(f"Robot pose: x={self.x:.2f}, y={self.y:.2f}, yaw={self.yaw:.2f} rad")

    def scan_callback(self, msg):
        if self.current_wp_index >= len(self.waypoints):
            twist = Twist()
            self.cmd_pub.publish(twist)  # stop robot
            self.get_logger().info("All waypoints reached!")
            return

        # Current goal
        x_goal, y_goal = self.waypoints[self.current_wp_index]

        # Check front obstacle
        front_angles = range(-10, 11)  # approx front in degrees
        # Ensure we only use valid indices
        front_ranges = [msg.ranges[i] for i in front_angles if 0 <= i < len(msg.ranges) and not math.isinf(msg.ranges[i])]
        front_min = min(front_ranges) if front_ranges else float('inf')

        twist = Twist()

        # Distance to current waypoint
        dist_to_goal = math.hypot(x_goal - self.x, y_goal - self.y)

        if dist_to_goal < self.goal_thresh:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            self.get_logger().info(f"Waypoint {self.current_wp_index+1} reached!")
            self.current_wp_index += 1
            self.following_wall = False
            return

        # Bug2 logic
        if not self.following_wall:
            if front_min < self.front_clear_dist:
                # Hit obstacle, start wall-following
                self.following_wall = True
                self.hit_point_distance = dist_to_goal
            else:
                # Move along M-line
                angle_to_goal = math.atan2(y_goal - self.y, x_goal - self.x)
                angle_diff = self.normalize_angle(angle_to_goal - self.yaw)
                twist.linear.x = 0.2
                twist.angular.z = 1.0 * angle_diff
        else:
            # Wall-following
            if front_min < self.front_clear_dist:
                twist.linear.x = 0.0
                twist.angular.z = 0.3  # turn left
            else:
                twist.linear.x = 0.1
                twist.angular.z = 0.0

            # Check if M-line reached closer to goal
            angle_to_goal = math.atan2(y_goal - self.y, x_goal - self.x)
            angle_diff = abs(self.normalize_angle(angle_to_goal - self.yaw))
            if angle_diff < 0.1 and dist_to_goal < self.hit_point_distance:
                self.following_wall = False

        self.cmd_pub.publish(twist)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = Bug2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
