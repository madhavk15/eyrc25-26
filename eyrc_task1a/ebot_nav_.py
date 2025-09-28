'''
# Team ID:          < eYRC#2293 >
# Theme:            < Krishi coBot >
# Author List:      < Madhav Kalra, Ashish Sinha, Tanay Maurya >
# Filename:         ebot_nav.py
# Functions:        odom_callback, scan_callback, proportional_move, proportional_forward_step,
#                   angular_control, stop_robot, is_obstacle_on_mline, is_front_blocked, normalize_angle, main
# Global variables: None
'''

'''
The following waypoint navigation algorithm is a modified version of the Bug1/Bug2 algorithm.
The bot creates a m-line which is the shortest line between its current position and the goal.
In a conventional Bug1/Bug2 algo, the bot would follow the m-line until it encounters an obstacle, and hug the obstacle until it reaches the m-line again.

But, here since this a rather less complex environment, we have modified the algorithm to suit our needs.
The environment consists of only corridors, surroundes by plants, and no complex obstacles.

We divided the approach into two main states:
1. If the m-line is clear, without any obstacle the bot aligns its yaw to the m-line and moves straight to the goal.
2. If the m-line is blocked, the bot moves forward along the y-axis until the absolute difference in y-coordinates of the bot and the goal is less than a certain threshold (0.05m).
If there is an obstacle in between, it would start hugging the obstacle until it can move forward again. But we don't require this approach in the given environment. 
Then, it rotates to align with the m-line and continues to move towards the goal.

This approach works because of the simple environment, where the adjacent obstacle walls are at 90 degrees
'''

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math


class EbotNav(Node):
    '''
    Purpose:
    ---
    This class implements the Bug2 navigation algorithm for a differential drive robot using ROS2. 
    It handles odometry, lidar-based obstacle detection, and waypoint following.
    '''

    def __init__(self):
        '''
        Purpose:
        ---
        Initializes the node, publishers, subscribers, parameters, and state variables.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        node = EbotNav()
        '''

        super().__init__('ebot_nav_node')

        # Publishers & Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)  # Publishes velocity commands
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)  # Subscribes to lidar
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)  # Subscribes to odometry

        # Robot pose variables
        self.x = 0.0      # Current x position from odometry
        self.y = 0.0      # Current y position from odometry
        self.yaw = 0.0    # Current yaw angle from odometry

        # Waypoints: list of (x, y, yaw) [given by eYantra]
        self.waypoints = [
            (-1.53, -1.95, 1.57),
            (0.13, 1.24, 0.0),
            (0.38, -3.32, -1.57)
        ]
        self.current_wp_index = 0  # Index of current waypoint being followed

        # Control parameters
        self.KP_LIN = 1.2      # Proportional gain for linear velocity
        self.KP_ANG = 2.0      # Proportional gain for angular velocity
        self.MAX_LIN = 1.0     # Maximum linear velocity
        self.MAX_ANG = 1.0     # Maximum angular velocity
        self.POS_THRESH = 0.08 # Position threshold to consider waypoint reached
        self.YAW_THRESH = 0.08 # Yaw threshold to consider orientation achieved
        self.SAFE_FRONT = 0.5  # Safe distance in front of robot to avoid obstacles

        # Control state machine
        self.state = "idle"            # Navigation state: idle, align_yaw_then_move, forward_then_turn
        self.forward_target_y = None   # Target y used in forward step for obstacle handling

        # Lidar info
        self.lidar_ranges = []             # Stores current lidar scan ranges
        self.angle_min = -math.pi/2        # Lidar minimum angle (-90°)
        self.angle_max = math.pi/2         # Lidar maximum angle (+90°)

        self.get_logger().info("EbotNav Node Started")

    def odom_callback(self, msg):
        '''
        Purpose:
        ---
        Updates the robot's current position (x, y) and yaw from odometry messages.

        Input Arguments:
        ---
        `msg` :  [ nav_msgs.msg.Odometry ]
            ROS2 Odometry message containing robot's pose.

        Returns:
        ---
        None

        Example call:
        ---
        self.odom_callback(msg)
        '''
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # Convert quaternion to yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        '''
        Purpose:
        ---
        Processes lidar scan data, checks obstacles, and performs navigation logic depending on the current state.

        Input Arguments:
        ---
        `msg` :  [ sensor_msgs.msg.LaserScan ]
            ROS2 LaserScan message containing lidar readings.

        Returns:
        ---
        None

        Example call:
        ---
        self.scan_callback(msg)
        '''
        self.lidar_ranges = msg.ranges

        # Check if all waypoints are completed
        if self.current_wp_index >= len(self.waypoints):
            self.stop_robot()
            self.get_logger().info("All waypoints reached!")
            return

        x_goal, y_goal, yaw_goal = self.waypoints[self.current_wp_index]
        dist_to_goal = math.hypot(x_goal - self.x, y_goal - self.y) #calculate distance to goal on m-line
        yaw_error = self.normalize_angle(yaw_goal - self.yaw) #calculate yaw error to goal

        # Waypoint reached condition. Prints the actual and goal pose in terminal
        if dist_to_goal < self.POS_THRESH and abs(yaw_error) < self.YAW_THRESH:
            self.get_logger().info(f"Waypoint {self.current_wp_index+1} reached")
            self.get_logger().info(f"Actual Pose: x={self.x:.2f}, y={self.y:.2f}, yaw={self.yaw:.2f}")
            self.get_logger().info(f"Waypoint: x={x_goal:.2f}, y={y_goal:.2f}, yaw={yaw_goal:.2f}")
            self.current_wp_index += 1
            self.state = "idle"
            return

        # State machine logic
        if self.state == "idle":
            # Check M-line for obstacles
            blocked = self.is_obstacle_on_mline(x_goal, y_goal)
            if not blocked:
                self.get_logger().info("M-line clear -> align yaw first")
                self.state = "align_yaw_then_move"
            else:
                self.get_logger().info("M-line blocked -> moving forward |dy| first")
                self.forward_target_y = y_goal
                self.state = "forward_then_turn"

        if self.state == "align_yaw_then_move":
            # Rotate to align with M-line before moving
            if abs(self.normalize_angle(math.atan2(y_goal - self.y, x_goal - self.x) - self.yaw)) > self.YAW_THRESH:
                self.get_logger().info("Aligning yaw to M-line")
                self.angular_control(math.atan2(y_goal - self.y, x_goal - self.x)) #P control to align yaw
            else:
                # Once aligned, move forward if safe
                self.get_logger().info("Moving along M-line to waypoint")
                if self.is_front_blocked():
                    self.get_logger().warn("Front blocked while moving straight -> stopping")
                    self.stop_robot()
                else:
                    self.proportional_move(x_goal, y_goal, yaw_goal) #P control to move to waypoint

        elif self.state == "forward_then_turn":
            # Move forward along y until dy condition satisfied
            if self.is_front_blocked():
                self.get_logger().warn("Front blocked during forward step -> stop")
                self.stop_robot()
            elif abs(self.y - self.forward_target_y) > 0.05: #threshold for forward step
                self.get_logger().info("Moving forward until |dy| reached")
                self.proportional_forward_step() #P control to move forward along y
            else:
                self.get_logger().info("Forward step complete -> rotate toward waypoint")
                self.angular_control(math.atan2(y_goal - self.y, x_goal - self.x)) #P control to align yaw to M-line
                self.state = "align_yaw_then_move" #Switch to align and move state

    def proportional_move(self, x_goal, y_goal, yaw_goal):
        '''
        Purpose:
        ---
        Generates linear and angular velocity commands to move robot proportionally towards a given waypoint - when no obstacle on m line.

        Input Arguments:
        ---
        `x_goal` :  [ float ]
            Target x-coordinate

        `y_goal` :  [ float ]
            Target y-coordinate

        `yaw_goal` :  [ float ]
            Target yaw orientation

        Returns:
        ---
        None
            Publishes a geometry_msgs.msg.Twist message to /cmd_vel

        Example call:
        ---
        self.proportional_move(1.0, 2.0, 0.0)
        '''
        dist_error = math.hypot(x_goal - self.x, y_goal - self.y) #Euclidean distance to goal
        angle_to_goal = math.atan2(y_goal - self.y, x_goal - self.x)
        yaw_error = self.normalize_angle(yaw_goal - self.yaw)
        angle_error = self.normalize_angle(angle_to_goal - self.yaw) #Angle difference to goal

        twist = Twist()
        twist.linear.x = min(self.KP_LIN * dist_error, self.MAX_LIN)
        twist.angular.z = min(self.KP_ANG * angle_error, self.MAX_ANG)
        self.cmd_pub.publish(twist)
        self.get_logger().info(f"Moving: dist_error={dist_error:.2f}, angle_error={angle_error:.2f}")

    def proportional_forward_step(self):
        '''
        Purpose:
        ---
        Moves robot forward along y-axis until forward target_y is reached.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None
            Publishes a geometry_msgs.msg.Twist message to /cmd_vel

        Example call:
        ---
        self.proportional_forward_step()
        '''
        dy_error = abs(self.forward_target_y - self.y)
        twist = Twist()
        twist.linear.x = min(self.KP_LIN * dy_error, self.MAX_LIN) #choose minimum of Kp*error and max linear vel
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        self.get_logger().info(f"Forward step: dy_error={dy_error:.2f}")

    def angular_control(self, target_yaw):
        '''
        Purpose:
        ---
        Rotates the robot to align with the given target yaw.

        Input Arguments:
        ---
        `target_yaw` :  [ float ]
            Target yaw angle

        Returns:
        ---
        None
            Publishes a geometry_msgs.msg.Twist message to /cmd_vel

        Example call:
        ---
        self.angular_control(1.57)
        '''
        yaw_error = self.normalize_angle(target_yaw - self.yaw)
        twist = Twist()
        twist.angular.z = min(self.KP_ANG * yaw_error, self.MAX_ANG) #choose minimum of Kp*error and max angular vel
        self.cmd_pub.publish(twist)
        self.get_logger().info(f"Rotating: yaw_error={yaw_error:.2f}")

    def stop_robot(self):
        '''
        Purpose:
        ---
        Publishes zero velocity to stop the robot.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        self.stop_robot()
        '''
        self.cmd_pub.publish(Twist())

    def is_obstacle_on_mline(self, x_goal, y_goal):
        '''
        Purpose:
        ---
        Checks if an obstacle lies on the M-line path between robot and the goal using lidar.

        Input Arguments:
        ---
        `x_goal` :  [ float ]
            Goal x-coordinate

        `y_goal` :  [ float ]
            Goal y-coordinate

        Returns:
        ---
        [ bool ] : True if obstacle detected on M-line, False otherwise

        Example call:
        ---
        blocked = self.is_obstacle_on_mline(1.0, 2.0)
        '''
        angle_to_goal = math.atan2(y_goal - self.y, x_goal - self.x)
        rel_angle = self.normalize_angle(angle_to_goal - self.yaw)
        if not (-math.pi/2 <= rel_angle <= math.pi/2):
            return False #Goal not in range of lidar
        idx = int((rel_angle - self.angle_min) / (self.angle_max - self.angle_min) * len(self.lidar_ranges))
        idx = max(0, min(idx, len(self.lidar_ranges)-1))
        obs_dist = self.lidar_ranges[idx] #maps relative angle to lidar index to find the lidar distance at that angle
        dist_to_goal = math.hypot(x_goal - self.x, y_goal - self.y)
        return (not math.isinf(obs_dist)) and obs_dist < dist_to_goal #check if obstacle is closer than goal

    def is_front_blocked(self):
        '''
        Purpose:
        ---
        Checks if an obstacle is directly in front of the robot.

        Input Arguments:
        ---
        None

        Returns:
        ---
        [ bool ] : True if obstacle within safe distance, False otherwise

        Example call:
        ---
        if self.is_front_blocked():
            self.stop_robot()
        '''
        if not self.lidar_ranges:
            return False
        mid_idx = len(self.lidar_ranges) // 2
        return self.lidar_ranges[mid_idx] < self.SAFE_FRONT

    def normalize_angle(self, angle):
        '''
        Purpose:
        ---
        Normalizes an angle to the range [-pi, pi].

        Input Arguments:
        ---
        `angle` :  [ float ]
            Input angle in radians

        Returns:
        ---
        [ float ] : Normalized angle

        Example call:
        ---
        angle = self.normalize_angle(3.5)
        '''
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    '''
    Purpose:
    ---
    Initializes the ROS2 node, spins it, and shuts it down after execution.

    Input Arguments:
    ---
    `args` :  [ list ]
        Command-line arguments (default: None)

    Returns:
    ---
    None

    Example call:
    ---
    main()
    '''
    rclpy.init(args=args)
    node = EbotNav()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
