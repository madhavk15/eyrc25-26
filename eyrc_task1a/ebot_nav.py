#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class ConstantVelocityPublisher(Node):
    def __init__(self):
        super().__init__('constant_velocity_publisher')
        # Create publisher to /cmd_vel
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        # Timer to call publish function at 10 Hz
        self.timer = self.create_timer(0.1, self.publish_velocity)

    def publish_velocity(self):
        msg = Twist()
        msg.linear.x = 0.1   # move forward at 0.1 m/s
        msg.angular.z = 0.0  # no rotation
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: linear.x={msg.linear.x}')


def main(args=None):
    rclpy.init(args=args)
    node = ConstantVelocityPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
