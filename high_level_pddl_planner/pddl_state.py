#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from custom_interfaces.srv import GetSetBool  

class StateManager(Node):
    def __init__(self):
        super().__init__('state_manager')

        # Internal state variables
        self.is_home = True
        self.is_ready = False
        self.gripper_is_open = False

        # Service servers
        self.is_home_srv = self.create_service(
            GetSetBool, '/is_home', self.is_home_callback
        )
        self.is_ready_srv = self.create_service(
            GetSetBool, '/is_ready', self.is_ready_callback
        )
        self.gripper_open_srv = self.create_service(
            GetSetBool, '/gripper_is_open', self.gripper_open_callback
        )

        self.get_logger().info("State Manager Node has started.")

    # -------- Service Callbacks --------
    def is_home_callback(self, request, response):
        response.success = True
        if request.set:
            self.is_home = request.value
            response.message = f"is_home set to {self.is_home}"
        else:
            response.message = f"is_home is {self.is_home}"
        response.value = self.is_home
        self.get_logger().info(response.message)
        return response

    def is_ready_callback(self, request, response):
        response.success = True
        if request.set:
            self.is_ready = request.value
            response.message = f"is_ready set to {self.is_ready}"
        else:
            response.message = f"is_ready is {self.is_ready}"
        response.value = self.is_ready
        self.get_logger().info(response.message)
        return response

    def gripper_open_callback(self, request, response):
        response.success = True
        if request.set:
            self.gripper_is_open = request.value
            response.message = f"gripper_is_open set to {self.gripper_is_open}"
        else:
            response.message = f"gripper_is_open is {self.gripper_is_open}"
        response.value = self.gripper_is_open
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = StateManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
