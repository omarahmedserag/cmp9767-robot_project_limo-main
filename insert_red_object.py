#!/usr/bin/env python3
"""Insert a red sphere into Gazebo for color detection testing."""

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
import os
import xacro

class RedObjectSpawner(Node):
    def __init__(self):
        super().__init__('red_object_spawner')
        self.client = self.create_client(SpawnEntity, '/spawn_entity')
        
    def spawn_red_sphere(self):
        # Simple SDF for a red sphere
        sdf_content = '''<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="red_sphere">
    <static>false</static>
    <link name="link">
      <inertial>
        <mass>0.5</mass>
        <inertia>
          <ixx>0.001</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.001</iyy>
          <iyz>0</iyz>
          <izz>0.001</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <sphere>
            <radius>0.1</radius>
          </sphere>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <sphere>
            <radius>0.1</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
          <specular>1 0 0 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''
        
        while not self.client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for spawn service...')
        
        request = SpawnEntity.Request()
        request.name = 'red_sphere'
        request.xml = sdf_content
        
        # Position in front of robot
        pose = Pose()
        pose.position.x = 0.5
        pose.position.y = 0.0
        pose.position.z = 0.2
        request.initial_pose = pose
        
        future = self.client.call_async(request)
        future.add_done_callback(self.spawn_callback)
        
    def spawn_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f'Spawned: {response.success}')
        except Exception as e:
            self.get_logger().error(f'Failed to spawn: {e}')

def main(args=None):
    rclpy.init(args=args)
    spawner = RedObjectSpawner()
    spawner.spawn_red_sphere()
    rclpy.spin(spawner)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
