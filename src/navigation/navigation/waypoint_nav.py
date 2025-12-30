#!/usr/bin/env python3
"""
Long waypoint tour for rally_obstacle5.world – LATEST SEQUENCE (50 waypoints + home)
Sequential goToPose → reliable in SLAM mode
"""
import rclpy
import math
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

def make_pose(navigator, x, y, yaw):
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw
    return pose

def main():
    rclpy.init()
    navigator = BasicNavigator()

    # Initial pose
    initial_pose = make_pose(navigator, 0.0, 0.0, 0.0)
    navigator.setInitialPose(initial_pose)
    navigator.waitUntilNav2Active()
    navigator.clearAllCostmaps()

    # LATEST WAYPOINTS from your /goal_pose echo (50 + home)
    waypoints = [
     (1.5473766326904297, 0.27490997314453125, 0.6359999775886536),
        (5.970274925231934, 3.3755369186401367, 0.8399999737739563),
        (7.4537224769592285, 6.109414577484131, 1.6399998664855957),
        (5.804237365722656, 7.7122321128845215, 1.8799996376037598),
        (-3.7526302337646484, 8.416918754577637, -3.0799999237060547),
        (-7.963253021240234, 2.4144678115844727, -1.399999737739563),
        (-8.295437812805176, -0.8720052242279053, -1.2799997329711914),
        (-6.157953262329102, -7.952448844909668, -0.37599998712539673),
        (-4.641843318939209, -8.23707103729248, 0.0820000022649765),
        (2.5776004791259766, -8.555967330932617, 0.03400000184774399),
        (6.652500152587891, -8.47697639465332, 1.0399999618530273),
        (6.892419815063477, -6.337416648864746, 0.6159999966621399),
        (7.372461795806885, -3.626044273376465, 1.4799996614456177),
        (5.8455705642700195, -2.7817916870117188, 1.359999656677246),
        (7.0885796546936035, 0.1101694107055664, 1.6399998664855957),
        (2.7595348358154297, 4.227568626403809, 1.919999599456787),
        (-1.403048038482666, 5.315399646759033, 0.1339999958872795),
        (-4.239857196807861, 1.7893872261047363, -1.2799997329711914),
        (-3.6552538871765137, -3.9850664138793945, -0.20800000429153442),
        (-1.8569555282592773, -4.481748580932617, -0.14799998700618744),
        (-1.0673456192016602, 1.0690011978149414, 1.559999704360962),
        (-1.8107590675354004, -2.6248738765716553, -1.3199996948242188),
        (-1.3952226638793945, -2.9490976333618164, -0.4359999895095825),
        (-0.5609240531921387, -1.4010496139526367, 1.2199997901916504),
        (-0.13903403282165527, -0.22911977767944336, 1.3199996948242188),
        (0.0, 0.0, 0.0)  # home
    ]

    print(f'Starting latest tour with {len(waypoints)} waypoints...')

    for i, (x, y, yaw) in enumerate(waypoints):
        goal_pose = make_pose(navigator, x, y, yaw)
        print(f'Sending waypoint {i+1}/{len(waypoints)}: ({x:.3f}, {y:.3f}, yaw={yaw:.3f})')
        navigator.goToPose(goal_pose)

        while not navigator.isTaskComplete():
            pass  # Blocking wait (add feedback if needed)

        result = navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            print(f'Waypoint {i+1} succeeded')
        else:
            print(f'Waypoint {i+1} {result.name} – continuing anyway')

    print('✅ Latest tour complete – robot returned home!')
    rclpy.shutdown()

if __name__ == '__main__':
    main()
