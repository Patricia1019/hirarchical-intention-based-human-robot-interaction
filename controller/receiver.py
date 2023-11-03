#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose,PoseStamped

kinova_control_pub = rospy.Publisher("/kinova_demo/pose_cmd", PoseStamped, queue_size=1)
def ComposePoseFromTransQuat(data_frame):
    # assert (len(data_frame.shape) == 1 and data_frame.shape[0] == 7)
    pose = Pose()
    pose.position.x = data_frame[0]
    pose.position.y = data_frame[1]
    pose.position.z = data_frame[2]
    pose.orientation.w = data_frame[3]
    pose.orientation.x = data_frame[4]
    pose.orientation.y = data_frame[5]
    pose.orientation.z = data_frame[6]
    return pose

def controller(data):
    intention = data.data
    # print(intention)
    if intention == "get long tubes":
        print(intention)
        kinova_control_msg = PoseStamped()
        waypointsDefinition=[   (0.13,0.001,0.209,0.0246653,0.7066765, 0.7066769, 0.024653),
                                (0.13,   0.278,  0.209,  0.0246043, 0.7066971, 0.7066575, 0.024677),
                                (-0.26, 0.260, 0.209, 0.024673, 0.7067194, 0.7066335, 0.0246577),
                                (-0.26, 0.260, -0.045, 0.0248318, 0.7066706, 0.7066775, 0.0246345)]
        for waypoints in waypointsDefinition:
            # waypoints = (-0.26, 0.260, -0.045,0.0248318, 0.7066706, 0.7066775, 0.0246345)
            kinova_control_msg.pose = ComposePoseFromTransQuat(waypoints)
            kinova_control_pub.publish(kinova_control_msg)

    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("chatter", String, controller)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()