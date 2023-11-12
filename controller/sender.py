import rospy
from std_msgs.msg import String
import argparse
from pathlib import Path
FILE_DIR = Path(__file__).parent


def send_intention_to_ros(intention):
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('intention', anonymous=True)
    rospy.loginfo(intention)
    pub.publish(intention)

def send_command_to_ros(command="stop"):
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('command', anonymous=True)
    rospy.loginfo(command)
    rate = rospy.Rate(10) # 10hz
    pub.publish(command)
    rate.sleep()
    rospy.loginfo(command)
    pub.publish(command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--intention', type=str,
                    help="intention send to rospy")
    parser.add_argument('--command', type=str,
                    help="command send to rospy")
    args = parser.parse_args()
    # while True:
    if args.intention:
        
        send_intention_to_ros(args.intention)
    if args.command:
        send_command_to_ros(args.command)
    