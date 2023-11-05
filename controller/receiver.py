#!/usr/bin/env python
import os,sys
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose,PoseStamped
import time
from pathlib import Path
FILE_DIR = Path(__file__).parent
import pdb

sys.path.append(f'{FILE_DIR}/../traj_intention')
from Dataset import INTENTION_LIST
# from speech import COMMAND_LIST
COMMAND_LIST = ["stop","short","long","spin","lift"]


RETRACT_POSITION = (0.2,0,0.19,0,-0.7,-0.7,0)
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

class PlanGraph:
    def __init__(self):
        self.SHORT_TUBES = 8
        self.LONG_TUBES = 4


class Receiver:
    def __init__(self):
        self.intention_list = []
        self.command_list = []
        self.command = None

    def receive_pose(self,data):
        pos_x = data.pose.position.x
        pos_y = data.pose.position.y
        pos_z = data.pose.position.z
        ort_w = data.pose.orientation.w
        ort_x = data.pose.orientation.x
        ort_y = data.pose.orientation.y
        ort_z = data.pose.orientation.z
        current_pose = (pos_x,pos_y,pos_z,ort_w,ort_x,ort_y,ort_z)
        self.current_pose = current_pose

    def receive_data(self,data):
        if data.data in INTENTION_LIST:
            self.intention_list.append(data.data)
        if data.data in COMMAND_LIST:
            # self.command_list.append(data.data)
            self.command = data.data
            # print(data.data)
        action = self.decide_send_action(data.data)
        if action[0]:
            print(action)
            self.execute_action(action)
        
    def execute_action(self,action):
        waypoints_list = self.generate_waypoints(action)
        kinova_control_msg = PoseStamped()
        kinova_control_msg.pose = ComposePoseFromTransQuat(waypoints_list[0])
        kinova_control_pub.publish(kinova_control_msg)
        i = 0
        if len(waypoints_list) == 0: return
        while i < len(waypoints_list):
            current_pose = self.current_pose
            command = self.command
            # pdb.set_trace()
            if command == "stop":
                print(command)
                # self.command_list.pop(-1)
                self.command = None
                break
            if self.reached(current_pose,waypoints_list[i]):
                i += 1
                if i < len(waypoints_list):
                    kinova_control_msg.pose = ComposePoseFromTransQuat(waypoints_list[i])
                    kinova_control_pub.publish(kinova_control_msg)
                else:
                    break
        return
    
    def generate_waypoints(self,action):
        waypoints_list = []
        if action[0] == "get_short_tubes":
            retract = RETRACT_POSITION
            ready = (0.2,0.32,0.19,0,-0.7,-0.7,0)
            x_interval = 0.08
            y_interval = 0.15
            row = action[1]%2
            col = action[1]//2
            get = (-0.26-x_interval*col,0.28-y_interval*row,0.1,0,-0.8,-0.7,0)
            grip = (-0.26-x_interval*col,0.28-y_interval*row,-0.05,0,-0.8,-0.7,0)
            deliver = (0.3,0.3,0.19,0,-0.7,-0.3,-0.1) # TODO: move with hand
            # [retract,ready,get,grip,(close gripper),get,ready,deliver,(open gripper),retract]
            waypoints_list = [retract,ready,get,grip,get,ready,deliver,retract]
        # TODO: other actions
        return waypoints_list

    def get_command(self):
        # TODO: NOT USED
        if len(self.command_list) > 0:
            return self.command_list[-1]
        return ""

    def decide_send_action(self,data):
        # TODO
        if data == "get_connectors":
            return ["get_short_tubes",0]
        return None,None
        # return ["",int]

    def reached(self,current_pose,target_pose):
        pos_gap = abs(current_pose[0]-target_pose[0])+abs(current_pose[1]-target_pose[1])+abs(current_pose[2]-target_pose[2])
        if pos_gap > 0.05:
            # print(f"pos_gap:{pos_gap}")
            return False
        else:
            w,x,y,z = current_pose[3],current_pose[4],current_pose[5],current_pose[6]
            sqrt_sum = (w**2+x**2+y**2+z**2)**0.5
            w,x,y,z = w/sqrt_sum,x/sqrt_sum,y/sqrt_sum,z/sqrt_sum
            quat_gap = abs(abs(w)-abs(target_pose[3]))+abs(abs(x)-abs(target_pose[4]))+abs(abs(y)-abs(target_pose[5]))+abs(abs(z)-abs(target_pose[6]))
            if quat_gap < 0.4:
                return True
            else:
                # print(f"quat_gap:{quat_gap}")
                return False
            
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    receiver = Receiver()
    rospy.init_node('listener', anonymous=True)

    # rospy.Subscriber("chatter", String, controller)
    rospy.Subscriber("chatter", String, receiver.receive_data)
    rospy.Subscriber("/kinova/pose_tool_in_base", PoseStamped, receiver.receive_pose, queue_size=1)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
    # print("ok")