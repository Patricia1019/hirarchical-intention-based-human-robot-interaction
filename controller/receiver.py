#!/usr/bin/env python
import os,sys
import rospy
from std_msgs.msg import String,Float64MultiArray
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
kinova_grip_pub = rospy.Publisher("/siemens_demo/gripper_cmd", Float64MultiArray, queue_size=1)
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
        self.TUBE_SUM = {"short":8,"long":4}
        self.tube_count = {"short":0,"long":0}
        self.screw_count = 0
        self.action_history = []
        self.stage = {"bottom":[],"four_tubes":[],"top":[]}


class Receiver:
    def __init__(self):
        self.intention_list = []
        self.command_list = []
        self.command = None
        self.GRIP_TIME = 1.5
        self.plangraph = PlanGraph()
        self.REVERT_LIST = {"grip":"open","open":"grip"}
        self.executing = False

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
            action = self.decide_send_action(data.data)
            if action[0] and not self.executing:
                print(action)
                self.execute_action(action)
        if data.data in COMMAND_LIST:
            # self.command_list.append(data.data)
            self.command = data.data
            # print(data.data)
            action = self.decide_send_action(data.data)
            if action[0] == "get_short_tubes" and not self.executing:
                print(action)
                for _ in range(4):
                    self.execute_action(action)
            elif action[0] and not self.executing:
                print(action)
                self.execute_action(action)
        
    def execute_action(self,action,index=0):
        waypoints_list,target_list,unique_actions = self.generate_waypoints(action)
        kinova_control_msg = PoseStamped()
        kinova_control_msg.pose = ComposePoseFromTransQuat(waypoints_list[index])
        kinova_control_pub.publish(kinova_control_msg)
        i = index
        if len(waypoints_list) == 0: return
        self.executing = True
        while i < len(waypoints_list):
            current_pose = self.current_pose
            command = self.command
            # pdb.set_trace()
            if command == "stop":
                print(command)
                self.retract(current_pose,target_list,unique_actions,i)
                self.command = None
                # kinova_control_msg.pose = ComposePoseFromTransQuat(current_pose)
                # kinova_control_pub.publish(kinova_control_msg)
                break
            if command in ["long","short"] and command not in action:
                print(command)
                action = self.decide_send_action(command)
                if action:
                    self.return_base(current_pose,target_list,unique_actions,i)
                    BASE_INDEX = 6
                    self.execute_action(action,BASE_INDEX+1)
                    self.command = None
                    break

            if self.reached(current_pose,target_list[i]):
                i += 1
                if i in unique_actions.keys():
                    kinova_control_msg.pose = ComposePoseFromTransQuat(target_list[i-1])
                    kinova_control_pub.publish(kinova_control_msg)
                    time.sleep(0.5)
                    self.execute_unique_actions(unique_actions[i])
                if i < len(waypoints_list):
                    kinova_control_msg.pose = ComposePoseFromTransQuat(waypoints_list[i])
                    kinova_control_pub.publish(kinova_control_msg)
                else: # end
                    kinova_control_msg.pose = ComposePoseFromTransQuat(target_list[i-1])
                    kinova_control_pub.publish(kinova_control_msg)
                    if "short" in action[0]:
                        self.plangraph.tube_count["short"] += 1
                    elif "long" in action[0]:
                        self.plangraph.tube_count["long"] += 1
                    self.plangraph.action_history.append(action[0])
        self.executing = False
        return
    
    def execute_unique_actions(self,unique_actions):
        for i in range(len(unique_actions)):
            print(unique_actions[i])
            if unique_actions[i] == "grip":
                kinova_grip_msg = Float64MultiArray()
                kinova_grip_msg.data = [0] # 0 for close, 1 for open
                kinova_grip_pub.publish(kinova_grip_msg)
                time.sleep(self.GRIP_TIME)
            elif "wait" in unique_actions[i]:
                wait_time = int(unique_actions[i][4:])
                time.sleep(wait_time)
            elif unique_actions[i] == "open":
                kinova_grip_msg = Float64MultiArray()
                kinova_grip_msg.data = [1] # 0 for close, 1 for open
                kinova_grip_pub.publish(kinova_grip_msg) 
                time.sleep(self.GRIP_TIME)
            else:
                print(f"Invalid unique actions[{i}]!")

    def retract(self,current_pose,ori_targets,ori_unique_actions,index):
        print("retract")
        kinova_control_msg = PoseStamped()
        kinova_control_msg.pose = ComposePoseFromTransQuat(current_pose)
        kinova_control_pub.publish(kinova_control_msg)
        # revert actions
        target_list = ori_targets[:index][::-1]
        waypoints_list = []
        for i in range(len(target_list)):
            if i == 0:
                tmp = []
                for j in range(len(target_list[i])):
                    tmp.append(target_list[i][j]+0.7*(target_list[i][j]-current_pose[j]))
                waypoints_list.append(tmp)
            if i > 0:
                tmp = []
                for j in range(len(target_list[i])):
                    tmp.append(target_list[i][j]+0.7*(target_list[i][j]-target_list[i-1][j]))
                waypoints_list.append(tmp)
        
        # revert unique actions
        unique_actions = {}
        for unique_index in ori_unique_actions.keys():
            if unique_index <= index:
                unique_actions[index-unique_index] = []
                for unique_action in ori_unique_actions[unique_index]:
                    if "wait" not in unique_action:
                        # pdb.set_trace()
                        unique_actions[index-unique_index].append(self.REVERT_LIST[unique_action])

        # pdb.set_trace()
        i = 0
        kinova_control_msg = PoseStamped()
        kinova_control_msg.pose = ComposePoseFromTransQuat(waypoints_list[0])
        kinova_control_pub.publish(kinova_control_msg)
        # pdb.set_trace()
        if i in unique_actions.keys():
            kinova_control_msg.pose = ComposePoseFromTransQuat(target_list[0])
            kinova_control_pub.publish(kinova_control_msg)
            time.sleep(0.5)
            self.execute_unique_actions(unique_actions[i])
        while i < len(waypoints_list):
            current_pose = self.current_pose
            if self.reached(current_pose,target_list[i]):
                i += 1
                if i in unique_actions.keys():
                    kinova_control_msg.pose = ComposePoseFromTransQuat(target_list[i-1])
                    kinova_control_pub.publish(kinova_control_msg)
                    time.sleep(0.5)
                    self.execute_unique_actions(unique_actions[i])
                if i < len(waypoints_list):
                    kinova_control_msg.pose = ComposePoseFromTransQuat(waypoints_list[i])
                    kinova_control_pub.publish(kinova_control_msg)
                else:
                    kinova_control_msg.pose = ComposePoseFromTransQuat(target_list[i-1])
                    kinova_control_pub.publish(kinova_control_msg)
        return
    
    def return_base(self):
        pass # TODO

    def generate_waypoints(self,action):
        waypoints_list = []
        if action[0] == "get_short_tubes":
            retract = RETRACT_POSITION
            ready = (0.2,0.32,0.25,0,-0.7,-0.7,0)
            # ready_way = (0.2,0.32+0.18,0.19,0,-0.7,-0.7,0)
            x_interval = 0.10
            y_interval = 0.16
            row = action[1]%2
            col = action[1]//2
            base = (-0.24,0.28,0.2,0,-0.7,-0.7,0)
            get = (-0.24-x_interval*col,0.28-y_interval*row,0.2,0,-0.7,-0.7,0)
            grip = (-0.24-x_interval*col,0.28-y_interval*row,-0.05,0,-0.7,-0.7,0)
            # grip_way = (-0.26-x_interval*col,0.28-y_interval*row,-0.1,0,-0.8,-0.7,0)
            deliver = (0.3,0.3,0.25,0,-0.7,-0.6,-0.2) # TODO: move with hand
            # [retract,ready,get,grip,(close gripper),get,ready,deliver,(open gripper),retract]
            # waypoints_list = [retract,ready_way,get,grip_way,get,ready_way,deliver,retract]
            target_list = [retract,ready,base,get,grip,get,base,ready,deliver,ready,retract]
            waypoints_list = []
            for i in range(len(target_list)):
                if i == 0:
                    waypoints_list.append(target_list[i])
                if i > 0:
                    tmp = []
                    for j in range(len(target_list[i])):
                        if j < 3: # only adjust position points
                            tmp.append(target_list[i][j]+0.7*(target_list[i][j]-target_list[i-1][j]))
                        else: # keep augular points stay as target points
                            tmp.append(target_list[i][j])
                    waypoints_list.append(tmp)
            unique_actions = {5:["grip"],9:["wait3","open"]}
        # TODO: other actions
        return waypoints_list,target_list,unique_actions

    def get_command(self):
        # TODO: NOT USED
        if len(self.command_list) > 0:
            return self.command_list[-1]
        return ""

    def decide_send_action(self,data):
        # TODO
        # for key in self.plangraph.tube_count.keys(): # tube sum check
        #     assert self.plangraph.tube_count[key] < self.plangraph.TUBE_SUM[key]
        if data == "get_connectors": # decide to get short tubes or get long tubes
            if len(self.plangraph.stage["bottom"]) < 4:
                return ["get_short_tubes",self.plangraph.tube_count["short"]]
            elif len(self.plangraph.stage["top"]) < 4:
                return ["get_short_tubes",self.plangraph.tube_count["short"]]
    
        if data == "short":
            if self.plangraph.tube_count["short"] < self.plangraph.TUBE_SUM["short"]:
                return ["get_short_tubes",self.plangraph.tube_count["short"]]
        if data == "long":
            if self.plangraph.tube_count["long"] < self.plangraph.TUBE_SUM["long"]:
                return ["get_long_tubes",self.plangraph.tube_count["long"]]
        return None,None
        # return ["",int]

    def reached(self,current_pose,target_pose):
        pos_gap = abs(current_pose[0]-target_pose[0])+abs(current_pose[1]-target_pose[1])+abs(current_pose[2]-target_pose[2])
        if pos_gap > 0.05:
            # print(f"pos_gap:{pos_gap}")
            return False
        else:
            w,x,y,z = target_pose[3],target_pose[4],target_pose[5],target_pose[6]
            sqrt_sum = (w**2+x**2+y**2+z**2)**0.5
            w,x,y,z = w/sqrt_sum,x/sqrt_sum,y/sqrt_sum,z/sqrt_sum
            quat_gap = 1 - (w*current_pose[3]+x*current_pose[4]+y*current_pose[5]+z*current_pose[6])**2
            # print(quat_gap)
            if quat_gap < 0.05:
                # print(f"quat_gap:{quat_gap}")
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