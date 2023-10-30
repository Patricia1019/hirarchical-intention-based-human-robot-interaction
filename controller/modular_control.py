from basic_control import check_for_end_or_abort,move_back,trajectory_cartesian,trajectory_angular,GripperCommand
import time
import threading
from kortex_api.autogen.messages import Base_pb2

class ModuleController:
    def __init__(self,router,base,base_cyclic):
        self.router = router
        self.base = base
        self.base_cyclic = base_cyclic
        self.gripper = GripperCommand(router,base)
        
    def making_module_T(self,success,tube_type="short",max_time=60):
        '''
        Input: 
            tube_type: "long" or "short"
            sleep_time: waiting time for human to complete his action, in seconds
        '''
        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(   check_for_end_or_abort(e),
                                                                Base_pb2.NotificationOptions())
        success &= move_back(self.base)
        self.gripper.SendGripperCommands(0.0)
        # move to get the tubes
        waypointsDefinition=[(0.13,   0.278,  0.209,  0.1, 176.006, -0.006, 89.997),
                                (-0.26, 0.260, 0.209, 0.1, 176.002, 0.001, 89.993),
                                (-0.26, 0.260, -0.045, 0.0, 175.991, 0.016, 90.000)]
        success &= trajectory_cartesian(self.base, self.base_cyclic, waypointsDefinition)
        # grip the tube
        self.gripper.SendGripperCommands(1.0)
        # go to the table
        waypointsDefinition=[(0.13,   0.278,  0.209,  0.0, 176.006, -0.006, 89.997),
                                (0.47, 0.439, 0.323, 0.0, 171.729, -40.517,88.936)]
        success &= trajectory_cartesian(self.base, self.base_cyclic, waypointsDefinition)
        # TODO: add intention recognition
        finished = e.wait(max_time)
        self.base.Unsubscribe(notification_handle)
        # let go the tube
        self.gripper.SendGripperCommands(0.4)
        return success