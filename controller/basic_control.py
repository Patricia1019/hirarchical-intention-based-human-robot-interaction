from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2
import time
import threading
# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 30
# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check
 
def move_back(base,action_name="Retract"):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready position
    print("Moving the arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == action_name:
            action_handle = action.handle

    if action_handle == None:
        print("Can't reach safe position. Exiting")
        return False

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Safe position reached")
    else:
        print("Timeout on action notification wait")
    return finished

def populateCartesianCoordinate(waypointInformation):
    
    waypoint = Base_pb2.CartesianWaypoint()  
    waypoint.pose.x = waypointInformation[0]
    waypoint.pose.y = waypointInformation[1]
    waypoint.pose.z = waypointInformation[2]
    # blending_radius is used for smoothly transfer to the next waypoint, typically 0.1 is good enough; 
    # for ending points, blending_radius must be set as zero
    waypoint.blending_radius = waypointInformation[3] 
    waypoint.pose.theta_x = waypointInformation[4]
    waypoint.pose.theta_y = waypointInformation[5]
    waypoint.pose.theta_z = waypointInformation[6] 
    waypoint.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
    # waypoint.maximum_linear_velocity = 0.1
    return waypoint

def populateAngularPose(jointPose,durationFactor):
    waypoint = Base_pb2.AngularWaypoint()
    waypoint.angles.extend(jointPose)
    waypoint.duration = durationFactor*1.0    
    
    return waypoint


def trajectory_cartesian(base, base_cyclic, waypointsDefinition=None):

    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    product = base.GetProductConfiguration()
    if not waypointsDefinition: # just for debugging
        waypointsDefinition = [ (0.13,   0.278,  0.209,  0.1, 176.006, -0.006, 89.997),
                                (-0.267, 0.241, 0.209, 0.1, 176.002, 0.001, 89.993),
                                (-0.267, 0.241, -0.045, 0.0, 175.991, 0.016, 90.000)]
    
    waypoints = Base_pb2.WaypointList()
    
    waypoints.duration = 0.0
    waypoints.use_optimal_blending = False
    
    index = 0
    for waypointDefinition in waypointsDefinition:
        waypoint = waypoints.waypoints.add()
        waypoint.name = "waypoint_" + str(index)   
        waypoint.cartesian_waypoint.CopyFrom(populateCartesianCoordinate(waypointDefinition))
        index = index + 1 

    # Verify validity of waypoints
    result = base.ValidateWaypointList(waypoints);
    if(len(result.trajectory_error_report.trajectory_error_elements) == 0):
        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(   check_for_end_or_abort(e),
                                                                Base_pb2.NotificationOptions())

        print("Moving cartesian trajectory...")
        
        base.ExecuteWaypointTrajectory(waypoints)

        print("Waiting for trajectory to finish ...")
        finished = e.wait(TIMEOUT_DURATION)
        base.Unsubscribe(notification_handle)

        if finished:
            print("Cartesian trajectory with no optimization completed ")
            return finished
        else:
            print("Timeout on action notification wait for non-optimized trajectory")

        return finished
        
    else:
        print("Error found in trajectory") 
        print(result.trajectory_error_report)


def trajectory_angular(base, base_cyclic):

    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    jointPoses = tuple(tuple())
    product = base.GetProductConfiguration()

    if(   product.model == Base_pb2.ProductConfiguration__pb2.MODEL_ID_L53 
    or product.model == Base_pb2.ProductConfiguration__pb2.MODEL_ID_L31):
        if(product.model == Base_pb2.ProductConfiguration__pb2.MODEL_ID_L31):
            jointPoses = (  (0.0,  344.0, 75.0,  360.0, 300.0, 0.0),
                            (0.0,  21.0,  145.0, 272.0, 32.0,  273.0),
                            (42.0, 334.0, 79.0,  241.0, 305.0, 56.0))
        else:
            # Binded to degrees of movement and each degrees correspond to one degree of liberty
            degreesOfFreedom = base.GetActuatorCount();
            if(degreesOfFreedom.count == 6):
                jointPoses = (  ( 360.0, 35.6, 281.8, 0.8,  23.8, 88.9 ),
                                ( 359.6, 49.1, 272.1, 0.3,  47.0, 89.1 ),
                                ( 320.5, 76.5, 335.5, 293.4, 46.1, 165.6 ),
                                ( 335.6, 38.8, 266.1, 323.9, 49.7, 117.3 ),
                                ( 320.4, 76.5, 335.5, 293.4, 46.1, 165.6 ),
                                ( 28.8,  36.7, 273.2, 40.8,  39.5, 59.8 ),
                                ( 360.0, 45.6, 251.9, 352.2, 54.3, 101.0 ))
            else: # degreesOfFreedom.count == 7
                homepose = (0.001,339.997,180.002,213.996,0.0,310.0,90.0)
                jointPoses = [ (332.785,15.968,129.522,240.518, 20.905,308.858,0.176 )]
            
    else:
        print("Product is not compatible to run this example please contact support with KIN number bellow")
        print("Product KIN is : " + product.kin())


    waypoints = Base_pb2.WaypointList()    
    waypoints.duration = 0.0
    waypoints.use_optimal_blending = False
    
    index = 0
    for jointPose in jointPoses:
        waypoint = waypoints.waypoints.add()
        waypoint.name = "waypoint_" + str(index)
        durationFactor = 1
        # Joints/motors 5 and 7 are slower and need more time
        if(index == 4 or index == 6):
            durationFactor = 6 # Min 30 seconds
        
        waypoint.angular_waypoint.CopyFrom(populateAngularPose(jointPose, durationFactor))
        index = index + 1 
    
    
   # Verify validity of waypoints
    result = base.ValidateWaypointList(waypoints);
    if(len(result.trajectory_error_report.trajectory_error_elements) == 0):

        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(
            check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        print("Reaching angular pose trajectory...")
        
        
        base.ExecuteWaypointTrajectory(waypoints)

        print("Waiting for trajectory to finish ...")
        finished = e.wait(TIMEOUT_DURATION)
        base.Unsubscribe(notification_handle)

        if finished:
            print("Angular movement completed")
        else:
            print("Timeout on action notification wait")
        return finished
    else:
        print("Error found in trajectory") 
        print(result.trajectory_error_report)
        # return finished


class GripperCommand:
    def __init__(self, router, base, proportional_gain = 2.0):

        self.proportional_gain = proportional_gain
        self.router = router

        # Create base client using TCP router
        self.base = base

    def SendGripperCommands(self,position):
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
        if gripper_measure.finger[0].value == position:
            return
        # Create the GripperCommand we will send
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()

        # Close the gripper with position increments
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger.finger_identifier = 1
        finger.value = position
        print("Going to position {:0.2f}...".format(finger.value))
        self.base.SendGripperCommand(gripper_command)
        time.sleep(0.8)



