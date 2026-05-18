from py_openshowvar import openshowvar
import numpy as np
import time

class KUKAControl:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.client = None
        print('kuka_control initialized')

    def run(self):
        while True:
            print("Robot is not connected ---> Trying again")
            if self.client is None:
                self.connect()
                time.sleep(1)
            # else:
            #     print("Robot is successfully connected")
            #     break
            else:
                self.read_go()
                if self.read_go() == '1':
                    print("Robot is successfully connected")
                    break
                else:
                    print("Robot is not connected ---> Trying again")
                    self.write_go('1')
                    time.sleep(1)     
        
    def connect(self):
        try:
            self.client = openshowvar(self.ip, self.port)
            if not self.client.can_connect:
                print(f"Warning: Could not connect to robot at {self.ip}:{self.port}")
                self.client = None
            else:
                print(f"Connected to robot at {self.ip}:{self.port}")
        except Exception as e:
            print(f"Error connecting to robot: {e}")
            self.client = None

    def close(self):
        if self.client:
            self.client.close()
            self.client = None

    def write_go(self, value: str = '1') -> None:
        #Write to the GO variable.
        if self.client:
            self.client.write('GO', value,debug=False)

    def read_go(self) -> str:
        #Read the GO variable.
        if self.client:
            return self.client.read('GO',debug=False).decode()
        return None

    def read_pose(self):
        if not self.client:
            # Attempt reconnect? Or just return None
            # For safety, just return None.
            return None
        try:
            # Reading all variables. Note:
            pose_x = float(self.client.read('$POS_ACT.X',debug=False).decode())
            pose_y = float(self.client.read('$POS_ACT.Y',debug=False).decode())
            pose_z = float(self.client.read('$POS_ACT.Z',debug=False).decode())
            pose_A = float(self.client.read('$POS_ACT.A',debug=False).decode())
            pose_B = float(self.client.read('$POS_ACT.B',debug=False).decode())
            pose_C = float(self.client.read('$POS_ACT.C',debug=False).decode())
            return np.array([pose_x, pose_y, pose_z, pose_A, pose_B, pose_C])
        except Exception as e:
            print(f"Error reading pose: {e}")
            return None

    def push_target(self, x: float, y: float, z: float, rz: float, ry: float, rx: float):
        if not self.client:
            return
        try:
            self.client.write('GTARGET_X', str(x), debug=False)
            self.client.write('GTARGET_Y', str(y), debug=False)
            self.client.write('GTARGET_Z', str(z), debug=False)
            self.client.write('GTARGET_A', str(rz), debug=False)
            self.client.write('GTARGET_B', str(ry), debug=False)
            self.client.write('GTARGET_C', str(rx), debug=False)
        except Exception as e:
            print(f"Error writing target: {e}")

    
    def read_joint(self):
        if not self.client:
            # Attempt reconnect? Or just return None
            # For safety, just return None.
            return None
        try:
            # Reading all variables.
            joint_1 = float(self.client.read('$AXIS_ACT.1').decode())
            joint_2 = float(self.client.read('$AXIS_ACT.2').decode())
            joint_3 = float(self.client.read('$AXIS_ACT.3').decode())
            joint_4 = float(self.client.read('$AXIS_ACT.4').decode())
            joint_5 = float(self.client.read('$AXIS_ACT.5').decode())
            joint_6 = float(self.client.read('$AXIS_ACT.6').decode())
            return [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]
        except Exception as e:
            print(f"Error reading joint: {e}")
            return None

    def push_3p3o(self, pose_6dof):
        self.push_target(pose_6dof[0], pose_6dof[1], pose_6dof[2], 
        pose_6dof[3], pose_6dof[4], pose_6dof[5])

    def push_3p(self, pose_3dof):
        current_pose = self.read_pose()
        self.push_target(pose_3dof[0], pose_3dof[1], pose_3dof[2], 
        current_pose[3], current_pose[4], current_pose[5])

    def push_3o(self, joint_3dof):
        current_joint = self.read_pose()
        self.push_target(current_joint[0], current_joint[1], current_joint[2],
        joint_3dof[0], joint_3dof[1], joint_3dof[2])

    def overideSpeed(self):
        while True:
            return self.client.read('$OV_PRO', debug=False)

    def overideSpeed(self, overide_speed):
        while True:
            self.client.write('$OV_PRO', str(overide_speed), debug=False)
            varify_speed = self.client.read('$OV_PRO', debug=False)
            if varify_speed.decode() == str(overide_speed):
                print("Overide speed set to", overide_speed)    
                break
            else:
                # pass
                print("Overide speed not set")

    
            
            
            
# client.read('$OV_PRO', debug=True)

# Home pose = [525, 0, 890, 0, 45, 0]

# if __name__ == "__main__":
#     robot = KUKAControl('172.31.1.147', 7000)
#     robot.run()
#     print("Push frist instruction")
#     robot.push_3p3o([570.44443589, -5.98803216, 625.04357543, 0, 90, 0])
#     time.sleep(5)
#     # print("Push third instruction")
#     # robot.push_3o([0, 90, 20])
#     # time.sleep(5)
#     # robot.overideSpeed(100)
#     # print("Push second instruction")
#     # robot.push_3p([525, 0, 890])
#     # time.sleep(5)
    
    
#     robot.close()   