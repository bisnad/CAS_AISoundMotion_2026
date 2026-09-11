import numpy as np
import analysis as ma


class MotionPipeline():
    def __init__(self, oscReceiver, mocap_config):
        
        self.mocap_config = mocap_config
            
        self.pos = oscReceiver.data[0]
        self.posDim = self.pos.shape[-1]

        if len(oscReceiver.data) > 1:
            self.rot = oscReceiver.data[1]
        else:
            self.rot = None
        self.jointWeigths = mocap_config["joint_weights"]
        self.updateInterval = 1.0 / mocap_config["fps"]
        self.jointCount = len(self.jointWeigths)
        self.ground_coordinates = mocap_config["ground_coordinates"]
        
        self.joint_hip_index = mocap_config["joint_names"].index(mocap_config["joint_name_hip"])
        self.joint_upperbody_indices = [ mocap_config["joint_names"].index(joint_name) for joint_name in mocap_config["joint_names_upperbody"] ]
        self.joint_lowerbody_indices = [ mocap_config["joint_names"].index(joint_name) for joint_name in mocap_config["joint_names_lowerbody"] ]
        self.joint_rightbody_indices = [ mocap_config["joint_names"].index(joint_name) for joint_name in mocap_config["joint_name_rightbody"] ]
        self.joint_leftbody_indices = [ mocap_config["joint_names"].index(joint_name) for joint_name in mocap_config["joint_names_leftbody"] ]
    
        
        # scale value (from cm to m)
        self.posScale = mocap_config["pos_scale"]
        self.posScaled = np.zeros_like(self.pos )
        
        # smooth
        self.posSmoothFactor = 0.9
        self.velSmoothFactor = 0.9
        self.accelSmoothFactor = 0.9
        self.jerkSmoothFactor = 0.9
        
        self.posSmooth = np.zeros_like(self.pos )
        self.velSmooth = np.zeros_like(self.pos )
        self.accelSmooth = np.zeros_like(self.pos )
        self.jerkSmooth = np.zeros_like(self.pos )
        
        # derivatives
        self.vel = np.zeros_like(self.pos)
        self.accel= np.zeros_like(self.pos)
        self.jerk = np.zeros_like(self.pos)
        
        # scalar
        self.posScalar = np.zeros(self.jointCount)
        self.velScalar = np.zeros(self.jointCount)
        self.accelScalar = np.zeros(self.jointCount)
        self.jerkScalar = np.zeros(self.jointCount)

        #quom
        self.quom = np.array([0.0])
        
        # bbox
        self.bbox = np.zeros([2, self.posDim])
        
        # bsphere
        self.bsphere = np.array([self.posDim + 1])
        
        # body volumes
        self.vol_fullbody = np.array([0])
        self.vol_upperbody = np.array([0])
        self.vol_lowerbody = np.array([0])
        self.vol_rightbody = np.array([0])
        self.vol_leftbody = np.array([0])
        
        # ring buffers
        self.ringSize = 25
        self.posRing = np.zeros([self.ringSize, self.jointCount, self.posDim])
        self.velScalarRing = np.zeros([self.ringSize, self.jointCount])
        self.accelScalarRing = np.zeros([self.ringSize, self.jointCount])
        self.jerkScalarRing = np.zeros([self.ringSize, self.jointCount])
        #self.ground_trajectory = np.zeros([self.ringSize, len(self.ground_coordinates)])
        
        self.ground_trajectory = np.random.uniform(low=-0.01, high=0.01, size=(self.ringSize, len(self.ground_coordinates)))
        
        # Laban Effort Factors
        self.windowLength = self.ringSize - 1
        self.flowEffort = np.array([0])
        self.timeEffort = np.array([0])
        self.weightEffort = np.array([0])
        self.timeEffort = np.array([0])
      
        # traversal through space
        self.travel_distance = np.array([0])
        self.area_covered = np.array([0])
        
    def setUpdateInterval(self, updateInterval):
        self.updateInterval = updateInterval

        
    def update(self):
        
        #print("DataPipeline update")
        #print("self.input_pos_data", self.input_pos_data )

        # pos scale
        _posScaled = self.pos * self.posScale
        
        # pos smooth
        _posSmooth = self.posSmooth * self.posSmoothFactor + _posScaled * (1.0 - self.posSmoothFactor)
        
        # hip pos ground
        _hipPosGround = _posSmooth[self.joint_hip_index, self.ground_coordinates]
        
        # body part positions
        _pos_upperbody = _posSmooth[self.joint_upperbody_indices]
        _pos_lowerbody = _posSmooth[self.joint_lowerbody_indices]
        _pos_rightbody = _posSmooth[self.joint_rightbody_indices]
        _pos_leftbody = _posSmooth[self.joint_leftbody_indices]
        
        # velocity 
        _vel = (_posSmooth - self.posSmooth) / self.updateInterval
        
        # velocity smooth
        _velSmooth = self.velSmooth * self.velSmoothFactor + _vel * (1.0 - self.velSmoothFactor)
        
        # acceleration
        _accel = (_velSmooth - self.velSmooth) / self.updateInterval
        
        # acceleration smooth
        _accelSmooth = self.accelSmooth * self.accelSmoothFactor + _accel * (1.0 - self.accelSmoothFactor)
        
        # jerk
        _jerk = (_accelSmooth - self.accelSmooth) / self.updateInterval
        
        # jerk smooth
        _jerkSmooth = self.jerkSmooth * self.jerkSmoothFactor + _jerk * (1.0 - self.jerkSmoothFactor)  
        
        # pos scalar
        _pos_scalar = np.linalg.norm(_posSmooth, axis=-1)
        
        # vel scalar
        _vel_scalar = np.linalg.norm(_velSmooth, axis=-1)
        
        # accel scalar
        _accel_scalar = np.linalg.norm(_accelSmooth, axis=-1)
        
        # jerk scalar
        _jerk_scalar = np.linalg.norm(_jerkSmooth, axis=-1)

        # quom
        _qom = np.array([np.sum(_vel_scalar * self.jointWeigths) / np.sum(self.jointWeigths)])
        
        # bbox
        _bbox = ma.bounding_box_rt(_posSmooth)

        # bsphere
        _bsphere = ma.bounding_sphere_rt(_posSmooth)
        
        # body volumes
        _vol_fullbody = ma.joint_volume(_posSmooth)
        _vol_upperbody = ma.joint_volume(_pos_upperbody)
        _vol_lowerbody = ma.joint_volume(_pos_lowerbody)
        _vol_rightbody = ma.joint_volume(_pos_rightbody)
        _vol_leftbody = ma.joint_volume(_pos_leftbody)

        # ring buffers
        _posRing = np.roll(self.posRing, shift=1, axis=0)
        _posRing[0] = _posSmooth
        
        _velScalarRing = np.roll(self.velScalarRing, shift=1, axis=0)
        _velScalarRing[0] = _vel_scalar

        _accelScalarRing = np.roll(self.accelScalarRing, shift=1, axis=0)
        _accelScalarRing[0] = _accel_scalar
        
        _jerkScalarRing = np.roll(self.jerkScalarRing, shift=1, axis=0)
        _jerkScalarRing[0] = _jerk_scalar
        
        _ground_trajectory = np.roll(self.ground_trajectory, shift=1, axis=0)
        _ground_trajectory[0] = _hipPosGround

        # Laban Effort Factors
        _flow_effort = ma.flow_effort_rt(_jerkScalarRing, self.jointWeigths)
        _time_effort = ma.time_effort_rt(_accelScalarRing, self.jointWeigths)
        _weight_effort = ma.weight_effort_rt(_velScalarRing, self.jointWeigths)
        _space_effort = ma.space_effort_v2_rt(_posRing, self.jointWeigths)
        
        # traversal through space
        _travel_distance =  ma.joint_travel_distance(self.ground_trajectory)
        _area_covered = ma.joint_volume(self.ground_trajectory)
        
        # update all member variables
        self.posScaled = _posScaled
        self.posSmooth = _posSmooth
        self.vel = _vel
        self.velSmooth = _velSmooth
        self.accel = _accel
        self.accelSmooth = _accelSmooth
        self.jerk = _jerk
        self.jerkSmooth = _jerkSmooth
        self.pos_scalar = _pos_scalar
        self.vel_scalar = _vel_scalar
        self.accel_scalar = _accel_scalar
        self.jerk_scalar = _jerk_scalar
        self.qom = _qom
        self.bbox = _bbox
        self.bsphere = _bsphere
        self.vol_fullbody = _vol_fullbody
        self.vol_upperbody = _vol_upperbody
        self.vol_lowerbody = _vol_lowerbody
        self.vol_rightbody = _vol_rightbody
        self.vol_leftbody = _vol_leftbody
        self.posRing = _posRing
        self.velScalarRing = _velScalarRing
        self.accelScalarRing = _accelScalarRing
        self.jerkScalarRing = _jerkScalarRing
        self.ground_trajectory = _ground_trajectory
        self.flow_effort = _flow_effort
        self.time_effort = _time_effort
        self.weight_effort = _weight_effort
        self.space_effort = _space_effort
        self.travel_distance = _travel_distance
        self.area_covered = _area_covered

        
        #print("posScaled ", self.posScaled)
        #print("posSmooth ", self.posSmooth)
        #print("vel ", self.vel)
        #print("velSmooth ", self.velSmooth)
        #print("accel ", self.accel)
        #print("accelSmooth ", self.accelSmooth)
        #print("jerk ", self.jerk)
        #print("jerkSmooth ", self.jerkSmooth)
        #print("pos_scalar ", self.pos_scalar)
        #print("vel_scalar ", self.vel_scalar)
        #print("accel_scalar ", self.accel_scalar)
        #print("jerk_scalar ", self.jerk_scalar)
        #print("qom ", self.qom)
        #print("bbox ", self.bbox)
        #print("bsphere ", self.bsphere)
        #print("vol_fullbody ", self.vol_fullbody)
        #print("vol_upperbody ", self.vol_upperbody)
        #print("vol_lowerbody ", self.vol_lowerbody)
        #print("vol_rightbody ", self.vol_rightbody)
        #print("vol_leftbody ", self.vol_leftbody)
        #print("posRing ", self.posRing)
        #print("velScalarRing ", self.velScalarRing)
        #print("accelScalarRing ", self.accelScalarRing)
        #print("jerkScalarRing ", self.jerkScalarRing)
        #print("flow_effort ", self.flow_effort)
        #print("time_effort ", self.time_effort)
        #print("weight_effort ", self.weight_effort)
        #print("space_effort ", self.space_effort)
        #print("travel_distance ", self.travel_distance)
        #print("area_covered ", self.area_covered)

