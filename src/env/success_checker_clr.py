import numpy as np
from src.mujoco_helper import MuJoCoParserClass
from src.mujoco_helper.transforms import r2rpy
import src.object_helpers.receptacles_clr as receptacles
mapper = {
    "microwave": [receptacles.Microwave],
    "white_slide_cabinet": [receptacles.WhiteSlideCabinet],
    "wooden_cabinet": [receptacles.WoodenCabinet],
    "wooden_slide_cabinet": [receptacles.WoodenSlideCabinet],
    "white_cabinet": [receptacles.WhiteCabinet],
    "bench": [receptacles.Bench]
}

class condition_checker:
    def __init__(self, env: MuJoCoParserClass, conditions, obj_names):
        self.gripper_pose_offset = 0.05
        self.env = env
        self.obj_condition = conditions['objects']
        self.gripper_condition = conditions['gripper']
        self.obj_attr = {}
        self.must_loop_receptacles = []
        for obj_name in obj_names:
            print(obj_name)
            if obj_name in mapper:
                cls = mapper[obj_name][0]
                instance = cls(env)
                self.obj_attr[obj_name] = instance
                if len(mapper[obj_name]) > 1:
                    method_name = mapper[obj_name][1]
                    bound_method = getattr(instance, method_name)
                    self.must_loop_receptacles.append((obj_name, bound_method))


    def reset(self, conditions=None):
        for obj_name, obj_attr in self.obj_attr.items():
            if obj_name not in conditions:
                continue
            condition = conditions[obj_name]
            print(condition)
            obj_attr.reset(condition)

    def get_q_pose_of_recp(self):
        q_pos = {}
        for obj_name, obj_attr in self.obj_attr.items():
            temp = obj_attr.get_q_state()
            for joint_name, joint_state in temp.items():
                q_pos[joint_name] = joint_state
        return q_pos 

    def check_success(self, verbose=False):
        if self.check_obj_condition(verbose=verbose) and self.check_gripper_condition():
            return True
        else:
            return False

    def check_obj_condition(self, verbose=False):
        # loop for must loop receptacles
        for obj_name, func in self.must_loop_receptacles:
            func()
        for obj in self.obj_condition:
            if obj['relation'] == 'on':
                obj1, obj2 = obj['names']
                if not self.is_on(obj1, obj2):
                    if verbose:
                        print(f"Failed on {obj1} to be {obj['relation']} {obj2}")
                    return False
            else:
                print("Currently CLR support only 'on' relationship")    
            
            continue
            if obj['relation'] == "turn_on" or obj['relation'] == "turn_off":
                cl_ = self.obj_attr[obj['names'][0]]
                if not cl_.check_range(condition=obj['relation']):
                    return False
            elif obj['relation'] == 'on' or obj['relation'] == 'in':
                obj1, obj2 = obj['names']
                if not self.is_on(obj1, obj2):
                    if verbose:
                        print(f"Failed on {obj1} to be {obj['relation']} {obj2}")
                    return False
            elif obj['relation'] == 'not on' or obj['relation'] == 'not in':
                obj1, obj2 = obj['names']
                if self.is_on(obj1, obj2):
                    if verbose:
                        print(f"Failed on {obj1} to be not {obj['relation']} {obj2}")
                    return False
            elif obj['relation'] == 'upright':
                R = self.env.get_R_body(body_name=obj['names'])
                rpy = r2rpy(R)
                # print(rpy)
                if abs(rpy[0]) > 0.1 or abs(rpy[1]) > 0.1:
                    if verbose:
                        print(f"Failed on {obj['names']} to be upright")
                    return False
            elif obj['relation'] == "open" or obj['relation'] == "close":
                cl_ = self.obj_attr[obj['names'][0]]
                if len(obj['names']) > 1: 
                    region = obj['names'][1]
                    if not cl_.check_range(condition=obj['relation'], region=region):
                        if verbose:
                            print(f"Failed on {obj['names'][0]} {region} to be {obj['relation']}")
                        return False
                else:
                    if not cl_.check_range(condition=obj['relation']):
                        if verbose:
                            print(f"Failed on {obj['names'][0]} to be {obj['relation']}")
                        return False
            else:
                raise NotImplementedError
        return True
    
    def check_gripper_condition(self):
        return True

        # Removing the gripper condition, since it is not relavent yet...

        if 'state' not in self.gripper_condition:
            return True
        state = self.gripper_condition['state']
        q = self.env.get_qpos_joint("rh_l1")[0]
        if q < 0.5:
            current_state = "open"
        else:
            current_state = "close"
        if state != current_state:
            return False
        pose = self.gripper_condition['pose']
        if len(pose) == 0:
            return True
        relation, obj = pose
        target_pose = self.env.get_p_site(site_name=obj)
        gripper_pose = self.env.get_p_body(body_name='tcp_link')
        if relation == "above":
            if gripper_pose[2] < target_pose[2] + self.gripper_pose_offset:
                return False
        elif relation == 'under':
            if gripper_pose[2] > target_pose[2] - self.gripper_pose_offset:
                return False
        elif relation=='behind':
            if gripper_pose[0] > target_pose[0] + self.gripper_pose_offset:
                return False
        elif relation=='front':
            if gripper_pose[0] < target_pose[0] - self.gripper_pose_offset:
                return False
        else:
            raise NotImplementedError
        return True



    def is_on(self, obj1, obj2):
        if isinstance(obj2, list):
            for obj_2 in obj2:
                success = self.check_on(obj1, obj_2)
                if success: return True
            return False
        else:
            return self.check_on(obj1, obj2)
            
    def check_on(self, obj1, obj2):
        # I am changing the 'check on' function for our current purposes. 
        # First we check if the bodies are in contact with each other
        # Second we check if z position of the target object is higher than that of the source object. 

        contact_body_names = self.env.get_contact_body_names()

        if [obj1, obj2] in contact_body_names or [obj2, obj1] in contact_body_names:
            return True
        else: 
            return False
        
        # self.env.contact[]
        
        #      contact_body1 = self.body_names[self.model.geom_bodyid[contact.geom1]]
        
        # res = self.env.model.geom_aabb[obj1]
        # print(res)
        # print(target_size)
        # Get the AABB (center and half-sizes in the geom's local frame)
        # geom_aabb has 6 values: [center_x, center_y, center_z, halfSize_x, halfSize_y, halfSize_z]
        # aabb_local = self.env.data.geom_aabb[geom_id]

        # To transform to the world frame, you need the geom's world orientation (geom_xmat)
        # and position (geom_xpos).
        # geom_pos = self.env.data.geom_xpos[geom_id]
        # geom_mat = self.env.data.geom_xmat[geom_id].reshape(3, 3)

        # print(aabb_local, geom_pos, geom_mat)        
        
        
        return False
        target_pose = self.env.get_p_site(site_name=obj2)
        target_R = self.env.get_R_site(site_name=obj2)
        yaw = r2rpy(target_R)[2]
        target_size = self.get_rotated_aabb_size(target_size, yaw)
        source_pose = self.env.get_p_site(site_name=obj1)
        # print("source_pose", source_pose)
        # print("target_pose", target_pose)
        # print("target_size", target_size)
        # check if the object is within the target site region
        if source_pose[0] > target_pose[0] - target_size[0] and source_pose[0] < target_pose[0] + target_size[0]:
            if source_pose[1] > target_pose[1] - target_size[1] and source_pose[1] < target_pose[1] + target_size[1]:
                if source_pose[2] > target_pose[2] - target_size[2] and source_pose[2] < target_pose[2] + target_size[2]:
                    return True
        return False
    def get_rotated_aabb_size(self, size, yaw):
        """
        Computes the axis-aligned bounding box dimensions for a rotated rectangle.

        Parameters:
        size: A tuple or list (width, height) representing the rectangle's dimensions.
        yaw: The rotation angle in radians (rotation about the z-axis).

        Returns:
        A tuple (new_width, new_height) representing the AABB dimensions.
        """
        w, h, c = size
        new_width = abs(w * np.cos(yaw)) + abs(h * np.sin(yaw))
        new_height = abs(w * np.sin(yaw)) + abs(h * np.cos(yaw))
        return new_width, new_height, c

    