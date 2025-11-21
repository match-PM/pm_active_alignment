from active_alignment_interfaces.msg import JointAlign 


class ControllerJoint:
    def __init__(self, 
                is_active_joint:bool,
                initial_joint_value:float,
                joint_align_msg:JointAlign):
        
        self.name = joint_align_msg.joint_name
        self.has_constraint = joint_align_msg.has_constraint
        self.upper_limit = joint_align_msg.upper_limit

        self.lower_limit = -(abs(joint_align_msg.lower_limit))
        self.initial_joint_value = initial_joint_value
        self.is_active_joint = is_active_joint  

        if self.has_constraint and self.upper_limit == self.lower_limit:
            raise ValueError(f"Joint {self.name} has constraints but upper and lower limits are equal.")

        if self.upper_limit < 0:
            raise ValueError(f"Joint {self.name} has constraints that is less than zero. This is not a valid configuration.")

        self._current_joint_value = self.initial_joint_value
        self._upper_joint_limit = None
        self._lower_joint_limit = None

        self._upper_joint_optimization_limit = None
        self._lower_joint_optimization_limit = None

    def set_current_value(self, value:float):
        if not self.is_active_joint:
            raise ValueError(f"Cannot set joint value for inactive joint {self.name}.")
        
        if self._upper_joint_limit is None or self._lower_joint_limit is None:
            raise ValueError(f"Joint limits not initialized for joint {self.name}.")
        
        if value > self._upper_joint_limit or value < self._lower_joint_limit:
            raise ValueError(f"Joint value {value} is out of bounds for joint {self.name}.")
        
        self._current_joint_value = value

    def get_current_value(self) -> float:   
        return self._current_joint_value

    def initialize_joint_limits(self, 
                                upper_limit:float, 
                                lower_limit:float):
        
        if not self.is_active_joint:
            raise ValueError(f"Cannot initialize joint limits for inactive joint {self.name}.")
        
        self._upper_joint_limit = upper_limit
        self._lower_joint_limit = lower_limit

        if self._upper_joint_limit < self._lower_joint_limit:
            raise ValueError(f"Upper limit {self._upper_joint_limit} is less than lower limit {self._lower_joint_limit} for joint {self.name}.")
        
        if self.initial_joint_value > self._upper_joint_limit or self.initial_joint_value < self._lower_joint_limit:
            raise ValueError(f"Initial joint value {self.initial_joint_value} is out of bounds for joint {self.name}.")

        if self.has_constraint:
            # Calculate optimization limits based on constraints
            if self.initial_joint_value + self.upper_limit > self._upper_joint_limit:
                self._upper_joint_optimization_limit = self._upper_joint_limit
            else:
                self._upper_joint_optimization_limit = self.initial_joint_value + self.upper_limit
            # lower limit
            if self.initial_joint_value - abs(self.lower_limit) < self._lower_joint_limit:
                self._lower_joint_optimization_limit = self._lower_joint_limit
            else:
                self._lower_joint_optimization_limit = self.initial_joint_value - abs(self.lower_limit)
        else:
            self._upper_joint_optimization_limit = self._upper_joint_limit
            self._lower_joint_optimization_limit = self._lower_joint_limit

    def get_debug_info(self):
        return {
            "name": self.name,
            "is_active_joint": self.is_active_joint,
            "has_constraint": self.has_constraint,
            "upper_limit": self.upper_limit,
            "lower_limit": self.lower_limit,
            "initial_joint_value": self.initial_joint_value,
            "current_joint_value": self._current_joint_value,
            "upper_joint_limit": self._upper_joint_limit,
            "lower_joint_limit": self._lower_joint_limit,
            "upper_joint_optimization_limit": self._upper_joint_optimization_limit,
            "lower_joint_optimization_limit": self._lower_joint_optimization_limit
        }

    def get_mapped_current_value(self) -> float:
        """
        Map the current joint value to a normalized [0, 1] range 
        based on the optimization limits.
        Handles negative joint ranges safely and clamps the output.
        """
        if not self.is_active_joint:
            raise ValueError(f"Cannot get mapped joint value for inactive joint {self.name}.")
        
        if self._upper_joint_optimization_limit is None or self._lower_joint_optimization_limit is None:
            raise ValueError(f"Optimization limits not initialized for joint {self.name}.")

        upper = self._upper_joint_optimization_limit
        lower = self._lower_joint_optimization_limit
        current = self._current_joint_value

        range_span = upper - lower
        if range_span <= 0:
            # Avoid division by zero or invalid range
            return 0.5  # neutral mid-point

        # Normalize and clamp to [0, 1]
        normalized = (current - lower) / range_span
        normalized_clamped = max(0.0, min(1.0, normalized))
        return normalized_clamped

    def set_current_value_from_mapped(self, normalized_value: float):
        """
        Set the current joint value from a normalized [0, 1] value 
        using the optimization limits.
        """
        if not self.is_active_joint:
            raise ValueError(f"Cannot set joint value for inactive joint {self.name}.")
        
        if self._upper_joint_optimization_limit is None or self._lower_joint_optimization_limit is None:
            raise ValueError(f"Optimization limits not initialized for joint {self.name}.")

        # Clamp normalized value to valid range
        normalized_clamped = max(0.0, min(1.0, normalized_value))

        lower = self._lower_joint_optimization_limit
        upper = self._upper_joint_optimization_limit

        self._current_joint_value = lower + normalized_clamped * (upper - lower)


class AlignmentController:
    def __init__(self, controller_name):
        self.controller_name = controller_name
        self.joints:list[ControllerJoint] = []

    def add_joint(self, joint:ControllerJoint):
        for existing_joint in self.joints:
            if existing_joint.name == joint.name:
                raise ValueError(f"Joint {joint.name} already exists in controller {self.controller_name}. Check for duplicates joints in your action request.")
        self.joints.append(joint)

    def is_joint_in_controller(self, joint_name:str) -> bool:
        for joint in self.joints:
            if joint.name == joint_name:
                return True
        return False

    def get_joint_states(self, only_active: bool=False) -> dict[str, float]:
        states = {}
        for joint in self.joints:
            if not only_active or joint.is_active_joint:
                states[joint.name] = joint.get_current_value()
        return states
    
    def get_joint_states_mapped(self) -> dict[str, float]:
        states = {}
        for joint in self.joints:
            if joint.is_active_joint:
                states[joint.name] = joint.get_mapped_current_value()
        return states

class AlignmentControllerManager:
    def __init__(self):
        self.controllers:list[AlignmentController] = []

    def add_controller(self, controller:AlignmentController):
        self.controllers.append(controller)

    def add_joint_to_controller(self, 
                                controller_name:str, 
                                joint:ControllerJoint):
        
        for controller in self.controllers:
            if controller.controller_name == controller_name:
                controller.add_joint(joint)
                return
        
        # create new controller
        new_controller = AlignmentController(controller_name)
        new_controller.add_joint(joint)
        self.controllers.append(new_controller)

    def get_debug_info(self):
        info = {}
        # num of controllers
        info["num_controllers"] = len(self.controllers)
        # joints per controller
        for controller in self.controllers:
            info[controller.controller_name] = [joint.get_debug_info() for joint in controller.joints]
        return info
    
    def is_joint_in_any_controller(self, joint_name:str) -> bool:
        for controller in self.controllers:
            if controller.is_joint_in_controller(joint_name):
                return True
        return False

    def get_controllers(self) -> list[AlignmentController]:
        return self.controllers

    def get_all_joint_states(self, only_active: bool=False) -> dict[str, float]:
        states = {}
        for controller in self.controllers:
            states.update(controller.get_joint_states(only_active=only_active))
        return states
    
    def get_all_joint_states_as_mapped(self) -> dict[str, float]:
        states = {}
        for controller in self.controllers:
            states.update(controller.get_joint_states_mapped())
        return states
    
    def set_joint_states_from_mapped(self, mapped_states:dict[str, float]):
        for controller in self.controllers:
            for joint in controller.joints:
                if joint.is_active_joint and joint.name in mapped_states:
                    joint.set_current_value_from_mapped(mapped_states[joint.name])

    def set_joint_states(self, joint_states:dict[str, float]):
        for controller in self.controllers:
            for joint in controller.joints:
                if joint.name in joint_states:
                    joint.set_current_value(joint_states[joint.name])