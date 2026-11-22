from kuafu.mpc.simulation.spatial_utils import Transform
from copy import deepcopy
import torch

class SimulationEntity:

    def __init__(self, name, description):
        self.name = name
        self.description = description
        self._str = str(self.name) + ":" + str(self.description)
        self.has_size = False
        self.prim_path = None
        self.transform_real_time = None
        self.transform_current = None
        self.transform_calculated = None
        self.size = None
        return

    def updateTransform(self, transform):
        self.transform_real_time = transform

    def saveTransform(self, env_mask=None):
        """
        Save the current real-time transform as a checkpoint.
        
        Args:
            env_mask: Optional boolean tensor [batch_size] indicating which environments
                     to save. If None, saves all environments (single-env mode).
        """
        if env_mask is None:
            # Single environment mode - save entire transform
            # Ensure we don't accidentally keep autograd graphs.
            self.transform_current = deepcopy(self.transform_real_time)
        else:
            # Batch mode - only update transforms for specified environments
            if self.transform_current is None:
                # First time saving in batch mode - initialize
                self.transform_current = deepcopy(self.transform_real_time)
            else:
                # Selectively update only the masked environments
                # The Transform.data is a batched tensor [batch_size, 4, 4]
                # Expand mask to match transform dimensions [batch_size, 4, 4]
                mask_expanded = env_mask.view(-1, 1, 1).expand_as(self.transform_real_time.data)
                self.transform_current.set_matrix(torch.where(
                    mask_expanded,
                    self.transform_real_time.data,
                    self.transform_current.data
                ))

    def calculateTransform(self, transform):
        self.transform_calculated = deepcopy(transform)

    def __str__(self):
        return self._str

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __add__(self, other):
        return str(self) + str(other)

    def __radd__(self, other):
        return str(other) + str(self)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __getnewargs_ex__(self):
        return ((self.name, self.description), {})


class VisibleEntity(SimulationEntity):

    def __init__(self, name, description, prime_path, transform: Transform):
        super().__init__(name, description)
        self.has_size = True
        self.prim_path = prime_path
        self.transform_real_time = transform

class SensedEntity(SimulationEntity):

    def __init__(self, name, description, prim_path, transform: Transform):
        super().__init__(name, description)
        self.has_size = False
        self.prim_path = prim_path
        self.transform_real_time = transform

class StateEntity(SimulationEntity):

    def __init__(self, name, index, description, is_continuous=False):
        super().__init__(name, description)
        self.index = index
        self.is_continuous = is_continuous
        self._str = str(self.index) + "." + str(self.name) + ":" + str(self.description)


class FrameEntity(SimulationEntity):
    def __init__(self, name, description):
        super().__init__(name, description)
        self._str = str(self.name) + ":" + str(self.description)


class ControlEntity(StateEntity):

    def __init__(
        self,
        name,
        index,
        description,
        is_continuous=False,
        upper_bound=1.0,
        lower_bound=-1.0,
    ):
        super().__init__(name, index, description, is_continuous)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
