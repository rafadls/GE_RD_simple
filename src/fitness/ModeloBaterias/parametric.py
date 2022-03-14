from interpolation import interp 
from numpy import array, isnan


class Interpolation(object):
    """Numpy Wrapper Fasttest One Dimension Inteporlation

    Args:
        object ([type]): [description]
    """

    def __call__(self, point):
        value = interp(self.xp, self.fp, point + self.scale)
        return 1.0 if isnan(value) else value


class FluidTemperature(Interpolation):
    """Fluid temperature estimation based on Fluid 
    Parametric Model

    Args:
        Interpolation ([type]): [description]
    """
    __slots__ = ('xp', 'fp', 'scale')

    def __init__(self):
        self.xp = array([250., 300., 350., 400., 450.])
        self.fp = array([1006., 1007., 1009., 1014., 1021.])
        self.scale = 273.15


class Density(Interpolation):
    """Density estimation based on Fluid 
    Parametric Model

    Args:
        Interpolation ([type]): [description]
    """
    __slots__ = ('xp', 'fp', 'scale')

    def __init__(self):
        self.xp = array([0.0, 20.0, 40.0])
        self.fp = array([1.293, 1.205, 1.127])
        self.scale = 0.


class RaynoldsNumber(Interpolation):
    """This class Estimate the Raynodls Number based on an 
    Interpolation

    Args:
        Interpolation ([type]): [description]

    Returns:
        [type]: [description]
    """

    __slots__ = ('xp', 'fp', 'scale')

    def __init__(self):
        self.xp = array([250., 300., 350., 400., 450.])
        self.fp = array(
            [159.6, 184.6, 208.2, 230.1, 250.7]
        ) * 1e-7 
        self.scale = 273.15
    
    def __call__(self, velocity, point, lenght, density):
          viscosity = super().__call__(point) 
          return (density * velocity * lenght)/viscosity 


class Conductivity(Interpolation):
    """This Class Estimate the Conductivity of the 
    system

    Args:
        Interpolation ([type]): [description]

    Returns:
        [type]: [description]
    """
    __slots__ = ('xp', 'fp', 'scale')

    def __init__(self):
        self.xp = array([250., 300., 350., 400., 450.])
        self.fp = array([22.3, 26.3, 30, 33.8, 37.3]) * 1e-3
        self.scale = 273.15
    
    def __call__(self, point):
          value = super().__call__(point)
          return 0.0001 if value < 0 else value


class DragConstants(object):
    """This class Estimate the Drag Constant of the system
    Args:
        object ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    __slots__ = ('xp', 'fp1', 'fp2', 'bound')

    def __init__(self):
        self.fp1 = array([0.039, 0.028, 0.027, 0.028, 0.005])
        self.fp2 = array([3.270, 2.416, 2.907, 2.974, 2.063])
        self.xp = array([0.10, 0.25, 0.50, 0.75, 1.00])
        self.bound = array([self.fp1[-1], self.fp2[-1], 0.653])
    
    def __call__(self, _input):
        _a1 = interp(self.xp, self.fp1, _input)
        _a2 = interp(self.xp, self.fp2, _input)
        return (self.bound
                if _input > 1.0 else array([_a1, _a2,  0.653]))


conductivity = Conductivity()
fluid_temperature = FluidTemperature()
density = Density()
raynolds_number = RaynoldsNumber()
drag_constants = DragConstants()
