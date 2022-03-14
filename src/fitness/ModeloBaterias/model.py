from fitness.ModeloBaterias import parametric
from math import pi
import numpy as np


class BaseModel(object):
    """This class define the phenomenological model structure,
    This could be used for define visual and structural 
    functionals

    Args:
        col_fluid ([int]): column of fluilds
        n_fluid ([int]): number of fluilds
        col_cell ([int]): column of cell
        n_cell ([int]): number of cells
        n (int, optional): fluodynamics iterations 
    """
    __slots__ = (
        'fluodynamics_iterations', 'max_err', 'col_fluid',
        'n_fluid', 'col_cell', 'n_cell','fixed', 'max_number'
    )

    def __init__(self, col_fluid, n_fluid, col_cell, n_cell, n=10):
        self.fluodynamics_iterations, self.max_err = n, 1e-3
        self.col_fluid, self.n_fluid = col_fluid, n_fluid
        self.col_cell, self.n_cell = col_cell, n_cell
        self.fixed = self.load_model_constants()
        self.max_number = 3.4028237 * 1e32
   
    @staticmethod
    def load_model_constants():
        """Load Defined Box Battery Building initial conditions

        Returns:
            [dict]: with defined model initial constants
        """
        constants =  dict(inner_resistance= 32.0 * 1e-3)
        constants.__setitem__('cell_length', 65 * 1e-3)
        constants.__setitem__('wall-cell_space', 15 * 1e-3)
        constants.__setitem__('study_cut', 5 * 1e-3)
        constants.__setitem__('atmosph_pressure', 0.0)
        return constants

    def compute_height(self, point):
        """Compute the height of the defined dimmeter 
        of the system

        Args:
            point ([dict]): row of the data as dict type

        Returns:
            [float]: [description]
        """
        space = 2 * self.fixed['wall-cell_space']
        cell_space = self.n_fluid + point['K'] * self.n_cell
        return space + point['Diametro'] * cell_space

    def compute_inner_area(self, point):
        """Compute the inner area of the defined cell length and 
        bounds

        Args:
            point ([dict]): row of the data as dict type

        Returns:
            [float]: [description]
        """
        pack_in = self.compute_height(point) * self.fixed['study_cut']
        numerator = point['Flujo']  * self.fixed['study_cut']
        denominator = self.fixed['cell_length'] * pack_in
        return numerator / denominator

    def compute_total_heat_flow(self, point, cell_area):
        """Compute the total heat flow area base on current and 
        area of the cell

        Args:
            point ([dict]): row of the data as dict type
            cell_area ([float]): cell area precomputed values

        Returns:
            [float]: [description]
        """
        cell_volume = self.fixed['cell_length'] * cell_area
        resistance = self.fixed['inner_resistance'] / cell_volume
        vol_flow_cell = (point['Current'] ** 2) * resistance
        return vol_flow_cell * self.fixed['study_cut'] * cell_area

    def compute_mass_flow(self, point, init_velocity, init_density):
        """Compute the mass flow of the system based on the diamenter
        of the system and some velocity and densities

        Args:
            point ([dict]):  row of the data as dict type
            init_velocity ([float]): [description]
            init_density ([float]): [description]

        Returns:
            [float]: [description]
        """
        space_dimater = point['Diametro'] * self.fixed['study_cut']
        factor = space_dimater * init_density * init_velocity
        return (point['K'] + 1) * factor, factor, space_dimater


    def get_dynamics_vars(self, point, constants):
        """Define the containers which will be used for dynamics
        evolution estimation 

        Args:
            point ([dict]): row of the data as dict type
            constants ([dict]): predefined fixed values

        Returns:
            [dict]: some precomputed containers
        """
        dynamics = dict(
            temperature=point['t_viento'] * np.ones(self.col_fluid),
            pressure=self.fixed['atmosph_pressure'] * np.ones(self.col_fluid),
            velocity=constants['_inner_area'] * np.ones(self.col_fluid),
            mass_velocity=constants['_inner_area'] * np.ones(self.col_fluid),
            fluid=np.zeros(self.col_fluid), 
            rem=np.zeros(self.col_fluid),
            _density=np.zeros(self.col_fluid),
            cell_temperature=point['t_viento'] * np.ones(self.col_cell)
        )
        initial_conductivity = parametric.conductivity(point['t_viento'])
        dynamics['_density'][0] = constants['_initial_density']
        dynamics['fluidk'] = initial_conductivity * np.ones(self.col_fluid)
        return dynamics
    
    def get_error_container(self):
        """Define the containers for error tracking

        Returns:
            [dict]: [description]
        """
        errors = dict(temperature=self.max_number * np.ones(self.col_fluid))
        errors['velocity'] = self.max_number * np.ones(self.col_fluid)
        errors['cell_temperature'] = self.max_number * np.ones(self.col_cell)
        errors['pressure'] = self.max_number * np.ones(self.col_cell)
        errors['temperature'][0] = 0.0
        return errors

        
class ParametricModel(BaseModel):
    """This class define the most relevant features for evolving a 
    phenomenological system.
    """

    def estimate_initial_conditions(self, point):
        """Define_intial conditions of the system based of the point

        Args:
            point ([dict]): row of the data as dict type
        """
        _diameter = self.fixed['study_cut'] * point['Diametro']
        _control_vol_area = (point['K'] + 1.) * _diameter
        _separation = point['K'] / (point['K'] + 1.)
        cell_area = (point['Diametro'] ** 2.) * (pi / 4.)
        total_heat_flow = self.compute_total_heat_flow(point, cell_area)

        _inner_area = self.compute_inner_area(point)

        _initial_density = parametric.density(point['t_viento'])

        _drag_args = parametric.drag_constants(point['K'])
        _initial_velocity = _drag_args[1] * _inner_area

        _mass_flow, factor, space_diameter = self.compute_mass_flow(
            point, _initial_velocity, _initial_density
        )
        
        _heat_per_area = total_heat_flow / (pi * space_diameter)
        _normalized_area = (pi * space_diameter) / _control_vol_area

        _initial_fluid_flow = 0.5 * factor * _initial_velocity
        _fluid_temperature = total_heat_flow / _mass_flow
        
        return dict((k, v) for k, v in locals().items() 
                    if str(k).startswith('_'))
    
    @staticmethod
    def set_error(idx, k, errors, dynamics):
        """Set defined values of errors for any key name vlues

        Args:
            idx ([int]): index
            k ([str]): name of the dynamics variable
            errors ([dict]): error container 
            dynamics ([dict]): dynamics containers
        """
        errors[k][idx] = np.abs(
            (errors[k][idx] - dynamics[k][idx])/dynamics[k][idx]
        )

    @staticmethod
    def set_pressure(idx, cons, dynamics, errors, function, k):
        """Set pressure values based on precomputed values and 
        dynamics evolution
        """
        mass_velocity = dynamics['mass_velocity'][idx]

        friction = function(
            dynamics['rem'][idx], k,
            mass_velocity/cons['_initial_velocity'],
            dynamics['_density'][idx] / 1.205
        )

        _next = dynamics['pressure'][idx + 1]
        v_density = (mass_velocity ** 2) * dynamics['_density'][idx]
        errors['pressure'][idx] = dynamics['pressure'][idx]
        dynamics['pressure'][idx] = _next + 0.5 * friction * v_density
        ParametricModel.set_error(idx, 'pressure', errors, dynamics)
  
    @staticmethod
    def set_fluid(idx, cons, dynamics, function, _next=False):
        """Set the fluid values base on some of features and individuals
        """
        drag = function(
            cons['_drag_args'][0], dynamics['rem'][idx], 
            cons['_normalized_area'], dynamics['_density'][idx] / 1.205
        )

        if _next:
            speed = dynamics['velocity'][idx] ** 2
            dynamics['fluid'][idx + 1] = (
              .5 * cons['_diameter'] * dynamics['_density'][idx] * speed * drag
            )
        else:
            dynamics['fluid'][idx] = cons['_initial_fluid_flow'] * drag

    @staticmethod
    def set_velocity(idx, cons, dynamics, errors, _next=False):
        """set velocity in containers such as dynamics and errors based
        of two conditions. 
        """
        if _next:
            dt = dynamics['pressure'][idx] - dynamics['pressure'][idx + 1]
            dt = cons['_control_vol_area'] * dt - dynamics['fluid'][idx + 1]
            dt = dt / cons['_mass_flow']
            errors['velocity'][idx + 1] = dynamics['velocity'][idx + 1]
            dynamics['velocity'][idx + 1] = dt + dynamics['velocity'][idx]
        else:
            base =  dynamics['fluid'][idx] / cons['_mass_flow']
            errors['velocity'][idx] = dynamics['velocity'][idx]
            dynamics['velocity'][idx] = cons['_initial_velocity'] - base

        ParametricModel.set_error(
            idx + int(_next), 'velocity', errors, dynamics
        )

    @staticmethod
    def set_raynolds(idx, cons, dynamics, point, _next=0):
        """set raynolds number in containers, based on point row data
        """
        dynamics['mass_velocity'][idx] = (
            dynamics['velocity'][idx] * cons['_separation']
        )
        
        dynamics['rem'][idx + _next] = parametric.raynolds_number(
            dynamics['mass_velocity'][idx], dynamics['temperature'][idx],
            point['Diametro'], dynamics['_density'][idx]
        )

    @staticmethod
    def set_fluid_temperature(i, cons, dynamics, errors):
        """set fluild temerature and dynamics and errors containers
        """
        current_temp = dynamics['temperature'][i]
        c_power = parametric.fluid_temperature(current_temp)
        dt = dynamics['velocity'][i + 1] ** 2 - dynamics['velocity'][i] ** 2
        errors['temperature'][i + 1] = dynamics['temperature'][i + 1]
        dynamics['temperature'][i + 1] = (
            current_temp + ( cons['_fluid_temperature'] - .5 * dt) / c_power
        )
        ParametricModel.set_error(i + 1, 'temperature', errors, dynamics)

    @staticmethod
    def set_fluid_cell(idx, cons, dynamics, errors, point, function):
        """Define the fluid cell based of previous and next temperature 
        and other elements
        """
        nusselt = function(
            2.0 * (idx + 1), dynamics['rem'][idx], cons['_drag_args'][2]
        )
        fluid =  parametric.conductivity(dynamics['temperature'][0])

        temp_n = dynamics['temperature'][idx]
        temp_n_1 = dynamics['temperature'][idx + 1]

        errors['cell_temperature'][idx] = dynamics['cell_temperature'][idx]
        heat = cons['_heat_per_area'] / (nusselt * fluid / point['Diametro'])
        dynamics['cell_temperature'][idx] = (heat + (temp_n + temp_n_1)/2.0)
        dynamics['fluidk'][idx] = fluid
        ParametricModel.set_error(idx, 'cell_temperature', errors, dynamics)

    def initial_states(self, cons, dynamics, errors, point, indv):
        """Initial conditions of the system
        """
        self.set_fluid(0, cons, dynamics, indv[0])
        self.set_velocity(0, cons, dynamics, errors)
        self.set_raynolds(0, cons, dynamics, point)
        self.set_pressure(0, cons, dynamics, errors, indv[1], point['K'])

    def evolve_states(self, idx, cons, dynamics, errors, point, indv):
        """System evolution parts for dynamics aproximation based on some
        properties of the system
        """
        self.set_raynolds(idx, cons, dynamics, point, 1)
        self.set_pressure(idx, cons, dynamics, errors, indv[1], point['K'])
        self.set_fluid(idx, cons, dynamics, indv[0], True)
        self.set_velocity(idx, cons, dynamics, errors, True)
        self.set_fluid_temperature(idx, cons, dynamics, errors)
        dynamics['_density'][idx + 1] = parametric.density(
            dynamics['temperature'][idx + 1]
        )
        self.set_fluid_cell(idx, cons, dynamics, errors, point, indv[2])
    

    def evolve(self, point, individuals):
        """Individuals Evolution based on Parametric Model construction

        Args:
            point ([dict]): [description]
            individuals ([type]): [description]
        """

        cons = self.estimate_initial_conditions(point)
        dynamics = self.get_dynamics_vars(point, cons)
        errors = self.get_error_container()

        for _ in range(self.fluodynamics_iterations):
            self.initial_states(cons, dynamics, errors, point, individuals)

            for idx in range(self.col_fluid - 1):
                self.evolve_states(
                    idx, cons, dynamics, errors, point, individuals
                )
            
            is_all_converged = sum(int(v.max() <= self.max_err) 
                                   for v in errors.values())
    
            if is_all_converged == len(errors):
                break

        return (
            dynamics['velocity'][1:], dynamics['pressure'][:-1], 
            dynamics['cell_temperature']
        )