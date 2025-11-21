__all__ = ["ContinuousBuildingControlEnvironment"]

import numpy as np
from scipy import signal
import pandas as pd
import gym
from gym import spaces, logger
from gym.utils import seeding
DATA_PATH = './data/'

# Environment
class ContinuousBuildingControlEnvironment(gym.Env):
    r"""
    Description:
        The environment includes the indoor thermal environment and the
        outdoor weather conditions. It can be described as
            s_{t+1} = f(s_t, a_t),
    Observation:
        Type: continuous space
        
        :T_env:     Envelope temperature, in C.
        :T_air:     Room air temperature, in C.
        :T_cor:     Corridor temperature, constant 22C.
        :T_out:     Outdoor air temperature, in C from data.
        :Qsg:       Solar gain to the room, in W from data.
        :Qint:      Internal heat gain to the room, in W from data.
        :Hour:      Time of the day, 0-23 hr, for setpoint scheduling
        Num Observation             Min         Max
        0   T_env                   10           35
        1   T_air                   18           27
        2   T_cor                   21           23
        3   T_out                  -40           40
        4   Q_SG                     0           1100
        5   Q_int                   50           180
        6   Hour                     0           23
    Actions:
        Type: Continouse space.box()
        Range -2kW - 2kW
        :a_t:       is the action of setting HVAC thermal input, in W.
        
        Num Actionz: 1
    Reward:
        Reward is the sum of (-1*electricity consumption) and occupants' utility for every time step,
        including termimnation step. The electricity consumption is in kWh.
        r_t = -1*abs(a_t/COP) + U,
        where U = beta*(T_air - T_ref)**2,
              COP(T_out) is the efficiency of the heating/cooling system.
    Starting State:
        T_env = 22. degC
        T_air = 22. degC
        T_cor = 22. degC
        t = start, is the starting time step
        T_out, Q_int, Q_SG are the values from data at the starting time step
    Episode Termination:
        When the end of the data is reached.
    Parameters:
        
        :param dt:      Transition time-step in sec
        :param data:    Weather data input in pandas format
        :param start:   Starting time of an episode as in the hour of year
        :param end:     Ending time of an episode as in the hour of year
        :param state:   Current environment states 
                        [T_env, T_air, T_cor, T_out, Qsg, Qint]
        :param t:       Current time, as in the hour of year, 0-8760
                        
        Parameters related to zone thermal properties:
        The parameters under this category will be inherited from the model universe class in the Meta-RL problem
        
        :param C_env:   Thermal capacitance of the zone envelope state, W/C
        :param C_air:   Thermal capacitance of the zone air state, W/C
        :param R_rc:    Thermal resistance between zone air and corridor, J/W
        :param R_oe:    Thermal resistance between outdoor and zone envelope, J/W
        :param R_er:    Thermal resistance between zone envelope and zone air state, J/W
        :param a_sol_env:  Fraction of solar heat gain on the zone envelope
        
        Parameters related to zone occupant (utility function parameter):
        The parameters under this category will be inherited from the model universe class in the Meta-RL problem
            
        :param beta:    Sensitivity to the distance between T_air and T_ref
        :param T_ref:   The preferred zone air temperature by occupant, degC
        Parameters related to building HVAC equipment:
        
        :param Tlv_heating:     Leaving water temperature of the heat pump in heating condition. degC constant.
        :param E_cf_heating:    Coefficients of the COP function of the heat pump in heating condition
        :param Tlv_cooling:     Leaving water temperature of the heat pump in cooling condition. degC constant.
        :param E_cf_cooling:    Coefficients of the COP function of the heat pump in cooling condition
        :param cp_air:          Specific heat of air
        :param m_dot_min        Minimum air flow rate to a room in kg/s.
        :param T_sup            Supply air temperature in deg C.
        :param m_design         Design flow rate based on 400 cfm for 1 room, kg/s.
        :param dP = 500         Design pressure increase, in Pa.
        :param e_tot            Fan efficiency.
        :param rho_air          Air density, kg/m^3.
        :param c_FAN            Fan coefficients.
        :param Qh_max           Maximum heating capacity of reheat
        :param m_dot_max        Maximum air flow rate to a room in kg/s
    """

    def __init__(self, data_file, dt = 1800, start = 0., end = 10000.,
                 C_env = None, C_air = None, R_rc = None, R_oe = None,
                 R_er = None, lb_set = 22., ub_set = 24.):
        
        
        # Initialize the parameters
        self.dt = dt
        self.data = pd.read_csv(DATA_PATH + data_file)
        self.start = start
        self.end = end
        self.num_step = int((self.end -self.start)/(self.dt/3600) + 1)        

        # Thermal capacitance, W/C
        self.C_env = C_env
        self.C_air = C_air
        # Thermal resistance, J/W
        self.R_rc = R_rc
        self.R_oe = R_oe
        self.R_er = R_er
        # Fraction of solar heat gain on the envelope
        self.a_sol_env = 0.90303

        # A and B matrices for state space
        A = np.zeros((2, 2))
        B = np.zeros((2, 5))

        A[0, 0] = (-1. / self.C_env) * (1. /\
                        self.R_er + 1./self.R_oe)
        A[0, 1] = 1. / (self.C_env * self.R_er)
        A[1, 0] = 1. / (self.C_air * self.R_er)
        A[1, 1] = (-1. / self.C_air) * \
                  (1. /self.R_er + 1. / self.R_rc)

        B[0, 1] = 1./(self.C_env * self.R_oe)
        B[0, 2] = self.a_sol_env / self.C_env
        B[1, 0] = 1. / (self.C_air * self.R_rc)
        B[1, 2] = (1. - self.a_sol_env) / self.C_air
        B[1, 3] = 1. / self.C_air
        B[1, 4] = 1. / self.C_air
        
        # C matrix
        C = np.array([[1, 0]])
        # D matrix
        D = np.zeros(5)
        
        # Discretize the matrices
        sys = signal.StateSpace(A, B, C, D)
        discrete_matrix = sys.to_discrete(dt = self.dt)
        self.A = discrete_matrix.A
        self.B = discrete_matrix.B        
        
        # Utility parameters
        self.beta = -0.05
        self.T_ref = 23.
        self.lb = lb_set
        self.ub = ub_set
        
        # HVAC parameters
        self.Tlv_cooling = 7.
        self.Tlv_heating = 35.
        self.E_cf_cooling =  np.array([14.8187, -0.2538, 0.1814, -0.0003, -0.0021, 0.002])
        self.E_cf_heating =  np.array([7.8885, 0.1809, -0.1568, 0.001068, 0.0009938, -0.002674])
        self.C_cf_cooling =  np.array([8471.4, 212.1, 782.6, -5.1, -3.7, -7.5 ])
        self.C_cf_heating =  np.array([7.8885, 0.1809, -0.1568, 0.001068, 0.0009938, -0.002674])
        self.m_dot_min = 0.080938984
        self.cp_air = 1004
        self.T_sup = 16.5
        self.m_design = 0.9264*0.4
        self.dP = 500
        self.e_tot = 0.6045
        self.rho_air = 1.225
        self.c_FAN = np.array([0.040759894, 0.08804497, -0.07292612, 0.943739823, 0])
        self.Qh_max = 1500
        self.m_dot_max = self.m_dot_min*550/140
        self.seed()
        
        # Define the action space in OpenAI gym format
        self.action_space = spaces.Box(low = np.array([0.]), high = np.array([1.]), dtype=np.float32)
        # Lower bound of the observation space
        self.low = np.array([10.0, 15.0, 21.0, -40.0, 0., 50., 0.])
        # Upper bound of the observation space
        self.high = np.array([35.0, 28.0, 23.0, 40.0, 1100., 180., 23.])
        # Define the state space in OpenAI gym format
        self.observation_space = spaces.Box(low = np.zeros((7)), high = np.ones((7)), dtype=np.float32)
        
        # Initialize state and time
        self.state = None
        self.t = None
        
    def seed(self, seed = None):
        """
        Setting the seed of the Environment
        """
        
        self.np_random, seed = seeding.np_random(seed)
        
        return [seed]    
    

    # The evolution of system states
    def step(self, a_t):

        r"""
        Outputs:
        Update self.state to be the next-time-step environmental states after the action taken.
        r is the reward for the agent based on energy consumption and temperature control.
        done is the indicator whether the end of epoch has been reached
        """
        
        s_t = self.state * (self.high - self.low) + self.low
        
        # Coefficient of performance of HVAC system
        T_out = s_t[3]  
                        
        EFh = self.E_cf_heating
        Tlvh = self.Tlv_heating

        EFc = self.E_cf_cooling
        Tlvc = self.Tlv_cooling
        
        COPh = 0.9#EFh[0] + T_out*EFh[1] + Tlvh*EFh[2] + (T_out**2)*EFh[3] + (Tlvh**2)*EFh[4] + T_out*Tlvh*EFh[5]
        COPc = EFc[0] + T_out*EFc[1] + Tlvh*EFc[2] + (T_out**2)*EFc[3] + (Tlvc**2)*EFc[4] + T_out*Tlvc*EFc[5]
        
        
        # Rescale the action
        if a_t == 0.5:
            u_t = self.m_dot_min*self.cp_air*(self.T_sup - s_t[1])
            u_c = u_t
            u_h = 0
            m_fan = self.m_dot_min
            energy = -1*u_t/COPc/1000/2
        elif a_t > 0.5:
            u_t = ((a_t - 0.5)/0.5)*self.Qh_max + self.m_dot_min*self.cp_air*(self.T_sup - s_t[1])
            u_c = self.m_dot_min*self.cp_air*(self.T_sup - s_t[1])
            u_h = ((a_t - 0.5)/0.5)*self.Qh_max
            m_fan = self.m_dot_min
            energy = (((a_t - 0.5)/0.5)*self.Qh_max/COPh + self.m_dot_min*self.cp_air*(s_t[1] - self.T_sup)/COPc)/1000/2             
        else:
            m_fan = (self.m_dot_max - self.m_dot_min)*((0.5 - a_t)/0.5)+self.m_dot_min
            u_t = m_fan*self.cp_air*(self.T_sup - s_t[1])
            u_c = u_t
            u_h = 0
            energy = -1*u_t/COPc/1000/2
        
        # Get states T_env, T_air
        s_next_room = self.A@s_t[:2] + self.B@np.append(s_t[2:6], u_t)
            
        # Fan power
        f_flow = m_fan/self.m_design
        f_pl = self.c_FAN[0]+self.c_FAN[1]*f_flow+self.c_FAN[2]*f_flow**2+self.c_FAN[3]*f_flow**3+self.c_FAN[4]*f_flow**4
        Q_fan = f_pl*self.m_design*self.dP/(self.e_tot*self.rho_air)
        energy += Q_fan/1000/2
        
        # Time t
        #t = s_t[-1] + 0.5
        self.t = self.t + 0.5
        # Exogenous states T_out, T_cor, Qsg, Qint, Hour.
        T_cor = 22.
        T_out = self.data.iloc[int(self.t*2)].Tout
        Qsg = self.data.iloc[int(self.t*2)].Qsg
        Qint = self.data.iloc[int(self.t*2)].Qint
        Hour = self.data.iloc[int(self.t*2)].Hour
        s_ext = np.array([T_cor, T_out, Qsg, Qint, Hour])
        
        
        # Utility function
        #Utility = self.beta*(s_next_room[-1] - self.T_ref)**2.
        if Hour >= 7 and Hour <= 20:
            lb = self.lb
            ub = self.ub
        
            if s_next_room[-1] < lb:
                Utility = -5.
                Temp_exceed = (lb - s_next_room[-1]) * 0.5
            elif s_next_room[-1] > ub:
                Utility = -5.
                Temp_exceed = (s_next_room[-1] - ub) * 0.5
            else:
                Utility = 0.
                Temp_exceed = 0.
                
        else:
            lb = 15.
            ub = 28.
                
            Utility = 0.
            Temp_exceed = 0.

        
        # Reward based on energy consumption and temperature control.
        #energy = -np.sqrt(a_t**2.)/COP/1000./2.
        r = -1*energy + Utility
        
        # Assemble the states
        self.state = (np.concatenate([s_next_room, s_ext]) - self.low)/(self.high - self.low)
        
        # Check if end of training period has been reached.
        done = bool((self.t + 0.5) > self.end)
        if done == True:
            logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
        
        return np.array(self.state), r, done, {'u_t': u_t, 'a_t': a_t, 'Energy': energy, 'Penalty': Utility/(-10), 'Exceedance': Temp_exceed, 'lb': lb, 'ub': ub}


    # Reset the environment to its initial condition
    def reset(self):
        
        # Initial condition of the environment at start time.
        self.t = self.start
        T_env_0 = 22.
        T_air_0 = 22.
        T_cor = 22.
        T_out = self.data.iloc[int(self.t*2)].Tout
        Qsg = self.data.iloc[int(self.t*2)].Qsg
        Qint = self.data.iloc[int(self.t*2)].Qint
        Hour = self.data.iloc[int(self.t*2)].Hour
        self.state = (np.array([T_env_0, T_air_0, T_cor, T_out, Qsg, Qint, Hour]) - self.low)/(self.high - self.low)
        
        return np.array(self.state)
