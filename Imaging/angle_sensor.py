import numpy as np
import attitude_utils as attu
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import env_utils as envu

class Angle_sensor(object):
    def __init__(self, eo_model, max_range_intensity=0.0, attitude_parameterization=attu.Quaternion_attitude,  use_range=True, 
                    pool_type='max', fixed_offset=None, use_ideal_offset=True, offset_init=None, state_type=None, optflow_scale=1.0, debug=False,
                    ignore_fov_vio=False, stabilized=True):
        self.ap =  attitude_parameterization
        self.ignore_fov_vio=ignore_fov_vio 
        self.stabilized = stabilized
        self.use_range = use_range
        self.eo_model = eo_model
        print(eo_model.get_optical_axis(np.identity(3)))
        self.max_range_intensity = max_range_intensity
        self.traj_pixel_coords = None
        self.cs_coords = None
        self.fov_violation = None
        self.pixel_int = None
        self.debug = debug
        self.use_ideal_offset = use_ideal_offset
        self.optflow_scale = optflow_scale
        self.fixed_offset = fixed_offset
        self.offset_init = offset_init
        if pool_type == 'ave':
            self.pool_func = self.ave_pool_forward_reshape
            print('using average pooling')
        else:
            self.pool_func = self.max_pool_forward_reshape 
            print('using max  pooling')

        if state_type is None:
            self.state_type=Angle_sensor.simple_state
        else:
            self.state_type = state_type

        if offset_init is None:
            self.offset_init = Angle_sensor.offset_init1
 
        print('Angle sensor:')
        print('\tOutput State type: ', state_type)
        print('\tOffset Init type: ', self.offset_init)
        print('\tFixed Offset: ', self.fixed_offset) 
            
    def reset(self, missile):
        self.initial_attitude = missile.state['attitude']
        self.eo_model.reset()
        self.du = 0.0
        self.dv = 0.0
        self.traj_pixel_coords = None
        self.cs_coords = None
        self.pixel_int = None
        self.image_f = None
        self.image_c = None
        self.last_pixel_coords = None
        self.last_pixel_int = None
      
        # hold location where object first appears as offset for duration of episode 
        if self.use_ideal_offset  :
            self.offset = np.zeros(2)   # just to get the actual pixel coords
            ideal_missile_state = {}
            ideal_missile_state['position'] = missile.state['position']
            ideal_missile_state['velocity'] = self.ideal_velocity 
            ideal_missile_state['attitude'] = missile.state['attitude']
            pixel_coords, _  = self.get_pixel_coords(ideal_missile_state, object_locations=missile.target.state['position'])
            self.offset = self.offset_init(np.squeeze(pixel_coords))
            #print('ideal: ', self.offset)
        elif self.fixed_offset is None:
            self.offset = np.zeros(2)   # just to get the actual pixel coords
            pixel_coords, _  = self.get_pixel_coords(missile.state, object_locations=missile.target.state['position'])
            self.offset = self.offset_init(np.squeeze(pixel_coords))
        else:
            self.offset = self.fixed_offset
        if self.debug:
            print('1: ',self.offset)

    @staticmethod
    def offset_init1(pixel_coords):
        offset = pixel_coords.copy()
        return offset
 
    @staticmethod
    def offset_init2(pixel_coords):
        u = pixel_coords[0]
        v = pixel_coords[1]
        #print('1: ', u,v)
        phi = np.arctan2(v,u)
        r = np.linalg.norm(pixel_coords)
        r = np.clip(r, 15, None)
        u = r*np.cos(phi)
        v = r*np.sin(phi)
        #print('2: ', u,v)
        offset = np.asarray([u,v]) 
        return offset

    @staticmethod
    def offset_init3(pixel_coords):
        u = pixel_coords[0]
        v = pixel_coords[1]
        phi = np.arctan2(v,u)
        r = np.linalg.norm(pixel_coords)
        #print('1: ', u,v,r)
        r = np.clip(r, 5, None)
        u = r*np.cos(phi)
        v = r*np.sin(phi)
        #print('2: ', u,v, r)
        offset = np.asarray([u,v])
        return offset

    @staticmethod
    def offset_init4(pixel_coords):
        u = np.clip(pixel_coords[0], 5, None)
        v = np.clip(pixel_coords[1], 5, None)
        offset = np.asarray([u,v])
        return offset
 
    def get_pixel_coords(self, agent_state,  object_locations=np.zeros(3), render=False ):
        agent_location = agent_state['position']
        agent_velocity = agent_state['velocity']
        out_of_fov = False
        if len(object_locations.shape) < 2:
            object_locations = np.expand_dims(object_locations,axis=0)
        object_intensities = np.linalg.norm(agent_location-object_locations,axis=1)
        if self.stabilized:
            agent_q = self.initial_attitude
        else:
            agent_q = agent_state['attitude']
        self.agent_q = agent_q
        pixel_coords, pixel_int = self.eo_model.get_pixel_coords(agent_location, agent_q, object_locations, object_intensities)
        if render:
            self.render(pixel_coords, pixel_int)
        #pixel_int = np.squeeze(pixel_int) 


        #print('sensor: ', pixel_int, np.linalg.norm(agent_location))

        self.fov_violation =  pixel_coords.shape[0] < 1

        if pixel_coords.shape[0] < 1:
            pixel_coords = np.expand_dims(1.1*self.eo_model.p_y//2*np.ones(2), axis=0)
        else:
            pixel_coords =  pixel_coords# - np.asarray([self.eo_model.p_y//2, self.eo_model.p_x//2])
 
        return pixel_coords, pixel_int

    def get_image_state(self, agent_state,  object_locations, update_dudv=True  ):

        agent_location = agent_state['position']
        agent_velocity = agent_state['velocity']

        pixel_coords,  pixel_int = self.get_pixel_coords( agent_state,  object_locations=object_locations )

        pixel_coords = np.squeeze(pixel_coords)

        self.traj_pixel_coords = pixel_coords.copy()

        pixel_int = np.squeeze(pixel_int)
        self.pixel_int = pixel_int
        #print('PIX: ', pixel_coords, pixel_coords.shape[0])
        if update_dudv:
            if self.fov_violation:
                self.du = 0.0
                self.dv = 0.0
                #print('0')
            elif self.last_pixel_coords is not None:
                #print('PC2: ', pixel_coords, self.last_pixel_coords)
                self.du = self.optflow_scale*(pixel_coords[0] - self.last_pixel_coords[0])
                self.dv = self.optflow_scale*(pixel_coords[1] - self.last_pixel_coords[1]) 
                #print('1: ', pixel_coords, self.last_pixel_coords, du,dv, pixel_vc)
            else:
                self.du = 0.0
                self.dv = 0.0
                #print('2: ', du,dv)
        #print(self.du, self.dv)
        if self.fov_violation < 1:
            pixel_int = 0.0
              
        if update_dudv: 
            self.last_pixel_coords = pixel_coords.copy()
            self.last_pixel_int = pixel_int

        self.cs_coords =  (pixel_coords - self.offset) / (self.eo_model.p_y//2)
        #print('PC: ', self.cs_coords, pixel_int, self.du, self.dv, self.offset)
        state = self.state_type( self.cs_coords, pixel_int, self.du, self.dv) 
        if self.debug and False:
            print('2:',pixel_coords, state, self.cs_coords * (self.eo_model.p_y//2))
        return state 

    @staticmethod
    def simple_state(pixel_coords, pixel_int, du, dv):
        state = pixel_coords
        return state 

    @staticmethod
    def optflow_state(pixel_coords, pixel_int, du, dv):
        #print(du,dv)
        if pixel_coords.shape[0] < 1:
            print(pixel_coords)
        state = np.hstack((pixel_coords,du,dv))
        return state

    @staticmethod
    def optflow_only_state(pixel_coords, pixel_int, du, dv):
        #print(du,dv)
        if pixel_coords.shape[0] < 1:
            print(pixel_coords)
        state = np.hstack((du,dv))
        return state

    @staticmethod
    def pix_state(pixel_coords, pixel_int, du, dv):
        #print(du,dv)
        state = pixel_coords 
        return state


    def check_for_vio(self):
        if self.ignore_fov_vio:
            return False
        else:
            return self.fov_violation
 
    def render(self, pixels, intensities):
        u = np.floor(pixels[:,1]).astype(int) + self.eo_model.p_x//2
        v = np.floor(pixels[:,0]).astype(int) - self.eo_model.p_y//2
        print('UV: ', u,v)
        print('INT: ', intensities)
        intensities = 100*np.ones(u.shape[0])
        image = self.max_range_intensity*np.ones((self.eo_model.p_x,self.eo_model.p_y))
        image[v,u] = np.squeeze(intensities)
        plt.figure()
        plt.imshow(image, interpolation='nearest',cmap='gray')
        plt.grid(True)

    def max_pool_forward_reshape(self, x, stride, pool_height, pool_width):
        """
        A fast implementation of the forward pass for the max pooling layer that uses
        some clever reshaping.

        This can only be used for square pooling regions that tile the input.
        """
        H, W = x.shape
    
        assert pool_height == pool_width == stride, 'Invalid pool params'
        assert H % pool_height == 0
        assert W % pool_height == 0
        x_reshaped = x.reshape(H // pool_height, pool_height,
                               W // pool_width, pool_width)
        out = x_reshaped.max(axis=1).max(axis=2)
        return out

    def ave_pool_forward_reshape(self, x, stride, pool_height, pool_width):
        """
        A fast implementation of the forward pass for the max pooling layer that uses
        some clever reshaping.

        This can only be used for square pooling regions that tile the input.
        """
        H, W = x.shape
    
        assert pool_height == pool_width == stride, 'Invalid pool params'
        assert H % pool_height == 0
        assert W % pool_height == 0
        x_reshaped = x.reshape(H // pool_height, pool_height,
                               W // pool_width, pool_width)
        out = x_reshaped.mean(axis=1).mean(axis=2)
        return out




