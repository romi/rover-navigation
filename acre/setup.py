import argparse
import cv2
    
from acre.rover import *
from acre.linefollowing import *
from acre.track import *
from acre.landmarks import *
from acre.particle import *
from acre.debug import *
        
class Setup():
    """The following setups are available:

    * no-sensors: The rover will drive freely and no steering
      correction is done. The noise in the drive system will likely
      steer the rover off track.

    * line-following: The top-camera will detect the position and
      orientation of the line of crops in the center of the bed to
      adjust the steering. No Kalman filtering is done.

    * line-following-soil: Similar to the previous setup but in this
      case the camera will detect the position and orientation of the
      soil between two lines of crops to adjust the steering. No
      Kalman filtering.

    * landmarks: The sensor measures the angles to the four landmarks
      and the absolute position and orientation of the rover is
      estimated. Based on this position and the predefined track, the
      error from the track is computed. No Kalman filtering.

    * landmarks-particle-filter: Similar to the previous
      setup. However, in this case, a particle filter is used to
      reduce the effects of noise of the angle sensor and rover
      steering.

    """ 
    SETUP_NO_SENSORS = "no-sensors"
    SETUP_PERFECT_SENSOR = "perfect-sensor"
    SETUP_LINE_FOLLOWING_CROP = "line-following"
    SETUP_LINE_FOLLOWING_SOIL = "line-following-soil"
    SETUP_LANDMARKS = "landmarks"
    SETUP_LANDMARKS_PARTICLE_FILTER = "landmarks-particle-filter"
    SETUP_TRACK_ERROR_PARTICLE_FILTER = "track-error-particle-filter"
    DEFAULT_SETUP = SETUP_LINE_FOLLOWING_CROP

    TRACK_LINE = "line"
    TRACK_CURVE = "curve"
    TRACK_FULL = "full"
    
    DEFAULT_WIDTH = 1.4 # m
    DEFAULT_LENGTH = 1.4 # m
    DEFAULT_STD_ROVER = 0.012 # m
    # position=(x,y,angle), with x, y in m, angle in radians from x to x-rover
    #DEFAULT_START_POSITION = [ 5.5, 2.0, np.radians(0.0) ] 
    DEFAULT_SPEED = 0.2 # m/s
    DEFAULT_UPDATE_FREQUENCY = 2.0
    DEFAULT_L1 = 3.0 # m
    DEFAULT_OUTDIR = "out"
    DEFAULT_NUMBER_PARTICLES = 1000
    DEFAULT_STD_ANGLES = 0.4  # degrees

    FIELD_WIDTH = 50.0
    FIELD_HEIGHT = 10.0
    CAMERA_PROJECTION_SIZE_X = 0.9
    CAMERA_PROJECTION_SIZE_Y = 1.6
    CAMERA_IMAGE_HEIGHT = 1080
    CAMERA_IMAGE_WIDTH = 1920
    CAMERA_IMAGE_HEIGHT = 1080
    
    def __init__(self):
        parser = argparse.ArgumentParser(description='ACRE navigation simulator')
        parser.add_argument('--setup',
                            type=str,
                            choices={self.SETUP_NO_SENSORS,
                                     self.SETUP_PERFECT_SENSOR,
                                     self.SETUP_LINE_FOLLOWING_CROP,
                                     self.SETUP_LINE_FOLLOWING_SOIL,
                                     self.SETUP_LANDMARKS,
                                     self.SETUP_LANDMARKS_PARTICLE_FILTER,
                                     self.SETUP_TRACK_ERROR_PARTICLE_FILTER},
                            default=self.DEFAULT_SETUP, 
                            help=f'The combination of sensor and filter to use. (Default: {self.DEFAULT_SETUP})')
        parser.add_argument('--track',
                            type=str,
                            choices={self.TRACK_LINE,
                                     self.TRACK_CURVE,
                                     self.TRACK_FULL},
                            default=self.TRACK_LINE, 
                            help=f'The track to follow. (Default: {self.TRACK_LINE})')
        parser.add_argument('--iterations',
                            type=int,
                            default=-1, 
                            help=f'The number of iterations to execute the navigation loop. The value -1 means that the simulation will continue until the end of the track is reached, or until the track error is more than 1 meter. (Default: -1)')
        parser.add_argument('--speed',
                            type=float,
                            default=self.DEFAULT_SPEED, 
                            help=f'The speed in m/s.(Default: {self.DEFAULT_SPEED})')
        parser.add_argument('--frequency',
                            type=float,
                            default=self.DEFAULT_UPDATE_FREQUENCY, 
                            help=f'The frequency at which the position and orientation are measured. (Default: {self.DEFAULT_UPDATE_FREQUENCY})')
        parser.add_argument('--width',
                            type=float,
                            default=self.DEFAULT_WIDTH, 
                            help=f'The width of the rover. (Default: {self.DEFAULT_WIDTH})')
        parser.add_argument('--length',
                            type=float,
                            default=self.DEFAULT_LENGTH, 
                            help=f'The length of the rover. (Default: {self.DEFAULT_LENGTH})')
        parser.add_argument('--l1',
                            type=float,
                            default=self.DEFAULT_L1, 
                            help=f'The L1 parameter, indicating the distance of the point ahead on the track (in meter) to which the rover should steer to correct errors. (Default: {self.DEFAULT_L1})')
        parser.add_argument('--std-rover',
                            type=float,
                            default=self.DEFAULT_STD_ROVER, 
                            help=f'The standiard deviation of the rover, in meters. (Default: {self.DEFAULT_STD_ROVER})')
        parser.add_argument('--std-angles',
                            type=float,
                            default=self.DEFAULT_STD_ANGLES, 
                            help=f'The standiard deviation of the angle measurements of the landmarks, in degrees. (Default: {self.DEFAULT_STD_ANGLES})')
        parser.add_argument('--particles',
                            type=int,
                            default=self.DEFAULT_NUMBER_PARTICLES, 
                            help=f'The number of particles. (Default: {self.DEFAULT_NUMBER_PARTICLES})')
        parser.add_argument('--out',
                            type=str,
                            default=self.DEFAULT_OUTDIR, 
                            help=f'The output directory. Use "False" to deactivate image output. (Default: {self.DEFAULT_OUTDIR})')
        parser.add_argument('--no-map', action='store_true')

        self.args = parser.parse_args()
        # print(self.args)
        self._init_values(self.args)
        self._rover = None
        self._top_camera = None
        self._image_segmentation = None
        self._landmarks = None
        self._track = None
        self._sensor = None
        self._filter = None
        self._navigation_controller = None
        self._steering = None
        if self.args.out and self.args.out != "False":
            debug_set_output_directory(self.args.out)
        
    def _init_values(self, args):
        self.width = args.width
        self.length = args.length
        self.speed = args.speed
        self.update_frequency = args.frequency
        #self.start_position = self.DEFAULT_START_POSITION
        self.delta_time = 1.0 / self.update_frequency
        self.output_directory = self.DEFAULT_OUTDIR
        self.iterations = args.iterations
        self.no_map = args.no_map
        
    @property
    def start_position(self):
        return self.track.start_position
        
    @property
    def rover(self):
        if self._rover is None:
            self._build_rover()
        return self._rover

    def _build_rover(self):
        self._rover = Rover(self.width, self.length,
                            self.start_position,
                            self.args.std_rover)

    @property
    def top_camera(self):
        if self._top_camera is None:
            self._build_top_camera()
        return self._top_camera
    
    def _build_top_camera(self):
        map_image = cv2.imread("acre.png") 
        self._top_camera = MapCamera(self.rover, map_image, 
                                     self.FIELD_WIDTH, self.FIELD_HEIGHT,
                                     self.CAMERA_PROJECTION_SIZE_X,
                                     self.CAMERA_PROJECTION_SIZE_Y,
                                     self.CAMERA_IMAGE_WIDTH, self.CAMERA_IMAGE_HEIGHT)

    @property
    def image_segmentation(self):
        if self._image_segmentation is None:
            self._build_image_segmentation()
        return self._image_segmentation

    def _build_image_segmentation(self):
        self._image_segmentation = SVM([-0.09421272925919766, 0.24514808436606472,
                                        -0.1509125064580985],
                                       -1.6243314759109384)

    @property
    def landmarks(self):
        if self._landmarks is None:
            self._build_landmarks()
        return self._landmarks

    def _build_landmarks(self):
        self._landmarks = np.array([[0.0, 0.0],
                                    [self.FIELD_WIDTH, 0.0],
                                    [self.FIELD_WIDTH, self.FIELD_HEIGHT],
                                    [0.0, self.FIELD_HEIGHT]])

    @property
    def track(self):
        if self._track is None:
            self._build_track()
        return self._track

    def _build_track(self):
        #self._track = Line([5.0, 2.0], [45.0, 2.0])
        if self.args.track == self.TRACK_LINE:
            self._track = Segment([5.0, 2.0], [45.0, 2.0])
        elif self.args.track == self.TRACK_CURVE:
            self._track = Waypoints([[5.0, 8.1875], [22.0, 8.1875],
                                     [28.0, 7.8125], [45.0, 7.8125]])
        elif self.args.track == self.TRACK_FULL:
            self._track = Waypoints([[5.0, 2.0],  # start straight line
                                     [45.0, 2.0],
                                     [47.0, 2.0],   # start U-turn
                                     [47.0, 7.8125],
                                     [45.0, 7.8125], # start curved line
                                     [28.0, 7.8125], # ... curve
                                     [22.0, 8.1875],  
                                     [5.0, 8.1875]]) 
        else: raise ValueError(f"Unknown track: {self.args.track}")
            
    @property
    def sensor(self):
        if self._sensor is None:
            self._build_sensor_and_filter()
        return self._sensor

    @property
    def filter(self):
        if self._filter is None:
            self._build_sensor_and_filter()
        return self._filter
    
    def _build_sensor_and_filter(self):
        if self.args.setup == self.SETUP_NO_SENSORS:
            self._build_dummy_sensor()
            self._build_dummy_filter()
            
        elif self.args.setup == self.SETUP_PERFECT_SENSOR:
            self._build_perfect_sensor()
            self._build_dummy_filter()
            
        elif self.args.setup == self.SETUP_LINE_FOLLOWING_CROP:
            # Look at plants in center
            self._build_line_detector(self.CAMERA_IMAGE_WIDTH/2, False)
            self._build_dummy_filter()
            
        elif self.args.setup == self.SETUP_LINE_FOLLOWING_SOIL:
            # Look at soil in between plants
            self._build_line_detector(741, True)
            self._build_dummy_filter()
            
        elif self.args.setup == self.SETUP_LANDMARKS:
            self._build_landmark_track_follower()
            self._build_dummy_filter()
            
        elif self.args.setup == self.SETUP_LANDMARKS_PARTICLE_FILTER:
            self._build_landmark_angle_sensor()
            self._build_landmark_particle_filter()
            
        elif self.args.setup == self.SETUP_TRACK_ERROR_PARTICLE_FILTER:
            self._build_line_detector(self.CAMERA_IMAGE_WIDTH/2, False)
            self._build_track_error_particle_filter()
            
        else: raise ValueError(f"Unknown setup: {self.args.setup}")
            
            
    def _build_line_detector(self, center_x, inverted):
        pixels_per_meter = self.CAMERA_IMAGE_WIDTH / self.CAMERA_PROJECTION_SIZE_Y
        self._sensor = LineDetector(self.top_camera, self.image_segmentation,
                                    center_x, self.CAMERA_IMAGE_HEIGHT/2,
                                    pixels_per_meter, inverted)

    def _build_landmark_track_follower(self):
        # TODO: Perfect sensor???
        angle_sensor = LandmarkAngleDetector(self.rover, self.landmarks, np.radians(self.args.std_angles)) 
        position_sensor = LandmarkPositioning(self.rover, self.landmarks, angle_sensor)
        self._sensor = PositionBasedTrackFollower(position_sensor, self.track)

    def _build_landmark_angle_sensor(self):
        self._sensor = LandmarkAngleDetector(self.rover, self.landmarks, np.radians(self.args.std_angles))
        
    def _build_dummy_sensor(self):
        self._sensor = DummySensor()
        
    def _build_perfect_sensor(self):
        position_sensor = PerfectSensor(self.rover)
        self._sensor = PositionBasedTrackFollower(position_sensor, self.track)
        
    def _build_dummy_filter(self):
        self._filter = DummyKalmanFilter()
        
    def _build_landmark_particle_filter(self):
        self._filter = LandmarksParticleFilter(self.rover,
                                               [0.2, 0.2, np.radians(2.0)],
                                               self.landmarks, self.track,
                                               self.args.particles)
        
    def _build_track_error_particle_filter(self):
        self._filter = TrackErrorParticleFilter(self.rover,
                                                [0.2, 0.2, np.radians(2.0)],
                                                self.track,
                                                self.args.particles)

    @property
    def navigation_controller(self):
        if self._navigation_controller is None:
            self._build_navigation_controller()
        return self._navigation_controller

    def _build_navigation_controller(self):
        self._navigation_controller = L1Controller(self.width, self.args.l1)

    @property
    def steering(self):
        if self._steering is None:
            self._build_steering()
        return self._steering

    def _build_steering(self):
        self._steering = DifferentialSteering(self.rover)
        return self._steering
