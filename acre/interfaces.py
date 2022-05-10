from abc import ABC, abstractmethod


class DataType:
    POSITION = "position"
    TRACK_ERROR = "track-error"
    LANDMARK_ANGLES = "landmark-angles"
    

class ICamera(ABC):
    def __init__(self): 
        pass

    def grab(self):
        pass


class ISegmentation(ABC):
    @abstractmethod
    def compute_mask(self, image):
        pass


class ISensor(ABC):
    def __init__(self): 
        pass

    @abstractmethod
    def measure(self):
        pass

    @property
    @abstractmethod
    def data_type(self):
        pass

    @property
    @abstractmethod
    def size(self):
        pass

    @property
    @abstractmethod
    def names(self):
        pass

    @property
    @abstractmethod
    def std(self):
        pass

    @property
    @abstractmethod
    def var(self):
        pass


class INavigationController(ABC):
    def __init__(self): 
        pass

    @abstractmethod
    def compute_steering(self, track_error):
        pass

    @abstractmethod
    def get_parameters(self):
        pass

    @abstractmethod
    def get_identifier(self):
        pass


class ISteering(ABC):
    def __init__(self): 
        pass

    @abstractmethod
    def drive(self, speed, radius, dt):
        pass


class ITrack(ABC):
    def __init__(self): 
        pass

    @abstractmethod
    def distance(self, p):
        """The distance to the track in meter. Always a positive value."""
        pass

    @abstractmethod
    def offset(self, p):
        """The offset to the track in meter. The absolute value of the offset
        is the same as the distance, but the sign depends on whether
        the point is left (positive) or right (negation) of the
        track.

        """
        pass

    @abstractmethod
    def point_at(self, p):
        pass
    
    @abstractmethod
    def orientation_at(self, p):
        pass

    @property
    @abstractmethod
    def start_position(self):
        """The starting point and orientation of the track."""
        pass
    

class IKalmanFilter(ABC):
    def __init__(self): 
        pass

    @property
    @abstractmethod
    def input_type(self):
        pass

    @property
    @abstractmethod
    def output_type(self):
        pass

    @abstractmethod
    def predict(self, speed, radius, dt):
        pass

    @abstractmethod
    def update(self, measurement, std):
        pass
    
    @abstractmethod
    def estimate(self):
        pass
    

