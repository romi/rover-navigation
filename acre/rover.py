import math
import numpy as np
    
from acre.interfaces import *
from acre.debug import *


def normalize_angle(angle):
    """ Return an angle between ]-pi, pi] """
    while angle <= -np.pi:
        angle += 2.0 * np.pi
    while angle > np.pi:
        angle -= 2.0 * np.pi
    return angle


def move_with_differential_speed(wheel_base, position, speed, dt, std):
    half_wheel_base = 0.5 * wheel_base
    
    # d_left and d_right are the distances travelled by the left and right
    # wheel.
    d_left = dt * (speed[0] + np.random.randn() * std)
    d_right = dt * (speed[1] + np.random.randn() * std)
    
    # dx and dy are the changes in the location of the rover, in
    # the frame of reference of the rover.
    if d_left == d_right:
        dx = d_left
        dy = 0.0
        alpha = 0.0
    else:
        radius = half_wheel_base * (d_left + d_right) / (d_right - d_left)
        if radius >= 0:
            alpha = d_right / (radius + half_wheel_base)
        else:
            alpha = -d_left / (-radius + half_wheel_base)
        dx = radius * math.sin(alpha)
        dy = radius - radius * math.cos(alpha)

    # Convert dx and dy to the changes in the last frame of
    # reference (i.e. relative to the current orientation).
    orientation = position[2]
    c = math.cos(orientation)
    s = math.sin(orientation)
    dx_ = c * dx - s * dy
    dy_ = s * dx + c * dy
    position[0] += dx_
    position[1] += dy_
    position[2] = normalize_angle(position[2] + alpha)
    return position


class Rover():
    def __init__(self, width, length, position, std):
        self._width = width
        self._length = length
        self._position = np.array(position)
        self._std = std

    @property
    def dimensions(self):
        return (self._width, self._length)

    @property
    def width(self):
        return self._width

    @property
    def length(self):
        return self._length

    @property
    def position(self):
        return self._position

    def set_position(self, position):
        self._position = np.array(position)

    @property
    def x(self):
        return self._position[0]

    def set_x(self, x):
        self._position[0] = x
    
    @property
    def y(self):
        return self._position[1]

    def set_y(self, y):
        self._position[1] = y
    
    @property
    def orientation(self):
        return self._position[2]

    def set_orientation(self, angle_radians):
        self._position[2] = normalize_angle(angle_radians)
    
    def move(self, dt, left_speed, right_speed):
        self._position = move_with_differential_speed(self._width,
                                                      self._position,
                                                      [left_speed, right_speed],
                                                      dt, self._std)

class MapCamera(ICamera):
    def __init__(self, rover, image,
                 field_width, field_height,
                 camera_projection_width, camera_projection_height,
                 camera_image_width, camera_image_height):
        super().__init__()
        self.rover = rover
        self.image = image
        # The size of the field, in meters
        self.field_width = field_width
        self.field_height = field_height
        # The size of the ground captures by the image, in meters
        self.camera_projection_width = camera_projection_width
        self.camera_projection_height = camera_projection_height
        # The size of the image produced by the camera, in pixels
        self.camera_image_width = camera_image_width
        self.camera_image_height = camera_image_height

    def grab(self):
        map_height, map_width, _ = self.image.shape
        # cv2.warpAffine expects shape in (length, height)
        shape = (map_width, map_height)
        pixels_per_meter_x = map_width / self.field_width
        pixels_per_meter_y = map_height / self.field_height
        rx = int(self.rover.x * pixels_per_meter_x)
        ry = int(map_height - self.rover.y * pixels_per_meter_y)
        # print(f"({self.rover.x}, {self.rover.y}) -> ({rx}, {ry})  ({pixels_per_meter_x}, {pixels_per_meter_y})")
        matrix = cv2.getRotationMatrix2D(center=(rx, ry),
                                         angle=-np.degrees(self.rover.orientation),
                                         scale=1)
        # print(matrix)
        rotated_image = cv2.warpAffine(src=self.image, M=matrix, dsize=shape)
        # debug_store_image("xxx-rotated", rotated_image, "jpg")

        w = int(self.camera_projection_width * pixels_per_meter_x)
        h = int(self.camera_projection_height * pixels_per_meter_y)
        x0 = rx
        x1 = rx + w
        y0 = int(ry - h / 2)
        y1 = y0 + h
        # print(f"ground {x0}-{x1}, {y0}-{y1}")
        ground_img = rotated_image[y0:y1, x0:x1]
        # debug_store_image("xxx-ground", ground_img, "jpg")

        # In the rover, the camera is rotated so that the image height
        # is along the x-axis and width is across the vegetable bed.
        camera_img = cv2.rotate(ground_img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        camera_img = cv2.resize(camera_img, (self.camera_image_width,
                                             self.camera_image_height))
        # debug_store_image("xxx-camera", camera_img, "jpg")
        return camera_img


class L1Controller(INavigationController):
    def __init__(self, width, l1):
        super().__init__()
        self._width = width
        self._l1 = l1

    def compute_steering(self, track_error):
        distance = track_error[0]
        orientation = track_error[1]
        radius = None  # None = straight forward
        if distance > self._l1:
            raise ValueError(f"L1Controller: distance > L1: {distance:0.6f} > {self._l1:0.6f}")
        gamma = -math.atan(distance / math.sqrt(self._l1**2 - distance**2))
        phi = orientation - gamma
        if phi != 0:
            radius = -self._l1 / (2.0 * math.sin(phi))
            # correction = self._width / (2.0 * R)
        return radius

    def get_parameters(self):
        return {'controller': 'L1', 'L': self._l1}

    def get_identifier(self):
        return f"l1-{self._l1:0.3f}"

def differential_speed(speed, width, radius):
    if radius is None:
        return speed, speed
    else:
        delta = width / (2.0 * radius)
        return speed * (1 - delta), speed * (1 + delta)

class DifferentialSteering(ISteering):
    def __init__(self, rover):
        super().__init__()
        self.rover = rover

    def drive(self, speed, radius, dt):
        left_speed, right_speed = differential_speed(speed, self.rover.width, radius)
        self.rover.move(dt, left_speed, right_speed)


class DummySensor(ISensor):
    def __init__(self): 
        super().__init__()

    @property
    def data_type(self):
        return DataType.TRACK_ERROR

    def measure(self):
        return [0.0, 0.0], [0.0, 0.0]        

    @property
    def size(self):
        return 2

    @property
    def names(self):
        return ['cross-track-error', 'orientation']

    @property
    def std(self):
        return [0.0, 0.0]

    @property
    def var(self):
        return [0.0, 0.0]
    

class PerfectSensor(ISensor):
    def __init__(self, rover): 
        super().__init__()
        self._rover = rover
        
    def measure(self):
        return self._rover.position, self.std

    @property
    def data_type(self):
        return DataType.POSITION

    @property
    def size(self):
        return 3

    @property
    def names(self):
        return ['x', 'y', 'orientation']

    @property
    def std(self):
        return [0.0, 0.0, 0.0]

    @property
    def var(self):
        return [0.0, 0.0, 0.0]
    

class DummyKalmanFilter(IKalmanFilter):
    def __init__(self): 
        super().__init__()
        self._measurement = None

    @property
    def input_type(self):
        return DataType.TRACK_ERROR

    @property
    def output_type(self):
        return DataType.TRACK_ERROR
        
    def predict(self, speed, radius, dt):
        pass

    def update(self, measurement, std):
        self._measurement = measurement
    
    def estimate(self):
        return self._measurement
    

