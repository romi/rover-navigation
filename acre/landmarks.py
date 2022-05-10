import numpy as np

from acre.interfaces import *
from acre.rover import Rover
from acre.track import *


def distance(x0, x1):
    return np.linalg.norm(x0 - x1)


def unit_vector(x0, x1):
    return (x1 - x0) / distance(x0, x1)


def orthogonal(v):
    return np.array([[0, -1], [1, 0]]) @ v


def compute_angle(p0, p1):
    d = p1 - p0
    return np.arctan2(d[1], d[0])
        

def compute_circle(x0, x1, angle):
    d = distance(x0, x1)
    ex = unit_vector(x0, x1)
    ey = orthogonal(ex)
    r = d / (2.0 * np.sin(angle))
    t = ex * d / 2.0
    v = ey * d * np.cos(angle) / (2.0 * np.sin(angle))
    c = x0 + t + v
    # print("t", t, "v", v)
    return c, r


def compute_intersections(c0, r0, c1, r1):
    d = distance(c0, c1)
    ex = unit_vector(c0, c1)
    ey = orthogonal(ex)
    x = (d**2 + r0**2 - r1**2) / (2.0 * d)
    y = np.sqrt(r0**2 - x**2)
    p0 = c0 + x * ex + y * ey
    p1 = c0 + x * ex - y * ey
    return p0, p1


class LandmarkAngleDetector(ISensor):
    def __init__(self, rover, landmarks, std):
        super().__init__()
        self._rover = rover
        self._landmarks = landmarks
        self._std = std
        self._angles = np.zeros(4)

    @property
    def data_type(self):
        return DataType.LANDMARK_ANGLES

    @property
    def size(self):
        return 4

    @property
    def names(self):
        return ['angle-1', 'angle-2', 'angle-3', 'angle-4']

    @property
    def std(self):
        return [self._std, self._std, self._std, self._std]

    @property
    def var(self):
        v = self._std**2.0
        return [v, v, v, v]
    
    def measure(self):
        for i in range(4):
            self._angles[i] = self._measure_angle(self._landmarks[i])
        noise = np.random.randn(self.size) * self._std
        # print(np.degrees(noise))
        self._angles += noise
        return self._angles, self.std
        
    def _measure_angle(self, p):
        # The angle from x-axis to the landmark, in the rover's coordinate system
        angle = compute_angle(self._rover.position[0:2], p)
        return angle - self._rover.orientation


class LandmarkPositioning(ISensor):
    def __init__(self, rover, landmarks, sensor, maxdist=0.2):
        super().__init__()
        self._rover = rover
        self._landmarks = landmarks
        self._sensor = sensor
        self._result = np.zeros(3)
        self._maxdist = maxdist

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
    
    def measure(self):
        return self._estimate_position(), self.std
    
    def _estimate_position(self):
        angles, std = self._sensor.measure()
        # print("angles", angles, np.degrees(angles))
        centers, radii = self._compute_circles(angles)
        # print("centers", centers)
        # print("radii", radii)
        intersections = self._compute_intersections_circles(centers, radii)
        # print("intersections", intersections)
        position = self._compute_position(intersections)
        orientation = self._compute_orientation(position, angles)
        # print("estimated position", position)
        self._result[0:2] = position
        self._result[2] = orientation
        return self._result

    def _compute_circles(self, angles):
        if not len(self._landmarks) == len(angles):
            print("landmarks", self._landmarks)
            print("angles", angles)
            raise ValueError("Different number of landmarks and angles")
        centers = np.zeros((4, 2))
        radii = np.zeros((4, 1))
        for i in range(len(self._landmarks)):
            j = (i + 1) % 4
            centers[i, :], radii[i] = compute_circle(self._landmarks[i],
                                                     self._landmarks[j],
                                                     angles[j]-angles[i])
        return centers, radii

    def _compute_intersections_circles(self, centers, radii):
        if not len(centers) == len(radii):
            raise ValueError("Different number of centers and radii")
        intersections = np.zeros((4, 2))
        for i in range(len(centers)):
            j = (i + 1) % 4
            p0, p1 = compute_intersections(centers[i], radii[i], centers[j], radii[j])
            if not self._is_landmark(p0):
                intersections[i, :] = p0
            elif not self._is_landmark(p1):
                intersections[i, :] = p1
            else:
                raise ValueError("Both circles intersect only in landmark!")
        return intersections

    def _is_landmark(self, p):
        result = False
        for i in range(len(self._landmarks)):
            if distance(p, self._landmarks) < self._maxdist:
                result = True
                break
        return result

    def _compute_position(self, points):
        return np.mean(points, axis=0)

    def _compute_orientation(self, position, angles):
        # At the position of the rover, the absolute angle (in fixed
        # coordinate system) from x-axis to the first landmark
        absolute_angle = compute_angle(position, self._landmarks[0])
        # At the position of the rover, the estimated angle (in the
        # rover's coordinate system) from x-axis to the first landmark
        measured_angle = angles[0]
        # The estimated orientation of the rover
        orientation = absolute_angle - measured_angle
        return orientation


class PositionBasedTrackFollower(ISensor):
    def __init__(self, sensor, track): 
        super().__init__()
        assert sensor.data_type == DataType.POSITION
        self._sensor = sensor
        self._track = track
        self._track_error = np.zeros(2)
        self._position = np.zeros(3)
        
    @property
    def data_type(self):
        return DataType.TRACK_ERROR
        
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
    
    def measure(self):
        return self._estimate_track_error(), self.std
    
    def _estimate_track_error(self):
        self._position, std = self._sensor.measure()
        self._track_error[0] = self._compute_cross_track_error()
        self._track_error[1] = self._compute_orientation_error() 
        return self._track_error

    def _compute_cross_track_error(self):
        return self._track.offset(self._position[0:2])

    def _compute_orientation_error(self):
        return self._position[2] - self._track.orientation_at(self._position[0:2])


class TestAngleDetector(ISensor):
    def __init__(self, rover, landmarks, std):
        super().__init__()
        self._std = std
        # Expected: [-168.6901  -11.3099   11.3099  168.6901]
        self._angles = np.radians(np.array([-167.7381, -11.8255, 11.1299, 168.1374]))

    @property
    def data_type(self):
        return DataType.LANDMARK_ANGLES

    @property
    def size(self):
        return 4

    @property
    def names(self):
        return ['angle-1', 'angle-2', 'angle-3', 'angle-4']

    @property
    def std(self):
        return [self._std, self._std, self._std, self._std]

    @property
    def var(self):
        v = self._std**2.0
        return [v, v, v, v]

    def measure(self):
        return self._angles, self.std


def test1(n, std_angles):
    rover = Rover(1.0, 1.0, [5.0, 2.0, np.radians(20.0)], 0.0)
    landmarks = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 10.0], [0.0, 10.0]])
    sensor = LandmarkAngleDetector(rover, landmarks, np.radians(std_angles))
    positioning = LandmarkPositioning(rover, landmarks, sensor)

    x = np.random.uniform(5.0, 45.0, n)
    y = np.random.uniform(1.0, 3.0, n)
    o = np.random.uniform(-np.pi/2.0, np.pi/2.0, n)
    error = np.zeros((n, 3))

    for i in range(n):
        # print("--------------------------------------------------")
        position = np.array([x[i], y[i], o[i]])
        rover.set_position(position)
        estimated_position, std = positioning.measure()
        error[i, :] = estimated_position - position
        # print(f"position ({position[0]:0.6f}, {position[1]:0.6f}, {np.degrees(o[i]):0.3f})"
        #      + f", estimated position ({estimated_position[0]:0.6f}, {estimated_position[1]:0.6f}, {np.degrees(estimated_position[2]):0.3f}), "
        #      + f", error ({error[0]:0.6f}, {error[1]:0.6f}, {np.degrees(error[2]):0.3f})")

    mean = np.mean(error, axis=0)
    std = np.std(error, axis=0)
    print(f"{std_angles}, {mean[0]}, {mean[1]}, {mean[2]}, {std[0]}, {std[1]}, {std[2]}")


def test2():
    rover = Rover(1.0, 1.0, [25.0, 5.0, np.radians(0.0)], 0.0)
    landmarks = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 10.0], [0.0, 10.0]])
    angle_sensor = LandmarkAngleDetector(rover, landmarks, np.radians(1))
    track = Segment([5.0, 2.0], [45.0, 2.0])
    position_sensor = LandmarkPositioning(rover, landmarks, angle_sensor)
    track_follower = PositionBasedTrackFollower(position_sensor, track)
    
    n = 20
    x = np.random.uniform(5.0, 45.0, n)
    y = np.random.uniform(1.0, 3.0, n)
    o = np.random.uniform(-np.pi/2.0, np.pi/2.0, n)

    for i in range(n):
        print("--------------------------------------------------")
        position = np.array([x[i], y[i], o[i]])
        rover.set_position(position)
        track_error, std = track_follower.measure()
        print(f"position ({position[0]:0.3f}, {position[1]:0.3f}, {np.degrees(o[i]):0.3f})"
              + f", track error ({position[1]-2.0:0.3f}, {np.degrees(o[i]):0.3f})"
              + f", estimated track error ({track_error[0]:0.3f}, {np.degrees(track_error[1]):0.3f})")


def test3():
    # angles +-11.3099°, +-168.6901°
    expected = [25.0, 5.0, np.radians(0.0)]
    rover = Rover(1.0, 1.0, expected, 0.0)
    landmarks = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 10.0], [0.0, 10.0]])
    # angle_sensor = TestAngleDetector(rover, landmarks, 0.01)
    angle_sensor = LandmarkAngleDetector(rover, landmarks, 0.01)
    position_sensor = LandmarkPositioning(rover, landmarks, angle_sensor)
    position = position_sensor.measure()
    
    print(f"expected expected ({expected[0]:0.3f}, {expected[1]:0.3f}, {np.degrees(expected[1]):0.3f}), "
          + f"measured position ({position[0]:0.3f}, {position[1]:0.3f}, {np.degrees(position[2]):0.3f})")


if __name__ == "__main__":
    test1(1000, 2.0)
