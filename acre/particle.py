import numpy as np
import scipy.stats
from scipy.stats import vonmises

from filterpy.monte_carlo import systematic_resample

from acre.rover import *
from acre.debug import *
from acre.landmarks import *


def create_gaussian_particles(mean, std, n):
    m = len(mean)
    particles = np.empty((n, m))
    for i in range(m):
        particles[:, i] = mean[i] + (np.random.randn(n) * std[i])
    return particles


def estimate_kappa(std):
    x = np.random.randn(1000) * std
    kappa, loc, scale = vonmises.fit(x, fscale=1)
    return kappa


def rgb2hex(r, g, b):
    r = int(r * 255.0)
    g = int(g * 255.0)
    b = int(b * 255.0)
    return f"#{r:02x}{g:02x}{b:02x}"


class ParticleFilter(IKalmanFilter):
    def __init__(self, rover, std, track, number): 
        super().__init__()
        self._rover = rover
        self._track = track
        self._n = number
        self._particles = create_gaussian_particles(rover.position, std, number)
        self._weights = np.ones(number) / number
        self._track_error = np.zeros(2)
        self._position = np.zeros(3)
        self._std = np.zeros(3)

    @property
    def output_type(self):
        return DataType.TRACK_ERROR
        
    def predict(self, speed, radius, dt):
        """ move according to control input u (left speed, right speed)
        with noise Q (std heading change, std velocity) """
        self._store_particles("before")
        left_speed, right_speed = differential_speed(speed, self._rover.width, radius)
        for i in range(self._n):
            self._particles[i, :] = move_with_differential_speed(self._rover.width,
                                                                 self._particles[i],
                                                                 [left_speed, right_speed],
                                                                 dt, self._rover._std)

    def update(self, measurements, std):
        self._update(measurements, std)
        self._resample()
        self._store_particles("after")

    @abstractmethod
    def _update(self, measurements, std):
        pass

    def _resample(self):
        if self._effective_n() < self._n / 2:
            indexes = systematic_resample(self._weights)
            self._resample_from_index(indexes)
            assert np.allclose(self._weights, 1 / self._n)

    def _effective_n(self):
        return 1.0 / np.sum(np.square(self._weights))

    def _resample_from_index(self, indexes):
        self._particles[:] = self._particles[indexes]
        self._n = len(self._particles)
        self._weights.resize(self._n)
        self._weights.fill(1.0 / self._n)
    
    def estimate(self):
        return self._estimate_track_error()
    
    def _estimate_track_error(self):
        self._estimate_position()
        self._compute_track_error()
        return self._track_error
    
    def _estimate_position(self):
        self._position = np.average(self._particles, weights=self._weights, axis=0)
        var = np.average((self._particles - self._position)**2,
                         weights=self._weights, axis=0)
        self._std = np.sqrt(var)
    
    def _compute_track_error(self):
        self._track_error[0] = self._compute_cross_track_error()
        self._track_error[1] = self._compute_orientation_error() 

    def _compute_cross_track_error(self):
        return self._track.offset(self._position[0:2])

    def _compute_orientation_error(self):
        return self._position[2] - self._track.orientation_at(self._position[0:2])

    def _store_particles(self, label):
        svg = self._svg_header()
        svg += self._svg_map()
        for i in range(self._n):
            svg += self._svg_particle(self._particles[i])
        svg += self._svg_rover()
        svg += self._svg_estimated_position()
        svg += self._svg_expected()
        svg += self._svg_footer()
        debug_store_svg(f"particles-{label}", svg)

    def _svg_header(self):
        return ('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
                + '<svg xmlns:svg="http://www.w3.org/2000/svg"\n'
                + '    xmlns="http://www.w3.org/2000/svg"\n'
                + '    xmlns:xlink="http://www.w3.org/1999/xlink"\n'
                + '    width="50cm" height="10cm" viewBox="0 0 50 10">\n')

    def _svg_map(self):
        # TODO: improve the link to the map
        return '<image xlink:href="../acre.png" x="0" y="0" width="50" height="10" />\n'

    def _svg_particle(self, position):
        color = rgb2hex(0.0, 0.0, 1.0)
        return self._svg_triangle(position, 0.2, 0.1, color)

    def _svg_rover(self):
        color = rgb2hex(1.0, 0.0, 0.0)
        return self._svg_triangle(self._rover.position, 0.2, 0.1, color)

    def _svg_estimated_position(self):
        color = rgb2hex(1.0, 1.0, 0.0)
        return self._svg_triangle(self._position, 0.2, 0.1, color)

    def _svg_expected(self):
        color = rgb2hex(0.0, 1.0, 0.0)
        p = self._track.point_at(self._rover.position)
        angle = self._track.orientation_at(self._rover.position)
        return self._svg_triangle([p[0], p[1], angle], 0.2, 0.1, color)

    def _svg_triangle(self, position, length, width, color):
        x = position[0]
        y = 10.0 - position[1]
        angle = np.degrees(position[2])
        return (f'<path d="m {x:0.3f},{y-width/2:0.3f} v {width} l {length},{-width/2} z" '
                + f'style="fill:{color};fill-opacity:1;stroke:none" '
                + f'transform="rotate({angle} {x:0.3f} {y:0.3f})" />\n')

    def _svg_footer(self):
        return ('</svg>\n')


class LandmarksParticleFilter(ParticleFilter):
    def __init__(self, rover, std, landmarks, track, number): 
        super().__init__(rover, std, track, number)
        self._landmarks = landmarks
        self._kappa = None

    @property
    def input_type(self):
        return DataType.LANDMARK_ANGLES

    def _update(self, measurements, std):
        if self._kappa is None:
            # FIXME: assumes that the std is the same for all angles,
            # and that it doesn't change throughout the simulation.
            self._kappa = estimate_kappa(std[0])   
        for i in range(self._n):
            position = self._particles[i, 0:2]
            orientation = self._particles[i, 2]
            weight = self._weights[i]
            for j in range(len(self._landmarks)):
                angle = compute_angle(position, self._landmarks[j]) - orientation
                angle = normalize_angle(angle)
                measured_angle = normalize_angle(measurements[j])
                #weight *= scipy.stats.norm(angle, std).pdf(measured_angle)
                p = vonmises.pdf(angle, self._kappa, loc=measurements[j])
                weight *= p
            self._weights[i] = weight
        self._weights += 1.0e-300  # avoid round-off to zero
        self._weights /= sum(self._weights)  # normalize


class TrackErrorParticleFilter(ParticleFilter):
    def __init__(self, rover, std, track, number): 
        super().__init__(rover, std, track, number)
        self._kappa = None

    @property
    def input_type(self):
        return DataType.TRACK_ERROR

    def _update(self, measurements, std):
        if self._kappa is None:
            # FIXME: assumes that the std doesn't change during the
            # simulation.
            self._kappa = estimate_kappa(std[1])   
        for i in range(self._n):
            position = self._particles[i]
            distance = self._track.offset(position)
            orientation = normalize_angle(position[2]
                                          - self._track.orientation_at(position))
            
            measured_distance = measurements[0]
            p = scipy.stats.norm(distance, std[0]).pdf(measured_distance)
            self._weights[i] *= p
            
            measured_orientation = normalize_angle(measurements[1])
            p = vonmises.pdf(orientation, self._kappa, loc=measured_orientation)
            self._weights[i] *= p
            
        self._weights += 1.e-300  # avoid round-off to zero
        self._weights /= sum(self._weights)  # normalize

    
