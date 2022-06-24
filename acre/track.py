import numpy as np

from acre.interfaces import *
from acre.util import *



# https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
#class Line(ITrack):
#    def __init__(self, p0, p1): 
#        super().__init__()
#        # ax + by + c = 0
#        self.a = p0[1] - p1[1]
#        self.b = p1[0] - p0[0]
#        assert self.a != 0.0 or self.b != 0.0
#        self.c = p0[0] * p1[1] - p1[0] * p0[1]
#        self.orientation = np.arctan2(-self.a, self.b)
#        
#    def distance_from_line(self, p):
#        return abs(self.offset(p))
#        
#    def offset(self, p):
#        return (self.a * p[0] + self.b * p[1] + self.c) / np.sqrt(self.a**2 + self.b**2)
#
#    def orientation_at(self, p):
#        return self.orientation


# https://math.stackexchange.com/questions/330269/the-distance-from-a-point-to-a-line-segment
# https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
class Segment(ITrack):
    def __init__(self, p0, p1): 
        super().__init__()
        self.p0 = p0
        self.p1 = p1
        self.len2 = self._dist2(p0, p1)
        # ax + by + c = 0
        self.a = p0[1] - p1[1]
        self.b = p1[0] - p0[0]
        self.c = p0[0] * p1[1] - p1[0] * p0[1]
        self.orientation = np.arctan2(-self.a, self.b)
        
    def orientation_at(self, p):
        return self.orientation
        
    def offset(self, p):
        return self._sign(p) * self.distance_from_line(p)
        
    def distance_from_line(self, p):
        pt = self.closest_point(p)
        return self._distance(p, pt)
        
    def closest_point(self, p):
        if self.len2 == 0.0:
            return self.p0
        t = (self.b * (p[0] - self.p0[0]) - self.a * (p[1] - self.p0[1])) / self.len2
        t = max(0.0, min(1.0, t))
        return [ self.p0[0] + t * self.b, self.p0[1] - t * self.a ]
        
    def _distance(self, p0, p1):
        return np.sqrt(self._dist2(p0, p1))
    
    def _dist2(self, p0, p1):
        return (p1[0] - p0[0])**2 + (p1[1] - p0[1])**2
    
    def _sign(self, p):
        result = 1.0
        v = self.a * p[0] + self.b * p[1] + self.c
        if v < 0.0:
            result = -1.0
        return result

    @property
    def start_position(self):
        return [self.p0[0], self.p0[1], self.orientation]

    def points_at_distance(self, reference_point, distance):
        pt = self.closest_point(reference_point)
        d = self._distance(reference_point, pt)
        # Line too far from circle, no intersections
        if d > distance:
            return []
        # Line intersects circle in one point
        elif d == distance:
            return [pt]
        d0 = self._distance(reference_point, self.p0)
        d1 = self._distance(reference_point, self.p1)
        # Segment lies inside circle, no intersections
        if d0 < distance and d1 < distance:
            return []
        
        xc = reference_point[0]
        yc = reference_point[1]
        
        # Find the intersections of the line a.x + b.y + c = 0 with the
        # circle (x-xc)² + (y-yc)² = R².
        # Solve:
        #   y = -(a.x + c)/b
        #   u.x² + v.x + w = 0

        if self.b != 0.0:
            u = 1 + (self.a / self.b)**2
            v = 2.0 * (self.a * self.c / self.b**2 + self.a * yc / self.b - xc)
            w = xc**2 + (self.c / self.b)**2 + yc**2 + 2 * self.c * yc / self.b - distance**2
            D = v**2 - 4 * u * w
            
            x0 = (-v - np.sqrt(D)) / (2 * u)
            y0 = -(self.a * x0 + self.c) / self.b
            p0 = [x0, y0]
            
            x1 = (-v + np.sqrt(D)) / (2 * u)            
            y1 = -(self.a * x1 + self.c) / self.b
            p1 = [x1, y1]
        else:
            # a.x + c = 0 => x = -c/a
            # (x-xc)² + (y-yc)² = R² => y = yc +- sqrt(R² - (x-xc)²)
            x0 = -self.c / self.a
            y0 = yc - np.sqrt(distance**2 - (x0 - xc)**2)
            p0 = [x0, y0]
            
            x1 = -self.c / self.a
            y1 = yc + np.sqrt(distance**2 - (x1 - xc)**2)
            p1 = [x1, y1]
            
        points = []
        if self.distance_from_line(p0) < 0.0001:
            points.append(p0)
        if self.distance_from_line(p1) < 0.0001:
            points.append(p1)
        return points


class Waypoints(ITrack):
    def __init__(self, waypoints): 
        super().__init__()
        self._check_waypoints(waypoints)
        self._waypoints = waypoints
        self._n = len(self._waypoints)
        self._segments = []
        for i in range(0, self._n - 1):
            segment = Segment(waypoints[i], waypoints[i+1])
            self._segments.append(segment)

    def _check_waypoints(self, waypoints):
        length = len(waypoints)
        assert length > 1
        for i in range(length):
            assert len(waypoints[i]) == 2

    @property
    def number_waypoints(self):
        return self._n
        
    def waypoints(self, index):
        return self._waypoints[index]

    @property
    def number_segments(self):
        return self._n - 1
        
    def segment(self, index):
        return self._segments[index]
        
    def offset_and_index(self, p):
        index = None
        offset = 1.0e100
        for i in range(self.number_segments):
            d = self.segment(i).offset(p)
            if abs(d) < abs(offset):
                offset = d
                index = i
        return offset, index
                
    def distance_from_line(self, p):
        offset, _ = self.offset_and_index(p)
        return abs(offset)
                
    def offset(self, p):
        offset, _ = self.offset_and_index(p)
        return offset

    def orientation_at(self, p):
        offset, index = self.offset_and_index(p)
        return self._segments[index].orientation_at(p)

    @property
    def start_position(self):
        return self.segment(0).start_position
        
    def closest_point(self, p):
        offset, index = self.offset_and_index(p)
        return self._segments[index].closest_point(p)

    def points_at_distance(self, reference_point, distance):
        result = []
        for i in range(self.number_segments):
            points = self.segment(i).points_at_distance(reference_point, distance)
            if len(points) > 0:
                result.append(*points)
        return result
