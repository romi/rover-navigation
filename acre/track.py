import numpy as np

from acre.interfaces import *


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
#    def distance(self, p):
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
        return self._sign(p) * self.distance(p)
        
    def distance(self, p):
        return np.sqrt(self._square_distance(p))
        
    def point_at(self, p):
        if self.len2 == 0.0:
            return self.p0
        t = (self.b * (p[0] - self.p0[0]) - self.a * (p[1] - self.p0[1])) / self.len2
        t = max(0.0, min(1.0, t))
        return [ self.p0[0] + t * self.b, self.p0[1] - t * self.a ]
        
    def _square_distance(self, p):
        pt = self.point_at(p)
        return self._dist2(p, pt)
        
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
                
    def distance(self, p):
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
        
    def point_at(self, p):
        offset, index = self.offset_and_index(p)
        return self._segments[index].point_at(p)
