import unittest
import math

from acre.track import *


class TestSegment(unittest.TestCase):
            
    def test_orientation_1(self):
        segment = Segment([0, 0], [1, 0])
        self.assertEqual(segment.orientation, 0.0)

    def test_orientation_2(self):
        segment = Segment([0, 0], [0, 1])
        self.assertAlmostEqual(segment.orientation, math.pi/2.0)

    def test_orientation_3(self):
        segment = Segment([0, 0], [0, -1])
        self.assertAlmostEqual(segment.orientation, -math.pi/2.0)

    def test_orientation_4(self):
        segment = Segment([0, 0], [-1, 0])
        self.assertAlmostEqual(segment.orientation, math.pi)

    def test_distance_1(self):
        segment = Segment([0, 0], [1, 0])
        self.assertEqual(segment.distance_from_line([0, 0]), 0.0)

    def test_distance_2(self):
        segment = Segment([0, 0], [1, 0])
        self.assertEqual(segment.distance_from_line([1, 0]), 0.0)

    def test_distance_3(self):
        segment = Segment([0, 0], [1, 0])
        self.assertEqual(segment.distance_from_line([0, 1]), 1.0)

    def test_distance_4(self):
        segment = Segment([0, 0], [1, 1])
        self.assertAlmostEqual(segment.distance_from_line([1, 0]), math.sqrt(2.0)/2.0)

    def test_distance_5(self):
        segment = Segment([0, 0], [0, 0])
        self.assertEqual(segment.distance_from_line([0, 0]), 0.0)

    def test_distance_6(self):
        segment = Segment([0, 0], [0, 0])
        self.assertEqual(segment.distance_from_line([1, 0]), 1.0)

    def test_offset_1(self):
        segment = Segment([0, 0], [1, 1])
        self.assertAlmostEqual(segment.offset([1, 0]), -math.sqrt(2.0)/2.0)

    def test_offset_2(self):
        segment = Segment([0, 0], [1, 1])
        self.assertAlmostEqual(segment.offset([0, 1]), math.sqrt(2.0)/2.0)

    def test_distance_outlier_point_1(self):
        segment = Segment([0, 0], [1, 0])
        self.assertEqual(segment.distance_from_line([2, 0]), 1.0)

    def test_distance_outlier_point_2(self):
        segment = Segment([0, 0], [1, 0])
        self.assertEqual(segment.distance_from_line([-1, 0]), 1.0)

    def test_offset_outlier_point_1(self):
        segment = Segment([0, 0], [1, 0])
        self.assertEqual(segment.offset([2, 0]), 1.0)

    def test_offset_outlier_point_2(self):
        segment = Segment([0, 0], [1, 0])
        self.assertEqual(segment.offset([-1, 0]), 1.0)

    def test_offset_outlier_point_3(self):
        segment = Segment([0, 0], [1, 0])
        self.assertEqual(segment.offset([2, 1]), math.sqrt(2.0))

    def test_offset_outlier_point_4(self):
        segment = Segment([0, 0], [1, 0])
        self.assertEqual(segment.offset([2, -1]), -math.sqrt(2.0))

    def test_no_points_when_segment_too_far(self):
        segment = Segment([0, 0], [1, 0])
        points = segment.points_at_distance([-1, 0], 0.5)
        self.assertListEqual(points, [])

    def test_one_point_at_circle_edge(self):
        segment = Segment([0, 0], [1, 0])
        points = segment.points_at_distance([-1, 0], 1)
        self.assertListEqual(points, [[0,0]])

    def test_no_points_if_segment_in_circle(self):
        segment = Segment([0, 0], [1, 0])
        points = segment.points_at_distance([0, 0], 2)
        self.assertListEqual(points, [])

    def compare_points(self, l0, l1):
        self.assertEqual(len(l0), len(l1))
        for i in range(len(l0)):
            self.assertAlmostEqual(l0[i][0], l1[i][0])
            self.assertAlmostEqual(l0[i][1], l1[i][1])

    def test_points_at_distance_1(self):
        segment = Segment([-1, 0], [1, 0])
        points = segment.points_at_distance([0, 0], 0.5)
        self.compare_points(points, [[-0.5, 0], [0.5, 0]])
        
    def test_points_at_distance_2(self):
        segment = Segment([-1, 0.5], [1, 0.5])
        points = segment.points_at_distance([0, 0], 1)
        self.compare_points(points, [[-np.cos(np.radians(30)), 0.5],
                                     [np.cos(np.radians(30)), 0.5]])
        
    def test_points_at_distance_with_only_one_point_too_far(self):
        segment = Segment([0, 0], [2, 0])
        points = segment.points_at_distance([0, 0], 1)
        self.compare_points(points, [[1, 0]])
        
    def test_points_at_distance_with_only_one_point_too_far(self):
        segment = Segment([-2, 0], [0, 0])
        points = segment.points_at_distance([0, 0], 1)
        self.compare_points(points, [[-1, 0]])
        

class TestWaypoints(unittest.TestCase):

    def test_constructor_1(self):
        with self.assertRaises(AssertionError):
            track = Waypoints([[0, 0]])

    def test_constructor_2(self):
        with self.assertRaises(AssertionError):
            track = Waypoints([[0]])

    def test_constructor_3(self):
        with self.assertRaises(AssertionError):
            track = Waypoints([[0,0], [0]])

    def test_offset_and_index_1(self):
        track = Waypoints([[0, 0], [1, 0], [1, 1]])
        offset, index = track.offset_and_index([0, 0])
        self.assertEqual(offset, 0.0)
        self.assertEqual(index, 0)

    def test_offset_and_index_2(self):
        track = Waypoints([[0, 0], [1, 0], [1, 1]])
        offset, index = track.offset_and_index([1, 0])
        self.assertEqual(offset, 0.0)
        self.assertEqual(index, 0)

    def test_offset_and_index_3(self):
        track = Waypoints([[0, 0], [1, 0], [1, 1]])
        offset, index = track.offset_and_index([1, 1])
        self.assertEqual(offset, 0.0)
        self.assertEqual(index, 1)

    def test_offset_and_index_4(self):
        track = Waypoints([[0, 0], [1, 0], [1, 1]])
        offset, index = track.offset_and_index([0, 1])
        self.assertEqual(offset, 1.0)
        self.assertEqual(index, 0)

    def test_offset_and_index_5(self):
        track = Waypoints([[0, 0], [1, 0], [1, 1]])
        offset, index = track.offset_and_index([0.5, 1])
        self.assertEqual(offset, 0.5)
        self.assertEqual(index, 1)

    def test_offset_and_index_5(self):
        track = Waypoints([[0, 0], [1, 0], [1, 1]])
        offset, index = track.offset_and_index([1.5, 1])
        self.assertEqual(offset, -0.5)
        self.assertEqual(index, 1)

    def test_offset_and_index_6(self):
        track = Waypoints([[0, 0], [1, 0], [1, 1]])
        offset, index = track.offset_and_index([1.0, -0.5])
        self.assertEqual(offset, -0.5)
        self.assertEqual(index, 0)

    def default_track(self):
        return Waypoints([[5.0, 2.0],  # start straight line
                          [45.0, 2.0],
                          [47.0, 2.0],   # start U-turn
                          [47.0, 7.8125],
                          [45.0, 7.8125], # start curved line
                          [28.0, 7.8125], # ... curve
                          [22.0, 8.1875],  
                          [5.0, 8.1875]]) 

    def compare_points(self, l0, l1):
        self.assertEqual(len(l0), len(l1))
        for i in range(len(l0)):
            self.assertAlmostEqual(l0[i][0], l1[i][0])
            self.assertAlmostEqual(l0[i][1], l1[i][1])
        
    def test_points_at_distance_1(self):
        segment = self.default_track()
        points = segment.points_at_distance([5, 2], 3)
        self.compare_points(points, [[8, 2]])

    def test_points_at_distance_2(self):
        # a² + b² = c²  -> b = sqrt(c² - a²)  
        # c=3, a=2, b = sqrt(9-4) = sqrt(5)
        segment = self.default_track()
        points = segment.points_at_distance([45, 2], 3)
        self.compare_points(points, [[42, 2], [47, 2 + math.sqrt(5)]])
        

if __name__ == '__main__':
    unittest.main()
