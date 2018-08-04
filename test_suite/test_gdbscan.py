import unittest
import gdbscan
import numpy as np


class TestGdbscan(unittest.TestCase):
    def test_get_centers(self):
        radius = 2
        dims = [10]*3
        offset = [1, 0, 0]

        centers = gdbscan.get_centers(radius, dims, offset)
        self.assertIn([1, 0, 0], centers)
        self.assertIn([6, 5, 0], centers)
        self.assertNotIn([1, 10, 5], centers)

    def test_get_cent_arr(self):
        data = np.stack([np.stack([np.arange(5)]*5)]*5)
        radius = 1
        offset = [0, 0, 0]

        centers = gdbscan.get_centers(radius, list(data.shape), offset)
        cent_arr = gdbscan.get_cent_arr(data, centers, radius)

        self.assertIn(0, cent_arr.flatten().tolist())
        self.assertNotIn(1, cent_arr.flatten().tolist())
        self.assertIn(3, cent_arr.flatten().tolist())

    def test_mult_centers(self):
        data = np.stack([np.stack([np.arange(5)]*5)]*5)
        radius = 1
        offset = [0, 0, 0]

        data = gdbscan.mult_centers(data, radius, offset)

        self.assertIn(0, data.flatten().tolist())
        self.assertNotIn(1, data.flatten().tolist())
        self.assertIn(12, data.flatten().tolist())

    def test_make_tiled_matrix(self):
        radius = 1
        dims = [5]*3

        arr_tiled = gdbscan.make_tiled_matrix(radius, dims)

        self.assertEqual(list(arr_tiled.shape), [6]*3)
        self.assertNotIn(0.5, arr_tiled.flatten().tolist())
        self.assertEqual(7*8, arr_tiled.flatten().tolist().count(1))

    def test_shift_tile(self):
        radius = 1
        dims = [5]*3
        offset = [2, 0, 0]

        arr_tiled = gdbscan.make_tiled_matrix(radius, dims)
        shifted = gdbscan.shift_tile(arr_tiled, offset, dims, radius)

        self.assertEqual(np.zeros((5, 5)).all(), shifted[0].all())
        self.assertEqual(dims, list(shifted.shape))

    def test_breadth_first_search(self):
        cores = np.full((3, 3, 3), False)
        cores[1] = np.full((3, 3), True)
        cores[1][1][1] = False

        start = [0, 0, 0]

        group, vis = gdbscan.breadth_first_search(cores, start)

        self.assertEqual(0, len(group))

        start = [1, 0, 0]
        group, vis = gdbscan.breadth_first_search(cores, start)

        self.assertEqual(8, len(group))
        self.assertNotIn([1, 1, 1], group)

    def test_search_all(self):
        cores = np.full([5]*3, False)
        cores[0] = np.full([5]*2, True)
        cores[2, 2] = np.full([5], True)
        cores[4, 4, 4] = True

        groups = gdbscan.search_all(cores)

        self.assertEqual(3, len(groups))
        lengths = [len(g) for g in groups]
        self.assertIn(1, lengths)
        self.assertIn(5, lengths)
        self.assertIn(25, lengths)

