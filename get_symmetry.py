import numpy as np
from multiprocessing.pool import Pool
import time
# Method for finding symmetries:
#   Load data
#   Sample small number of points (make sure that they are not nan)
#   Begin with yz plane -> xz -> xy
#   Assume that the symmetry will be in 50% + or - 5%
#   Multiprocess multiple different reflections
#   Check that 95% of points are within 10% of opposite value
#   If so, sample a bunch more points just to make sure
#   Do for all planes
#   Return position of each plane reflection

def symmetries(data):
    print("Finding symmetries . . .")
    start = time.time()

    num_points = 1000
    dims = [[0, data.shape[0]], [0, data.shape[1]], [0, data.shape[2]]]
    points = get_points(num_points, dims)

    syms = [-1] * 3

    num_workers = 8
    proc = Pool(processes=num_workers)
    for direction in ['x', 'y', 'z']:
        print("Finding " + direction + " symmetry . . .")
        dist_step = 0.5 * num_workers
        start_dists = list(range(0, dist_step, step=0.5))

        # Build the arguments for each worker
        args = []
        for i in range(num_workers):
            args.append((points, data, direction, start_dists[i], dist_step))

        results = proc.starmap_async(find_sym, args)
        # Flatten the list
        results = [ item for sublist in results for item in sublist ]

        axis = {'x': 0, 'y': 1, 'z': 2}[direction]
        while -1 in results:
            results.remove(-1)

        # Try more intensive symmetry finding.
        # Get 100 times more points to retest the supposed symmetries
        new_points = get_points(num_points*100, dims)
        # Build more arguments
        args = []
        for i in range(len(results)):
            args.append((new_points, data, direction, results[i]))

        large_results = proc.starmap_async(are_symmetric, args)

        # Choose position closest to enter if there happen to be multiple


                print("Found symmetry along " + direction + " axis")
                syms[axis] = res
                break
        else:
            print("No symmetry found along " + direction + " axis")

    print("Total time elapsed: ", (time.time() - start))

    return syms


def find_sym(points, data, inversion, start_dist, dist_step):
    # Maximum distance from center plane to try to find symmetries
    maxdist = 50

    axis = {'x': 0, 'y': 1, 'z': 2}[inversion]
    mid = data.shape[axis]/2.0

    sym_position = [-1]*int((maxdist - start_dist)/dist_step)
    index = 0
    for dist in range(start_dist, maxdist, step=dist_step):
        if (mid + dist) % 1 != 0:
            continue

        if are_symmetric(points, data, inversion, mid + dist) != -1:
            sym_position[index] = mid + dist
            if dist == 0:
                continue
        elif are_symmetric(points, data, inversion, mid - dist) != -1:
            sym_position[index] = mid - dist

        # List order does not actually matter.
        index += 1

    return sym_position


def are_symmetric(points, data, inversion, position, percentage_same=0.9):
    """Test if points are semi-symmetric across some plane.

    Parameters
    ----------
    points : numpy.array of ints
        Array of points to test.
    data : DataSet
        Set of all data
    inversion : str in {'x', 'y', 'z'}
        Direction to invert upon
    position : float
        Position of plane to invert on. If inversion == 'x' then the position
        is where the yz plane is located in the x dimension.

    Returns
    -------
    symmetric : bool
    """
    invert_dir = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
    if inversion not in invert_dir.keys():
        raise KeyError("Illegal inversion key.")

    inv = invert_dir[inversion]

    if position % 0.5 != 0:
        raise ValueError("Illegal position. Should be multiple of 0.5.")

    num_symmetric = 0
    for point in points:
        newpoint = [0] * 3
        for i in range(3):
            dist = inv[i] * (point[i] - position)
            newpoint[i] = point[i] - 2 * dist
        point_val = data[point[0], point[1], point[2]]
        newpoint_val = data[newpoint[0], newpoint[1], newpoint[2]]
        if semi_equal(point_val, newpoint_val):
            num_symmetric += 1

    if num_symmetric >= len(points) * percentage_same:
        return position
    else:
        return -1


def semi_equal(a, b, epsilon=None, percentage=0.05):
    """Test if two values are almost equal.

    Parameters
    ----------
    a, b : float
        Values to test
    epsilon : float, optional
        Static value to test if |a - b| < epsilon. If None then use percentages.
    percentage : float, optional
        Test if (1 - percentage)*a < b < (1 + percentage)*a.

    Returns
    -------
    equal : bool
        True or False depending if they are semi-equal or not.
    """
    if a == b:
        return True

    if a is None and b is None:
        return True
    elif a is None or b is None:
        return False

    if a > b:
        large = a
        small = b
    else:
        large = b
        small = a

    if epsilon is not None:
        # Use epsilon only if percentage dif is less.
        if large * percentage < epsilon:
            if abs(large - small) < epsilon:
                return True
            else:
                return False

    if (1 - percentage) * large <= small <= (1 + percentage) * large:
        return True
    else:
        return False


def get_points(n, dims):
    """Get random points for testing.

    Parameters
    ----------
    n : int
        Number of points to find
    dims : list of ints
        Should be a 3x2 list. Each row contains then min value and max value to choose.
        Example:
            dims = [[0, 501], [0, 100], [0, 100]]

    Returns
    -------
    points : numpy.array
        The random points
    """
    # create a 2D array, n points x 3 dims
    points = np.array([np.random.randint(dims[0][0], high=dims[0][1], size=n),
                       np.random.randint(dims[1][0], high=dims[1][1], size=n),
                       np.random.randint(dims[2][0], high=dims[2][1], size=n)])
    return points.T
