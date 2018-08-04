import numpy as np
import math
from multiprocessing.pool import Pool


def gdbscan(data, radius, minPts, minVal):
    cores = get_cores(data, radius, minPts, minVal)
    groups = search_all(cores)
    return groups


def breadth_first_search(cores, start):
    # Search in 3x3x3 box around each core
    group = []
    visited = np.full(cores.shape, False)
    if not cores[start[0], start[1], start[2]]:
        return group, visited
    visited[start[0], start[1], start[2]] = True

    queue = [[start[0], start[1], start[2]]]
    while len(queue) != 0:
        pos = queue.pop(0)
        group.append(pos)
        for i in range(-1, 2):
            if pos[0] + i < 0 or pos[0] + i >= cores.shape[0]:
                continue

            for j in range(-1, 2):
                if pos[1] + j < 0 or pos[1] + j >= cores.shape[1]:
                    continue

                for k in range(-1, 2):
                    if pos[2] + k < 0 or pos[2] + k >= cores.shape[2]:
                        continue
                    if i == 0 and j == 0 and k == 0:
                        continue
                    curr_pos = [pos[0] + i, pos[1] + j, pos[2] + k]
                    if cores[curr_pos[0], curr_pos[1], curr_pos[2]] and not visited[curr_pos[0], curr_pos[1], curr_pos[2]]:
                        queue.append(curr_pos)
                        visited[curr_pos[0], curr_pos[1], curr_pos[2]] = True

    return group, visited


def search_all(cores):
    groups = []
    vis = np.full(cores.shape, False)

    for i in range(cores.shape[0]):
        for j in range(cores.shape[1]):
            for k in range(cores.shape[2]):
                if not vis[i, j, k]:
                    ngroup, nvis = breadth_first_search(cores, [i, j, k])
                    groups.append(ngroup)
                    vis += nvis

    return groups


def get_cores(data, radius, minPts, minVal, num_workers=32):
    tiled = make_tiled_matrix(radius, list(data.shape))
    # Make sure that the data and tiled array can not be changed.
    data.flags.writeable = False
    tiled.flags.writeable = False

    proc = Pool(processes=num_workers)
    # Build arguments for each worker
    args = []
    for i in range(2*radius + 1):
        for j in range(2*radius + 1):
            for k in range(2*radius + 1):
                args.append(data, tiled, radius, minPts, minVal, [i, j, k])

    results = proc.starmap_async(offset_cores, args)
    cores = np.zeros(data.shape)
    for res in results:
        cores += res
    return cores


def offset_cores(data, tiled_arr, radius, minPts, minVal, offset):
    dims = list(data.shape)

    shifted_tile = shift_tile(tiled_arr, offset, dims, radius)

    values = np.multiply(data, shifted_tile)
    values = mult_centers(values, radius, offset)

    center = get_centers(radius, dims, offset)

    off_core = np.full(dims, False)
    for c in center:
        area = values[max(c[0] - radius, 0):min(c[0] + radius + 1, dims[0]), max(c[1] - radius, 0):min(c[1] + radius + 1, dims[1]), max(c[2] - radius, 0):min(c[2] + radius + 1, dims[2])]
        if is_core(area, minPts, minVal):
            off_core[c[0], c[1], c[2]] = True

    return off_core


def shift_tile(tiled_arr, offset, dims, radius):
    # point at radius + 1 should be at 0 if offset is 0
    lower_bound = [radius - offset[i] for i in range(len(offset))]
    shifted_tile = tiled_arr[max(lower_bound[0], 0):, max(lower_bound[1], 0):, max(lower_bound[2], 0):]

    # if offset is greater than radius + 1 then padding 0's is needed
    padding = []
    for i in range(len(lower_bound)):
        # last 0 because do not want to pad end
        padding.append([abs(min(lower_bound[i], 0)), 0])

    shifted_tile = np.pad(shifted_tile, padding, 'constant')
    shifted_tile = shifted_tile[:dims[0], :dims[1], :dims[2]]
    return shifted_tile


def is_core(vals, minPts, minVal):
    if len(np.where(vals > minVal)[0]) >= minPts:
        return True
    return False


def num_iter(rad, dim):
    return math.ceil(dim/(2*rad + 1))


def make_tiled_matrix(radius, dims):
    """
    Make a matrix that is tiled with smaller matrices of size 2*radius + 1 and have multiplicative
    scaling factors of 1/dist from center point.
    :param radius: Size of circle to search in for dbscan
    :param dims: Size of array this will be applied to
    :return arr_tiled: Array of tiled factors
    """

    # First make a single tile.
    tile = np.zeros([2*radius + 1]*3)

    for i in range(len(tile)):
        for j in range(len(tile[i])):
            for k in range(len(tile[i][j])):
                dist2 = (i - radius)**2 + (j - radius)**2 + (k - radius)**2
                if dist2 > radius**2:
                    tile[i][j][k] = 0
                elif dist2 == 0:
                    tile[i][j][k] = 1
                else:
                    tile[i][j][k] = 1/(dist2**.5)

    tile_num = [num_iter(radius, dim) for dim in dims]

    tile_num = np.array(tile_num)

    arr_tiled = np.tile(tile, tile_num)
    return arr_tiled


def mult_centers(data, radius, offset):
    # Create matrix to multiply by then do multiplication
    center_index = get_centers(radius, list(data.shape), offset)
    cent_arr = get_cent_arr(data, center_index, radius)

    return np.multiply(data, cent_arr)


def get_cent_arr(data, centers, radius):
    cent_arr = np.zeros(data.shape)

    for c in centers:
        lower_bound = [max(c[i] - radius, 0) for i in range(len(data.shape))]
        upper_bound = [min(c[i] + radius + 1, data.shape[i]) for i in range(len(data.shape))]

        val = data[c[0], c[1], c[2]]
        cent_arr[lower_bound[0]:upper_bound[0], lower_bound[1]:upper_bound[1], lower_bound[2]:upper_bound[2]] = val

    return cent_arr


def get_centers(radius, dims, offset):
    indices = offset.copy()

    center_pos = []
    while indices[0] < dims[0]:
        while indices[1] < dims[1]:
            while indices[2] < dims[2]:
                center_pos.append(indices.copy())
                indices[2] += 2*radius + 1
            indices[2] = offset[2]
            indices[1] += 2*radius + 1
        indices[1] = offset[1]
        indices[0] += 2*radius + 1

    return center_pos
