import numpy as np
import math

def make_tiled_matrix(radius, dims):
    """
    Make a matrix that is tiled with smaller matrices of size 2*radius + 1 and have multiplicative
    scaling factors of 1/dist from center point.
    :param radius: Size of circle to search in for dbscan
    :param dims: Size of array this will be applied to
    :return arr_tiled: Array of tiled factors
    """

    # First make a single tile.
    tile = np.ndarray([2*radius + 1]*3)

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

    tile_num = []
    num_f = lambda d, r: math.ceil(d/(2*r + 1))
    for dim in dims:
        tile_num.append(num_f(dim, radius))

    tile_num = np.array(tile_num)

    arr_tiled = np.tile(tile, tile_num)
    print(arr_tiled)


def mult_centers(data, radius, offset):
    iterations =


if __name__ == '__main__':
    make_tiled_matrix(1, [4]*3)
