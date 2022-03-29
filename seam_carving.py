from typing import Dict, Any, Tuple

import numpy as np

import utils

NDArray = Any


def delete_seam(image: NDArray, idxs: NDArray, seam: NDArray) -> NDArray:
    h, w = image.shape[0], image.shape[1]
    mask = np.ones((h, w), dtype=bool)
    mask[range(h), seam] = False
    return image[mask].reshape(h, w - 1), idxs[mask].reshape(h, w - 1)


def delete_seams(image: NDArray, seam: NDArray, k: int) -> NDArray:
    h, w = image.shape[0], image.shape[1]
    mask = np.ones((h, w), dtype=bool)
    rows = np.arange(h).reshape(h, 1)
    mask[rows, seam] = False
    return image[mask].reshape(h, w - k)


def find_best_seam(keys: NDArray, last_idx: int, idxs: NDArray, seam: NDArray) -> NDArray:
    h = keys.shape[0]
    next = last_idx
    for i in range(h, -1, -1):
        seam[i] = idxs[i, next]
        next = keys[i, next]

def color_seams(image: NDArray, seams: NDArray, col: str) -> NDArray:
    h = image.shape[0]
    rows = np.arange(h).reshape(h, 1)
    color = np.array([255, 0, 0]) if col == 'red' else np.array([0, 0, 0])
    image[rows, seams] = color


def find_vrt_seams(image: NDArray, k: int) -> NDArray:
    """
    :param image:
    :param k:
    :return:
    """
    # pad the image 1 column each side
    # todo

    seams = np.zeros((image.shape[0], k))
    c = np.zeros((3, image.shape[1]))
    m = np.zeros_like(image)
    keys = np.zeros_like(image)
    energy = utils.get_gradients(image)
    for i in range(k):
        for row in range(1, m.shape[0]):
            # calculate the cost of the last row
            c[:, :] = np.absolute(image[row, :-2] - image[row, 2:])
            c[0, :] += np.absolute(image[row - 1, 1:-1] - image[row, :-2]) + m[row - 1, :-2]
            c[1, :] += m[row - 1, 1:-1]
            c[2, :] += np.absolute(image[row - 1, 1:-1] - image[row, 2:]) + m[row - 1, 2:]
            # add a pixel energy to the cost
            m[row, :] = energy[row, :] + np.min(c, axis=0)
            # save the indexes for later
            keys[row, :] = np.argmin(c, axis=0)  # use unused rows of m


def resize(image: NDArray, out_height: int, out_width: int, forward_implementation: bool) -> Dict[str, NDArray]:
    """

    :param image: ِnp.array which represents an image.
    :param out_height: the resized image height
    :param out_width: the resized image width
    :param forward_implementation: a boolean flag that indicates whether forward or basic implementation is used.
                                    if forward_implementation is true then the forward-looking energy is used otherwise
                                    the basic implementation is used.
    :return: A dictionary with three elements, {'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3},
            where img1 is the resized image and img2/img3 are the visualization images
            (where the chosen seams are colored red and black for vertical and horizontal seams, respectively).
    """
    o_height, o_width, _ = image.shape
    gray = utils.to_grayscale(image)
    idx_mat = np.arange(o_height * o_width).reshape(o_height, o_width)

    def o_idx(idx: int) -> Tuple[int, int]:
        return idx // o_width, idx % o_width

    vrt_k, hrz_k = abs(out_width - o_width), abs(out_height - o_height)
    vrt_seams = []
    hrz_seams = []

    # TODO: return { 'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3}
