from typing import Dict, Any, Tuple

import numpy as np

import utils

NDArray = Any


def delete_seam(image: NDArray, idxs: NDArray, seam: NDArray) -> NDArray:
    h, w = image.shape
    mask = np.ones((h, w), dtype=bool)
    mask[range(h), seam] = False
    return image[mask].reshape(h, w - 1), idxs[mask].reshape(h, w - 1)


def delete_seams(image: NDArray, seams: NDArray, k: int) -> NDArray:
    h, w = image.shape
    mask = np.ones((h, w), dtype=bool)
    rows = np.arange(h).reshape(h, 1)
    mask[rows, seams] = False
    return image[mask].reshape(h, w - k)


def dup_seams(im: NDArray, seams: NDArray, k: int) -> NDArray:
    h, w = im.shape[0], im.shape[1] + k
    new_img = np.zeros((h, w, 3))
    for i in range(3):
        image = im[:, :, i]
        flat = image.flatten()
        doubler = np.ones_like(flat, dtype=int)
        doubler[seams.flatten()] = 2
        new = np.repeat(flat, doubler)
        h, w = image.shape[0], image.shape[1] + k
        new_img[:, :, i] = new.reshape(h, w)

    return new_img


def find_best_seam(keys: NDArray, last_idx, idxs: NDArray, seam: NDArray) -> NDArray:
    h = keys.shape[0]
    next = last_idx
    relative_seam = np.zeros(h, dtype=int)
    for i in range(h - 1, -1, -1):
        seam[i] = idxs[i, next]
        relative_seam[i] = next
        next = keys[i, next]
    return relative_seam


def color_seams(image: NDArray, seams: NDArray, col: str) -> NDArray:
    image = np.copy(image)
    h = image.shape[0]
    rows = np.arange(h).reshape(h, 1)
    color = np.array([255, 0, 0]) if col == 'red' else np.array([0, 0, 0])
    image[np.unravel_index(seams, image.shape[:-1])] = color
    return image


def find_vrt_seams(image: NDArray, k: int, idx_mat: NDArray) -> NDArray:
    """
    :param image:
    :param k:
    :return:
    """

    seams = np.zeros((image.shape[0], k), dtype=int)
    keys = np.zeros_like(image, dtype=int)
    idx = np.arange(keys.shape[1], dtype=int)
    c = np.zeros((3, image.shape[1]))

    # pad the image 1 column each side
    pad = np.zeros((image.shape[0], 1)) + np.nan
    image = np.hstack((pad, image, pad))
    m = np.zeros(image.shape[1])
    m[[0, -1]] = np.nan
    for i in range(k):
        # first row is only the energy
        energy = utils.get_gradients(image[:, 1:-1])
        m[1:-1] = energy[0, :]
        for row in range(1, keys.shape[0]):
            # calculate the cost of the last row for each option
            c[:, :] = np.absolute(image[row, :-2] - image[row, 2:])
            np.nan_to_num(c, nan=255, copy=False)
            c[0, :] += np.absolute(image[row - 1, 1:-1] - image[row, :-2]) + m[:-2]
            c[1, :] += m[1:-1]
            c[2, :] += np.absolute(image[row - 1, 1:-1] - image[row, 2:]) + m[2:]
            # add a pixel energy to the cost
            # print(row)
            m[1:-1] = energy[row, :] + np.nanmin(c, axis=0)

            # save the indexes for forward
            keys[row, :] = \
                idx + np.nanargmin(c, axis=0) - 1
        best_idx = np.nanargmin(m)
        # find the best seam save it to seams
        corr_seam = find_best_seam(keys, best_idx, idx_mat, seams[:, i])
        # delete the seam
        image, idx_mat = delete_seam(image[:, 1:-1], idx_mat, corr_seam)
        # re-pad the image
        image = np.hstack((pad, image, pad))
        c = c[:, 1:]
        m = m[1:]
        keys = keys[:, :-1]
        idx = idx[:-1]

    return seams


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

    # img1, img2, img3 = None, None, None

    def o_idx(idx: int) -> Tuple[int, int]:
        return idx // o_width, idx % o_width

    o_height, o_width, _ = image.shape
    gray = utils.to_grayscale(image)
    idx_mat = np.arange(o_height * o_width).reshape(o_height, o_width)
    k = out_width - o_width
    if forward_implementation:
        seams = find_vrt_seams(gray, abs(k), idx_mat)
    else:
        # todo implement the find_vrt_seams without forward
        seams = find_vrt_seams(gray, abs(k), idx_mat)
    vertical_seams = color_seams(image, seams, "red")

    if k > 0:
        image = dup_seams(image, seams, k)
    else:
        image = delete_seams(image, seams, -k)

    image = np.rot90(image)
    idx_mat = np.arange(out_width * o_height).reshape(out_width, o_height)
    u = out_height - o_height
    gray = utils.to_grayscale(image)
    if forward_implementation:
        seams = find_vrt_seams(gray, abs(u), idx_mat)
    else:
        # todo implement the find_vrt_seams without forward
        seams = find_vrt_seams(gray, abs(u), idx_mat)
    horizontal_seams = color_seams(image, seams, "black")
    horizontal_seams = np.rot90(horizontal_seams,k=3)
    if u > 0:
        image = dup_seams(image, seams, u)
    else:
        image = delete_seams(image, seams, -u)
    image = np.rot90(image, k=3)
    print("done")
    return {'resized': image, 'vertical_seams': vertical_seams, 'horizontal_seams': horizontal_seams}


if __name__ == '__main__':
    a = np.arange(20).reshape(4, 5)
    b = np.array([[3, 1], [9, 6], [13, 10], [19, 15]])
    dup_seams(a, b, 2)
