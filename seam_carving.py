from typing import Dict, Any, Tuple

import numpy as np

import utils

NDArray = Any


def find_vrt_seams(image: NDArray, k: int) -> NDArray:
    pd = np.zeros(image.shape[0], k)



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
