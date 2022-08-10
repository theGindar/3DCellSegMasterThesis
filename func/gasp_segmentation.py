import time
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from elf.segmentation import GaspFromAffinities
from elf.segmentation.watershed import apply_size_filter

from plantseg.pipeline.steps import AbstractSegmentationStep
from plantseg.segmentation.dtws import compute_distance_transfrom_watershed
from plantseg.segmentation.utils import shift_affinities


def process_gasp(pmaps):
    # save_directory = "GASP"
    gasp_linkage_criteria = 'average'
    beta = 0.5
    run_ws = True
    ws_2d = False
    ws_threshold = 0.4
    ws_minsize = 100
    ws_sigma = 0.3
    ws_w_sigma = 0
    post_minsize = 100
    n_threads = 6
    state = True

    dt_watershed = partial(compute_distance_transfrom_watershed,
                           threshold=ws_threshold, sigma_seeds=ws_sigma,
                           stacked=ws_2d, sigma_weights=ws_w_sigma,
                           min_size=ws_minsize, n_threads=n_threads)
    # start real world clock timer
    runtime = time.time()

    if run_ws:
        # In this case the agglomeration is initialized with superpixels:
        # use additional option 'intersect_with_boundary_pixels' to break the SP along the boundaries
        # (see CREMI-experiments script for an example)
        ws = dt_watershed(pmaps)

        def superpixel_gen(*args, **kwargs):
            return ws

    else:
        superpixel_gen = None

    print('Clustering with GASP...')
    # Run GASP
    run_GASP_kwargs = {'linkage_criteria': gasp_linkage_criteria,
                       'add_cannot_link_constraints': False,
                       'use_efficient_implementations': False}

    # pmaps are interpreted as affinities
    affinities = np.stack([pmaps, pmaps, pmaps], axis=0)

    offsets = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    # Shift is required to correct aligned affinities
    affinities = shift_affinities(affinities, offsets=offsets)

    # invert affinities
    affinities = 1 - affinities

    # Init and run Gasp
    gasp_instance = GaspFromAffinities(offsets,
                                       superpixel_generator=superpixel_gen,
                                       run_GASP_kwargs=run_GASP_kwargs,
                                       n_threads=n_threads,
                                       beta_bias=beta)
    # running gasp
    segmentation, _ = gasp_instance(affinities)

    # init and run size threshold
    if post_minsize > ws_minsize:
        segmentation, _ = apply_size_filter(segmentation.astype('uint32'), pmaps, post_minsize)

    # stop real world clock timer
    runtime = time.time() - runtime
    print(f"Clustering took {runtime:.2f} s")

    return segmentation