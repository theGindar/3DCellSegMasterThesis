import time
from functools import partial

import numpy as np
from elf.segmentation import GaspFromAffinities
from elf.segmentation.watershed import apply_size_filter

from plantseg.pipeline import gui_logger
from plantseg.pipeline.steps import AbstractSegmentationStep
from plantseg.segmentation.dtws import compute_distance_transfrom_watershed
from plantseg.segmentation.utils import shift_affinities


class WSSegmentationFeeder:
    def __init__(self, segmentation):
        self.segmentation = segmentation

    def __call__(self, *args, **kwargs):
        return self.segmentation


class GaspFromPmaps(AbstractSegmentationStep):
    def __init__(self,
                 predictions_paths,
                 save_directory="GASP",
                 gasp_linkage_criteria='average',
                 beta=0.5,
                 run_ws=True,
                 ws_2D=True,
                 ws_threshold=0.4,
                 ws_minsize=50,
                 ws_sigma=0.3,
                 ws_w_sigma=0,
                 post_minsize=100,
                 n_threads=6,
                 state=True,
                 **kwargs):

        super().__init__(input_paths=predictions_paths,
                         save_directory=save_directory,
                         file_suffix='_gasp_' + gasp_linkage_criteria,
                         state=state)

        assert gasp_linkage_criteria in ['average',
                                         'mutex_watershed'], f"Unsupported linkage criteria '{gasp_linkage_criteria}'"

        # GASP parameters
        self.gasp_linkage_criteria = gasp_linkage_criteria
        self.beta = beta

        # Watershed parameters
        self.run_ws = run_ws
        self.ws_2d = ws_2D
        self.ws_threshold = ws_threshold
        self.ws_minsize = ws_minsize
        self.ws_sigma = ws_sigma

        # Post processing size threshold
        self.post_minsize = post_minsize

        self.n_threads = n_threads

        self.dt_watershed = partial(compute_distance_transfrom_watershed,
                                    threshold=ws_threshold, sigma_seeds=ws_sigma,
                                    stacked=ws_2D, sigma_weights=ws_w_sigma,
                                    min_size=ws_minsize, n_threads=n_threads)

    def process(self, pmaps):
        # start real world clock timer
        runtime = time.time()

        if self.run_ws:
            # In this case the agglomeration is initialized with superpixels:
            # use additional option 'intersect_with_boundary_pixels' to break the SP along the boundaries
            # (see CREMI-experiments script for an example)
            ws = self.dt_watershed(pmaps)

            def superpixel_gen(*args, **kwargs):
                return ws

        else:
            superpixel_gen = None

        gui_logger.info('Clustering with GASP...')
        # Run GASP
        run_GASP_kwargs = {'linkage_criteria': self.gasp_linkage_criteria,
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
                                           n_threads=self.n_threads,
                                           beta_bias=self.beta)
        # running gasp
        segmentation, _ = gasp_instance(affinities)

        # init and run size threshold
        if self.post_minsize > self.ws_minsize:
            segmentation, _ = apply_size_filter(segmentation.astype('uint32'), pmaps, self.post_minsize)

        # stop real world clock timer
        runtime = time.time() - runtime
        gui_logger.info(f"Clustering took {runtime:.2f} s")

        return segmentation
