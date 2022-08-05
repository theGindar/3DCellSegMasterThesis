import time
from functools import partial

import nifty
import nifty.graph.rag as nrag
import numpy as np
from elf.segmentation.features import compute_rag
from elf.segmentation.multicut import multicut_kernighan_lin, transform_probabilities_to_costs
from elf.segmentation.watershed import distance_transform_watershed, apply_size_filter

from plantseg.pipeline import gui_logger
from plantseg.pipeline.steps import AbstractSegmentationStep
from plantseg.segmentation.dtws import compute_distance_transfrom_watershed


class MulticutFromPmaps(AbstractSegmentationStep):
    def __init__(self,
                 predictions_paths,
                 save_directory="MultiCut",
                 beta=0.5,
                 run_ws=True,
                 ws_2D=True,
                 ws_threshold=0.4,
                 ws_minsize=50,
                 ws_sigma=2.0,
                 ws_w_sigma=0,
                 post_minsize=50,
                 n_threads=6,
                 state=True,
                 **kwargs):

        super().__init__(input_paths=predictions_paths,
                         save_directory=save_directory,
                         file_suffix='_multicut',
                         state=state)

        self.beta = beta

        # Watershed parameters
        self.run_ws = run_ws
        self.ws_2D = ws_2D
        self.ws_threshold = ws_threshold
        self.ws_minsize = ws_minsize
        self.ws_sigma = ws_sigma
        self.ws_w_sigma = ws_w_sigma

        # Post processing size threshold
        self.post_minsize = post_minsize

        # Multithread
        self.n_threads = n_threads

        self.dt_watershed = partial(compute_distance_transfrom_watershed,
                                    threshold=ws_threshold, sigma_seeds=ws_sigma,
                                    stacked=ws_2D, sigma_weights=ws_w_sigma,
                                    min_size=ws_minsize, n_threads=n_threads)

    def process(self, pmaps):
        runtime = time.time()
        segmentation = self.segment_volume(pmaps)

        if self.post_minsize > self.ws_minsize:
            segmentation, _ = apply_size_filter(segmentation, pmaps, self.post_minsize)

        # stop real world clock timer
        runtime = time.time() - runtime
        gui_logger.info(f"Clustering took {runtime:.2f} s")

        return segmentation

    def segment_volume(self, pmaps):
        ws = self.dt_watershed(pmaps)

        gui_logger.info('Clustering with MultiCut...')
        rag = compute_rag(ws)
        # Computing edge features
        features = nrag.accumulateEdgeMeanAndLength(rag, pmaps, numberOfThreads=1)  # DO NOT CHANGE numberOfThreads
        probs = features[:, 0]  # mean edge prob
        edge_sizes = features[:, 1]
        # Prob -> edge costs
        costs = transform_probabilities_to_costs(probs, edge_sizes=edge_sizes, beta=self.beta)
        # Creating graph
        graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
        graph.insertEdges(rag.uvIds())
        # Solving Multicut

        node_labels = multicut_kernighan_lin(graph, costs)
        return nifty.tools.take(node_labels, ws)
