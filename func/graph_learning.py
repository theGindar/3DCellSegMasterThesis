import dgl
import networkx as nx
import numpy as np
import torch
from dgl.data import DGLDataset

from func.run_pipeline_super_vox import get_outlayer_of_a_3d_shape, get_crop_by_pixel_val


class SuperVoxToNxGraph():
    def __init__(self, boundary_extend=2):
        super(SuperVoxToNxGraph, self).__init__
        self.boundary_extend = boundary_extend

        self.UN_PROCESSED = 0
        self.LONELY_POINT = -1
        self.A_LARGE_NUM = 100000000

    def get_neighbors_and_touching_area(self, input_3d_img, restrict_area_3d=None):
        """
        Parameters
        ----------
        input_3d_img
        restrict_area_3d

        Returns numpy array with each column representing two super voxels touching
                -> shape: supervoxel_1, neighbor_1, touching_area(between supervoxel_1 and neighbor_1)
                          supervoxel_1, neighbor_2, touching_area(between supervoxel_1 and neighbor_2)
                          ...
        -------

        """
        self.input_3d_img = input_3d_img

        if restrict_area_3d is None:
            self.restrict_area_3d = np.array(input_3d_img==0, dtype=np.int8)
        else:
            self.restrict_area_3d = restrict_area_3d

        unique_vals, unique_val_counts = np.unique(self.input_3d_img, return_counts=True)

        unique_val_counts = unique_val_counts[unique_vals>0]
        unique_vals = unique_vals[unique_vals>0]
        sort_locs = np.argsort(unique_val_counts)[::-1]

        self.unique_vals = unique_vals[sort_locs]

        self.val_labels = dict()
        for unique_val in self.unique_vals:
            self.val_labels[unique_val] = self.UN_PROCESSED

        self.val_outlayer_area = dict()
        for idx, unique_val in enumerate(self.unique_vals):
            # print("get val_outlayer area of all vals: "+str(idx/len(self.unique_vals)))
            self.val_outlayer_area[unique_val] = self.A_LARGE_NUM

        """
        neighborhoods:
        np array:
        supervoxel1, neighbor_1, touching_area
        supervoxel1, neighbor_2, touching_area
        ...
        """
        neighborhoods = []
        for idx, current_val in enumerate(self.unique_vals):
            # print('processing: '+str(idx/len(self.unique_vals))+' pixel val: '+str(current_val))
            #if self.val_labels[current_val]!=self.UN_PROCESSED:
            #    continue
            valid_neighbor_vals = self.regionQuery(current_val)
            if len(valid_neighbor_vals) != 0:
                neighborhoods.append(valid_neighbor_vals)
            # if len(valid_neighbor_vals)>0:
            #     # print('Assign label '+str(current_val)+' to current val\'s neighbors: '+str(valid_neighbor_vals))
            #    self.val_labels[current_val] = current_val
            #    self.growCluster(valid_neighbor_vals, current_val)
            # else:
            #    self.val_labels[current_val] = self.LONELY_POINT

        # self.output_3d_img = self.input_3d_img
        neighborhoods = np.vstack(neighborhoods)

        # remove duplicate combinations
        neighborhoods_sorted_pairs = neighborhoods[:,0:2][:, neighborhoods[:,0:2][0, :].argsort()]


        ind = np.argsort(neighborhoods[:,0:2], axis=1)
        neighborhoods_sorted_pairs = np.take_along_axis(neighborhoods[:,0:2], ind, axis=1)

        _, indices_unique = np.unique(neighborhoods_sorted_pairs, axis=0, return_index=True)

        neighborhoods_deduplicated = neighborhoods[indices_unique]

        return neighborhoods_deduplicated

    def get_outlayer_area(self, current_val):
        current_crop_img, current_restrict_area = get_crop_by_pixel_val(self.input_3d_img, current_val,
                                                                        boundary_extend=self.boundary_extend,
                                                                        crop_another_3d_img_by_the_way=self.restrict_area_3d)
        current_crop_img_onehot = np.array(current_crop_img==current_val, dtype=np.int8)
        current_crop_img_onehot_outlayer = get_outlayer_of_a_3d_shape(current_crop_img_onehot)

        assert current_crop_img_onehot_outlayer.shape == current_restrict_area.shape

        current_crop_img_onehot_outlayer[current_restrict_area>0]=0
        current_crop_outlayer_area = np.sum(current_crop_img_onehot_outlayer)

        return current_crop_outlayer_area

    def regionQuery(self, current_val):
        current_crop_img, current_restrict_area = get_crop_by_pixel_val(self.input_3d_img, current_val,
                                                                        boundary_extend=self.boundary_extend,
                                                                        crop_another_3d_img_by_the_way=self.restrict_area_3d)

        current_crop_img_onehot = np.array(current_crop_img==current_val, dtype=np.int8)
        current_crop_img_onehot_outlayer = get_outlayer_of_a_3d_shape(current_crop_img_onehot)

        assert current_crop_img_onehot_outlayer.shape == current_restrict_area.shape

        current_crop_img_onehot_outlayer[current_restrict_area>0]=0
        current_crop_outlayer_area = np.sum(current_crop_img_onehot_outlayer)

        neighbor_vals, neighbor_val_counts = np.unique(current_crop_img[current_crop_img_onehot_outlayer>0], return_counts=True)
        neighbor_val_counts = neighbor_val_counts[neighbor_vals>0]
        neighbor_vals = neighbor_vals[neighbor_vals>0]

        # print("current_crop_outlayer_area: "+str(current_crop_outlayer_area))

        valid_neighbor_vals = self.neighborCheck(current_val, neighbor_vals, neighbor_val_counts, current_crop_outlayer_area)


        # print("valid_neighbor_vals: "+str(valid_neighbor_vals))

        return valid_neighbor_vals

    def neighborCheck(self, current_val, neighbor_vals, neighbor_val_counts, current_crop_outlayer_area):
        neighbor_val_counts = neighbor_val_counts[neighbor_vals>0]
        neighbor_vals = neighbor_vals[neighbor_vals>0]

        valid_neighbor_vals = np.empty((len(neighbor_vals), 3))
        valid_neighbor_vals[:,0] = current_val
        for idx, neighbor_val in enumerate(neighbor_vals):
            # print("touching_area: "+str(neighbor_val_counts[idx]), end="\r")
            # valid_neighbor_vals_dict[neighbor_val] = neighbor_val_counts[idx]
            valid_neighbor_vals[idx, 1] = neighbor_val
            valid_neighbor_vals[idx, 2] = neighbor_val_counts[idx]

        # double_checked_valid_neighbor_vals = []
        # for valid_neighbor_val in valid_neighbor_vals_dict.keys():
        #    if self.val_labels[valid_neighbor_val]==self.UN_PROCESSED or \
        #     self.val_labels[valid_neighbor_val]==self.LONELY_POINT:
        #        double_checked_valid_neighbor_vals.append(valid_neighbor_val)

        return valid_neighbor_vals

    def add_ground_truth_node_labels(self, input_3d_img, groundtruth_img, neighbors_and_touching_area):
        # add ground truth column to neighbors_and_touching_area matrix
        neighbors_and_touching_area = np.c_[(neighbors_and_touching_area,
                                                np.zeros(len(neighbors_and_touching_area)))]
        unique_values = np.unique(np.append(neighbors_and_touching_area[:,0], neighbors_and_touching_area[:,1]))

        # get the ground truth cell label for each super voxel
        groundtruth_labels = {}
        for idx, value in enumerate(unique_values):
            # get values of groundtruth that overlap with each supervoxel
            overlapping_voxels = groundtruth_img[np.where(input_3d_img == value)]
            # get the most occuring groundtruth voxel label for each supervoxel
            gt_label = np.bincount(overlapping_voxels.astype(int)).argmax()

            groundtruth_labels[value] = gt_label

        # set ground 4th column of neighbors_and_touching_area to 1
        # if both supervoxels have the same groundtruth cell label, 0 otherwise
        for idx, col in enumerate(neighbors_and_touching_area):
            if groundtruth_labels[col[0]] == groundtruth_labels[col[1]]:
                neighbors_and_touching_area[idx, 3] = 1

        return neighbors_and_touching_area

    def get_edges_with_voxel_size(self, neighbors, input_3d_img):
        """

        Parameters
        ----------
        neighbors: numpy array with pair_id, voxel1, voxel2, touching_area

        Returns
        -------
        numpy array with pair_id_1, voxel1(pair1), voxel2(pair1), pair_id_2, voxel(pair2), voxel(pair2), size of shared voxel_x <- pairs that share voxel_x

        """

        unique_vals_all, unique_val_counts = np.unique(input_3d_img, return_counts=True)

        # np array: voxel_id, size
        voxel_sizes = np.transpose(np.stack((unique_vals_all, unique_val_counts)))

        # get the unique values that are really present in nodes
        unique_values = np.unique(np.append(neighbors[:,1], neighbors[:,2]))
        all_combinations = []
        for idx, unique_val in enumerate(unique_values):
            # get the entries (combinations of voxels) that include that voxel
            shared_entries = neighbors[np.where(np.logical_or(neighbors[:,1] == unique_val, neighbors[:,2]==unique_val))][:,0:3]
            if len(shared_entries) > 1:
                # combinations = np.stack(np.meshgrid(shared_entries), -1).reshape(-1, 2)
                combinations = []
                for i in range(0, len(shared_entries)-1):
                    for j in range(i+1,len(shared_entries)):
                        combinations.append(np.hstack((shared_entries[i], shared_entries[j])))
                combinations = np.stack(combinations, axis=0)
                # add voxel size
                new_col = np.ones(len(combinations)) * voxel_sizes[np.where(voxel_sizes[:, 0] == unique_val)][0, 1]
                combinations = np.c_[combinations, new_col]
                all_combinations.append(combinations)

        return np.vstack(all_combinations)

    def build_networkx_graph(self, neighbors, edges_with_voxel_size, with_ground_truth=False):
        G = nx.DiGraph()

        # add nodes
        G.add_nodes_from(list(neighbors[:,0]))

        # add node attributes
        # feat = touching area
        # label = 1 if the node belongs to the same cell, 0 otherwise
        attributes_dict = {}
        if with_ground_truth:
            for neighbor_pair in neighbors:
                attributes_dict[neighbor_pair[0]] = {"feat": neighbor_pair[3],
                                                     "label": neighbor_pair[4]}
        else:
            for neighbor_pair in neighbors:
                attributes_dict[neighbor_pair[0]] = {"feat": neighbor_pair[3]}
        nx.set_node_attributes(G, attributes_dict)

        # add edges of shared super voxels. the weight of each edge is the size of the shared super edges_with_voxel_size
        for edge in edges_with_voxel_size:
            G.add_edge(edge[0], edge[3], weight=edge[6])
            G.add_edge(edge[3], edge[0], weight=edge[6])

        return G

    def get_nx_graph_from_ws_with_gt(self, seg_foreground_super_voxel_by_ws, hand_seg):
        """

        Parameters
        ----------
        seg_foreground_super_voxel_by_ws: supervoxels obtained by local watershed
        hand_seg: the ground truth segmentation

        Returns
        -------

        """
        print("getting neighbor pairs")
        neighbors = self.get_neighbors_and_touching_area(seg_foreground_super_voxel_by_ws)
        print("adding ground truth")
        neighbors_with_gt = self.add_ground_truth_node_labels(seg_foreground_super_voxel_by_ws,
                                                                            hand_seg,
                                                                            neighbors)
        print("adding neighbor ids")
        neighbors = np.c_[np.arange(len(neighbors)), neighbors]
        neighbors_with_gt = np.c_[np.arange(len(neighbors_with_gt)), neighbors_with_gt]
        print("calculate edges")
        edges_with_voxel_size = self.get_edges_with_voxel_size(neighbors, seg_foreground_super_voxel_by_ws)
        print("build networkx graph")
        graph = self.build_networkx_graph(neighbors_with_gt, edges_with_voxel_size, with_ground_truth=True)
        return graph


class VoxelGraphDataset(DGLDataset):
    def __init__(self, nx_graph_list):
        self.nx_graph_list = nx_graph_list
        super().__init__(name='voxel_graph')

    def process(self):
        self.graphs = []
        for nx_graph in self.nx_graph_list:
            n_nodes = nx.number_of_nodes(nx_graph)
            graph = dgl.from_networkx(nx_graph, node_attrs=["feat", "label"], edge_attrs=["weight"])
            graph = dgl.add_self_loop(graph)

            # unsqueeze features since they are only scalars
            graph.ndata['feat'] = torch.unsqueeze(graph.ndata['feat'], dim=1)

            graph.ndata['label'] = graph.ndata['label'].type(torch.LongTensor)
            graph.ndata['feat'] = graph.ndata['feat'].float()

            # is it a good idea to use validation set?
            n_train = int(n_nodes * 0.8)
            n_val = int(n_nodes * 0.2)

            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)

            train_mask[:n_train] = True
            val_mask[n_train:n_train + n_val] = True

            graph.ndata['train_mask'] = train_mask
            graph.ndata['val_mask'] = val_mask

            self.graphs.append(graph)

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)
