import dgl
import networkx as nx
import numpy as np
import torch
from dgl.data import DGLDataset

import torch.nn.functional as F

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

    def add_ground_truth_node_labels(self, input_3d_img, groundtruth_img, neighbors_and_touching_area,
                                     data_has_background=True):
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
                if data_has_background:
                    if groundtruth_labels[col[0]] != 0.:
                        neighbors_and_touching_area[idx, 3] = 1
                else:
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

    def build_networkx_graph(self, neighbors, edges_with_voxel_size, input_3d_img, with_ground_truth=False):
        G = nx.DiGraph()

        # add nodes
        G.add_nodes_from(list(neighbors[:,0]))

        unique_vals_all, unique_val_counts = np.unique(input_3d_img, return_counts=True)

        # np array: voxel_id, size
        voxel_sizes = np.transpose(np.stack((unique_vals_all, unique_val_counts)))

        # add node attributes
        # feat = [size of bigger voxel, size of smaller voxel, touching area]
        # label = 1 if the node belongs to the same cell, 0 otherwise
        attributes_dict = {}
        if with_ground_truth:
            for neighbor_pair in neighbors:

                # size of first voxel of the pair
                # print(voxel_sizes)
                v_1_size = voxel_sizes[np.where(voxel_sizes[:, 0] == neighbor_pair[1])][0][1]
                # print(f"v_1_size: {v_1_size}")
                # print(f"neighbor_pair[1]: {neighbor_pair[1]}")
                v_2_size = voxel_sizes[np.where(voxel_sizes[:, 0] == neighbor_pair[2])][0][1]
                # print(f"v_2_size: {v_2_size}")

                # the bigger voxel should come first
                if v_1_size >= v_2_size:
                    feature = np.array([v_1_size, v_2_size, neighbor_pair[3]])
                else:
                    feature = np.array([v_2_size, v_1_size, neighbor_pair[3]])

                attributes_dict[neighbor_pair[0]] = {"feat": feature,
                                                     "label": neighbor_pair[4]}
        else:
            for neighbor_pair in neighbors:

                # size of first voxel of the pair
                # print(voxel_sizes)
                v_1_size = voxel_sizes[np.where(voxel_sizes[:, 0] == neighbor_pair[1])][0][1]
                # print(f"v_1_size: {v_1_size}")
                # print(f"neighbor_pair[1]: {neighbor_pair[1]}")
                v_2_size = voxel_sizes[np.where(voxel_sizes[:, 0] == neighbor_pair[2])][0][1]
                # print(f"v_2_size: {v_2_size}")

                # the bigger voxel should come first
                if v_1_size >= v_2_size:
                    feature = np.array([v_1_size, v_2_size, neighbor_pair[3]])
                else:
                    feature = np.array([v_2_size, v_1_size, neighbor_pair[3]])

                attributes_dict[neighbor_pair[0]] = {"feat": feature}
        nx.set_node_attributes(G, attributes_dict)

        # add edges of shared super voxels. the weight of each edge is the size of the shared super edges_with_voxel_size
        for edge in edges_with_voxel_size:
            G.add_edge(edge[0], edge[3], weight=edge[6])
            G.add_edge(edge[3], edge[0], weight=edge[6])

        return G

    def get_nx_graph_from_ws_with_gt(self, seg_foreground_super_voxel_by_ws, hand_seg, data_has_background=True):
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
                                                                            neighbors,
                                                                            data_has_background)
        print("adding neighbor ids")
        neighbors = np.c_[np.arange(len(neighbors)), neighbors]
        neighbors_with_gt = np.c_[np.arange(len(neighbors_with_gt)), neighbors_with_gt]
        print("calculate edges")
        edges_with_voxel_size = self.get_edges_with_voxel_size(neighbors, seg_foreground_super_voxel_by_ws)
        print("build networkx graph")
        graph = self.build_networkx_graph(neighbors_with_gt, edges_with_voxel_size, seg_foreground_super_voxel_by_ws,
                                          with_ground_truth=True)
        return graph

from torchvision.transforms import Normalize

def normalize_tensor(tensor):
    min = torch.min(tensor)
    max = torch.max(tensor)
    tensor_normalized = (tensor - min) / (max - min)
    return tensor_normalized

class VoxelGraphDataset(DGLDataset):
    def __init__(self, nx_graph_list, with_ground_truth_labels=True, with_edge_weights=True):
        self.nx_graph_list = nx_graph_list
        self.with_ground_truth_labels = with_ground_truth_labels
        self.with_edge_weights = with_edge_weights
        super().__init__(name='voxel_graph')

    def process(self):
        self.graphs = []
        for nx_graph in self.nx_graph_list:
            n_nodes = nx.number_of_nodes(nx_graph)
            if self.with_ground_truth_labels:
                if self.with_edge_weights:
                    graph = dgl.from_networkx(nx_graph, node_attrs=["feat", "label"], edge_attrs=["weight"])
                else:
                    graph = dgl.from_networkx(nx_graph, node_attrs=["feat", "label"])
            else:
                if self.with_edge_weights:
                    graph = dgl.from_networkx(nx_graph, node_attrs=["feat"], edge_attrs=["weight"])
                else:
                    graph = dgl.from_networkx(nx_graph, node_attrs=["feat"])

            graph = dgl.add_self_loop(graph)


            # unsqueeze features since they are only scalars
            # not needed anymore, since features are now a vector
            # graph.ndata['feat'] = torch.unsqueeze(graph.ndata['feat'], dim=1)
            if self.with_edge_weights:
                graph.edata['weight'] = torch.unsqueeze(graph.edata['weight'], dim=1)
                # graph.edata['weight'] = F.normalize(graph.edata['weight'], p=2.0, dim=0)
                graph.edata['weight'] = normalize_tensor(graph.edata['weight'])
                # print(torch.unique(graph.edata['weight']))
                # graph.edata['weight'] = Normalize(graph.edata['weight'])

            # print(graph.edata['weight'])

            # normalize the features
            # graph.ndata['feat'][0:2] = F.normalize(graph.ndata['feat'][0:2], p=2.0)
            # graph.ndata['feat'][2] = F.normalize(graph.ndata['feat'][2], p=2.0)

            # print(f"feat shape: {graph.ndata['feat'][:, 2].shape}")
            # print(f"feat shape: {graph.ndata['feat'][:, 0:2].shape}")

            # graph.ndata['feat'][: ,0:2] = F.normalize(graph.ndata['feat'][:, 0:2], p=2.0, dim=0)
            # graph.ndata['feat'][:, 2] = F.normalize(graph.ndata['feat'][:, 2], p=2.0, dim=0)

            graph.ndata['feat'][:, 0:2] = normalize_tensor(graph.ndata['feat'][:, 0:2])
            graph.ndata['feat'][:, 2] = normalize_tensor(graph.ndata['feat'][:, 2])

            """
            print("feat...")
            print(f"max 1: {torch.max(graph.ndata['feat'][: ,0])}")
            print(f"max 2: {torch.max(graph.ndata['feat'][:, 1])}")
            print(f"max 3: {torch.max(graph.ndata['feat'][:, 2])}")
            print(f"min 1: {torch.min(graph.ndata['feat'][:, 0])}")
            print(f"min 2: {torch.min(graph.ndata['feat'][:, 1])}")
            print(f"min 3: {torch.min(graph.ndata['feat'][:, 2])}")
            print(graph.ndata['feat'][: ,0:2])
            """
            # graph.ndata['feat'] = Normalize(graph.ndata['feat'])
            # print(graph.number_of_nodes())
            # print(graph.ndata['feat'].shape)
            # print(graph.edata['weight'].shape)

            if self.with_ground_truth_labels:
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


class Cluster_Super_Vox_Graph():
    def __init__(self, model):
        super(Cluster_Super_Vox_Graph, self).__init__
        self.model = model
        self.super_vox_to_nx_graph = SuperVoxToNxGraph()

        self.UN_PROCESSED = 0
        self.PROCESSED = 1
        self.LONELY_POINT = -1
        self.A_LARGE_NUM = 100000000

    def fit(self, input_3d_img, fake_predictions=False, image_has_only_foreground=False):
        self.input_3d_img = input_3d_img
        print("getting neighbor pairs")
        neighbors = self.super_vox_to_nx_graph.get_neighbors_and_touching_area(input_3d_img)
        """
        neighbors:
            -> shape: supervoxel_1, neighbor_1, touching_area(between supervoxel_1 and neighbor_1)
                      supervoxel_1, neighbor_2, touching_area(between supervoxel_1 and neighbor_2)
                      ...
        """
        print("adding neighbor ids")
        neighbors = np.c_[np.arange(len(neighbors)), neighbors]
        """
        neighbors:
            -> shape: neighbors_id_1, supervoxel_1, neighbor_1, touching_area(between supervoxel_1 and neighbor_1)
                      neighbors_id_2, supervoxel_1, neighbor_2, touching_area(between supervoxel_1 and neighbor_2)
                      ...
        """
        # neighbors_with_gt = np.c_[np.arange(len(neighbors)), neighbors]
        print("calculate edges")
        edges_with_voxel_size = self.super_vox_to_nx_graph.get_edges_with_voxel_size(neighbors, input_3d_img)
        """
        edges_with_voxel_size:
            -> shape: pair_id_1, voxel1(pair1), voxel2(pair1), pair_id_2, voxel(pair2), voxel(pair2), size of shared voxel_x <- pairs that share voxel_x
        """
        print("build networkx graph")
        graph = self.super_vox_to_nx_graph.build_networkx_graph(neighbors, edges_with_voxel_size, input_3d_img,
                                          with_ground_truth=False)

        # ugly, but first create a dataset to get normalization
        dataset = VoxelGraphDataset([graph], with_ground_truth_labels=False)

        voxel_graph = dataset[0]

        self.model.eval()

        print("predict...")

        # for model with two outputs!
        # with torch.no_grad():
        #     predictions = self.model(voxel_graph, voxel_graph.ndata['feat']).argmax(1).numpy()

        # for model with one output!
        print("using prediction for single-output model")
        with torch.no_grad():
            model_output = torch.sigmoid(self.model(voxel_graph, voxel_graph.ndata['feat']))
            predictions = (model_output > 0.5).type(torch.FloatTensor)

        print(f"number of negatives: {len(predictions[predictions==0])}")

        if fake_predictions:
            print("FAKE PREDICTIONS")
            predictions = np.ones_like(predictions)

        # add predictions column to edges_with_voxel_size
        neighbors_w_prediction = np.c_[predictions, neighbors]

        """
        neighbors_w_prediction:
            -> shape: prediction, neighbors_id_1, supervoxel_1, neighbor_1, touching_area(between supervoxel_1 and neighbor_1)
                      prediction, neighbors_id_2, supervoxel_1, neighbor_2, touching_area(between supervoxel_1 and neighbor_2)
                      ...
        """


        # with open('neighbors_w_prediction_LRP.npy', 'wb') as f:
        #     np.save(f, neighbors_w_prediction)


        # remove the voxel pairs that are predicted as not sharing the same cell
        prediction_mask = (neighbors_w_prediction[:,0] == 1)
        neighbors_w_prediction = neighbors_w_prediction[prediction_mask, :]


        # TODO predictions of valid neighbors should be np arrays

        unique_vals, unique_val_counts = np.unique(self.input_3d_img, return_counts=True)
        # if not image_has_only_foreground:
        unique_val_counts = unique_val_counts[unique_vals > 0]
        unique_vals = unique_vals[unique_vals > 0]
        sort_locs = np.argsort(unique_val_counts)[::-1]
        self.unique_vals = unique_vals[sort_locs]

        self.val_labels = dict()
        for unique_val in self.unique_vals:
            self.val_labels[unique_val] = self.UN_PROCESSED

        def get_valid_neighbors(value_to_check):
            valid_neighbor_list = []

            # search column 2
            check_mask_1 = (neighbors_w_prediction[:, 2] == value_to_check)
            val_neighbors_1 = neighbors_w_prediction[check_mask_1, 3]

            # TODO way too inefficient
            for i in val_neighbors_1:
                valid_neighbor_list.append(i)

            # search column 3
            check_mask_2 = (neighbors_w_prediction[:, 3] == value_to_check)
            val_neighbors_2 = neighbors_w_prediction[check_mask_2, 2]

            # TODO way too inefficient
            for i in val_neighbors_2:
                valid_neighbor_list.append(i)

            # make sure the values in the list are unique.
            # this should be the case, otherwise the graph would probably contain duplicates!
            valid_neighbor_set = set(valid_neighbor_list)

            # assert len(valid_neighbor_set) == len(valid_neighbor_list)

            return valid_neighbor_list

        # TODO probably useless
        # self.val_outlayer_area = dict()
        # for idx, unique_val in enumerate(self.unique_vals):
        #     # print("get val_outlayer area of all vals: "+str(idx/len(self.unique_vals)))
        #     self.val_outlayer_area[unique_val] = self.A_LARGE_NUM
        """ v1
        for idx, current_val in enumerate(self.unique_vals):
            # print('processing: '+str(idx/len(self.unique_vals))+' pixel val: '+str(current_val))
            if self.val_labels[current_val] != self.UN_PROCESSED:
                continue
            valid_neighbor_vals = get_valid_neighbors(current_val)
            # print(f"number of valid neighbors: {len(valid_neighbor_vals)}")
            if len(valid_neighbor_vals) > 0:
                for val_neighbor in valid_neighbor_vals:
                    # print("merged super voxels!")
                    self.input_3d_img[self.input_3d_img == val_neighbor] = current_val

            self.val_labels[current_val] = self.PROCESSED
        """
        # v2
        for idx, current_val in enumerate(self.unique_vals):
            # print('processing: '+str(idx/len(self.unique_vals))+' pixel val: '+str(current_val))
            if self.val_labels[current_val] != self.UN_PROCESSED:
                continue
            valid_neighbor_vals = get_valid_neighbors(current_val)
            # print(f"number of valid neighbors: {len(valid_neighbor_vals)}")
            if len(valid_neighbor_vals) > 0:
                for val_neighbor in valid_neighbor_vals:
                    # print("merged super voxels!")
                    self.input_3d_img[self.input_3d_img == val_neighbor] = current_val

                    check_mask_1 = (neighbors_w_prediction[:, 2] == val_neighbor)
                    neighbors_w_prediction[check_mask_1, 2] = current_val

                    check_mask_2 = (neighbors_w_prediction[:, 3] == val_neighbor)
                    neighbors_w_prediction[check_mask_2, 3] = current_val

            self.val_labels[current_val] = self.PROCESSED


        print("everything predicted!")

        self.output_3d_img = self.input_3d_img

from func.run_pipeline_super_vox import semantic_segment_crop_and_cat_3_channel_output, img_3d_erosion_or_expansion, \
    generate_super_vox_by_watershed, delete_too_small_cluster, assign_boudary_voxels_to_cells_with_watershed, \
    semantic_segment_crop_and_cat_3_channel_output_edge_gated_model, semantic_segment_crop_and_cat_2_channel_output, \
    semantic_segment_crop_and_cat_2_channel_output_edge_gated_model
def segment_super_vox_3_channel_graph_learning(raw_img, model, graph_model, device,
                                crop_cube_size=128, stride=64,
                                how_close_are_the_super_vox_to_boundary=2,
                                min_touching_area=30, min_touching_percentage=0.51,
                                min_cell_size_threshold=10,
                                transposes=[[0, 1, 2], [2, 0, 1], [0, 2, 1], [1, 0, 2]],
                                reverse_transposes=[[0, 1, 2], [1, 2, 0], [0, 2, 1], [1, 0, 2]]):
    # feed the raw img to the model
    print('Feed raw img to model. Use different transposes')
    raw_img_size = raw_img.shape

    seg_background_comp = np.zeros(raw_img_size)
    seg_boundary_comp = np.zeros(raw_img_size)

    for idx, transpose in enumerate(transposes):
        print(str(idx + 1) + ": Transpose the image to be: " + str(transpose))
        with torch.no_grad():
            seg_img = \
                semantic_segment_crop_and_cat_3_channel_output(raw_img.transpose(transpose), model, device,
                                                               crop_cube_size=crop_cube_size, stride=stride)
        seg_img_background = seg_img['background']
        seg_img_boundary = seg_img['boundary']
        seg_img_foreground = seg_img['foreground']
        torch.cuda.empty_cache()

        # argmax
        print('argmax', end='\r')
        seg = []
        seg.append(seg_img_background)
        seg.append(seg_img_boundary)
        seg.append(seg_img_foreground)
        seg = np.array(seg)
        seg_argmax = np.argmax(seg, axis=0)
        # probability map to 0 1 segment
        seg_background = np.zeros(seg_img_background.shape)
        seg_background[np.where(seg_argmax == 0)] = 1
        seg_foreground = np.zeros(seg_img_foreground.shape)
        seg_foreground[np.where(seg_argmax == 2)] = 1
        seg_boundary = np.zeros(seg_img_boundary.shape)
        seg_boundary[np.where(seg_argmax == 1)] = 1

        seg_background = seg_background.transpose(reverse_transposes[idx])
        seg_foreground = seg_foreground.transpose(reverse_transposes[idx])
        seg_boundary = seg_boundary.transpose(reverse_transposes[idx])

        seg_background_comp += seg_background
        seg_boundary_comp += seg_boundary
    print("Get model semantic seg by combination")
    seg_background_comp = np.array(seg_background_comp > 0, dtype=np.int)
    seg_boundary_comp = np.array(seg_boundary_comp > 0, dtype=np.int)
    seg_foreground_comp = np.array(1 - seg_background_comp - seg_boundary_comp > 0, dtype=np.int)

    # Generate super vox by watershed
    seg_foreground_erosion = 1 - img_3d_erosion_or_expansion(1 - seg_foreground_comp,
                                                             kernel_size=how_close_are_the_super_vox_to_boundary + 1,
                                                             device=device)
    seg_foreground_super_voxel_by_ws = generate_super_vox_by_watershed(seg_foreground_erosion,
                                                                       connectivity=min_touching_area)

    # Super voxel clustering
    cluster_super_vox = Cluster_Super_Vox_Graph(graph_model)
    cluster_super_vox.fit(seg_foreground_super_voxel_by_ws, fake_predictions=False)
    seg_foreground_single_cell_with_boundary = cluster_super_vox.output_3d_img

    # Delete too small cells
    seg_foreground_single_cell_with_boundary = delete_too_small_cluster(seg_foreground_single_cell_with_boundary,
                                                                        threshold=min_cell_size_threshold)

    # Assign boudary voxels to their nearest cells
    seg_final = assign_boudary_voxels_to_cells_with_watershed(seg_foreground_single_cell_with_boundary,
                                                              seg_boundary_comp, seg_background_comp, compactness=1)

    # Reassign unique numbers
    # seg_final=reassign(seg_final)

    return seg_final


def segment_super_vox_2_channel_graph_learning(raw_img, model, graph_model, device,
                                crop_cube_size=128, stride=64,
                                how_close_are_the_super_vox_to_boundary=2,
                                min_touching_area=30, min_touching_percentage=0.51,
                                min_cell_size_threshold=10,
                                transposes=[[0, 1, 2], [2, 0, 1], [0, 2, 1], [1, 0, 2]],
                                reverse_transposes=[[0, 1, 2], [1, 2, 0], [0, 2, 1], [1, 0, 2]]):
    # feed the raw img to the model
    print('Feed raw img to model. Use different transposes')
    raw_img_size = raw_img.shape

    # seg_background_comp = np.zeros(raw_img_size)
    seg_boundary_comp = np.zeros(raw_img_size)

    for idx, transpose in enumerate(transposes):
        print(str(idx + 1) + ": Transpose the image to be: " + str(transpose))
        with torch.no_grad():
            seg_img = \
                semantic_segment_crop_and_cat_2_channel_output(raw_img.transpose(transpose), model, device,
                                                               crop_cube_size=crop_cube_size, stride=stride)
        seg_img_boundary = seg_img['boundary']
        seg_img_foreground = seg_img['foreground']
        torch.cuda.empty_cache()

        # argmax
        print('argmax', end='\r')


        seg_foreground = np.array(seg_img_foreground - seg_img_boundary > 0, dtype=np.int)
        seg_boundary = 1 - seg_foreground

        seg_foreground = seg_foreground.transpose(reverse_transposes[idx])
        seg_boundary = seg_boundary.transpose(reverse_transposes[idx])

        seg_boundary_comp += seg_boundary
    print("Get model semantic seg by combination")
    seg_boundary_comp = np.array(seg_boundary_comp > 0, dtype=float)
    seg_foreground_comp = 1 - seg_boundary_comp

    # Generate super vox by watershed
    seg_foreground_erosion = 1 - img_3d_erosion_or_expansion(1 - seg_foreground_comp,
                                                             kernel_size=how_close_are_the_super_vox_to_boundary + 1,
                                                             device=device)
    seg_foreground_super_voxel_by_ws = generate_super_vox_by_watershed(seg_foreground_erosion,
                                                                       connectivity=min_touching_area)


    # with open('seg_foreground_supervoxel_LRP_graph.npy', 'wb') as f:
    #     np.save(f, seg_foreground_super_voxel_by_ws)

    # Super voxel clustering
    cluster_super_vox = Cluster_Super_Vox_Graph(graph_model)
    cluster_super_vox.fit(seg_foreground_super_voxel_by_ws, fake_predictions=False)
    seg_foreground_single_cell_with_boundary = cluster_super_vox.output_3d_img

    # Delete too small cells
    seg_foreground_single_cell_with_boundary = delete_too_small_cluster(seg_foreground_single_cell_with_boundary,
                                                                        threshold=min_cell_size_threshold)

    # with open('seg_final_LRP_graph_wo_boundary.npy', 'wb') as f:
    #     np.save(f, seg_foreground_single_cell_with_boundary)

    # Assign boudary voxels to their nearest cells
    seg_final = assign_boudary_voxels_to_cells_with_watershed(seg_foreground_single_cell_with_boundary,
                                                              seg_boundary_comp, compactness=1)

    # with open('seg_final_LRP_graph.npy', 'wb') as f:
    #     np.save(f, seg_final)

    # Reassign unique numbers
    # seg_final=reassign(seg_final)

    return seg_final


def segment_super_vox_3_channel_graph_learning_edge_gated_model(raw_img, model, graph_model, device,
                                                                crop_cube_size=128, stride=64,
                                                                how_close_are_the_super_vox_to_boundary=2,
                                                                min_touching_area=30, min_touching_percentage=0.51,
                                                                min_cell_size_threshold=10,
                                                                transposes=[[0, 1, 2], [2, 0, 1], [0, 2, 1], [1, 0, 2]],
                                                                reverse_transposes=[[0, 1, 2], [1, 2, 0], [0, 2, 1], [1, 0, 2]]):
    # feed the raw img to the model
    print('Feed raw img to model. Use different transposes')
    raw_img_size = raw_img.shape

    seg_background_comp = np.zeros(raw_img_size)
    seg_boundary_comp = np.zeros(raw_img_size)

    for idx, transpose in enumerate(transposes):
        print(str(idx + 1) + ": Transpose the image to be: " + str(transpose))
        with torch.no_grad():
            seg_img = \
                semantic_segment_crop_and_cat_3_channel_output_edge_gated_model(raw_img.transpose(transpose), model, device,
                                                                                crop_cube_size=crop_cube_size, stride=stride)
        seg_img_background = seg_img['background']
        seg_img_boundary = seg_img['boundary']
        seg_img_foreground = seg_img['foreground']
        torch.cuda.empty_cache()

        # argmax
        print('argmax', end='\r')
        seg = []
        seg.append(seg_img_background)
        seg.append(seg_img_boundary)
        seg.append(seg_img_foreground)
        seg = np.array(seg)
        seg_argmax = np.argmax(seg, axis=0)
        # probability map to 0 1 segment
        seg_background = np.zeros(seg_img_background.shape)
        seg_background[np.where(seg_argmax == 0)] = 1
        seg_foreground = np.zeros(seg_img_foreground.shape)
        seg_foreground[np.where(seg_argmax == 2)] = 1
        seg_boundary = np.zeros(seg_img_boundary.shape)
        seg_boundary[np.where(seg_argmax == 1)] = 1

        seg_background = seg_background.transpose(reverse_transposes[idx])
        seg_foreground = seg_foreground.transpose(reverse_transposes[idx])
        seg_boundary = seg_boundary.transpose(reverse_transposes[idx])

        seg_background_comp += seg_background
        seg_boundary_comp += seg_boundary
    print("Get model semantic seg by combination")
    seg_background_comp = np.array(seg_background_comp > 0, dtype=np.int)
    seg_boundary_comp = np.array(seg_boundary_comp > 0, dtype=np.int)
    seg_foreground_comp = np.array(1 - seg_background_comp - seg_boundary_comp > 0, dtype=np.int)

    # Generate super vox by watershed
    seg_foreground_erosion = 1 - img_3d_erosion_or_expansion(1 - seg_foreground_comp,
                                                             kernel_size=how_close_are_the_super_vox_to_boundary + 1,
                                                             device=device)
    seg_foreground_super_voxel_by_ws = generate_super_vox_by_watershed(seg_foreground_erosion,
                                                                       connectivity=min_touching_area)

    # Super voxel clustering
    cluster_super_vox = Cluster_Super_Vox_Graph(graph_model)
    cluster_super_vox.fit(seg_foreground_super_voxel_by_ws, fake_predictions=False)
    seg_foreground_single_cell_with_boundary = cluster_super_vox.output_3d_img

    # Delete too small cells
    seg_foreground_single_cell_with_boundary = delete_too_small_cluster(seg_foreground_single_cell_with_boundary,
                                                                        threshold=min_cell_size_threshold)

    # Assign boudary voxels to their nearest cells
    seg_final = assign_boudary_voxels_to_cells_with_watershed(seg_foreground_single_cell_with_boundary,
                                                              seg_boundary_comp, seg_background_comp, compactness=1)

    # Reassign unique numbers
    # seg_final=reassign(seg_final)

    return seg_final


def segment_super_vox_2_channel_graph_learning_edge_gated_model(raw_img, model, graph_model, device,
                                                                crop_cube_size=128, stride=64,
                                                                how_close_are_the_super_vox_to_boundary=2,
                                                                min_touching_area=30, min_touching_percentage=0.51,
                                                                min_cell_size_threshold=10,
                                                                transposes=[[0, 1, 2], [2, 0, 1], [0, 2, 1], [1, 0, 2]],
                                                                reverse_transposes=[[0, 1, 2], [1, 2, 0], [0, 2, 1], [1, 0, 2]]):
    # feed the raw img to the model
    print('Feed raw img to model. Use different transposes')
    raw_img_size = raw_img.shape

    seg_background_comp = np.zeros(raw_img_size)
    seg_boundary_comp = np.zeros(raw_img_size)

    for idx, transpose in enumerate(transposes):
        print(str(idx + 1) + ": Transpose the image to be: " + str(transpose))
        with torch.no_grad():
            seg_img = \
                semantic_segment_crop_and_cat_2_channel_output_edge_gated_model(raw_img.transpose(transpose), model, device,
                                                                                crop_cube_size=crop_cube_size, stride=stride)
        # seg_img_background = seg_img['background']
        seg_img_boundary = seg_img['boundary']
        seg_img_foreground = seg_img['foreground']
        torch.cuda.empty_cache()

        # argmax
        print('argmax', end='\r')
        seg = []
        # seg.append(seg_img_background)
        seg.append(seg_img_boundary)
        seg.append(seg_img_foreground)
        seg = np.array(seg)
        seg_argmax = np.argmax(seg, axis=0)
        # probability map to 0 1 segment
        # seg_background = np.zeros(seg_img_background.shape)
        # seg_background[np.where(seg_argmax == 0)] = 1
        seg_foreground = np.zeros(seg_img_foreground.shape)
        seg_foreground[np.where(seg_argmax == 2)] = 1
        seg_boundary = np.zeros(seg_img_boundary.shape)
        seg_boundary[np.where(seg_argmax == 1)] = 1

        # seg_background = seg_background.transpose(reverse_transposes[idx])
        seg_foreground = seg_foreground.transpose(reverse_transposes[idx])
        seg_boundary = seg_boundary.transpose(reverse_transposes[idx])

        # seg_background_comp += seg_background
        seg_boundary_comp += seg_boundary
    print("Get model semantic seg by combination")
    # seg_background_comp = np.array(seg_background_comp > 0, dtype=np.int)
    seg_boundary_comp = np.array(seg_boundary_comp > 0, dtype=np.int)
    seg_foreground_comp = np.array(1 - seg_background_comp - seg_boundary_comp > 0, dtype=np.int)

    # Generate super vox by watershed
    seg_foreground_erosion = 1 - img_3d_erosion_or_expansion(1 - seg_foreground_comp,
                                                             kernel_size=how_close_are_the_super_vox_to_boundary + 1,
                                                             device=device)
    seg_foreground_super_voxel_by_ws = generate_super_vox_by_watershed(seg_foreground_erosion,
                                                                       connectivity=min_touching_area)

    # Super voxel clustering
    cluster_super_vox = Cluster_Super_Vox_Graph(graph_model)
    cluster_super_vox.fit(seg_foreground_super_voxel_by_ws, fake_predictions=False)
    seg_foreground_single_cell_with_boundary = cluster_super_vox.output_3d_img

    # Delete too small cells
    seg_foreground_single_cell_with_boundary = delete_too_small_cluster(seg_foreground_single_cell_with_boundary,
                                                                        threshold=min_cell_size_threshold)

    # Assign boudary voxels to their nearest cells
    seg_final = assign_boudary_voxels_to_cells_with_watershed(seg_foreground_single_cell_with_boundary,
                                                              seg_boundary_comp, compactness=1)

    # Reassign unique numbers
    # seg_final=reassign(seg_final)

    return seg_final