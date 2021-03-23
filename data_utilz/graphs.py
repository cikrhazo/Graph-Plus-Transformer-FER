import numpy as np


class Graph():
    def __init__(self, max_hop=3, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        # get edges
        self.num_node, self.edge, self.parts = self._get_edge()

        # get adjacency matrix
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        num_node = 68 - 17
        neighbor_link = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
            (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),  # face
            (17, 18), (18, 19), (19, 20), (20, 21),  # left eyebrow
            (22, 23), (23, 24), (24, 25), (25, 26),  # right eyebrow
            (27, 28), (28, 29), (29, 30), (30, 33), (31, 32), (32, 33), (33, 34), (34, 35),  # nose
            (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),  # left eye
            (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),  # right eye
            (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
            (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),  # lip
            (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60),  # teeth
            (27, 38), (27, 39), (27, 40), (27, 21), (27, 22), (27, 42), (27, 43), (27, 47),  # additional
            (17, 36), (21, 39), (22, 42), (26, 45),  # additional
            (48, 60), (54, 64),  # additional
            (36, 31), (41, 31), (40, 31), (39, 31), (48, 31), (49, 31),  # additional
            (42, 35), (47, 35), (46, 35), (45, 35), (54, 35), (53, 35)  # additional
        ]
        neighbor_link = [(i[0]-17, i[1]-17) for i in neighbor_link if i[0]-17 >= 0 and i[1]-17 >= 0]
        parts = [
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),  # face
            np.array([17, 18, 19, 20, 21]),  # left eyebrow
            np.array([22, 23, 24, 25, 26]),  # right eyebrow
            np.array([27, 28, 29, 30, 31, 32, 33, 34, 35]),  # nose
            np.array([36, 37, 38, 39, 40, 41]),  # left eye
            np.array([42, 43, 44, 45, 46, 47]),  # right eye
            np.array([48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]),  # lip
            np.array([60, 61, 62, 63, 64, 65, 66, 67]),  # teeth
        ]
        parts = [i-17 for i in parts if (i-17).min() >= 0]
        self_link = [(i, i) for i in range(num_node)]
        edge = self_link + neighbor_link
        return num_node, edge, parts

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):
        hop_dis = self._get_hop_distance()
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
        return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD
