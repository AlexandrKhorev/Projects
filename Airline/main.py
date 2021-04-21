import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from scipy.cluster import hierarchy

AV = 25


class AirlineNetwork:

    def __init__(self, random):
        self.air_matrix = np.genfromtxt("air500matrix.csv", delimiter=",")
        self.Graph = nx.Graph(self.air_matrix)
        self.nodes = self.Graph.number_of_nodes()
        self.links = self.Graph.number_of_edges()
        self.k_average = self.links / self.nodes
        self.dendrogram = None
        if random:
            self.Graph = nx.gnm_random_graph(self.nodes, self.links, None, True)

    def graph(self):
        nx.draw(self.Graph,
                node_size=2.0, width=0.1,
                node_color='red', edge_color='black',
                alpha=0.7)
        plt.show()

    def info1(self):
        print("Число узлов N = " + str(self.nodes))
        print("Число связей L = " + str(int(self.links)))
        print("Средняя степень <k> = " + str(self.k_average))
        print("Ориентированная сеть")
        print("Количество компонент связности = " + str(nx.is_weakly_connected(self.Graph)))
        print("Количество компонент связности = " + str(nx.is_strongly_connected(self.Graph)))
        print("Количество компонент связности = " + str(nx.node_connected_component(self.Graph, 0)))

    def distribution_degrees(self):
        degree_dict = sorted((dict(nx.degree(self.Graph))).values())
        N_k = np.zeros(self.nodes)
        for i in range(self.nodes):
            for j in range(self.nodes):
                if degree_dict[i] == j:
                    N_k[j] += 1
        N_k = (N_k / self.nodes)

        plt.figure(1)
        plt.xlabel("k")
        plt.ylabel("Pk")
        plt.plot([np.average(N_k[i:i + AV]) for i in range(0, self.nodes - AV)], '.')
        plt.grid()

        plt.figure(2)
        plt.xlabel("(log) k")
        plt.ylabel("(log) Pk")
        plt.plot([np.average(N_k[i:i + AV]) for i in range(0, self.nodes - AV)], '.')
        plt.loglog()
        plt.grid()
        plt.show()

    def distribution_cluster_coefficient(self):
        clustering = list(nx.clustering(self.Graph).values())
        degree = list(dict(nx.degree(self.Graph)).values())

        plt.figure(1)
        plt.plot([np.average(clustering[i:i + AV]) for i in range(0, self.nodes - AV)], '.')
        plt.xlabel("i")
        plt.ylabel("C_i")
        # plt.loglog()
        plt.grid()

        plt.figure(2)
        plt.plot([np.average(degree[i:i + AV]) for i in range(0, self.nodes - AV)],
                 [np.average(clustering[i:i + AV]) for i in range(0, self.nodes - AV)], '.')
        plt.xlabel("k_i")
        plt.ylabel("C_i")
        # plt.loglog()
        plt.grid()
        plt.show()

    def distribution_distance(self):
        distance = list(nx.all_pairs_dijkstra_path_length(self.Graph))
        dij = np.zeros(10)
        for i in range(len(distance)):
            d = list(dict(distance[i][1]).values())
            for j in range(len(d)):
                for k in range(1, len(dij)):
                    if d[j] == k:
                        dij[k] += 1
        dij = np.trim_zeros(dij)
        print(np.sum(dij))
        index = np.arange(1, len(dij) + 1, 1)
        plt.bar(index, dij)
        plt.xlabel("d_ij")
        plt.ylabel("Number of node pairs")
        plt.show()

    def info2(self):
        C = nx.average_clustering(self.Graph)
        d_mean = nx.average_shortest_path_length(self.Graph)
        d_max = nx.diameter(self.Graph)

        print("Кластеризация сети C = " + str(C))
        print("Средняя длина пути <d> = " + str(d_mean))
        print("Диаметр сети d_max = " + str(d_max))

    def scale_invariance(self):
        return

    def assortative(self):
        degree = list(dict(nx.degree(self.Graph)).values())
        k_nn = list(dict(nx.average_neighbor_degree(self.Graph)).values())

        degree = [np.average(degree[i:i + AV]) for i in range(0, self.nodes - AV)]
        k_nn = [np.average(k_nn[i:i + AV]) for i in range(0, self.nodes - AV)]

        a, b = np.polyfit(degree, k_nn, 1)
        x = np.array([np.min(degree), np.max(degree)])
        y = np.array(a * x + b)

        plt.figure()
        plt.plot(degree, k_nn, '.')
        plt.plot(x, y)
        plt.xlabel("(log) k")
        plt.ylabel("(log) k_nn")
        # plt.loglog()
        plt.grid()
        plt.show()

    def similarity(self, matrix):

        graph = nx.DiGraph(matrix)
        nodes = graph.number_of_nodes()

        Xij0 = np.zeros(nodes ** 2 - nodes)
        Jij = np.zeros(nodes ** 2 - nodes)
        Hev = 0
        for i in range(nodes):
            for j in range(nodes):
                if i == j:
                    continue
                J = np.intersect1d(list(nx.neighbors(graph, i)),
                                   list(nx.neighbors(graph, j)))
                Jij[(nodes - 1) * i + j] = len(J)
                if matrix[i, j] == 1:
                    Hev = 1
                    Jij[(nodes - 1) * i + j] += 1
                K = np.min([nx.degree(graph, i), nx.degree(graph, j)])
                Xij0[(nodes - 1) * i + j] = Jij[(nodes - 1) * i + j] / (K + 1 - Hev)

        i1, j1 = divmod(np.argmax(Xij0), (self.nodes - 1))
        matrix = np.delete(matrix, [i1, j1], axis=0)
        matrix = np.delete(matrix, [i1, j1], axis=1)
        n1 = np.zeros(len(matrix))
        n2 = np.zeros(len(matrix) + 1)
        l1 = np.intersect1d(list(nx.neighbors(graph, i1)), list(nx.neighbors(graph, j1)))
        for i in l1:
            n1[i] = 1
            n2[i] = 1
        matrix = np.column_stack((matrix, n1))
        matrix = np.row_stack((matrix, n2))

        inf = np.array([i1, j1, Xij0[i1, j1], ])

        self.dendrogram = np.row_stack((self.dendrogram, inf))
        return matrix

    def matrix_dendrogram( self):
        matrix = self.air_matrix
        while len(matrix) > 2:
            matrix = self.similarity(matrix)

        # z = hierarchy.linkage(self.air_matrix, method='centroid')
        # print(z)
        # dn = hierarchy.dendrogram(z, 6, truncate_mode='level')
        # dn = hierarchy.dendrogram(Xij0)
        # print(dn)
        # plt.show()


air1 = AirlineNetwork(False)
air1.graph()
# air1.info1()
air1.distribution_degrees()
air1.distribution_cluster_coefficient()
air1.distribution_distance()
# air1.info2()
# air1.assortative()
air1.dendrogram()

air2 = AirlineNetwork(True)
# air2.graph()
# air2.info1()
# air2.distribution_degrees()
# air2.distribution_cluster_coefficient()
# air2.distribution_distance()
# air2.info2()
# air2.assortative()
# air2.dendrogram()

