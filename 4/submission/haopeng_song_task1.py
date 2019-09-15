from pyspark import SparkContext, SparkConf
import sys, time, queue, operator, copy
import matplotlib.pyplot as plt


class GraphNode:

    def __init__(self, ids, parents, children, shortest_path, weight, level):
        self.ids = ids
        self.parents = parents
        self.children = children
        self.shortest_path = shortest_path
        self.weight = weight
        self.level = level

    # def __hash__(self):
    #     return hash(self.ids)
    #
    # def __eq__(self, other):
    #     return self.__class__ == other.__class__ and self.ids == other.ids


class BFSTree:

    def __int__(self, root, tree):
        self.root = root
        self.tree = tree


def construct_graph(x):

    neighbor = set()

    same_review_num = 0

    for k,v in user_business_map.items():

        if k != x[0]:

            same_review_num = len(v.intersection(x[1]))

            if same_review_num >= filter_threshold:

                neighbor.add(k)

    return (x[0], neighbor)


def bfs(root, graph):

    visited = {}

    root_node = GraphNode(root, set(), set(), 0, 0, 0)

    visited[root] = root_node

    level = 1

    q = queue.Queue()

    q.put(root)

    while not q.empty():

        q_size = q.qsize()

        for x in range(q_size):

            curr_node = q.get()

            for nei in graph[curr_node]:

                if nei not in visited:

                    node = GraphNode(nei, set(), set(), 0, 0, level)

                    node.parents.add(visited[curr_node])

                    visited[curr_node].children.add(node)

                    q.put(nei)

                    visited[nei] = node

                else:

                    if visited[curr_node].level != visited[nei].level and visited[nei] not in visited[curr_node].parents:

                        visited[curr_node].children.add(visited[nei])

                        visited[nei].parents.add(visited[curr_node])
        level += 1

    return visited


def assign_shortest_path(tree, root):

    # assign root 1

    tree[root].shortest_path = 1

    q = queue.Queue()

    q.put(tree[root])

    visited = set()

    visited.add(root)

    level_map = {}

    while not q.empty():

        curr_node = q.get()

        if curr_node.level not in level_map:

            level_map[curr_node.level] = set()

            level_map[curr_node.level].add(curr_node)

        else:

            level_map[curr_node.level].add(curr_node)

        # root node

        if len(curr_node.parents) == 0:

            curr_node.shortest_path = 1

        else:

            sp = 0

            for p in curr_node.parents:

                sp += (p.shortest_path)

            curr_node.shortest_path = sp

        for ch in curr_node.children:

            if ch.ids not in visited:

                visited.add(ch.ids)

                q.put(ch)

    return level_map


def get_edge(c, p):

    edge = [c.ids, p.ids]

    edge.sort()

    return tuple(edge)


def assign_outgoing_edge(node, edge_map):

    # credit for outgoing edges

    if len(node.parents) > 0:

        totoal_sp = 0.0

        for p in node.parents:

            totoal_sp += (p.shortest_path)

        for p in node.parents:

            edge = get_edge(node, p)

            edge_map[edge] = ( p.shortest_path / float(totoal_sp) ) * node.weight


def assign_credit(level_tree, edge_map):

    for pair in level_tree:

        nodes = pair[1]

        for node in nodes:

            # leaf

            if len(node.children) == 0:

                node.weight = 1.0

                # credit for outgoing edges

                assign_outgoing_edge(node, edge_map)

            else:

                edge_sum = 0.0

                for ch in node.children:

                    edge = get_edge(node, ch)

                    edge_sum += edge_map[edge]

                node.weight = 1 + edge_sum

                # credit for outgoing edges

                assign_outgoing_edge(node, edge_map)


def output_betweenness(of, btn):

    for ele in btn:

        line = str(ele[0]) + ', ' + str(ele[1])

        of.write(line + '\n')

    of.close()


def edge_remove(g, edge_r):

    g[edge_r[0][0]].discard(edge_r[0][1])

    g[edge_r[0][1]].discard(edge_r[0][0])


def calculate_q(graph, m, a, sss):

    Q = []

    for ss in sss:

        sums = 0.0

        for i in ss:

            for j in ss:

                A = 0

                if j in a[i]:

                    A = 1

                else:

                    A = 0

                expected_edges = (len(a[i]) * len(a[j]) / (2.0 * m))

                sums += (A - expected_edges) / (2.0 * m)

        Q.append(sums)

    return sum(Q)


def calculate_betweenness(graph):

    # 1. for each node, build bfs tree

    tree_map = {}

    for k, v in graph.items():

        curr_tree = bfs(k, graph)

        tree_map[k] = curr_tree

    # 2. label each node by the number of shortest paths that reach it from the root. start by labeling the root 1.

    level_map = {}

    for k, v in tree_map.items():

        gn_tree = tree_map[k]

        level_map[k] = assign_shortest_path(gn_tree, k)

    # 3. calculate for each edge e the sum over all nodes Y of the
    # fraction of shortest paths from the root X to Y that go through e.

    credit_map = {}

    for k, v in level_map.items():

        edge_map = {}

        level_ordered = sorted(level_map[k].items(), key = operator.itemgetter(0), reverse = True)

        assign_credit(level_ordered, edge_map)

        credit_map[k] = edge_map

    betweenness_map = {}

    for k, v in credit_map.items():

        for e, w in v.items():

            if e not in betweenness_map:

                betweenness_map[e] = []

                betweenness_map[e].append(w)

            else:

                betweenness_map[e].append(w)

    for k, v in betweenness_map.items():

        betweenness_map[k] = (sum(v)) / 2.0

    btn_sorted = [(k, v) for k , v in betweenness_map.items()]

    btn_sorted.sort()

    btn_sorted.sort(key = lambda x : x[1], reverse=True)

    return btn_sorted


def community_detection(node, graph, curr_node):

    visited = set()

    visited.add(node)

    q = queue.Queue()

    q.put(node)

    while not q.empty():

        queue_size = q.qsize()

        for x in range(queue_size):

            curr = q.get()

            for nei in graph[curr]:

                if nei not in visited:

                    visited.add(nei)

                    q.put(nei)

                    curr_node.add(nei)

    return visited


def find_communities(g):

    curr_graph = copy.deepcopy(g)

    # num_nodes = len(curr_graph)

    curr_node = set()

    communities = []

    for k, v in curr_graph.items():

        if k not in curr_node:

            curr_node.add(k)

            comm = community_detection(k, curr_graph, curr_node)

            communities.append(list(comm))

    return communities


def comm_detection(time_iteration, m, edge_highest_btn, graph, A, qs):

    comms = None

    m_num_edges = m

    for r in range(time_iteration):

        q = 0

        if r == 0:

            edge_remove(graph, edge_highest_btn)

            # find communities, with filter

            # compute q for each community

            # re compute btn

            comms = find_communities(graph)

            q = calculate_q(graph, m_num_edges, A, comms)

            qs.append(q)

            edge_highest_btn = calculate_betweenness(graph)[0]

        else:

            edge_remove(graph, edge_highest_btn)

            comms = find_communities(graph)

            q = calculate_q(graph, m_num_edges, A, comms)

            qs.append(q)

            if r != m_num_edges - 1:

                edge_highest_btn = calculate_betweenness(graph)[0]

    return comms


def output_communities(of, comms):

    for coms in comms:

        of.write(str(coms)[1:-1] + '\n')

    of.close()


if __name__ == '__main__':

    if len(sys.argv) < 5:

        print('Error: not enough arguments')

        exit(-1)

    start = time.time()

    sc = SparkContext().getOrCreate()

    filter_threshold = int(sys.argv[1])

    input_path = sys.argv[2]

    # '/Users/haopengsong/Downloads/sample_data.csv'

    output_path_btn = sys.argv[3]

    # '/Users/haopengsong/PycharmProjects/inf553hw4/betweenness.txt'

    output_path_comms = sys.argv[4]

    #'/Users/haopengsong/PycharmProjects/inf553hw4/comms.txt'

    data = sc.textFile(input_path)

    data_firstLine = data.take(1)

    data = data.filter(lambda x : x != data_firstLine[0])

    user_business = data.map(lambda x : x.split(",")) \
                        .map(lambda x : (x[0], [x[1]])) \
                        .reduceByKey(lambda a, b : a + b)

    user_business = user_business.map(lambda x : (x[0], set(x[1])))

    user_business_map = user_business.collectAsMap()

    s1 = time.time()

    graph = user_business.map(construct_graph) \
                        .filter(lambda x : len(x[1]) != 0) \
                        .collectAsMap()

    print('\ntime: ' + str(time.time() - s1))



    # output betweenness

    output_file = open(output_path_btn, 'w')

    btn_sorted = calculate_betweenness(graph)

    output_betweenness(output_file, btn_sorted)

    # community detection
    # 1. compute Q

    # A = graph, adjacent matrix of the original graph

    # new graph after remove one edge

    m_num_edges = len(btn_sorted)

    A = copy.deepcopy(graph)

    qs = []

    edge_highest_btn = btn_sorted[0]

    comm_detection(m_num_edges, m_num_edges, edge_highest_btn, graph, A, qs)

    #find largest q

    q_index = qs.index(max(qs))

    qs = []

    graph = copy.deepcopy(A)

    output_comms = comm_detection(q_index + 1, m_num_edges, btn_sorted[0], graph, A, qs)

    #print(q_index)

    for com in output_comms:

        com.sort()


    output_comms.sort()

    output_comms.sort(key = lambda x : len(x))

    output_file = open(output_path_comms, 'w')

    output_communities(output_file, output_comms)

    print('\ntime: ' + str(time.time() - s1))
