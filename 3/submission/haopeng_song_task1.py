from pyspark import SparkContext, SparkConf
from itertools import combinations
import sys, time


def file_output(fp, data):

    fp.write('business_id_1,business_id_2,similarity' + '\n')

    for pair in data:

        fp.write(pair[0][0] + ',' + pair[0][1] + ',' + pair[1] + '\n')


def prime_generator(n, pl):

    if n >= 2:
        pl.append(2)
    for i in range(3, n + 1):
        is_prime = True
        for x in range(2, int(i ** 0.5) + 1):
            if i % x == 0:
                is_prime = False
                break

        if is_prime:
            pl.append(i)


def update_sig_matrix_hash(a, b, m, char_row, row_num, sig_matrix):
    compute_hash_value = []

    for i in range(sig_matrix_num_rows):
        compute_hash_value.append(((row_num * a[i] + b[i])) % m)

    char_col_ctn = 0
    for col in char_row:
        if col == 1:
            compute_hash_value_ctn = 0
            for sig_row in sig_matrix:
                if sig_row[char_col_ctn] > compute_hash_value[compute_hash_value_ctn]:
                    sig_row[char_col_ctn] = compute_hash_value[compute_hash_value_ctn]
                compute_hash_value_ctn += 1
        char_col_ctn += 1


if __name__ == '__main__':

    config = SparkConf() \
        .set('spark.executor.memory', '4g') \
        .set('spark.driver.memory', '4g')

    sc = SparkContext(conf=config).getOrCreate()

    if len(sys.argv) < 3:

        print('Error: not enough arguments')

        exit(-1)

    start1 = time.time()

    input_path = sys.argv[1]

    output_path = sys.argv[2]

    data = sc.textFile(input_path)

    data = data.map(lambda x: x.split(','))

    first = data.take(1)

    data = data.filter(lambda x: x != first[0])

    users = data.filter(lambda x: x != first[0]) \
        .map(lambda x: x[0])

    business = data.filter(lambda x: x != first[0]) \
        .map(lambda x: x[1])

    user_business = data.map(lambda x: (x[0], [x[1]])) \
        .reduceByKey(lambda a, b: a + b)

    user_business_map = user_business.collectAsMap()

    business_user = data.map(lambda x: (x[1], [x[0]])) \
        .reduceByKey(lambda a, b: a + b) \
        .mapValues(lambda x: set(x)) \
        .mapValues(lambda x: list(x))

    business_user_map = business_user.collectAsMap()

    # user - index
    user_index = {}
    index_user = {}
    count = 0
    for k, v in user_business_map.items():
        user_index[count] = k
        index_user[k] = count
        count += 1

    business = business.distinct().collect()

    # business - index
    business_index = {}
    index_business = {}
    count = 0
    for ele in business:
        business_index[ele] = count
        index_business[count] = ele
        count += 1

    business_user_map_index = {}
    for k, v in business_user_map.items():
        if business_index[k] not in business_user_map_index:
            user_list = []
            for user in v:
                user_list.append(index_user[user])
            business_user_map_index[business_index[k]] = user_list
        else:
            for user in v:
                business_user_map_index[business_index[k]].append(index_user[user])

    num_business = len(business_index)

    num_user = len(user_index)

    characteristic_matrix = [[0] * num_business for i in range(num_user)]

    business_col = {}
    for k, v in user_index.items():
        row_index_user = k
        for business in user_business_map[v]:
            col_index_business = business_index[business]
            characteristic_matrix[row_index_user][col_index_business] = 1


    # pick 100 hash functions
    # y = ((ax + b) % p) % m

    prime_list = []

    prime_generator(10000, prime_list)

    m = num_user

    ab = []
    a = prime_list[0:120]
    b = prime_list[-120:]

    sig_matrix_num_rows = len(b)
    sig_matrix_num_cols = num_business

    #signature matrix
    INT_MAX = 1000000000
    sig_matrix = [[INT_MAX] * sig_matrix_num_cols for i in range(sig_matrix_num_rows)]


    row_ctn = 0
    for row in characteristic_matrix:
        update_sig_matrix_hash(a, b, m, row, row_ctn, sig_matrix)
        row_ctn += 1

    #divide matrix
    start = time.time()
    b = 40
    r = 3
    band = {}
    num_cand = 0
    coli = 0
    cand_pair = []
    cand_pair_set = set()
    for i in range(0, b):
        row_range_counter = i * r
        partition = {}
        band[i] = partition
        for col in range(sig_matrix_num_cols):
            x = str(i)
            for row in range(row_range_counter, row_range_counter + r):
                x += str(sig_matrix[row][col])
            if x not in partition:
                cand_list = []
                cand_list.append(col)
                partition[x] = cand_list
            else:
                partition[x].append(col)
        #print(partition)
        for k, v in partition.items():
            if len(v) >= 100:
                coli += 1
            if len(v) >= 2:
                num_cand += 1
                for pair in combinations(v, 2):
                    if frozenset(pair) not in cand_pair_set:
                        cand_pair_set.add(frozenset(pair))
                        cand_pair.append(pair)

    true_cand = []

    for index in cand_pair:

        s1 = set(business_user_map_index[index[0]])
        s2 = set(business_user_map_index[index[1]])
        inter = len(s1.intersection(s2))
        union = len(s1.union(s2))
        if inter / union >= 0.5:
            true_cand.append((index[0], index[1], inter/union))

    print('Duration: ' + str(time.time() - start1))

    output_pair = []
    for comb in true_cand:
        b1 = index_business[comb[0]]
        b2 = index_business[comb[1]]
        b3 = comb[2]
        business_pair = [b1, b2]
        output_pair.append(( sorted(business_pair) , str(b3)))
    output_pair.sort(key = lambda x : x[0])
    # for p in output_pair:
    #     print(p)

    output_file = open(output_path, 'w')

    file_output(output_file , output_pair)

    output_file.close()

