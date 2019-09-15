from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from itertools import combinations
import sys, time, operator


def addition_entry(x):
    if x[0] not in user_index:
        return (x[0], x[1], x[2])
    elif x[1] not in business_index:
        return (x[0], x[1], x[2])
    return None


def output_testset(x):
    if x[0] in user_index and x[1] in business_index:
        return (int(user_index[x[0]]), int(business_index[x[1]]), float(x[2]))


def map_business_rating(x):
    business_rating = {}
    for tup in x:
        business_rating[tup[0]] = tup[1]
    return business_rating


def rearrange_result(x):
    if x > 5:
        x = 5
    return x


def cos_sim(x, u, y, v, cos):
    inter = x.intersection(y)

    unio = x.union(y)

    result = 0

    if len(inter) > 0:

        numerator = [user_business_rating_map[u][i] * user_business_rating_map[v][i] for i in inter]

        denominator_u = [user_business_rating_map[u][i] ** 2 for i in x]

        denominator_v = [user_business_rating_map[v][i] ** 2 for i in y]

        if denominator_u != 0 and denominator_v != 0:
            result = sum(numerator) / ((sum(denominator_u) ** (1 / 2)) * (sum(denominator_v) ** (1 / 2)))

            cos[frozenset({u, v})] = result

            return result

    cos[frozenset({u, v})] = result

    return result


def train_case3(x):
    N = 15

    user_u = x[0]

    business_u = x[1]

    u_businesses = list(user_business_map[user_u])

    if business_u in u_businesses:
        return ((x[0], x[1]), user_business_rating_map[user_u][business_u])

    business_u_ratings = [user_business_rating_map[i][business_u] for i in business_user_map[business_u]]

    # print(business_u_ratings)

    business_u_ratings_avg = sum(business_u_ratings) / float(len(business_u_ratings))

    # print(business_u_ratings_avg)

    # similarity between user_u and u_businesses

    business_u_users = business_user_map[business_u]

    pearson = {}

    u_business_all = [user_business_rating_map[user_u][i] for i in u_businesses]

    u_business_all_avg = sum(u_business_all) / float(len(u_business_all))

    for business_v in u_businesses[:N]:

        # print('business : ' + str(business_v))

        business_v_users = business_user_map[business_v]

        business_v_ratings = [user_business_rating_map[i][business_v] for i in business_user_map[business_v]]

        # print(business_v_ratings)

        business_v_ratings_avg = sum(business_v_ratings) / float(len(business_v_ratings))

        # print(business_v_ratings_avg)

        inter = business_u_users.intersection(business_v_users)

        if len(inter) > 0:

            dot_product = 0.0

            distance_to_avg_u = 0.0

            distance_to_avg_v = 0.0

            for user in inter:
                coeff_u = float(user_business_rating_map[user][business_u] - business_u_ratings_avg)

                # print(coeff_u)

                coeff_v = float(user_business_rating_map[user][business_v] - business_v_ratings_avg)

                # print(coeff_v)

                dot_product += coeff_u * coeff_v

                distance_to_avg_u += (coeff_u ** 2)

                distance_to_avg_v += (coeff_v ** 2)

            pearson_u_v = 0.0

            if dot_product == 0 or distance_to_avg_u == 0 or distance_to_avg_v == 0:

                pearson_u_v = 0.0

            else:

                pearson_u_v = dot_product / ((distance_to_avg_u ** (1 / 2)) * (distance_to_avg_v ** (1 / 2)))

            # print('pearson: ' + str(pearson_u_v))

            pearson[(business_v, business_u)] = pearson_u_v

    pearson_total = 0.0

    numerator = 0.0

    pearson_sorted = sorted(pearson.items(), key=operator.itemgetter(1), reverse=True)

    # print(pearson_sorted)

    for pair in pearson_sorted[:(N - 1)]:
        b_choosen = pair[0][0]

        b_rating_user_u = user_business_rating_map[user_u][b_choosen]

        numerator += b_rating_user_u * pair[1]

        pearson_total += abs(pair[1])

    rating = 0

    if pearson_total != 0:
        rating = numerator / pearson_total

    # print(rating)

    return ((x[0], x[1]), u_business_all_avg)


def train_case2(x):

    user_train = x[0]

    business_train = x[1]

    # all users who have rated business_train, list
    users_who_rated_business = business_user_map[business_train]

    # if user_train has already rated business_train, return the rating
    if user_train in users_who_rated_business:
        return ((x[0], x[1]), user_business_rating_map[user_train][business_train])

    # otherwise
    # calculate pearson
    # average rating on co-rated business
    # businesses that were rated by user_train
    user_train_rated_businesses = user_business_map[user_train]

    user_train_ratings = [user_business_rating_map[user_train][i] for i in user_train_rated_businesses]

    user_train_ratings_avg = sum(user_train_ratings) / float(len(user_train_ratings))

    pearson = {}

    for u in users_who_rated_business:

        # businesses that rated by u
        u_rated_businesses = user_business_map[u]

        u_ratings = [user_business_rating_map[u][i] for i in u_rated_businesses]

        u_ratings_avg = sum(u_ratings) / float(len(u_ratings))

        # co-rated businesses between u and user_train
        inter = u_rated_businesses.intersection(user_train_rated_businesses)

        if len(inter) > 0:

            dot_product = 0.0

            distance_to_avg_u = 0.0

            distance_to_avg_user_train = 0.0

            for business in inter:
                coeff_u = float(user_business_rating_map[u][business] - u_ratings_avg)

                coeff_user_train = float(user_business_rating_map[user_train][business] - user_train_ratings_avg)

                dot_product += (coeff_u) * (coeff_user_train)

                distance_to_avg_u += ((coeff_u ** 2))

                distance_to_avg_user_train += ((coeff_user_train ** 2))

            pearson_u_user_train = 0.0

            if dot_product == 0 or distance_to_avg_u == 0 or distance_to_avg_user_train == 0:

                pearson_u_user_train = 0.0

            else:

                pearson_u_user_train = dot_product / (
                            (distance_to_avg_u ** (1 / 2)) * (distance_to_avg_user_train ** (1 / 2)))

            pearson[(u, user_train)] = pearson_u_user_train

    # pick top N user
    pearson_total = 0.0

    numerator = 0.0

    N = 13

    pearson_sorted = sorted(pearson.items(), key=operator.itemgetter(1), reverse=True)

    for pair in pearson_sorted[:N]:

        u_choosen = pair[0][0]

        u_rating_business_train = user_business_rating_map[u_choosen][business_train]

        u_sum_all_other = []

        for k, v in user_business_rating_map[u_choosen].items():

            # calculate prediction numerator , avg Ru
            if k != business_train:
                u_sum_all_other.append(float(v))

        u_avg_all_other_rated_items = sum(u_sum_all_other) / float(len(u_sum_all_other))

        numerator += pair[1] * (u_rating_business_train - u_avg_all_other_rated_items)

        pearson_total += abs(pair[1])

    r_a_user_train_sum = [user_business_rating_map[user_train][i] for i in user_train_rated_businesses]

    r_a_user_train_avg = sum(r_a_user_train_sum) / float(len(r_a_user_train_sum))

    rating = r_a_user_train_avg

    if pearson_total > 0:
        rating = r_a_user_train_avg + numerator / pearson_total

    rating = rearrange_result(rating)

    return ((x[0], x[1]), rating)


def output_prediction(fp, r1, index_user, index_business):

    fp.write('user_id,business_id,prediction' + '\n')

    output_r1 = []

    #r1.sort(key = lambda x : x[0][0])

    for pair in r1:

        u1 = pair[0][0]

        if u1 in index_user:

            u1 = index_user[pair[0][0]]

        b1 = pair[0][1]

        if b1 in index_business:

            b1 = index_business[pair[0][1]]

        ra = pair[1]

        #fp.write(u1 + ',' + b1 + ',' + str(ra) + '\n')

        output_r1.append(((u1, b1), ra))

    output_r1.sort(key = lambda x : (x[0][0], x[0][1]))

    for pair in output_r1:

        fp.write(pair[0][0] + ',' + pair[0][1] + ',' + str(pair[1]) + '\n')


if __name__ == '__main__':

    if len(sys.argv) < 5:

        print('Error: not enough arguments')

        exit(-1)

    config = SparkConf() \
        .set('spark.executor.memory', '4g') \
        .set('spark.driver.memory', '4g')

    sc = SparkContext(conf=config).getOrCreate()

    train_file = sys.argv[1]

    test_file = sys.argv[2]

    case_id = int(sys.argv[3])

    output_file = sys.argv[4]

    # task2 case #

    data = sc.textFile(train_file)

    testdata = sc.textFile(test_file)

    data = data.map(lambda line: line.split(','))

    firstLine = data.take(1)

    data = data.filter(lambda x: x != firstLine[0]).map(lambda x: (x[0].strip(), x[1].strip(), x[2].strip()))

    user_index = data.map(lambda x: x[0].strip()).distinct().zipWithIndex().collectAsMap()

    business_index = data.map(lambda x: x[1].strip()).distinct().zipWithIndex().collectAsMap()

    index_business = {}

    for k, v in business_index.items():
        index_business[v] = k

    index_user = {}

    for k, v in user_index.items():
        index_user[v] = k

    ratings = data.map(lambda x: (int(user_index[x[0]]), int(business_index[x[1]]), float(x[2])))

    testdata = testdata.map(lambda line: line.split(','))

    testdata_firstline = testdata.take(1)

    testdata = testdata.filter(lambda x: x != testdata_firstline[0]) \
                        .map(lambda x: (x[0].strip(), x[1].strip(), float(x[2].strip())))

    additional_entry = testdata.map(addition_entry).filter(lambda x: x != None)

    # assign ratings for additional entries using avg rating over all addition entries
    num_additional_entry = additional_entry.distinct().count()

    avg_additional_entry_rating = 0

    if num_additional_entry > 0:
        avg_additional_entry_rating = additional_entry.map(lambda x: float(x[2])) \
                                          .sum() / num_additional_entry

    # additional entry with diff rating
    additional_entry_avg = additional_entry \
        .map(lambda x: ((x[0], x[1]), avg_additional_entry_rating))

    additional_entry_orig = additional_entry \
        .map(lambda x: ((x[0], x[1]), x[2]))

    testdata_case12 = testdata.map(output_testset).filter(lambda x: x != None)

    testset_ratings_case12 = testdata_case12 \
        .map(lambda x: (x[0], x[1], x[2]))

    testset_case12 = testdata_case12.map(lambda x: (x[0], x[1]))

    user_business_map = data.map(lambda x: (int(user_index[x[0]]), [int(business_index[x[1]])])) \
        .reduceByKey(lambda a, b: a + b) \
        .map(lambda e: (e[0], set(e[1]))) \
        .collectAsMap()

    business_user_map = data.map(lambda x: (int(business_index[x[1]]), [int(user_index[x[0]])])) \
        .reduceByKey(lambda a, b: a + b) \
        .map(lambda e: (e[0], set(e[1]))) \
        .collectAsMap()

    user_business_rating_map = data \
        .map(lambda x: (int(user_index[x[0]]), [(int(business_index[x[1]]), float(x[2]))])) \
        .reduceByKey(lambda a, b: a + b) \
        .mapValues(map_business_rating) \
        .collectAsMap()

    # case 1

    if case_id == 1:

        print('case 1 Running...')

        time_case1 = time.time()

        features = 2

        iterations = 10

        model = ALS.train(ratings, features, iterations, seed=10)

        preds_case1 = model.predictAll(testset_case12).map(lambda x: ((x[0], x[1]), x[2]))

        preds_case1 = preds_case1.union(additional_entry_avg).collect()

        predsCompare_case1 = testset_ratings_case12 \
            .map(lambda x: ((x[0], x[1]), x[2])).union(additional_entry_orig) \
            .join(sc.parallelize(preds_case1))

        case1RMSE = pow(predsCompare_case1.map(lambda x: pow(x[1][0] - x[1][1], 2)).mean(), 1 / 2)

        print('\n')

        print('case 1 RMSE: ' + str(case1RMSE))

        fp = open(output_file, 'w')

        output_prediction(fp, preds_case1, index_user, index_business)

        fp.close()

        print('task2 case 1 finish time: ' + str(time.time() - time_case1))

    elif case_id == 2:

        print('case 2 Running...')

        time_case2 = time.time()

        preds_case2 = testset_case12.map(train_case2)

        preds_case2 = preds_case2.union(additional_entry_avg).collect()

        print(len(preds_case2))

        predsCompare_case2 = testset_ratings_case12 \
            .map(lambda x: ((x[0], x[1]), x[2])).union(additional_entry_orig) \
            .join(sc.parallelize(preds_case2))

        case2RMSE = pow(predsCompare_case2.map(lambda x: ((x[1][0] - x[1][1]) ** 2)).mean(), 1 / 2)

        print('\n')

        print('case 2 RMSE: ' + str(case2RMSE))

        fp = open(output_file, 'w')

        output_prediction(fp, preds_case2, index_user, index_business)

        fp.close()

        print('task2 case 2 finish time: ' + str(time.time() - time_case2))

    elif case_id == 3:

        print('case 3 Running...')

        time3 = time.time()

        preds_case3 = testset_case12.map(train_case3)

        preds_case3 = preds_case3.union(additional_entry_avg).collect()

        predsCompare_case3 = testset_ratings_case12 \
            .map(lambda x: ((x[0], x[1]), x[2])).union(additional_entry_orig) \
            .join(sc.parallelize(preds_case3))

        case3RMSE = pow(predsCompare_case3.map(lambda x: pow(x[1][0] - x[1][1], 2)).mean(), 1 / 2)

        print('\n')

        print('case 3 RMSE: ' + str(case3RMSE))

        fp = open(output_file, 'w')

        output_prediction(fp, preds_case3, index_user, index_business)

        fp.close()

        print('task2 case 3 finish time: ' + str(time.time() - time3))

    else:

        print('Error: no such case')

        exit(-1)






