from pyspark import SparkContext, SparkConf
import json, sys



def mapper(s):
    element = json.loads(s)
    if int(element['useful']) > 1:
        return (element["review_id"], 1)

def mapper_stars(s):
    element = json.loads(s)
    if int(element['stars']) >= 5:
        return (element['review_id'], 1)

def mapper_text(s):
    element = json.loads(s)
    return (element['review_id'], len(element['text']))

def mapper_text_exists(s):
    element = json.loads(s)
    if element['text'] != None and len(element['text']) > 0:
        return element['user_id']

def mapper_user_reviews(s):
    element = json.loads(s)
    if element['text'] != None and len(element['text']) > 0:
        return (element['user_id'], 1)

def mapper_business_id(s):
    element = json.loads(s)
    return element['business_id']

def mapper_business_reviews(s):
    element = json.loads(s)
    if element['text'] != None and len(element['text']) > 0:
        return (element['business_id'], 1)

if __name__ == '__main__':

    sc = SparkContext.getOrCreate()

    filePath = sys.argv[1]
    outputPath = sys.argv[2]

    # "/Users/haopengsong/Downloads/yelp_dataset/review.json"

    review = sc.textFile(filePath)


    #task 1.A
    print("1. a")
    review_map = review.map(mapper)
    review_result = review_map.filter(lambda x : x != None).distinct()
    print(review_result.count())

    #task 1.A checker
    sumUsefulReview = review_result.map(lambda x : x[1]).sum()
    print(sumUsefulReview)

    #task 1.B
    print("1. b")
    review_star = review.map(mapper_stars).filter(lambda x : x != None).distinct()
    print(review_star.count())

    #task 1.B checker
    sumStartReview = review_star.map(lambda x : x[1]).sum()
    print(sumStartReview)

    #task 1.C
    print("1. c")
    longestReviewLen = review.map(mapper_text).map(lambda x : x[1]).max()
    print(longestReviewLen)

    #task 1.D
    print("1. d")
    numUserReviews = review.map(mapper_text_exists).distinct().count()
    print(numUserReviews)

    #task 1.E
    print("1. e")
    topUserWithReview = review.map(mapper_user_reviews).reduceByKey(lambda a, b: a + b)
    topUserResult = topUserWithReview.takeOrdered(20, key = lambda x : -x[1])
    print(topUserResult)

    #task 1.F
    print("1. f")
    numBusinessReviewed = review.map(mapper_business_id).distinct().count()
    print(numBusinessReviewed)

    #task 1.G
    print("1. g")
    topBusinessWithReview = review.map(mapper_business_reviews).reduceByKey(lambda a, b: a + b).takeOrdered(20, key = lambda x : -x[1])
    print(topBusinessWithReview)

    file = open(outputPath, 'w')
    file.write('haha')

