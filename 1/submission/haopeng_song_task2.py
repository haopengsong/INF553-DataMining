

import json, sys, time
from pyspark import SparkContext, SparkConf

filePath1 = sys.argv[1]
filePath2 = sys.argv[2]

outputFileA = sys.argv[3]
outputFileB = sys.argv[4]

#"/Users/haopengsong/Downloads/yelp_dataset/review.json"
#"/Users/haopengsong/Downloads/yelp_dataset/business.json"

sc = SparkContext.getOrCreate()

review = sc.textFile(filePath1)
business = sc.textFile(filePath2)


businessToStates = business \
				.map(lambda x : (json.loads(x)['business_id'], json.loads(x)['state'])) 


businessToStarAndState = review \
						.map(lambda s : (json.loads(s)['business_id'], json.loads(s)['stars'])) \
						.filter(lambda x : x is not None) \
						.leftOuterJoin(businessToStates) 
                        

stateToStars = businessToStarAndState  \
				.map(lambda x : (x[1][1], x[1][0]))  \
				.mapValues(lambda x : (x, 1))   \
				.reduceByKey(lambda x , y : (x[0] + y[0], x[1] + y[1]))   \
				.mapValues(lambda x : x[0] / x[1])   \
				.sortBy(lambda x : x[1], False)
#THE FINAL ANSWER !!!!!!!

#output A
outputA = stateToStars.collect()
sorted(outputA, key = lambda i : i[1], reverse = True)
fileA = open(outputFileA, 'w')
fileA.write('state,stars' + '\n')
for x in outputA:
	fileA.write(x[0]+','+str(x[1])+'\n')
fileA.close()

#m1
start_time = time.time()
m1_data = stateToStars.collect()
for x in range(5):
    print(m1_data[x][0])
end_time_m1 = time.time() - start_time
print('{}s seconds'.format(end_time_m1))

#m2
start_time = time.time()
m2_data = stateToStars.take(5)
for x in m2_data:
    print(x[0])
end_time_m2 = time.time() - start_time
print('{}s seconds'.format(end_time_m2))

outputB = {}
outputB['m1'] = end_time_m1
outputB['m2'] = end_time_m2
outputB['explanation'] = '1. the collect() function would fetch all outputs to the main memory whereas' + \
						' take(n) would only fetch certain numbers (n) of output to the main memory' +  \
						  '2. Depends on the size of the output, there could be a huge different of running time between those two function'
fileB = open(outputFileB, 'w')
fileB.write(json.dumps(outputB))
fileB.close()
