from pyspark import *
import numpy as np


def get_info(line):
    x=line[0].split("_")
    stock_name=x[2]
    data=line[1].split('\r\n')
    vector=[]

    # todo: formalize the vector. here I just simply add close price into the vector without any preprocessing.
    count = 0;
    for record in data:
        tmp=record.split(",")
        if len(tmp)>=6:
            count+=1
            vector.append(float(tmp[5]))
            # mannually restrict the length of all vectors. Otherwise correlation cannot be computed.
            # TODO: time alignment processing. Make sure that vectors share the same size.
            if count==6000: break;

    print(len(vector))
    return [stock_name, vector]


# after applying cartesian function, it would have result like this:
# [(('AALB', [40.25, 39.63]), ('AGN', [4.12, 3.677])), (('AGN', [4.12, 3.677]), ('AALB', [40.25, 39.63]))]
# filter the same records
def filter_same_record(x,y):
    key=""
    if x>y :
        key+=x+"&"
        key+=y
    else:
        key+=y+"&"
        key+=x
    return key


def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))
    return H


# Calculate mutual information for two vectors.
def calc_MI(stocks):
    X=stocks[0]
    Y=stocks[1]
    bins=7000
    c_X = np.histogram(X,bins)[0]
    c_Y = np.histogram(Y,bins)[0]
    c_XY = np.histogram2d(X, Y, bins)[0]

    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    MI = MI/np.sqrt(H_X*H_Y) # normal mutual information. Delete this step and it will be mutual information.
    return MI


if __name__ == '__main__':
    # totally six threads.
    conf = SparkConf().setAppName('test').setMaster('local[6]')
    sc = SparkContext(conf=conf)

    # read data by wholeTextFiles : (filename, content)
    # use get_info to clear and extract data
    # get_info function help to extract stock names, and close price.
    # reduceByKey function merges time series data of different month into one list. TODO: Watch out the merge order! Not sure it is correct.
    vector_representation=sc.wholeTextFiles("dataset/2020/test").map(get_info).reduceByKey(lambda s,t: s+t)

    # mutual information
    # possible combination
    # .filter(lambda x:x[0][0]!=x[1][0])
    pair_wise=vector_representation.cartesian(vector_representation).map(lambda x: (filter_same_record(x[0][0],x[1][0]),[x[0][1],x[1][1]])).reduceByKey(lambda s,t:s)
    mutual_info=pair_wise.map(lambda x: (x[0],calc_MI(x[1])))

    # count=distFile.map(get_info)
    # print(pair_wise.collect())
    print(mutual_info.collect())
