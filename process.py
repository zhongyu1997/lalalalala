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
            if count==4000: break;

    print(len(vector))
    return [stock_name, vector]


def endMonth(month):
    if month == 2:
        return 29
    elif month == 4 or month == 6 or month == 9 or month == 11:
        return 30
    else:
        return 31

def get_info_by_month(line):
    print("KLK")
    x = line[0].split("_")
    stock_name = x[2] + x[0].split('20200')[1]
    data = line[1].split('\r\n')
    vector = []
    prepData = {}

    day = 0
    oldDay = 0
    for record in data:
        # if it is an entry
        if len(record.split(",")) > 6:
            tmp = record.split(",")
            print(tmp[0])
            day = int(tmp[0].split("/")[1])
            if oldDay == 0:  # check for missing values before the 1st day, the price will be the opening price
                for i in range(1, day):
                    prepData[i] = float(tmp[2])
            else: # we check for missing days between last day known closing price and opening current one, interpolate
                n=day-oldDay
                for i in range(1, n):
                    prepData[oldDay+i] = prepData[oldDay]*((n-i)/n) + float(tmp[2])*(i/n)

            prepData[day] = float(tmp[5])
            oldDay=day

    # check for missing values after last day, the price will be the last closing price
    month = int(x[0].split('20200')[1])
    endDay = endMonth(month)
    if day != endDay:
        for i in range(day + 1, endDay + 1):
            prepData[i] = float(tmp[5])
    for key in prepData:
        vector.append([prepData[key]])

    print(vector)
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
    bins=40
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
    vector_representation=sc.wholeTextFiles("dataset/2020/test").map(get_info_by_month).reduceByKey(lambda s,t: s+t)

    # mutual information
    # possible combination
    pair_wise=vector_representation.cartesian(vector_representation).filter(lambda x:x[0][0]!=x[1][0]).map(lambda x: (filter_same_record(x[0][0],x[1][0]),[x[0][1],x[1][1]])).reduceByKey(lambda s,t:s)
    mutual_info=pair_wise.map(lambda x: (x[0],calc_MI(x[1])))

    # print(vector_representation.collect())
    # print(pair_wise.collect())
    print(mutual_info.collect())