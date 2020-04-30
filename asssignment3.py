from pyspark import *
from math import sqrt
import numpy as np

def data_formatting(f):
    name = f[0]
    data = f[1]

    stock = name.split('_')[2]
    data = data.split('\r\n')
    price = []
    date = []
    time = []
    for point in data:
        tmp = point.split(",")
        price.append(float(tmp[2]))
        time.append(tmp[1])
        date.append(tmp[0])
        if(len(date) == 600):
            break
    return((stock,date,time,price))

def corr(x,y):
    n = len(x)
    sumx = sum(x)
    sumy = sum(y)
    xy = sum(map(lambda a,b: a*b,x,y))
    xx = sum(map(lambda f: f*f,x))
    yy = sum(map(lambda f: f*f,y))
    a = (n*xy) - (sumx*sumy)
    b = sqrt((n*xx) - (sumx*sumx))*sqrt((n*yy) - (sumy*sumy))
    return(a/b)

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

sc = SparkContext()
rdd=sc.wholeTextFiles("D:\\test\\").map(data_formatting)
prices = rdd.map(lambda x: (x[0],x[3]))
stocks = prices.map(lambda x: (x[0]))

prices_pairs = prices.cartesian(prices).filter(lambda x: x[0][0] < x[1][0]).flatMap(lambda x: [(x[0][0]+ '&' + x[1][0],x[0][1]),(x[0][0] + '&' + x[1][0],x[1][1])])

corrs_pearson = prices_pairs.reduceByKey(lambda acc,x: corr(acc,x))
corrs_mutual_info = prices_pairs.reduceByKey(lambda acc,x: calc_MI([acc,x]))

print(corrs_pearson.count())
print(stocks.collect())

for x in corrs_pearson.collect():
    print(x)

for x in corrs_mutual_info.collect():
    print(x)
