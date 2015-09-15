#! /usr/bin/env python
"""
@author: ruonan weituo
this is the training model and feature extraction.
"""

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors, SparseVector
from pyspark import SparkContext, SparkConf

def learn(examples,depth,bin):
    global model
    model = DecisionTree.trainRegressor(examples, categoricalFeaturesInfo={},
                                        impurity='variance', maxDepth=depth, maxBins=bin)

def predict(examples):
    return model.predict(examples.map(lambda x: x.features))

def rmse(y_true, y_pred):
    labelAndPredicted = y_true.zip(y_pred)
    trainMSE = labelAndPredicted.map(lambda (v, p): (v - p) * (v - p)).sum() / float(y_true.count())
    import math
    return math.sqrt(trainMSE)

def validate(train,test,d = 5, b =100):
    learn(train, d,b)
    predicted = predict(test)
    return rmse(test.map(lambda x: x.label), predicted)

if __name__ == "__main__":
    import music
    train_examples = music.load_examples('data/train.pkl')
    fileName = 'data/temp.data'
    music.save_libsvm(fileName,train_examples)

    conf = SparkConf()
    conf.setMaster('spark://qwan-ThinkPad-T430s:7077').setAppName('EMIMusic').set("spark.executor.memory", "2g").set("spark.storage.memoryFraction", "0.5").set("spark.kryoserializer.buffer.max.mb","128").set("spark.default.parallelism","12")
    sc = SparkContext(conf = conf)

    fw = open("output.txt","w")

    for i in range(3,8):
        train = MLUtils.loadLibSVMFile(sc,"%s.%s"%(fileName,"train")).cache()
        test = MLUtils.loadLibSVMFile(sc,"%s.%s"%(fileName,"test")).cache()
        dep = i*2
        score = validate(train,test, dep)
        print "RMSE: %0.6f" % (score)
        fw.write("%d,%0.6f\n" % (dep,score))
        fw.flush()

    for i in range(8,12):
        train = MLUtils.loadLibSVMFile(sc,"%s.%s"%(fileName,"train")).cache()
        test = MLUtils.loadLibSVMFile(sc,"%s.%s"%(fileName,"test")).cache()
        bin = i*10
        score = validate(train,test,b = bin)
        print "RMSE: %0.6f" % (score)
        fw.write("%d,%0.6f\n" % (bin,score))
        fw.flush()

    fw.close()
