from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression  #encontraremos una recta (h) para decidir si ciertos valores se clasifican para un lado o para el otro
from pyspark.ml.evaluation import BinaryClassificationEvaluator

'''  
    Defensivo: 0 
    Ofensivo: 1
'''

def main():
    conf = SparkConf().setAppName('fifaModel').setMaster('local')
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    rdd = sc.textFile("data/FullData-modif.csv").map(lambda linea: linea.split(","))

    rdd_data = rdd.map(lambda x: [int(x[0]), int(x[1]), int(x[2]), int(x[3]),int(x[4]),int(x[5]),int(x[6]),int(x[7]),int(x[8]),int(x[9]),int(x[10]),int(x[11]),int(x[12]),int(x[13]),int(x[14]),int(x[15]),int(x[16]),int(x[17]),int(x[18]),int(x[19]),int(x[20]),int(x[21]),int(x[22]),int(x[23]),int(x[24]),int(x[25]),int(x[26]),int(x[27]),int(x[28]),int(x[29]),int(x[30]),int(x[31]),int(x[32])])
    #rdd.foreach(lambda x: print(x))
    headers = ["PRE_POS","WF","SM","BC","DRI","MA","SLT","STT","AG","REACT","AP","INT","VI","CO","CRO","SP","LP","ACC","SPEED","STA","STR","BA","AGI","JU","HE","SHP","FI","LS","CU","FA","PE","VOL","RA"]

    data = spark.createDataFrame(rdd_data, headers)
    #data.show()
    
    train, test = data.randomSplit([0.7,0.3], seed=12345)
    #train.show()

    features = ["WF","SM","BC","DRI","MA","SLT","STT","AG","REACT","AP","INT","VI","CO","CRO","SP","LP","ACC","SPEED","STA","STR","BA","AGI","JU","HE","SHP","FI","LS","CU","FA","PE","VOL","RA"]
    output = "features"
    assembler = VectorAssembler(inputCols= features, outputCol= output)
    
    train_data = assembler.transform(train).select("features","PRE_POS")
    test_data = assembler.transform(test).select("features","PRE_POS")

    # train_data.show()
    # print("test_data assembler")
    # test_data.show()

    print("Encontrando h ...")

    lr = LogisticRegression(
        maxIter=100, regParam=0.3, elasticNetParam=0.8, 
        labelCol='PRE_POS', family='binomial')

    lr_model = lr.fit(train_data)

    print("Coefficients: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))

    print("Testing model ...")

    data_to_validate = lr_model.transform(test_data)

    evaluator1 = BinaryClassificationEvaluator(labelCol='PRE_POS', metricName='areaUnderROC', rawPredictionCol='rawPrediction')

    print("{}:{}".format("areaUnderROC",evaluator1.evaluate(data_to_validate)))

    evaluator2 = BinaryClassificationEvaluator(labelCol='PRE_POS', metricName='areaUnderPR', rawPredictionCol='rawPrediction')

    print("{}:{}".format("areaUnderPR",evaluator2.evaluate(data_to_validate)))    

if __name__ == '__main__':
    main()