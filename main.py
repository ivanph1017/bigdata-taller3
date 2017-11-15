from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression  #encontraremos una recta (h) para decidir si ciertos valores se clasifican para un lado o para el otro
from pyspark.ml.evaluation import BinaryClassificationEvaluator

'''  
    Defensivo: 0 
    Ofensivo: 1
'''

## split y parseo
def cleanProcess(sc, url):
    rdd = sc.textFile(url).map(lambda linea: linea.split(","))
    rdd_data = rdd.map(lambda x: [int(x[1]), int(x[2]), int(x[3]),int(x[4]),int(x[5]),int(x[6]),int(x[7]),int(x[8]),int(x[9]),int(x[10]),int(x[11]),int(x[12]),int(x[13]),int(x[14]),int(x[15]),int(x[16]),int(x[17]),int(x[18]),int(x[19]),int(x[20]),int(x[21]),int(x[22]),int(x[23]),int(x[24]),int(x[25]),int(x[26]),int(x[27]),int(x[28]),int(x[29]),int(x[30]),int(x[31]),int(x[32]),int(x[33])])
    #rdd.foreach(lambda x: print(x))
    return rdd_data

## VectorAssembler: Convertir el dataframe en uno que tenga una columna llamada output_col compuesta por las columnas @features_array. 
def convert_dataframe(data, features_array, output_col):        
    assembler = VectorAssembler(inputCols= features_array, outputCol= output_col)    
    return assembler.transform(data).select("features","PRE_POS")    
    #test_data = assembler.transform(test).select("features","PRE_POS")

## Obtenemos un modelo de Regresion Logistica dependiendo de familyName
def logistic_regression(data, label_col, familyName):
    lr = LogisticRegression(
        maxIter=100, regParam=0.3, elasticNetParam=0.8, 
        labelCol= label_col, family=familyName)
    return lr.fit(data)

##Mostrar parametros multiclase
def parameters_lr_multiclass(lr_model):
    print("Coefficients: " + str(lr_model.coefficientMatrix))
    print("Intercept: " + str(lr_model.interceptVector))
    
##Mostrar parametros binomial
def parameters_lr_binomial(lr_model):
    print("Coefficients: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))
    
##Metodo que realizar una evaluacion
def evaluate_model_regression(label_col, name, data_to_validate):
    evaluator = BinaryClassificationEvaluator(labelCol=label_col, metricName=name, rawPredictionCol='rawPrediction')
    print("{}:{}".format(name,evaluator.evaluate(data_to_validate)))

##Ejecuciòn de modelo  binomial de regresion logistica
def execute_logistic_regression_binomial(sc, url, spark):
    ''' Regresion binomial '''
    print("------------ Regresion binomial --------------")

    rdd_data = cleanProcess(sc, url)
    headers = ["PRE_POS","WF","SM","BC","DRI","MA","SLT","STT","AG","REACT","AP","INT","VI","CO","CRO","SP","LP","ACC","SPEED","STA","STR","BA","AGI","JU","HE","SHP","FI","LS","CU","FA","PE","VOL","RA"]
    data = spark.createDataFrame(rdd_data, headers)
    #data.show()    
    train, test = data.randomSplit([0.7,0.3], seed=12345)
    #train.show()

    features = ["WF","SM","BC","DRI","MA","SLT","STT","AG","REACT","AP","INT","VI","CO","CRO","SP","LP","ACC","SPEED","STA","STR","BA","AGI","JU","HE","SHP","FI","LS","CU","FA","PE","VOL","RA"]
    output = "features"
    train_data = convert_dataframe(train, features, output)
    # train_data.show() 

    print("Encontrando h ...")
    label_col = 'PRE_POS'

    lr_model_binomial = logistic_regression(train_data, label_col, 'binomial')
    parameters_lr_binomial(lr_model_binomial)

    test_data = convert_dataframe(test, features, output)

    print("Testing model ...")

    data_to_validate = lr_model_binomial.transform(test_data)
    data_to_validate.show()
    
    evaluate_model_regression(label_col, 'areaUnderROC',data_to_validate)
    evaluate_model_regression(label_col, 'areaUnderPR',data_to_validate)
    
##Ejecuciòn de modelo  multiclase de regresion logistica
def execute_logistic_regression_multiclass(sc, url, spark):
    ''' Regresion Multiclase '''
    print("------------ Regresion multiclase --------------")

    rdd_data = cleanProcess(sc, url)
    headers = ["PRE_POS","WF","SM","BC","DRI","MA","SLT","STT","AG","REACT","AP","INT","VI","CO","CRO","SP","LP","ACC","SPEED","STA","STR","BA","AGI","JU","HE","SHP","FI","LS","CU","FA","PE","VOL","RA"]
    data = spark.createDataFrame(rdd_data, headers)
    #data.show()    
    train, test = data.randomSplit([0.7,0.3], seed=12345)
    #train.show()

    features = ["WF","SM","BC","DRI","MA","SLT","STT","AG","REACT","AP","INT","VI","CO","CRO","SP","LP","ACC","SPEED","STA","STR","BA","AGI","JU","HE","SHP","FI","LS","CU","FA","PE","VOL","RA"]
    output = "features"
    train_data = convert_dataframe(train, features, output)
    # train_data.show() 

    print("Encontrando h ...")
    label_col = 'PRE_POS'

    lr_model_multiclass = logistic_regression(train_data, label_col, 'multinomial')
    parameters_lr_multiclass(lr_model_multiclass)    

    print("Testing model ...")

    test_data = convert_dataframe(test, features, output)

    data_to_validate = lr_model_multiclass.transform(test_data)
    data_to_validate.show()

    evaluate_model_regression(label_col, 'areaUnderROC',data_to_validate)
    evaluate_model_regression(label_col, 'areaUnderPR',data_to_validate)

def main():
    
    conf = SparkConf().setAppName('fifaModel').setMaster('local')
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    
    ##Llamada a funciones de regresiones logisticas
    execute_logistic_regression_binomial(sc, "data/dataConRating1.csv", spark)
    execute_logistic_regression_multiclass(sc, "data/dataConRating2.csv", spark)

if __name__ == '__main__':
    main()
