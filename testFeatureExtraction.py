import collections
import pickle
from tsfresh import extract_relevant_features, extract_features, feature_selection
import utilFeatExtr as util
import pandas as pd
import os
import multiprocessing as mp
import time

if __name__ == '__main__':
    totalTime = time.time()
    # Choice of the number of features to use
    numberFeatUse = 20
    featuNum = [x for x in range(1,numberFeatUse+1)]

    # Name of the dataset
    datasetUsed = ["ECG200"]

    # Percentage for Cross Validation
    trainPerc = 0.5

    # Threshold of the distance
    threshold = 0.8

    # Choice of the algorithm (Greedy Algorithm default)
    chooseAlgorithm = 0
    testKClique = 0
    trainKClique = 0
    if chooseAlgorithm == 0:
        algorithm = "Greedy Modularity"
    elif chooseAlgorithm == 1:
        testKClique = 10
        trainKClique = 5
        algorithm = "KClique(" + trainKClique + "," + testKClique + ")"
    else:
        algorithm = "SLPA.find_communities(G, 20, 0.01)"

    print("Choosed Algorithm: " + algorithm)

    if os.path.isfile("./" + datasetUsed[0] + '/SFS/' + datasetUsed[0] + 'RankAlgorithm.csv'):
        os.remove("./" + datasetUsed[0] + '/SFS/' + datasetUsed[0] + 'RankAlgorithm.csv')

    for name in datasetUsed:
        nameDataset = name
        # If necessary, adapt the time series for the software and I create the folder for the dataset
        util.augmentationTrain(nameDataset, trainPerc)

        # Take the path of the folders
        trainPath, testPath = util.getPathNamDatTrainPrec(nameDataset, trainPerc)

        # Create the dataframe for the extraction of the features
        listOut, series = util.adaptTimeSeries(testPath)



        # Extraction or loading of  features of the number of time series select for the train cross validation.

        if os.path.isfile(
                "./" + nameDataset + "/Train" + str(trainPerc) + "/featureALL" + nameDataset + ".pk1") == False:
            # Se non le tengo, le estraggo e le salvo in un pickle
            features_filtered_direct = extract_features(listOut, column_id='id', column_sort='time')
            features_filtered_direct.to_pickle(
                "./" + nameDataset + "/Train" + str(trainPerc) + "/featureALL" + nameDataset + ".pk1")
        else:
            features_filtered_direct = pd.read_pickle(
                "./" + nameDataset + "/Train" + str(trainPerc) + "/featureALL" + nameDataset + ".pk1")

        # Extract the relevance for each features and it will be ordered by importance
        features_filtered_direct = features_filtered_direct.dropna(axis='columns')
        ris = feature_selection.relevance.calculate_relevance_table(features_filtered_direct,series,ml_task="classification")
        print("Feature choosed: " + str(len(features_filtered_direct.keys())))
        ris = ris.sort_values(by='p_value')
        print(ris[["p_value"]])
        print(list(ris[["p_value"]]["p_value"]))




        dictOfFeat = {}
        # Creation of group of features

        for t in featuNum:
            print("Threshold: " + str(threshold))
            timeTot = 0
            listDict = {}
            if os.path.isdir("./" + nameDataset + "/SFS") == False:
                os.mkdir("./" + nameDataset + "/SFS")
                os.mkdir("./" + nameDataset + "/SFS/SingleIterationInfo/")

            f = open("./" + nameDataset + "/Train" + str(trainPerc) + "/"+nameDataset+"__NF" + str(t) + "_th" + str(threshold) + "_algSFS.tsv", "w+")
            f.write("Threshold: \t" + str(threshold) + "\n")
            f.write("Algoritmo Usato:\t " + algorithm + " \n")
            listOfId = set(listOut["id"])
            clusterK = len(set(list(series)))
            vettoreIndici = list(set(list(series)))
            print(set(list(series)))
            print(len(set(list(series))))


            dictOfT = {}

            # Create of dataframe where there are the values of the features take in consideration
            for value in set(list(series)):
                dictSing = {value: pd.Series([0], index=["count"])}
                dictOfT.update(dictSing)

            dictSing = {}
            df = pd.DataFrame(dictOfT)
            print(df)

            listOfNumCom = []
            dictOfInfoTrain = {}

            def collect_result_Train(result):
                dictOfInfoTrain.update(result)

            pool = mp.Pool(mp.cpu_count())
            print(t)
            numbOfUse =  t
            listOfFeat = []


            # Creation of the features that we want to use
            for feature in ris["feature"]:
                if len(listOfFeat) < numbOfUse:
                    print(feature)
                    listOfFeat.append(feature)
                else:
                    break
            print(listOfFeat)



            listOfClustering = []

            start = time.time()
            # Creation of graph and extraction of community detection
            for feature in listOfFeat:
                pool.apply_async(util.getCommunityDetectionTrain, args=(feature, features_filtered_direct, listOfId, threshold, clusterK,chooseAlgorithm,trainKClique,nameDataset,"SFS"),callback=collect_result_Train)

            pool.close()
            pool.join()
            end = time.time()
            print("Time Training: " + str(end - start))
            timeTot += end - start
            listDict.update({"TimeTrainCommDetect":end - start})
            f.write("Time Community Detection: \t" + str(end - start) + "\n")


            print(str(len(dictOfInfoTrain.keys())))

            setCluster = list()
            # Creation of list with all the cluster and their weights, used for the creation of CoOccurrence Matrix
            for key in dictOfInfoTrain.keys():
                for clusterInside in dictOfInfoTrain[key]["cluster"]:
                    dictSing = {'list':list(clusterInside),'weight':dictOfInfoTrain[key]["weightFeat"]}
                    print(dictSing)
                    setCluster.append(dictSing)

            start = time.time()
            print("Computation Matrix....")
            # Creation of CoOccurrence Matrix
            matrixNsym = util.getTabNonSym(setCluster,list(listOfId))
            end = time.time()
            timeTot += end - start
            print("Time Computation Matrix: " + str(end - start))
            listDict.update({"TimeTrainMatrix": end - start})

            f.write("Time Computation Matrix: \t" + str(end - start) + "\n")
            start = time.time()

            print("Compute Clustering....")
            # List of the cluster created in the training set. It will be used later for the intersaction
            # with the cluster extract from the testing.
            print("Train")
            listOfCommFindTest = util.getCluster(matrixNsym, listOfId, clusterK)
            end = time.time()

            timeTot += end - start
            print("Time Computation Cluster: " + str(end - start))
            listDict.update({"TimeTrainClus": end - start})
            print("Test")
            listOfCommFindTest = util.createSet(listOfCommFindTest, clusterK)

            # Modify the index of the TimeSeries with their classes
            listOfAllClass = []
            for value in listOfCommFindTest:
                listOfClass = []
                for ind in value["cluster"]:
                    listOfClass.append(series[ind])
                listOfAllClass.append(listOfClass)

            resultCount = {}
            for value in set(list(series)):
                dictSingA = {value: pd.Series([0], index=["count"])}
                resultCount.update(dictSingA)
            dfA = pd.DataFrame(resultCount)

            # Creation of confusion matrix
            confMatrix = []
            for listOfClassS in listOfAllClass:
                if (listOfClassS != []):
                    counter = collections.Counter(listOfClassS)
                    print(counter.most_common())
                    dictConfMatrix = {}
                    for a, b in counter.most_common():
                        dictConfMatrix.setdefault(a, []).append(b)
                    listConfusion = []
                    for i in range(0, clusterK):
                        if vettoreIndici[i] in dictConfMatrix.keys():
                            listConfusion.insert(i, dictConfMatrix.get(vettoreIndici[i])[0])
                        else:
                            listConfusion.insert(i, 0)
                    confMatrix.append(listConfusion)

            # Calculatio of Accuracy, Precision, Recall and FScore
            accuracy, precision, recall, fScore, migliorM = util.getBestAccuracy(confMatrix)
            print("Accuracy:" + str(accuracy))
            print("Precision:" + str(precision))
            print("Recall:" + str(recall))
            print("FScore:" + str(fScore))
            print("Miglior M:" + str(migliorM))
            countTP = 0
            for i in dfA.keys():
                countTP += dfA[i]["count"]

            listDict.update({"Accuracy": accuracy})
            listDict.update({"Precision": precision})
            listDict.update({"Recall": recall})
            listDict.update({"F1Score": fScore})

            for i in range(0, len(listOfFeat)):
                f.write("Feature " + str(i) + ": \t" + str(listOfFeat[i]) + "\n")

            f.close()

            print()
            # Calculation and creation file for Ranking Algorithm
            for feat in listOfFeat:
                util.calcSSW(feat, features_filtered_direct[feat], listOfCommFindTest, len(listOfFeat), accuracy,
                             nameDataset)

            listDict.update({"Threshold": threshold})
            listDict.update({"Cluster": clusterK})
            dictOfFeat.update({t: listDict})

        dbfile = open("./" + nameDataset + "/SFS/SingleIterationInfo/" + nameDataset + "_th" + str(
            threshold) + "_alg_" + algorithm + "SUMMARYRES.csv", 'w+')

        for key in dictOfFeat.keys():
            dbfile.write("Feature")
            for keys in dictOfFeat[key]:
                dbfile.write("," + keys)
            dbfile.write("\n")
            break
        for key in dictOfFeat.keys():
            dbfile.write(str(key))
            for keys in dictOfFeat[key]:
                dbfile.write("," + str(dictOfFeat[key][keys]))
            dbfile.write("\n")

        dbfile.close()

    f = open("./" + nameDataset + "/SFS/SingleIterationInfo/" + nameDataset + "__NF" + str(
        t) + "_th" + str(threshold) + "_alg" + algorithm + "TIMETOTAL.tsv", "w+")
    f.write(str(time.time() - totalTime))

























