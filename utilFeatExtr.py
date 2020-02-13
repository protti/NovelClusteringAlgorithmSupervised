import csv
import os
import time
import random
import pandas as pd
import numpy as np
import itertools
from sklearn.metrics.pairwise import pairwise_distances
from pyclustering.cluster.kmedoids import kmedoids
import matplotlib.pyplot as plt

import pickle
import networkx as nx
import multiprocessing as mp
import SLPA
import utilityUCR as util
matrixSym = []


def adaptTimeSeries(path):
    with open(path, 'r') as csvFile:
        reader = csv.reader(csvFile)
        id = 0
        listOfValue = []
        listOfId = []
        listOfTime = []
        listOfClass = []
        listGeneric = []
        for row in reader:

            splitted = row[0].split('\t')
            listOfClass.append(splitted[0])
            for i in range(1,len(splitted)):
                listOfValue.append(float(splitted[i]))
                listOfTime.append(i)
                listOfId.append(id)
                listGeneric.append((id,i,(float(splitted[i]))))

            id += 1

        df = pd.DataFrame(listGeneric, columns=['id', 'time','value'])
        series = pd.Series((i for i in listOfClass))
        return df,series

def getMedianDistance(threshold,listOfValue):
    try:
        listOfDistance = []
        for i in range(0,len(listOfValue)):
            for j in range(i + 1, len(listOfValue)):
                listOfDistance.append(abs(listOfValue[i] - listOfValue[j]))
        listOfDistance.sort(reverse=False)


        # plt.show()



    except Exception as e:
        print(e)
    if listOfDistance[int(len(listOfDistance) * threshold)] == 0:
        print("Tutti i valori sono uguali")
    return listOfDistance[int(len(listOfDistance) * threshold)]




def getTabNonSym(setCluster,listId):


    w = len(listId)
    matrixSym = [[0 for x in range(w)] for y in range(w)]


    def matrixCalcParal(result):
        # print(len(result))
        for val in result:
            matrixSym[val["i"]][val["j"]] = val["value"]



    pool = mp.Pool(mp.cpu_count())
    totRig = int(len(listId)/mp.cpu_count())

    for i in range(0,mp.cpu_count()):
        start = i * int((len(listId)/mp.cpu_count()))

        if i == mp.cpu_count() - 1:
            totRig += int(len(listId)%mp.cpu_count())

        pool.apply_async(getValueMatrix, args=(start,listId,totRig,setCluster),callback=matrixCalcParal)

    pool.close()
    pool.join()
    print("Lunghezza" + str(len(matrixSym)))
    for i in range(len(matrixSym)):
        maxVal = max(matrixSym[i])
        for j in range(len(matrixSym)):
            matrixSym[i][j] = abs(matrixSym[i][j] - maxVal)
    return matrixSym



def getValueMatrix(start,listId,totRig,listOfClust):
    try:
        dictOfValueIJ = []
        for i in range(0,totRig):
            for j in range(0, len(listId)):
                resultCouple = numOfRipetitionCouple(listId[i+start], listId[j], listOfClust)
                resultCluster = numOfClusterPres(listOfClust, listId[i+start])
                if resultCluster[1] == resultCouple[1]:
                    value = 1
                elif resultCouple[1] == 0:
                    value = 0
                else:
                    value = resultCouple[0] / resultCluster[0]

                dictSingle = {"value":value,
                              "i":i+start,"j":j}

                dictOfValueIJ.append(dictSingle)
        return dictOfValueIJ
    except Exception as e:
        print("Exception in getValueMatrix:")


def getCluster(matrixsym,setCluster,numClust):

    dictTotal = {}
    for x in setCluster:
        listOfDist = []
        for y in setCluster:
            if x != y:

                dictSing = {"id":y,"distance":matrixsym[x][y]}
                listOfDist.append(dictSing)
        dictTotal[x] = listOfDist


    idChoose = util.getInitialIndex(dictTotal,numClust)
    D = pairwise_distances(matrixsym, metric='correlation')


    kmedoids_instance = kmedoids(D, idChoose, tolerance=0.000001)
    kmedoids_instance.process()
    Cl = kmedoids_instance.get_clusters()
    # show allocated clusters

    dictClu = {}
    for i in range(0, len(Cl)):
        dictApp = {i: Cl[i]}
        dictClu.update(dictApp)
    print(dictClu)


    listOfCommFind = []

    for label in dictClu:
        for point_idx in dictClu[label]:
            # print('label {0}:　{1}'.format(label, list(setCluster)[point_idx]))
            dictSing = {"label": label, "cluster": list(setCluster)[point_idx]}
            listOfCommFind.append(dictSing)
    return listOfCommFind

def numOfClusterPres(setCluster,id):
    countId = 0
    countTimes = 0
    for i in range(0,len(setCluster)):
        if id in (setCluster[i]["list"]):
            countId += (setCluster[i]["weight"])
            countTimes += 1
    return countId, countTimes

def numOfRipetitionCouple(id1,id2,setCluster):
    countId = 0
    countTimes = 0
    for i in range(0,len(setCluster)):
        if id1 in (setCluster[i]["list"]) and id2 in (setCluster[i]["list"]):
            countId += setCluster[i]["weight"]
            countTimes += 1
    return countId,countTimes


def listOfId(setCluster):
    listId = set()
    for value in setCluster:
        for id in value:
            listId.add(id)
    return list(listId)

def createSet(listOfCommFind,clusterK):
    listOfCluster = []
    for i in range(0,clusterK):
        dictSing = {"cluster":[],"label":i}
        listOfCluster.append(dictSing)
    for value in listOfCommFind:
        listApp = listOfCluster[value["label"]]["cluster"]
        listApp.append(value["cluster"])
        listOfCluster.remove(listOfCluster[value["label"]])
        dictSing = {"cluster":listApp,"label":value["label"]}
        listOfCluster.insert(value["label"],dictSing)

    return listOfCluster



def augmentationTrain(nameDataset,percNeed):


    if os.path.isfile("./" +nameDataset +"/Train"+str(percNeed)+"/"+ nameDataset+"_"+str(percNeed) +"_NewTrain.tsv") == False:

        if os.path.isdir("./" +nameDataset) == False:
            os.mkdir(nameDataset)
            os.rename("./"+nameDataset+".tsv", "./" +nameDataset + "/"+nameDataset+".tsv")

        if os.path.isdir("./" +nameDataset +"/Train"+str(percNeed)) == False:
            os.mkdir(nameDataset +"/Train"+str(percNeed))

        listOut, series = adaptTimeSeries("./" +nameDataset + "/"+nameDataset+".tsv")

        dictOfT = {}
        for value in set(list(series)):
            dictSing = {value: pd.Series([int(list(series).count(value) * percNeed)], index=["count"])}
            dictOfT.update(dictSing)
        df = pd.DataFrame(dictOfT)

        fTr = open("./" +nameDataset+"/Train"+str(percNeed)+"/"+ nameDataset+"_"+str(percNeed) +"_NewTrain.tsv", "w+")

        with open("./" +nameDataset+"/"+nameDataset+".tsv", 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                # print(df)
                ind = row[0][0]
                if(ind == "-"):
                    ind = "-1"
                if df[ind]["count"] != 0:
                    fTr.write(row[0])
                    fTr.write("\n")
                    df[ind]["count"] -= 1
        fTr.close()


def getPathNamDatTrainPrec(nameDataset, percNeed):
    trainPath = "./" +nameDataset+"/Train"+str(percNeed)+"/"+ nameDataset+"_"+str(percNeed) +"_NewTrain.tsv"
    testPath = "./" +nameDataset+"/"+nameDataset+".tsv"
    return trainPath,testPath



def getCommunityDetectionTrain(feature,features_filtered_direct,listOfId,threshold,clusterK,chooseAlgorithm,trainKClique, nameDataset, algorithmFeat):

    listOfDictInfoFeat = {}
    if os.path.isdir("./" + nameDataset + "/" + algorithmFeat + "/CommunityDetection") == False:
        if os.path.isdir("./" + nameDataset + "/" + algorithmFeat + "/CommunityDetection") == False:
            os.mkdir("./" + nameDataset + "/" + algorithmFeat + "/CommunityDetection")

    if os.path.isfile("./" + nameDataset + "/" + algorithmFeat + "/CommunityDetection/TrainListOfComm.pkl") == False:
        with open("./" + nameDataset + "/" + algorithmFeat + "/CommunityDetection/TrainListOfComm.pkl", 'wb') as f:
            pickle.dump(listOfDictInfoFeat, f)

    with open("./" + nameDataset + "/" + algorithmFeat + "/CommunityDetection/TrainListOfComm.pkl", 'rb') as f:
        listOfDictInfoFeat = pickle.load(f)

    if not feature in listOfDictInfoFeat.keys():
        dictOfInfo = {}
        G = nx.Graph()
        H = nx.path_graph(listOfId)
        G.add_nodes_from(H)
        distanceMinAccept = getMedianDistance(threshold, features_filtered_direct[feature])

        for i in range(0, len(listOfId)):
            for j in range(i + 1, len(listOfId)):
                if abs(features_filtered_direct[feature][i] - features_filtered_direct[feature][j]) < distanceMinAccept:
                    G.add_edge(i, j)

        try:

            if chooseAlgorithm == 0:
                coms = list(nx.algorithms.community.greedy_modularity_communities(G))
            elif chooseAlgorithm == 1:
                coms = list(nx.algorithms.community.k_clique_communities(G,trainKClique))
            else:
                extrC = SLPA.find_communities(G, 20, 0.01)
                coms = []
                for val in extrC:
                    coms.append(frozenset(extrC[val]))

            for value in coms:
                if len(coms) > clusterK:
                    dictOfInfo[feature] = {"distance": distanceMinAccept, "cluster": coms,
                                         "weightFeat": clusterK / len(coms)}
                else:
                    dictOfInfo[feature] = {"distance": distanceMinAccept, "cluster": coms,
                                         "weightFeat":  len(coms)/clusterK}


        except Exception as e:
            print("Exception in CommDectTrain")
            pass
        with open("./" + nameDataset + "/"+algorithmFeat + "/CommunityDetection/TrainListOfComm.pkl", 'wb') as f:
            listOfDictInfoFeat[feature] = dictOfInfo
            pickle.dump(listOfDictInfoFeat, f)
            f.close()
    else:
        dictOfInfo = listOfDictInfoFeat[feature]

    return dictOfInfo



def getBestAccuracy(confMatrix):
    bestAcc = 0
    bestM = []
    print("ConfMatrix " +str(confMatrix))
    for m in itertools.permutations(confMatrix):
        if bestAcc == 0:
            bestAcc = calcAcc(m)
            bestM = m
        else:
            val = calcAcc(m)
            if bestAcc < val:
                bestAcc = val
                bestM = m

    precision = calcPrecision(bestM)
    recall = calcRecall(bestM)
    fScore = 2 * ((precision*recall)/(precision+recall))


    return bestAcc,precision,recall,fScore,bestM

def calcAcc(m):
    sumDiag = 0
    sumRig = 0
    for i in range(len(m)):
        sumDiag += m[i][i]
        sumRig += np.sum(m, axis=0)[i]

    return sumDiag/sumRig

def calcPrecision(m):
    sumAcc = 0
    for i in range(len(m)):
        sumDiag = m[i][i]
        sumAll = np.sum(m, axis=0)[i]
        sumAcc += sumDiag/sumAll
    return sumAcc/len(m)

def calcRecall(m):
    sumAcc = 0
    for i in range(len(m)):
        sumDiag = m[i][i]
        sumAll = np.sum(m, axis=1)[i]
        sumAcc += sumDiag / sumAll
    return sumAcc / len(m)


def calcSSW(feat,feature,listOfCommFindTest,numFeat,accuracy,nameDataset,):
    from sklearn import metrics
    featureRes = np.reshape(list(feature), (len(feature),1 ))
    clusterLab = []
    for x in range(0, len(feature)):
        indexClust = 0
        for clusterInd in listOfCommFindTest:
            if x in clusterInd["cluster"]:
                clusterLab.append(indexClust)
                break
            else:
                indexClust += 1


    silhouette_avg = metrics.silhouette_score(featureRes, clusterLab)
    calinski_harabasz_score = metrics.calinski_harabaz_score(featureRes, clusterLab)
    davies_bouldin_score = metrics.davies_bouldin_score(featureRes, clusterLab)
    print(feat)
    print("Silhouette meglio quando è 1, quando è 0 indica overlapping: " + str(silhouette_avg))
    print("calinski_harabasz_score, più alto è meglio è : " + str(calinski_harabasz_score))
    print("davies_bouldin_score, meglio quando vicino a 0: " + str(davies_bouldin_score))
    print("   ")



    with open("./" + nameDataset + '/SFS/'+nameDataset+'RankAlgorithm.csv', 'a') as csvfile:
        filewriter = csv.writer(csvfile, delimiter='#', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([feat, str(numFeat), silhouette_avg, calinski_harabasz_score,davies_bouldin_score,accuracy])
