# Clustering using Supervised Features Selection

## Running 

In the `testFeatureExtraction.py` we can found the main file where we can set the parameter for launch the code. 

Being Supervised, it isn't necessery specify the Supervised Method for choice the features.

```python
# Choice of the number of features to use
numberFeatUse = 20
featuNum = [x for x in range(1,numberFeatUse+1)]

# Name of the dataset
datasetUsed = ["Coffee"]

# Percentage for Cross Validation
trainPerc = 0.5

# Threshold of the distance
threshold = 0.8

# Choice of the algorithm (Greedy Algorithm default)
chooseAlgorithm = 0
```

## Configuration File

For test some other dataset it's very important to create a *.tsv* file where the first column will be the class of the time series
and then all the points of the latter:
<table>
  <tr>
    <th>Classe</th>
    <th>1</th>
    <th>2</th>
    <th>3</th>
    <th>4</th>
    <th>5</th>
    <th>...</th>
    
  </tr>
  <tr>
    <td>0</td>
    <td>2.5</td>
    <td>2.8</td>
    <td>2.2</td>
    <td>2.1</td>
    <td>3.8</td>
    <td>...</td>
  </tr>
  
  <tr>
    <td>1</td>
    <td>10.5</td>
    <td>12.1</td>
    <td>11.2</td>
    <td>10.3</td>
    <td>14.8</td>
    <td>...</td>
  </tr> 
  
  <tr>
    <td>0</td>
    <td>1.5</td>
    <td>1.9</td>
    <td>2.2</td>
    <td>2.9</td>
    <td>3.3</td>
    <td>...</td>
  </tr> 
  <tr>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
    <td>...</td>
  </tr> 
</table>

The `.tsv` should have the same name of the folder where it is contained, so if for example the name is `dataset.tsv` it should be in
the folder named `dataset`. And for test the code just put `dataset` in this way:
```python
# Name of the dataset
datasetUsed = ["dataset"]
```

## Output File

The file of output will be in the folder `./dataset/SFS/...` of the dataset chosed to test. The name will be 
<p align="center">
<i> nameDataset + `RankAlgorithm.csv`* </i>
</p>

This file will be the input for the [Ranking Algorithm](https://github.com/DonaTProject/RankingAlgorithm) 
