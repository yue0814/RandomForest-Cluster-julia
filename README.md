Code Overview
========
Algorithm simulation
--------
#### 1. Julia packages


```julia
using DataFrames
using PyPlot
using StatsBase
using Distributions
using RCall
using Clustering
using DecisionTree
using MultivariateStats
using Distances
```

Since the "RandomForests.jl" and "Survival.jl" packages cannot be used, we choose "RCall.jl" to achieve random forest and survival anaylysis with R 3.3.3 environment.

<code>R"library(randomForest)"</code> 

<code>R"library(survival)"</code>

<code>R"library(MASS)"</code>  

<code>R"library(cluster)"</code>

#### 2. Distributions to obtain pseudo-labeled data directly from the given dataset
* Addcl1: Synthetic data are added by randomly sampling from the product of empirical marginal distributions of the variables.

<code>sample(X, length(X), replace = true)</code>

* Addcl2: Synthetic data are added by randomly sampling from the hyperrectangle that contains observed data, that is, with random uniform distribution of each columns.

<code>rand(Uniform(minimum(X), maximum(X)), length(X))</code>

#### 3. Random Forest predictor to classify "original" and "synthetic" data
number of trees: 2000
number of forests: 100
<code>RF1 = R"randomForest(factor(yy)~.,data=datRFsyn[,-1], ntree=no_tree, proximity=TRUE,do.trace=F,mtry=mtry1)"</code>

<code>RF2 = R"randomForest(factor(yy)~.,data=datRFsyn[,-1], ntree=no_tree, proximity=TRUE,do.trace=F,mtry=mtry1)"</code>

#### 4. Calculate the average Random Forest dissimilarity 
<code>distRF[:"addcl1"] = sqrt(cleandist(1.0 - RFproxAddcl1/no_forest))</code>

<code>distRF[:"addcl2"] = sqrt(cleandist(1.0 - RFproxAddcl2/no_forest))</code>

Euclidean Approximation

<code>cl1 = distRF[:"addcl1"]</code>

<code>cl2 = distRF[:"addcl2"]</code>

<code>@rput cl1</code>

<code>@rput cl2</code>

<code>R"d1 <- as.dist(cl1)"</code>

<code>R"d2 <- as.dist(cl2)"</code>

Six different kinds of clustering
----------
* #### k-means

<code>label1 = kmeans(cl1, 2).assignments</code>

<code>label2 = kmeans(cl2, 2).assignments</code>

* #### k-medoid (partitioning around medoid function which corrects the clustering membership based on the sillhouette strength.)

<code>R"pam1 <- pam(cmd1,3,diss=T,metric=\"euclidean\")"</code>

<code>R"silinfo1 <- pam1$silinfo$widths"</code>

<code>R"index1 <- as.numeric(as.character(row.names(silinfo1)))"</code>

<code>R"silinfo2 <- silinfo1[order(index1),]"</code>

<code>R"label3 <- ifelse(silinfo2[,3]<0, silinfo2[,2], silinfo2[,1])"</code>


Result visualization & analysis
--------

* Kaplan-Meier Plot with R survival package

<code>R"par(mfrow=c(2,3))"</code>

<code>R"plot(survfit(T ~ label1, data = dat), conf.int=F, col=c(1:2), xlab = c(\"Time to death (years)\"), ylab = c(\"Survival\"), main = c(\"K-M curves\"))"</code>

* Clustering Result Visualization with Multidimension Scaling 

<code>R"plot(cmd1, type = \"n\", xlab = \"Scaling Dimension 1\", ylab = \"Scaling Dimension 2\")</code>

<code>text(cmd1, label = ifelse(dat$labelRF==1, \"1\", \"2\"), col=label1)"</code>

* Rand index

<code>rand1 = [i for i in randindex(dat[:,13],label1)]</code>

<code>Rand1 = (rand1[1]+rand1[2])/sum(rand1)</code>



### Introduction

In the machine learning or data mining domain, the types of data are categorized as ‘labeled’ or ‘unlabeled’. In practice, however, there are many obstacles when collecting patient labels because of the limitations of time, cost, and confidentiality conflicts. Therefore, researchers have been attracted to predictive models that can also utilize unlabeled patient data, which are relatively more abundant. It may be possible to obtain more labeled data by generating labels for unlabeled data and treating them as if they were labeled. These may be referred to as **_‘pseudo-labeled’_** data. Note that labeled and unlabeled patient data are obtained directly from a given dataset, whereas pseudo-labeled data are generated artificially by the proposed model in the paper. This is the motivation of our study.
Tumor marker expressions which are often scored by the percentage of cells staining, can have semicontinuous, highly skewed distributions since many observations may take on the value 0 or 100 percent. 
![Marker](https://github.com/jerry0814/juliagit/blob/master/Marker.PNG)

### Method 
We will use a dataset with 366 patients and 8 tumor markers expressions levels. It also has three extra columns, the event, survival time and optimal labels used to result analysis. In this project, we will first create two labels, “original” and “synthetic”, by sampling a same-size dataset from original dataset based on specific distribution.
In my project, the following distributions will be used in synthetic process:

Addcl1: Synthetic data are added by randomly sampling from the product of empirical marginal distributions of the variables.

Addcl2: Synthetic data are added by randomly sampling from the hyperrectangle that contains observed data, that is, with random uniform distribution of each columns.

Then, I use the proximity matrices generated from a Random Forest Predictor that distinguishes “original” from “synthetic” data to calculate the Random Forest dissimilarity.
After using euclidean approximation for the dissimilarity matrix, we will choose one multidimensional scaling (MDS) to visualize the dissimilarities in a low-dimensional space.

### Results
Addcl1 and Addcl2 dissimilarities are used as input of k-means clustering, then we will obtain two labels. Also, Addcl1 and Addcl2 dissimilarities are used as input of classical MDS and isotonic MDS, then we will use a new PAM clustering function which corrects the clustering membership based on the silhouette strength to create four clustering result labels. After doing visualizations, the K-M curves for label 1 is closest to the optimal label in the dataset. Also, the label 1 has highest Rand index so that we choose it to plot out the clustering result.

### Discussion
Unsupervised machine learning is the machine learning task of inferring a function to describe hidden structure from "unlabeled" data, such as clustering. Random Forest is a supervised learning method which handles highly skewed variables well. However, this dataset used for clustering is unlabeled. Random Forest dissimilarity which has been found to be useful for tumor class discovery was used as input in partitioning around medoid (PAM) and Kmeans clustering to group the patients into clusters. Using Kaplan-Meier Plot to visualize the survival time distributions of tumor sample groups and Rand Index to measure the similarity between two data clusterings. Since the Survival and randomForest Package cannot be used in Julia, we use RCall Package to run the survival analysis and random forest predictor in R environment. With a hundred of forests and two thousand trees, using the “for” loop in Julia Language is faster than running in R.
