using DataFrames
using PyPlot
using StatsBase
using Distributions
using RCall
using Clustering
using DecisionTree
using MultivariateStats
using Distances
# using RandomForests
# using Survival
R"library(randomForest)"
R"library(survival)"
R"library(MASS)"
R"library(cluster)"


dat = readtable("testData.csv")
datRF = dat[:,2:9]

# Ploting the histograms for each Markers score
n_row, n_col = 2, 4
fig, axes = subplots(n_row, n_col, figsize=(16, 6))
subplot_num = 0
for i in 1:n_row
	for j in 1:n_col
		ax= axes[i, j]
		subplot_num += 1
		ax[:hist](datRF[:, subplot_num], alpha = 0.6)
		ax[:set_title]("Marker $subplot_num")
        ax[:set_xlabel]("Score in %")
	end
end	
# savefig("/Users/PY/GoogleDrive/Spring2017/PHP2561/juliagit/Marker.PNG")


# Random Forest dissimilarity

# synthetic label1
synthetic1 = function(dat)

    sample1 = function(X)
        sample(X, length(X), replace = true)
    end

    g1 = function(dat)
        result = sample1(dat[:,1])
        for i in 2:size(dat)[2]
            result = hcat(result, sample1(dat[:,i]))
        end
        return result
    end

    nrow1 = size(dat)[1]
    @rput nrow1
    R"yy <- rep(c(1,2),c(nrow1, nrow1))"
    @rget yy
    return hcat(yy, vcat(Array(dat), g1(dat)))
end


# synthetic label2
synthetic2 = function(dat)
        
    sample2 = function(X)
        rand(Uniform(minimum(X), maximum(X)), length(X))
    end

    g2 = function(dat)
        result = sample2(dat[:,1])
        for i in 2:size(dat)[2]
            result = hcat(result, sample2(dat[:,i]))
        end
        return result
    end

    nrow1 = size(dat)[1]
    @rput nrow1
    R"yy = rep(c(1,2),c(nrow1, nrow1))"
    @rget yy
    return hcat(yy, vcat(Array(dat), g2(dat)))
end

# Calculate the dissimilarity based on Randome Forest 
nrow1 = size(datRF)[1]
@rput nrow1
R"rep1 <- rep(0, 2*nrow1)"
@rget rep1
RFproxAddcl1 = 0
RFproxAddcl2 = 0
no_tree = 2000
no_forest = 100
mtry1 = 3
@rput RFproxAddcl1
@rput RFproxAddcl2
@rput no_tree
@rput no_forest
@rput mtry1    
    
for i in 0:no_forest
    index1 = sample(collect(1:(2*nrow1)), 2*nrow1, replace = false)
    rep1[index1] = collect(1:(2*nrow1))
    datRFsyn = synthetic1(datRF)[index1,:]
    @rput datRFsyn
    yy = datRFsyn[:,1]
    @rput yy
    @rput mtry1
    @rput no_tree
    RF1 = R"randomForest(factor(yy)~.,data=datRFsyn[,-1], ntree=no_tree, proximity=TRUE,do.trace=F,mtry=mtry1)"
    @rput RF1
    @rput rep1 
    RF1prox = R"RF1$proximity[rep1, rep1]"
    @rput RF1prox
    R"RFproxAddcl1 <- RFproxAddcl1 + (RF1prox[c(1:nrow1), c(1:nrow1)])"
end
@rget RFproxAddcl1


for i in 0:no_forest
    index1 = sample(collect(1:(2*nrow1)), 2*nrow1, replace = false)
    rep1[index1] = collect(1:(2*nrow1))
    datRFsyn = synthetic2(datRF)[index1,:]
    @rput datRFsyn
    yy = datRFsyn[:,1]
    @rput mtry1
    RF2 = R"randomForest(factor(yy)~.,data=datRFsyn[,-1], ntree=no_tree, proximity=TRUE,do.trace=F,mtry=mtry1)"
    @rput RF2
    @rput rep1 
    RF2prox = R"RF2$proximity[rep1, rep1]"
    @rput RF2prox
    R"RFproxAddcl2 <- RFproxAddcl2 + (RF2prox[c(1:nrow1), c(1:nrow1)])"
end
@rget RFproxAddcl2 

# eliminate the negative result
cleandist = function(x)
    @rput x
    R"x1 <- as.dist(x)"
    R"x1[x1 <= 0] <- 0.000000001"
    R"x1 <- as.matrix(x1)"
    @rget x1
    return x1
end

# get the average of dissimilarity matrix
distRF = Dict()
distRF[:"addcl1"] = sqrt(cleandist(1.0 - RFproxAddcl1/no_forest))
distRF[:"addcl2"] = sqrt(cleandist(1.0 - RFproxAddcl2/no_forest))


# Clustering
@rput dat
cl1 = distRF[:"addcl1"]
cl2 = distRF[:"addcl2"]
@rput cl1
@rput cl2

# Euclidean Approximation
R"d1 <- as.dist(cl1)"
R"d2 <- as.dist(cl2)"


# Kmeans in Julia Clustering Package
label1 = kmeans(cl1, 2).assignments
label2 = kmeans(cl2, 2).assignments
@rput label1
@rput label2
R"dat$label1 <- label1"
R"dat$label2 <- label2"



# MDS
R"par(mfrow=c(2,2))"
R"cmd1 <- cmdscale(d1,2)"  # better fit

R"plot(cmd1, main=\"CMD1\")"
R"cmd2 <- cmdscale(d2,2)"  
R"plot(cmd2, main=\"CMD2\")"

R"iso1 <- isoMDS(d1,k=2)$points"
R"plot(iso1, main=\"ISO1\")"
R"iso2 <- isoMDS(d2,k=2)$points"
R"plot(iso2, main=\"ISO2\")"


# A new pam clustering function which corrects the clustering membership based on the sillhouette strength.
# choose to use cmd1 
R"pam1 <- pam(cmd1,3,diss=T,metric=\"euclidean\")"
R"silinfo1 <- pam1$silinfo$widths"
R"index1 <- as.numeric(as.character(row.names(silinfo1)))"
R"silinfo2 <- silinfo1[order(index1),]"
R"label3 <- ifelse(silinfo2[,3]<0, silinfo2[,2], silinfo2[,1])"
@rget label3
label3 = floor(Int64, label3)

R"pam1 <- pam(cmd2,3,diss=T,metric=\"euclidean\")"
R"silinfo1 <- pam1$silinfo$widths"
R"index1 <- as.numeric(as.character(row.names(silinfo1)))"
R"silinfo2 <- silinfo1[order(index1),]"
R"label4 <- ifelse(silinfo2[,3]<0, silinfo2[,2], silinfo2[,1])"
@rget label4
label4 = floor(Int64, label4)


R"pam1 <- pam(iso1,3,diss=T,metric=\"euclidean\")"
R"silinfo1 <- pam1$silinfo$widths"
R"index1 <- as.numeric(as.character(row.names(silinfo1)))"
R"silinfo2 <- silinfo1[order(index1),]"
R"label5 <- ifelse(silinfo2[,3]<0, silinfo2[,2], silinfo2[,1])"
@rget label5
label5 = floor(Int64, label5)


R"pam1 <- pam(iso2,3,diss=T,metric=\"euclidean\")"
R"silinfo1 <- pam1$silinfo$widths"
R"index1 <- as.numeric(as.character(row.names(silinfo1)))"
R"silinfo2 <- silinfo1[order(index1),]"
R"label6 <- ifelse(silinfo2[,3]<0, silinfo2[,2], silinfo2[,1])"
@rget label6
label6 = floor(Int64, label6)


# visualization
R"plot(cmd1, type = \"n\", xlab = \"Scaling Dimension 1\", ylab = \"Scaling Dimension 2\")
text(cmd1, label = ifelse(dat$labelRF==1, \"1\", \"2\"), col=label1)"

R"plot(cmd1, type = \"n\", xlab = \"Scaling Dimension 1\", ylab = \"Scaling Dimension 2\")
text(cmd1, label = ifelse(dat$labelRF==1, \"1\", \"2\"), col=label3)"


R"dat$label3 <- label3"
R"dat$label4 <- label4"
R"dat$label5 <- label5"
R"dat$label6 <- label6"


# Result Analysis
# Rand index
# The function computes the Rand index between two partitions


rand1 = [i for i in randindex(dat[:,13],label1)]
Rand1 = (rand1[1]+rand1[2])/sum(rand1)

rand2 = [i for i in randindex(dat[:,13],label2)]
Rand2 = (rand2[1]+rand2[2])/sum(rand2)

rand3 = [i for i in randindex(dat[:,13],label3)]
Rand3 = (rand3[1]+rand3[2])/sum(rand3)

rand4 = [i for i in randindex(dat[:,13],label4)]
Rand4 = (rand4[1]+rand4[2])/sum(rand4)

rand5 = [i for i in randindex(dat[:,13],label5)]
Rand5 = (rand5[1]+rand5[2])/sum(rand5)

rand6 = [i for i in randindex(dat[:,13],label6)]
Rand6 = (rand6[1]+rand6[2])/sum(rand6)

Rand_index = [Rand1,Rand2,Rand3,Rand4,Rand5,Rand6]
for i in eachindex(Rand_index)
    print("Rand index for label$(i) is $(round(Rand_index[i],2))\n")
end

# K-M Plots in R
R"dat$T <- Surv(dat$time, dat$event)"

R"plot(survfit(T ~ labelRF, data = dat), conf.int=F, col=c(1:2), xlab = c(\"Time to death (years)\"), ylab = c(\"Survival\"), main = c(\"Best K-M curves\"))"

R"par(mfrow=c(2,3))"
# Best model Fitting
R"plot(survfit(T ~ label1, data = dat), conf.int=F, col=c(1:2), xlab = c(\"Time to death (years)\"), ylab = c(\"Survival\"), main = c(\"K-M curves\"))"
R"plot(survfit(T ~ label2, data = dat), conf.int=F, col=c(1:2),, xlab = c(\"Time to death (years)\"), ylab = c(\"Survival\"), main = c(\"K-M curves\"))"
R"plot(survfit(T ~ label3, data = dat), conf.int=F, col=c(1:2),, xlab = c(\"Time to death (years)\"), ylab = c(\"Survival\"), main = c(\"K-M curves\"))"
R"plot(survfit(T ~ label4, data = dat), conf.int=F, col=c(1:2),, xlab = c(\"Time to death (years)\"), ylab = c(\"Survival\"), main = c(\"K-M curves\"))"
R"plot(survfit(T ~ label5, data = dat), conf.int=F, col=c(1:2),, xlab = c(\"Time to death (years)\"), ylab = c(\"Survival\"), main = c(\"K-M curves\"))"
R"plot(survfit(T ~ label6, data = dat), conf.int=F, col=c(1:2),, xlab = c(\"Time to death (years)\"), ylab = c(\"Survival\"), main = c(\"K-M curves\"))"

