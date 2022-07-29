################################################################################
###                                                                          ###
#                           Raster Band Stack within R                         #
###                                                                          ###
################################################################################

################################################################################
###
#                           Visulizing Data within R
##
################################################################################


install.packages("rgdal")
install.packages("raster")
install.packages("sp")
install.packages("sf")
library(sp)
library(raster)
library("sf")
#The raster package offers '
#layer() - one file, one band
#stack() - multiple files, multiple bands
#brick() - one file, multiple bands

library(raster)
path <- "C:\\Users\\kobus\\Desktop\\RRProject\\RiverRidgeDeadTreeDL\\Mask_MosaicRanch05042020.tif"

img <- brick(path)
img

plotRGB(img,
        r = 3, g = 3, b = 3,
        stretch = "lin"
)

plotRGB(img,
        r = 3, g = 2, b = 1,
        stretch = "lin"
)

green <- img[[2]]

hist(green,
     breaks = 200,
     xlim = c(0, 2000),
     ylim = c(0, 120000),
     xlab = "band 2 reflectance value [DN * 0.01]",
     ylab = "frequency",
     main = "histogram MIca sense Red-Edge  MX band 2 (green)"
)


################################################################################
###                                                                          ###
#                           Preparing Classification Samples                   #
###                                                                          ###
################################################################################
library(rgdal)
library(sf)
library(sp)
library(raster)

setwd("C:\\Users\\kobus\\Desktop\\RRProject\\RRRImagery\\RRRImagery\\06132021\\Block1\\Programming")

getwd()
#Tells what is in the folder directory
dir()

#Load the Image
img <- brick("Mask_MosaicRanch05042020.tif")
img

shp <- shapefile("TrainingData_050420.shp")
shp

#Check CRS
#Comparing if corrdinate systems are matching
compareCRS(img,shp)

#Plot RGB, While also plotting the training data as a red shapefile over the top of the RGB image.
plotRGB(img, r = 3, g = 2, b = 1, stretch = "lin")
plot(shp, col="red", add=TRUE)

################################################################################
###                                                                          ###
#                       Preparing Training Data for Classification             #
###                                                                          ###
################################################################################

#Conversion of class-characters
#needs to happen because you can only do analysis with int data
levels(as.factor(shp$Classvalue))

for (i in 1:length(unique(shp$Classvalue))) {cat(paste0(i, " ", levels(as.factor(shp$Classvalue))[i]), sep="\n")}

#Rename bands of UAV image
names(img)
names(img) <- c("b1", "b2", "b3", "b4", "b5")
names(img)

#Create dataframe
smp = extract(img, shp, df = TRUE)

#Matching ID of smp and class of shp to new column "cl", delete "ID"-column
smp$cl <- as.factor(shp$Classvalue[match(smp$ID, seq(nrow(shp)))])
smp <- smp[-1]

#Save dataframe to wd as 
save(smp, file = "smp2.rda")

#Load datafram from your wd
# this process makes it much faster to reload for future analysis
load(file = "smp2.rda")

#Check out the summary of the class-column smp
summary(smp$cl)
str(smp)

################################################################################
###                                                                          ###
#                       Spectral Profile Visualization                         #
###                                                                          ###
################################################################################




#Aggregate cl-column, This combines all of the rows of the same class in the column cl, then calculates the mean
sp <- aggregate( . ~ cl, data = smp, FUN = mean, na.rm = TRUE )

#Plot empty plot of a defined size
plot(0,
     ylim = c(min(sp[2:ncol(sp)]), max(sp[2:ncol(sp)])), 
     xlim = c(1, ncol(smp)-1), 
     type = 'n', 
     xlab = "UAV Imagery", 
     ylab = "reflectance [% * 100]"
)

#Define colors for class representation
mycolors <- c("#734C00", "#38A800", "#7e7e7e", "#CDAA66", "#b7ffba", "#000000")

#Draw one line for each class, plotting a visualization of the spectral signatures for the 
for (i in 1:nrow(sp)){
  lines(as.numeric(sp[i, -1]), 
        lwd = 4, 
        col = mycolors[i]
  )
}

#Add a grid
grid()

#Add a legend
legend(as.character(sp$cl),
       x = "topleft",
       col = mycolors,
       lwd = 4,
       bty = "n"
)

################################################################################
###                                                                          ###
#                       Classification: Random Forest (RF)                     #
###                                                                          ###
################################################################################

### https://www.rdocumentation.org/packages/randomForest/versions/4.6-14/topics/randomForest  ###
### https://cran.rstudio.com/web/packages/randomForestExplainer/vignettes/randomForestExplainer.html  ###

#install additional packages
#Should be able to assist in helping with over sampling through the process of the tree structures
install.packages("randomForest")
library(randomForest)

#Access training samples using the cl column that contains integars for class names of each sample/variable
summary(smp$cl)

#Down-Sampling via minority class and identify the number of available training samples per class
smp.size <- rep(min(summary(smp$cl)), nlevels(smp$cl))
smp.size

#Hyperparameter tuning uses tune() to perform a grid search over specified parameter ranges
#doBest to run a forest using the optimal geometry found
#Out-of-Bag score are the samples not used in the model, but are used almost as truthing data but not validation
#plot to plot the OOB error as function of geometry
rfmodel <- tuneRF(x = smp[-ncol(smp)],
                  y = smp$cl,
                  sampsize = smp.size,
                  strata = smp$cl,
                  ntree = 250,
                  importance = TRUE,
                  doBest = TRUE,                  
                  plot = TRUE                     
)

# OBB class error is calculated by counting however many points in the training set were misclassified and dividing this number by the total number of observation

#Print useful information about our model
rfmodel

#Access/Plot importance variables
varImpPlot(rfmodel)

#Plot rfmodel
plot(rfmodel, col = c("#734C00", "#38A800", "#7e7e7e", "#CDAA66", "#b7ffba", "#000000"), lwd = 3)

#Save model-file
save(rfmodel, file = "rfmodel.RData")

#Load model-file
load("rfmodel.RData")

#Predict all pixels/run classification
#Using this method, we obtain predictions from the model, as well as decision values from the binary classifiers.
result <- predict(img,
                  rfmodel,
                  filename = "RFclassification.tif",
                  overwrite = TRUE
)

#Plot Classification
plot(result, 
     axes = FALSE, 
     box = FALSE,
     col = mycolors
)


################################################################################
###                                                                          ###
#                       Classification: Support Vector Machine (SVM)           #
###                                                                          ###
################################################################################

###               https://data-flair.training/blogs/e1071-in-r/              ###

#Install required packages
install.packages("e1071")
library(e1071)

#Load dataframe from your wd
load(file = "smp2.rda")

#Get distribution of available training samples
summary(smp$cl)

#Print head of smp
head(smp)

#Shuffle/Sample all rows of smp
smp <- smp[sample(nrow(smp)),]

#Print head of smp
head(smp)

#Get distribution of available training samples
summary(smp$cl)

#Get min value to determine max sample-size
smp.maxsamplesize <- min(summary(smp$cl))
smp.maxsamplesize

#Select each class in the same size via smp.maxsamplesize
smp <- smp[ave(1:(nrow(smp)), smp$cl, FUN = seq) <= smp.maxsamplesize, ]

#Get distribution of available training samples
summary(smp$cl)

#Find hyperparameters
#All the kernels except the linear one require the gamma parameter
gammas = 2^(-8:5)
gammas


#The cost of constraints violation (default: 1)—it is the 
#‘C’-constant of the regularization term in the Lagrange formulation.
costs = 2^(-5:8)
costs

#Run SVM-Classification, Hyperparameter tuning uses tune() to perform a grid search over specified parameter ranges
svmgs <- tune(svm,
              train.x = smp[-ncol(smp)],
              train.y = smp$cl,
              type = "C-classification",
              kernel = "radial", 
              scale = TRUE,
              ranges = list(gamma = gammas, cost = costs),
              tunecontrol = tune.control(cross = 5)
)

#Check output
svmgs

#Plot gridsearch
plot(svmgs)

#Extract best model
svmmodel <- svmgs$best.model
svmmodel

#Save/load file
save(svmmodel, file = "svmmodel.RData")
#load("svmmodel.RData")

#Using this method, we obtain predictions from the model, as well as decision values from the binary classifiers.
#Predict entire dataset
result <- predict(img,
                  svmmodel,
                  filename = "SVM_classification.tif",
                  overwrite = TRUE
)

#Visualize data in a plot
#Visualizing data, support vectors and decision boundaries, if provided.
plot(result, 
     axes = FALSE, 
     box = FALSE,
     col = mycolors
)

################################################################################
###                                                                          ###
#                           Deep Learning Analysis                            #
###                                                                          ###
################################################################################

##Setting Up the environment
install.packages("keras")
install.packages("tensorflow")
install.packages("tfdatasets")
install.packages("purr")
install.packages("ggplot2")
install.packages("rsample")
install.packages("raster")
install.packages("stars")
install.packages("reticultae")
install.packages("mapview")
install.packages(c("keras","tfdatasets","mapview","stars","rsample","gdalUtils","purrr", "magick", "jpeg"))
library(keras)
library(tensorflow)
library(tfdatasets)
library(purrr)
library(ggplot2)
library(rsample)
library(stars)
library(raster)
library(reticulate)
library(mapview)
install_keras()
install_tensorflow()

input_img <- stack("falsecolor.tif")
training1 <- stack("falsecolorTrainingData.tif")

#initiate an empty model
first_model <- keras_model_sequential()
#add first layer, the expected input is of shape 128 by 128 on three channels (we will be dealing with RGB images)
#I made a false color image of 5.2020 with on NIR, RED, n BLUE
layer_conv_2d(first_model,filters = 32,kernel_size = 3, activation = "relu",input_shape = c(128,128,3))

#Plot the summary of the model
summary(first_model)


layer_max_pooling_2d(first_model, pool_size = c(2, 2)) 
layer_conv_2d(first_model, filters = 64, kernel_size = c(3, 3), activation = "relu") 
layer_max_pooling_2d(first_model, pool_size = c(2, 2)) 
layer_conv_2d(first_model, filters = 128, kernel_size = c(3, 3), activation = "relu") 
layer_max_pooling_2d(first_model, pool_size = c(2, 2)) 
layer_conv_2d(first_model, filters = 128, kernel_size = c(3, 3), activation = "relu")
layer_max_pooling_2d(first_model, pool_size = c(2, 2)) 
layer_flatten(first_model) 
layer_dense(first_model, units = 256, activation = "relu")
layer_dense(first_model, units = 1, activation = "sigmoid")
######################################
#ADDITIONAL NOTATION
######################################
#ALL Funcitons for plot()

#x – An object of class svm.
#Formula – Formula selecting the visualized two dimensions. Only needed, when we use more than two input variables.
#Fill – Switch indicating whether a contour plot for the class regions should be added.
#Grid – Granularity for the contour plot.
#Slice – A list of named numeric values for the dimensions are held constant. If dimensions are not specified, we can fix it at 0.
#Model – Represents an object of class svm data, resulting from the svm() function.
#Data – Represents the data to visualize. It should use the same data used for building the model in the svm() function.
#symbolPalette – Color palette used for the class the data points and support vectors belong to.
#svSymbol – Symbol used for support vectors.
#dataSymbol – Symbol used for data points (other than support vectors).

#Keras functions
#layer_conv_2d(): adds a convolutional layer
#layer_dense(): adds a dense layer, i.e. fully connected layer
#layer_max_pooling_2d(): adds a maxpooling layer
#layer_flatten(): adds a flattening layer

