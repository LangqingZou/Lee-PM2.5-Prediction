# Introduction
This is an assignment of Mechine Learnign class provided by Hongyi Lee in YouTube: https://www.youtube.com/playlist?list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49 <br>

The train set is the data for the first 20 days of each month, and the test set is a sample of the remaining data from the Fengyuan District, Taiwan, China.<br>
**train.csv**: 12 months of meteorological data for each hour of the first 20 days of each month (18 features per hour) <br>
**test.csv**: The remaining data are sampled for 10 consecutive hours, the first nine hours of all observations are treated as features, and the tenth hour of PM2.5 is treated as answers. 240 non-repeated test data are taken, and the PM2.5 of these 240 data are predicted according to the linear model I created.<br>


# How to run it
pandas, matplotlib, sklearn are needed. <br>
Prefer to run on the Jupyter Notebook.

# Discussion
**Here I try two versions of features selected.**

### According to the heapmap, I selected PM2.5, PM10, NO2 as features predicted.
<img width="602" alt="image" src="https://user-images.githubusercontent.com/55254825/147159937-dce84995-f576-4d95-8ed1-50327c090cff.png">

After 100 epoches and learning rate = 0.0001, the loss value on test set is 4.973021068410819e+299.

The details of version one could be found in main_version1.ipynb

### I try to select only PM2.5 as a feature predicted

Still need to be finished. 

The details of version two could be found in main_version2.ipynb

# Improvement
In the next step, I will try to normalize data.
