% Project Iota Progress Report
% Zeyu Li, Jie Li, Qingyuan Zhang, Chuan Yun 
% November 12, 2015


## The Paper

We choose "Working memory in healthy and schizophrenic individuals" from OpenFMRI.org, and we are interested in functional brain connectivity, which is known as neural networks. Here are the 4 particular regions of interest (ROIs) in the paper:

- Dorsal fronto-parietal network (FP).
- Cingulo-opercular network (CO).
- Cerebellar network (CER).
- "Default mode" network (DMN).

They showed that individuals with schizophrenia have reduced connectivity between neural networks.


## The Data

The data for this paper has 102 subjects. For each subject, they used fMRI scanning to find blood oxygen level dependent (BOLD) during resting state (R), and after 0-back (0B), 1-back (1B) and 2-back (2B) working memory task. Subjects are in four groups: 

- Individuals with schizophrenia (SCZ).
- Siblings of schizophrenia (SCZ-SIB).
- Healthy controls (CON).
- Siblings of controls (CON-SIB).

Since we don't have time to go through all of the subjects, we choose the first subject from SCZ group. If we have time, we will work on another subject from CON group and do a comparison between these two.


## Data Processing

We downloaded data from OpenFMRI.org and looked through all the files it has. We found that "ds115_sub001-005.tgz" contains the brain image of the subject we chose. 

- First, we checked all the files in it and found there are three tasks for our subject. 
- Second, we found that under each task there are four brain images. We thought one of them should be the raw image and the others were some-how processed by the authors. 
- Then, we plotted these four brain images and chose the most blurred one (This is probably because it is after smoothing), which is ``BOLD_mcf_brain.nii''. 


## Data Processing

![Raw Data Brain Image](../code/utils/sample_image.png)


## Data Processing

![Single Voxel Behavior](../code/utils/single_voxel_behavior.png)


## Convolving with the Hemodynamic Response

Similar to what we did in class, we used HRF function to convolve our neural prediction, This will give us a hemodynamic prediction, under the linear-time-invariant assumptions of the convolution. This is a very important step because we will use convolution data in our linear modeling process in the future. Here is the comparison between the original neural prediction and convolved.


## Convolving with the Hemodynamic Response

![Neural Prediction vs Convolved](../data/convo/task001_run001_conv005.png)


## Simplified Objectives

The paper's objective is finding connectivity within and between each ROI on different tasks. They extracted these ROIs and worked directly on them. Since we do not have the knowledge about partition the brain into ROIs, we could not do the same analysis as them. Therefore, we will simply focus on the entire brain and find the region related to these tasks. 


## Statistical Analysis: Linear Modeling

- Data cleaning: focusing on cleaning outliers in order to keep normality assumption

- Perform a statistical test (p-value map) to determine whether a task related blood pressure in the each slice

- Non-constant variance of error: construct a generalized linear regression (WLS) to weighted different variance for each voxel of the brain


## Statistical Analysis: Linear Modeling

![P-value Map](p_value_map.png)


## Statistical Analysis: Sparse Inverse Covariance

The matrix inverse of the covariance matrix (the precision matrix) is proportional to the partial correlation matrix. It gives the partial independence relationship. In other words, if two features are independent conditionally on the others, the corresponding coefficient in the precision matrix will be zero. By learning independence relations from the data, the estimation of the covariance matrix is better conditioned. 

Since the estimate of covariance matrix of beta_hat might not be precious, so we think sparse inverse covariance should be a good way to solve this issue.


## Statistical Analysis: Time Series

- Based on the beta_hat generated from linear regression, we are finding the sub-area has the significantly effect between blood pressure and nerual signal.

- Using differecing method to remove the noise between each lag and plot the ACF and PACF to understand how the sub-area works when time is moving on.

- Using boostraping method to simulate our samples from the sub-area, and develop the time series model (ARMA, ARIMA) to predict future blood pressure.


## Our Process

1. Task assignment
2. Git workflow, python
3. Issues that we faced and how we addressed them 
	- Data organization
	- fMRI 
	- Coding multiple regression model and tests 





