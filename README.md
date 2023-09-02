# KNN Covid Analysis
Project Presentation: https://docs.google.com/presentation/d/1NUrXpARbfYUEleM5kn4CVs9BD6nU360qjHn2lP3g0t0/edit?usp=sharing

# I. Abstract
The COVID-19 pandemic has been an unprecedented global health crisis, affecting millions of people worldwide. The virus responsible for the pandemic, SARS-CoV-2, was first identified in December 2019 and has since spread rapidly, causing significant mortality. Given the critical circumstances of this pandemic, identifying medical conditions that indicate a higher risk of death from COVID-19 is extremely important. Our research paper presents a study on the use of the K-Nearest Neighbors (KNN) algorithm to cluster patients with COVID-19 based on their medical information and predict their classification of death(lived or deceased). Moreover, we utilized KNN’s feature importance to identify the most important attribute that correlates with the death of a patient. Our KNN model resulted in a 0.713 percent accuracy in predicting whether or not a patient will die based on their medical information. Finally, we determined that Age and Pneumonia are the most significant health conditions in determining whether or not a patient will survive their infection.
_Keywords_: Covid-19, KNN, Feature importance.


# II. Introduction
COVID-19, the coronavirus, is a highly infectious disease caused by a virus, SARS-CoV-2, and was initially discovered in Wuhan, China in 2019. According to the World Health Organization(WHO), this disease has led to more than 6.8 million deaths globally, with the United States alone with roughly over 1 million deaths. The pandemic has had a significant impact on the world population, resulting in a widespread lockdown in 2019 to 2020. The long-term effects of COVID-19 are serious, including organ damage affecting the heart, kidneys, skin, and brain. Due to various factors such as previous health conditions, various strands, treatment, and hospitalization conditions, patients with of COVID-19 respond differently to the illness. As of 2023, COVID-19 continues to pose a significant global threat, with new cases continuously being reported. 

The motivation for our research is to be able to adequately analyze and predict COVID-19 patient data, specifically the health conditions of those patients who lived and those who died. This research topic has personal significance to the researchers because we have personally been diagnosed with COVID-19 and have experienced both immediate and long-term symptoms. Furthermore, family members from China, including elderly relatives, have suffered severe health consequences from COVID-19. Over 1.1 million people have died from COVID-19 in the United States, and 6.86 million people worldwide. Being able to accurately explore and predict this data can allow for tailored and improved treatment, care, and sufficient supervisions of current and future COVID-19 patients in hospitals. The significance in this research lies specifically in the death prediction given the specific attributes of each patient. Moreover, we want to be able to identify health conditions that may signal higher risk of mortality. We will do so using the KNN algorithm to cluster patients in our dataset and generate a KNN predictive model to predict whether or not a patient will die based on their pre-existing and current medical conditions. Moreover, we will utilize the Standard deviation feature importance score to determine the relative importance of each feature in a K-Nearest Neighbors (KNN) model.

There is no one-size-fits all treatment for numerous patients with COVID-19, thus analyzing this dataset with over 1 million observations will allow researchers to learn more about the various health and medical factors that highly affect or do not affect COVID-19 deaths. Our long-term goal of our research is to be able to alleviate the consequences from this illness and improve the understanding of COVID-19 deaths. 


 
## Research Questions
The research questions that lie at the foundation of our investigation and guide our study are as follows: 
  1.  Can we use the full set of medical information about patients to predict whether the individual will live or die? 
  2.  Can we use information about the patients’ death and medical information to identify the most common disease of those who died from COVID-19?
As such, these research questions serve as the driving factor for our analysis which will be further discussed later in the paper.


# III. Materials and Methods
Exploratory Data analysis and cleaning were done using the Pandas library. All data pre-processing was performed in Jupyter Notebook, an open-sourced integrated development environment (IDE). Libraries that we utilized include Sklearn, Matplotlib, Pandas, numpy, and Seaborn (Figure 1). 

<img width="592" alt="Screenshot 2023-09-02 at 10 30 52 AM" src="https://github.com/WilliamChenn/Covid-Project/assets/85557718/fd94ac29-db65-472b-8691-914a97b100f1">

**Figure 1:** _imported libraries_

Methods involved in our procedure were first cleaning the data by filtering out the unnecessary attributes in the dataset with the drop method in Python. Next, we utilized Matplotlib in the descriptive analysis section of our research, in which we bar graphs, pie charts, box and whisker plots and correlation heatmaps to identify any possible trends in the data. Finally, we implemented machine learning with K-Nearest Neighbors for the predictive analysis section, where we determined the predictability and accuracy of our statistical models. These methods will be covered more in-depth in the upcoming sections.


## Description of the Dataset
The datasets utilized for our research investigation were provided by the Mexican government and contain anonymized patient-related information including pre-existing conditions. The raw dataset consists of 21 unique attributes and 1,048,576 observations, with each observation representing a single patient.  Most of the attributes contain binary data, with zeros representing “no” and ones representing “yes.” Null values or missing values are denoted with a 97 and 99. Below is a detailed description provided by Kaggle on each attribute of the dataset and the meaning of the values. 

- Sex: 1 for female and 2 for male.
- Age: of the patient.
- Classification: Covid test findings. Values 1-3 mean that the patient was diagnosed with covid in different degrees. 4 or higher means that the patient is not a carrier of covid or that the test is inconclusive.
- Patient type: 1 means the patient home and 2 for hospitalization.
- Pneumonia: whether the patient already has air sacs inflammation or not. 
- Pregnancy: whether the patient is pregnant or not.
- Diabetes: whether the patient has diabetes or not.
- Copd: Indicates whether the patient has Chronic obstructive pulmonary disease or not.
- Asthma: whether the patient has asthma or not.
- Inmsupr: whether the patient is immunosuppressed or not.
- Hypertension: whether the patient has hypertension or not.
- Cardiovascular: whether the patient has heart or blood vessels related disease.
- Renal Chronic: whether the patient has chronic renal disease or not.
- Other disease: whether the patient has another disease or not.
- Obesity: whether the patient is obese or not.
- Tobacco: whether the patient is a tobacco user.
- Usmr: Indicates whether the patient treated medical units of the first, 2nd/3rd level.
- Medical unit: type of institution of the National Health System that provided the care.
- Intubed: whether the patient was connected to the ventilator.
- Icu: Indicates whether the patient had been admitted to an Intensive Care Unit.
- Date died: If the patient died, indicate the date of death, and 9999-99-99 otherwise.

<img width="1189" alt="Screenshot 2023-09-02 at 10 28 56 AM" src="https://github.com/WilliamChenn/Covid-Project/assets/85557718/555bba54-5ddd-4e95-acb3-f1f1e47fb84c">

**Figure 2:** _Dataframe.head()_ (first five observations of the preprocessed dataset).

## Exploratory Data Analysis and Visualization

As demonstrated by Figure 3, the histogram diagram demonstrates a right skew for the age distribution. This suggests that the majority of the patients in our dataset fall within the age range of lower to upper adulthood. 

<img width="547" alt="Screenshot 2023-09-02 at 10 32 28 AM" src="https://github.com/WilliamChenn/Covid-Project/assets/85557718/371d2f25-fc5f-411c-8189-42fa482612f2">

**Figure 3:** _Age Distribution Bar Graph_

Figure 4 displays a pie chart illustrating the distribution of patients who survived and those who did not. Based on the diagram, we can infer that the larger proportion of patients in the dataset survived.

<img width="985" alt="Screenshot 2023-09-02 at 10 33 23 AM" src="https://github.com/WilliamChenn/Covid-Project/assets/85557718/ac6fa497-d6a6-442b-9743-8f51d230724b">

**Figure 4:** _Death Distribution Pie Chart_

Figure 5 is a heatmap of variables and attributes for our COVID-19 dataset.
This diagram indicates there are particularly weak relationships between all predictors and DEATH, which is our target variable. The negative correlations indicate that the variable and DEATH have an inverse relationship with each other, while the position correlations indicate that the variable and DEATH have a direct relationship. From the heatmap, we can infer that PATIENT_TIME and PNEUMONIA have the strongest correlations to the variable, DEATH. Moreover, by examining the relationship between the predictors, we were able to drop the predictors that had an extremely weak correlation to DEATH in order to increase the precision of our predictive KNN analysis.

<img width="736" alt="Screenshot 2023-09-02 at 10 34 43 AM" src="https://github.com/WilliamChenn/Covid-Project/assets/85557718/4a3e6d83-d1f9-4d46-bba9-0e3eaf2f5b93">

**Figure 5:** _Correlation Heatmap_

We also generated Death vs. Age Correlation Boxplots to examine the relationship between the two predictors (​​Figure 6). From the diagrams, we can infer that the median age of the deceased patients is greater than that of the surviving patients. Additionally, there are outliers for the patients who passed away. The range of ages for the surviving patients is notably smaller than that of the deceased patients. These boxplots provide a clear representation of the relationship between these two predictor variables.  

<img width="688" alt="Screenshot 2023-09-02 at 10 35 25 AM" src="https://github.com/WilliamChenn/Covid-Project/assets/85557718/bc064fe2-3b53-42e1-a0bb-509e91faf126">

**Figure 6:** _Death vs Age Correlation Boxplots_

## Data Cleaning/Processing
After analyzing the distribution of values for each  predictor, we identified two variables that contained a considerable amount of null values. Figure 7 displays the two bar plots which demonstrate that the majority of these two variables contain only missing values. From this information, we discovered that the majority of ICU and INTUBED are null/missing (denoted by the 97 and 99). Given this, we determined that these variables could not provide an accurate representation of our dataset or patients as a whole, and therefore decided to remove them from the analysis.

<img width="782" alt="Screenshot 2023-09-02 at 10 36 34 AM" src="https://github.com/WilliamChenn/Covid-Project/assets/85557718/9257e56a-cff4-4721-9653-81fae0c0f42e">

**Figure 7:** _Intubed and ICU value distribution_

The following variables were dropped for the low correlation scores based off of our heatmap. We utilized the .drop method in the pandas library to drop all of the attributes we deemed as insignificant (See Figures 5 and 8):

<img width="629" alt="Screenshot 2023-09-02 at 10 37 23 AM" src="https://github.com/WilliamChenn/Covid-Project/assets/85557718/57f51af7-2008-4108-9b28-84c98ec6bf51">

**Figure 8:** _Dropping low correlation attributes._

Finally, normalizing our predictors allows for improvement of the accuracy and precision for our predictions and analysis. After normalizing our data values between 0 and 1 through min max scaling, we plotted the distribution of the values for all our predictors, shown by Figure 9.

<img width="718" alt="Screenshot 2023-09-02 at 10 37 57 AM" src="https://github.com/WilliamChenn/Covid-Project/assets/85557718/964725b8-2f62-47a6-b005-1f0cb6909044">

**Figure 9:** _Dataset value distribution after normalization and scaling_

# KNN Analysis:
For our predictive analysis portion of our research, we utilized the KNN algorithm to cluster and predict whether or not a patient will die based on their list of medical conditions. For the entirety of our investigation, we implemented a train/test split of 80:20 respectively since our dataset contains a relatively large quantity of observations. Furthermore, after the trial of 90:10 and 70:30 splits, which are also conventional train/test splits alongside 80:20, we achieved weaker model scores compared to 80:20 (See Figure 10). 

<img width="817" alt="Screenshot 2023-09-02 at 10 38 59 AM" src="https://github.com/WilliamChenn/Covid-Project/assets/85557718/dae82e29-f95d-4b3c-b3d3-3573e55a4260">

**Figure 10:** _Train_test_split_

The KNN algorithm is based on the principle of similarity, where it finds the k closest points to a given data point and assigns the class or value of the majority of those k points to the new data point. Choosing the value of k is an essential hyperparameter in the KNN algorithm. The optimal value of k can significantly affect the performance of our algorithm. In this research paper analysis, we will discuss the reason for choosing k=5 for our KNN algorithm based on the silhouette score, elbow plot, and testing accuracy scores.
The silhouette score is a measure of how well each data point fits into the assigned cluster. It takes into account the distance between each point and the points in its assigned cluster, as well as the distance between each point and the points in the nearest neighboring cluster. To find the most effective value of k, we can calculate the silhouette score for different values of k and choose the k that maximizes the average silhouette score. We calculated the silhouette score for k-values ranging from 2 to 8 and plotted the results. From the silhouette scores we generated, we observed that the highest silhouette score was achieved when k = 5 with a silhouette value of 0.439. (See Figure 11)

<img width="958" alt="Screenshot 2023-09-02 at 10 39 49 AM" src="https://github.com/WilliamChenn/Covid-Project/assets/85557718/211e9294-231b-4e7d-a75d-2828b7600d2b">

**Figure 11:** _Silhouette Scores with respective K-value_

The elbow plot is another method for determining the optimal value of k. The elbow plot shows the relationship between the value of k and the sum of squared distances between each data point and its assigned cluster. From the elbow plot, we observed a clear elbow at k=3 and a less optimal elbow at k=5. However, after the elbow at k=5, the drop in the sum of squared distances becomes less optimal (See figure 12).

<img width="488" alt="Screenshot 2023-09-02 at 10 40 22 AM" src="https://github.com/WilliamChenn/Covid-Project/assets/85557718/86d344cb-7e0a-44b4-a603-509bfd4745d4">

**Figure 12:** _The elbow graph shows WCSS (Within-Cluster Sum of Square) values corresponding to the different values of K(on the x-axis)._

After selecting k=5 as the optimal value for our KNN algorithm based on the silhouette score and elbow plot, we used k-means clustering to cluster the dataset into 5 clusters. Since KNN plots are multidimensional, we have to use the principal component analysis (PCA) algorithm from the scikit-learn library to reduce the data into 2 dimensions (See Figure 13).

<img width="904" alt="Screenshot 2023-09-02 at 10 41 14 AM" src="https://github.com/WilliamChenn/Covid-Project/assets/85557718/f8476e91-3fea-410f-981f-6c55d3355e51">

**Figure 13:**  _KNN Clusters visualized using PCA visualization method_

We also implemented a testing accuracy plot to determine the optimal k-value. From our testing accuracy plot and table, we determined that there is a significant increase in testing accuracy from k = 1 to k = 5, with the testing accuracy reaching 0.713. Although there is a gradual increase in testing accuracy as k approaches 40, the increase in accuracy is not as significant (See Figure 14). Moreover, this gradual increase in testing accuracy may also be due to overfitting. Therefore, we can conclude that the optimal k value is equal to 5.

<img width="616" alt="Screenshot 2023-09-02 at 10 42 02 AM" src="https://github.com/WilliamChenn/Covid-Project/assets/85557718/115991e1-58e1-40eb-90e8-f4c5b2073918">

**Figure 14:** _Testing accuracy plot with corresponding value of K_

After we fit our dataset into the KNeighborsClassifier with a K value of 5, we plotted our confusion matrix to observe the results of our classification. Our model correctly predicted 10943 of the 15348 test observations which yields a 0.7129 accuracy rate (See Figure 15). Through our Confusion matrix, we can observe that our model correctly predicted significantly more individuals who died compared to lived (8738 and 2205 respectively). This confusion matrix indicates that we are able to, with relatively high accuracy, utilize our model to assign new patients with their list of medical conditions to their corresponding cluster (whether or not the patient will live or die).

<img width="560" alt="Screenshot 2023-09-02 at 10 42 45 AM" src="https://github.com/WilliamChenn/Covid-Project/assets/85557718/1b0f9957-0256-4e29-bbec-20f0ed9a174f">

**Figure 15:** _Confusion matrix for K = 5_

To determine the feature importance for our model, we utilized the standard deviation feature importance score. Standard deviation importance score measures how much the feature values vary across the dataset and how much that variation affects the model's accuracy.
The basic idea behind the standard deviation feature importance score is that features with higher variance tend to have a larger impact on the model's accuracy. To calculate the standard deviation importance score, the standard deviation of each feature is computed across the entire dataset. Features with a higher standard deviation are considered more important, as they have a greater impact on the model's accuracy.
While evaluating the most important feature in influencing our predictive analysis, we determined that the attributes “Age”, “Medical Unit”, and “Pneumonia,” were the most important features (See Figure 16). This indicates that Age and Pneumonia may be considered the most significant health conditions for prediction of death from COVID-19 (as the medical unit is not a health condition).

<img width="481" alt="Screenshot 2023-09-02 at 10 43 26 AM" src="https://github.com/WilliamChenn/Covid-Project/assets/85557718/dbe0f382-e538-4756-b390-e1167bff9d2a">

**Figure 16:** _Standard deviation feature importance score visualization_

# Discussion & Conclusion

From our COVID-19 dataset, we concluded that Age and Pneumonia as the two most significant health features that contribute to our prediction of death for KNN. Furthermore, after selecting our k-cluster of 5, we concluded that the results of our KNN model yielded a 0.713 percent accuracy rate in predicting whether or not a patient will live or die based on the attributes. This research project utilized over 1 million patient observations with various health predictors such as diabetes, obesity, age, gender, and pneumonia. The results of our dataset concluded that our model has a reasonably high accuracy rate in predicting a patient’s death classification. The findings of our research can assist hospitals and medical officials in mitigating the continuous COVID-19 deaths and can enhance identification of which patients are considered at a high risk of death. Furthermore, with efficient resource allocations and treatments for these patients, we can continue to specialize and tailor care towards those patients who need it and ultimately minimize the negative effects of the COVID-19 virus.

## Limitations of the Study 
Although the KNN model that we built provides a decent accuracy rate for prediction, there are some limitations of our study that could be refined and researched for further studies. The dataset utilized in this research was provided by the Mexican government, thus it was strictly limited to their population and cannot be generalized to the human population. For future studies, samples and observations can be collected from an accurate representative population. Furthermore, the dataset that was provided contained many missing and null values that were essentially removed from our dataset, and potentially could affect the accuracy of our results. For future studies, a more expanded dataset including more features could be used to predict COVID-19 deaths with a higher accuracy prediction. There may be more significant health attributes that contribute to COVID-19 deaths that we did not take into account for this research study. For some patients, it is unclear whether the death was caused primarily by the COVID-19 virus or by other underlying diseases that the patient had. Furthermore, the attributes with negative feature importance could have been removed to improve the accuracy of our research. The negative attributes only worsened the results of the research and were not removed in our KNN prediction. In conclusion, our KNN model and research study provided insightful results in predicting COVID-19 deaths, but there are limitations that should be taken into account and potentially expanded on in the future. Further research is required to verify and expand the conclusions of this study to a more representative population, include more health features, and investigate different machine learning algorithms.







