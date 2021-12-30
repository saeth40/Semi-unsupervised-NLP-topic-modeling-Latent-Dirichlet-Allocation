# Unsupervised-NLP-Latent-Dirichlet-Allocation (LDA)
Thailand Development Research Institute (TDRI)| Feb 2021 - April 2021

Education reform policy and sustainable development policy department

***Objective:***
1. Query non-stem jobs from MongoDB database.
2. Preparing test and train dataset.
3. Train LDA models to classify job titles from their description.
4. Evauate and fine tune models.
5. Deploy models and analyze the results to define education policy for the newer Thai generations to have the skills demanded by the job market trend.

## Preparing test and train dataset
### Query non-STEM jobs from MongoDB and test the LDA algorithm

***Table***

![](Images/ldaNonStemTable.png)

***LDA model***

![](Images/ldaNonStemModel.png)

***Link to the code:*** [Click here](https://github.com/saeth40/Unsupervised-NLP-Latent-Dirichlet-Allocation/blob/main/Preparing%20test%20and%20train%20dataset/Query_non_STEM_plus_LDAtesting.ipynb).

### Shuffle selected data and split it to test (5,000) and train (the rest)

(link to the code)

### Divide test dataset into 10 files allocated to members to label job titles

(link to the code)
(the excel sample of title label)

### Combine job description with job title into a 'combine' column

(the output excel combine)

### Preprocess a dictionary of the train dataset
1. without bigram and trigram.

(link to the code)

2. with bigram and trigram.

(link to the code)

## Training LDA models
