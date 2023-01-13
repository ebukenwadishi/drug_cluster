# -*- coding: utf-8 -*-
"""

The dataset provides patient reviews on specific drugs along with related conditions. Reviews and ratings are grouped into reports on the three aspects benefits, side effects and overall comment.

The dataset was downloaded from the UCI Machine Learning repository as recommended during the learning phase.
The data used is CSV (comma-separated values) file that had been provided for this analysis and for building the clustering model or algorithms.

Prototype-based clustering is a type of clustering algorithm that uses existing data points as prototypes for new clusters. This means that, rather than starting with a predetermined number of clusters, the algorithm will create new clusters on the fly as it processes the data.

To build an application with prototype-based clustering, you would need to first select a dataset to work with. This could be any type of dataset, such as a collection of customer data, financial data, or even text data.

Next, you would need to preprocess the data to ensure that it is in a suitable format for the clustering algorithm. This might involve cleaning the data, scaling it, or transforming it in some way to make it more suitable for analysis.

Once the data is ready, you would then need to implement the prototype-based clustering algorithm. This typically involves iterating over the data points and assigning them to clusters based on their similarity to existing prototypes. As new data points are added, the algorithm may create new prototypes and form new clusters as needed.

Finally, once the data has been clustered, you can use the resulting clusters to perform further analysis or to build models that can make predictions or provide insights about the data.

Overall, building an application with prototype-based clustering involves selecting a dataset, preprocessing the data, implementing the clustering algorithm, and using the resulting clusters for further analysis.
"""

#importing neccearily libraries 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import stats

#importing the downloaded dataset 
drug_df = pd.read_csv('drugsComTest_raw.csv', sep='\t')

#check top 5 reocrds in the dataset
drug_df.head()

#check the correlation and relationship between the features using cor() function and ploting an heatmaps 
corr_mat = drug_df.corr()

cmap = sns.diverging_palette(240, 20, as_cmap = True)
sns.heatmap(corr_mat, vmax = 1, vmin = -1, cmap = cmap)

plt.show()

#check the rows and columns of the dataset
drug_df.shape

#check the name of the columns 
drug_df.columns

#view the value counts of the rating given to every drugs 
drug_df['rating'].value_counts()

plt.figure(figsize=(8,5))
sns.countplot('rating',data=drug_df,palette='ocean')

#checking information about the features in the dataframe, the condition has some missing values 
drug_df.info()

# calculate the mode of the "condition" column
mode = drug_df['condition'].mode()

# fill the missing values with the mode of the "condition" column
drug_df['condition']= drug_df['condition'].fillna(mode)

#converting the category features to numeric 
drug_df['condition'] =drug_df['condition'].astype('category').cat.codes

drug_df.columns

selected_columns = drug_df[['Unnamed: 0', 'condition', 'rating', 'usefulCount']]

#intialize the Kmeans library
model = KMeans()

#building the Kmean model
model.fit(selected_columns)

#predict on the model 
predictions = model.predict(selected_columns)

#using the rating column as train_label to check the performance of the model using adjusted rand score 
train_label = drug_df['rating']

from sklearn.metrics.cluster import adjusted_rand_score

#checking model performance 
adjusted_rand_score(train_label,predictions)

