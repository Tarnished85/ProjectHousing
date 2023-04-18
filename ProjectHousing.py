# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt

filepath = "C:/Users/KMarg/Downloads/assignment_rev2.csv"

data = pd.read_csv(filepath)

data.columns

data['price_per_sq'] = data['price']/data['sq_meters']

#Because of the situation with id's and the agent id, we will subset the dataframe in order to proceed.

#As a first step we will filter the dataframe and keep the columns that we want.
Subset1 = data[["geography_name", "subtype", "sq_meters", "price", "year_of_construction", "price_per_sq"]]

#Now we drop duplicates
Subset1 = Subset1.drop_duplicates()

#Now we check for na's in our final dataframe
if Subset1.isnull().sum().sum() > 0:
    print("Warning: missing values detected in final subset.")
    
#Finally, we create a pivot table with the needed descriptive statistcs in our new dataframe
table = Subset1.pivot_table(values='price_per_sq', index=['subtype'], columns=['geography_name'], aggfunc= ['mean', 'median', 'std'])

print(table)

#We save the pivot table in an excel file to send it to the marketing department. We can process the excel file to make it look better.
table.to_excel("C:/Users/KMarg/Documents/pivot_table.xlsx")

#Now we will create some metrics to calculate the competitiveness of the area.
#First of all, we will see will check the concertration if ad types per neighborhood.
#The bigger the presence of star, premium and up, the harder for a simple listing to get high

Count_Of_adtype_per_Region = data.groupby('geography_name')['ad_type'].value_counts()/data.groupby('geography_name')['ad_type'].count()*100

print(Count_Of_adtype_per_Region)

#Now we are going to create a bar chart

# create a bar chart of the counts
Count_Of_adtype_per_Region.unstack().plot(kind='bar', figsize=(8,6))

# set the title and axis labels
plt.title('Count of Ad Types per Region')
plt.xlabel('Region')
plt.ylabel('Percentage of Ad Types')

# show the plot
plt.show()

#Based on this gentrification area and northern sub seem like more competitive.
#On the other hand the bigger concertration of simples listings in beesy neighborhood makes
#the environment more competitive for the simple.

# create a pie chart of the counts
fig, axs = plt.subplots(1, len(Count_Of_adtype_per_Region.index.levels[0]), figsize=(15, 6), squeeze=False)
plt.subplots_adjust(wspace=0.2)

for i, region in enumerate(Count_Of_adtype_per_Region.index.levels[0]):
    wedges, texts, autotexts = axs[0, i].pie(Count_Of_adtype_per_Region[region], labels=Count_Of_adtype_per_Region[region].index, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
    axs[0, i].set_title(region, fontsize=14)
    # adjust the title position
    title = axs[0, i].title
    title.set_position([0.5, 1.2])

# add a common title and adjust the layout
fig.suptitle('Count of Ad Types per Region', fontsize=18)
fig.tight_layout()

# show the plot
plt.show()

#Now based on the fact that ranking is not enough, we will check the average ranking per type per region.

#intra-Category rating
Average_Ranking_per_Ad_type = data.groupby(['geography_name', 'ad_type'])['ranking_score'].agg(['mean', 'median'])


print(Average_Ranking_per_Ad_type)

# Plot a bar chart of the means and medians
ax = Average_Ranking_per_Ad_type.plot(kind='bar')
ax.set_xlabel('(Geography Name, Ad Type)')
ax.set_ylabel('Ranking Score')
ax.legend(['Mean', 'Median'])
plt.show()

#Here we see another story. The mean rating in beesy neighborhood is very close for each ad type
#This means that really rating cannot help a simple add in such a case. In any case, more research is needed.


