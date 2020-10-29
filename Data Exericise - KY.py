# Katherine Yu
# Burning Glass Data Exercise
# 09/07/2020

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

###file source
os.chdir('/Users/katherine/Downloads/db_25_0_excel')
pd.set_option('display.max_columns', None)

###import datasets
taskRatings = pd.read_excel('Task Ratings.xlsx')
tasksDWA = pd.read_excel('Tasks to DWAs.xlsx')
#print(taskRatings.info())
#print(taskRatings.head())
#print(tasksDWA.head())
#print(tasksDWA.info())

##############

###identify the task as either analytical or not
###if the DWA Title or the Task itself contains any of the keywords identified below, then flag the task as analytical
###Checking DWA Title as a supplement to Task in case the wording is different

###convert relevant strings to lowercase
tasksDWA['DWA Title'] = tasksDWA['DWA Title'].str.lower()
tasksDWA['Task'] = tasksDWA['Task'].str.lower()
taskRatings['Task'] = taskRatings['Task'].str.lower()

###create keywords related to analytical and search the tasks and DWAs if they contain any key words
###create a flag that is 0 if keywords don't appear in statements and 1 if they do; group by task ID afterwards
keywordsList = ['analyze', 'research', 'evaluate', 'recommend', 'investigate', 'advise',
                'consult','forecast','statistic','analyses']
keywords = '|'.join(keywordsList)
tasksDWA['tasksDWAFlag'] = (tasksDWA['DWA Title'].str.contains(keywords) |
                            tasksDWA['Task'].str.contains(keywords)).astype(int)
tasksDWA = tasksDWA.groupby(['Task ID'], as_index=False).sum()

###found out through commands below that Task Ratings has more task ID's than Tasks to DWA
# print(tasksDWA['Task ID'].nunique())
# print(taskRatings['Task ID'].nunique())
###solution: search tasks in Task Ratings for keywords and after merging, add the two flags together
###any task with a flag greater than 0 will be considered analytical
taskRatings['taskRatingsFlag'] = taskRatings['Task'].str.contains(keywords).astype(int)

###merge tasksDWA with taskRatings and get the "final" version of categorizing analytical tasks
tasks = pd.merge(taskRatings,
                 tasksDWA,
                 on='Task ID',
                 how='left')
tasks.tasksDWAFlag = tasks.tasksDWAFlag.fillna(value=0)
tasks['analyticalFlag'] = tasks['taskRatingsFlag'] + tasks['tasksDWAFlag']
#print(tasks.info())
tasks.to_csv('tasks before.csv')

#################

###calculate the "analytical score" for each occupation using importance and relevance
###drop rows with frequency data
tasks = tasks.loc[tasks['Scale ID'] != 'FT', :]
###pivot the data to make the dataframe a multiindex and be able to work with IM and RT data values
unstacked = tasks.pivot(index=['O*NET-SOC Code','Task ID','analyticalFlag'],columns=['Scale ID'],values=['Data Value'])
#print(unstacked.head(100))

###scale importance data from 1-5 to 0-100 so it's the same as relevance data range
scaler = MinMaxScaler(feature_range=(0,100))
unstacked[('Data Value','imScaled')] = scaler.fit_transform(unstacked[[('Data Value','IM')]])

###create a separate occupations dataframe on the occupational to merge with later, format and get rid of unnecessary rows
###get the number of tasks for each occupation
occupations = unstacked.count(level = 'O*NET-SOC Code')
occupations.columns = occupations.columns.droplevel('Scale ID')
occupations = occupations.reset_index()
occupations = occupations.iloc[:,0:2]
occupations.rename(columns={'Data Value':'taskCount'}, inplace=True)

###if analyticalFlag is >0, that means keywords were found i.e. the task is analytical, and so we calculate the taskScore
unstacked = unstacked.reset_index()
condition = unstacked['analyticalFlag'] > 0
unstacked[('Data Value','taskScore')] = np.where(condition,
                                                       np.sqrt(unstacked[('Data Value','imScaled')]*unstacked[('Data Value','RT')]),
                                                       0)

unstacked = unstacked.groupby(by='O*NET-SOC Code').sum()
unstacked.columns = unstacked.columns.droplevel()
unstacked = unstacked.reset_index()
unstacked.rename(columns={'taskScore':'taskTotalScore'}, inplace=True)

###merge taskTotalScore with the separate occupations dataset with the number of tasks for occupation already
occupations = pd.merge(occupations,
                       unstacked[['O*NET-SOC Code', 'taskTotalScore']],
                       on='O*NET-SOC Code',
                       how='left')
#print(occupations)

###calculate analytical score for every sub-occupation
###change 8 digit code to 6 digit code
###simple average over 6 digit codes
occupations['occupationScore'] = (occupations.taskTotalScore/(occupations.taskCount*100))*100
occupations['O*NET-SOC Code'] = occupations['O*NET-SOC Code'].str.slice(0, -3)
occupations = occupations.groupby(by='O*NET-SOC Code').mean().reset_index()

###import gender and employment datasets
gender = pd.read_excel('/Users/katherine/Downloads/Occ_Gender_Distribution.xlsx')
gender.rename(columns={'SOC':'O*NET-SOC Code'}, inplace=True)
employ = pd.read_excel('/Users/katherine/Downloads/Occ_Employment.xlsx')
employ.rename(columns={'OCC_CODE':'O*NET-SOC Code'}, inplace=True)

###merge gender and occupations
occupations = pd.merge(occupations,
                       gender[['O*NET-SOC Code','SOCName','% Female (ACS)']],
                       on = 'O*NET-SOC Code',
                       how='left')
occupations = pd.merge(occupations,
                       employ[['O*NET-SOC Code', 'TOT_EMP']],
                       on='O*NET-SOC Code',
                       how='left')

###calculate the actual number of females by occupation
occupations['# Female'] = (occupations['% Female (ACS)']/100)*occupations['TOT_EMP']

###reorder and export occupations as a csv file
occupationsExport = occupations.sort_values(by='occupationScore', ascending=False)
occupationsExport.to_csv('occupationsExport.csv')
#print(occupations.info())
#print(occupations)

##################

###creating different graphs for visualization
###occupation score distribution
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(occupations['occupationScore'])

###% and number females on a scatter plot with analytical score
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.scatter(occupations['% Female (ACS)'], occupations['occupationScore'])
ax2.set_xlabel('% Female (ACS)')
ax2.set_ylabel('Occupation Analytical Score')
ax2.set_title('% Female vs. Analytical Score by Occupation')

fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)
ax3.scatter(occupations['# Female'], occupations['occupationScore'])
ax3.set_xlabel('Number of Females (ACS)')
ax3.set_ylabel('Occupation Analytical Score')
ax3.set_title('Number of Females vs. Analytical Score by Occupation')

plt.show()

###############

###supplemental analysis for gender gap
subset = occupations.loc[occupations.occupationScore>0,:]
subsetMean = (subset.occupationScore.mean())
subset2 = occupations.loc[occupations['occupationScore']>subsetMean,:]
subset2.to_csv('subset2.csv')
print(subset2['% Female (ACS)'].mean())
subset3 = occupations.loc[occupations['occupationScore']<=subsetMean,:]
print(subset3['% Female (ACS)'].mean())
subset4 = subset.loc[subset['occupationScore']<=subsetMean,:]
print(subset4['% Female (ACS)'].mean())


