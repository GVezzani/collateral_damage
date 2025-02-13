
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:15:44 2024

@author: gabrielevezzani
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import ruptures as rpt 
import random
from statistics import stdev
from scipy.stats import linregress
from tqdm import tqdm
import random

#clean dataset
df = pd.read_excel('data/dataset_processed.xlsx', index_col=0)




#test one
reference_date = datetime(2007, 1, 1)
start_war = datetime(2022, 2, 1)
delta = start_war - reference_date
threshold = delta.days

pre = []
post = []

for i, date in enumerate(df['date_integers']):
    if date > threshold:
        post.append(df.iloc[i,3])
    elif date < threshold:
        pre.append(df.iloc[i,3])
            

group1 = pre
group2 = post
_, p_value_group1 = stats.shapiro(group1)
_, p_value_group2 = stats.shapiro(group2)

alpha = 0.05 

if any([p_value_group1<alpha, p_value_group2<alpha]):
    normality_check = False
else:
    normality_check = True


if normality_check:
    _, p_value_levene = stats.levene(group1, group2)
    
    if p_value_levene < alpha:
        t_one, p_value_test_one = stats.ttest_ind(group1, group2, equal_var=False)
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.std(group1, ddof=1) ** 2 + (n2 - 1) * np.std(group2, ddof=1) ** 2) / (n1 + n2 - 2))
        eff_size_test_one = (np.mean(group1) - np.mean(group2)) / pooled_std
        
    else:
        t_one, p_value_test_one = stats.ttest_ind(group1, group2, equal_var=True)
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.std(group1, ddof=1) ** 2 + (n2 - 1) * np.std(group2, ddof=1) ** 2) / (n1 + n2 - 2))
        eff_size_test_one = (np.mean(group1) - np.mean(group2)) / pooled_std

else:
    t_one, p_value_test_one = stats.mannwhitneyu(group1, group2)
    grouping = [1]*len(list(group1)) + [0]*len(list(group2))
    values = list(group1) + list(group2)
    eff_size_test_one = stats.pointbiserialr(grouping,values)[0]
    

data = {'Group': ['pre'] * len(group1) + ['post'] * len(group2),
        'Values': group1 + group2}
sns.boxplot(x='Group', y='Values', data=data)
plt.savefig('results/boxplot_ratings.png')


with open('results/statistical_tests_ratings.txt', 'w') as f:
    f.write('\n\n\nTEST ONE\n')
    if normality_check:
        f.write('t-test\n')
    else:
        f.write('mann-whitney\n')
    f.write(f'mean score pre war: {np.mean(pre)}, mean score post war: {np.mean(post)}\ndiff: {np.mean(pre)-np.mean(post)}')
    f.write(f't/u: {t_one}\neffect size: {eff_size_test_one}\np value: {p_value_test_one}')
    


# permutation

eff_sizes = []
x = []
y = []

diff = abs(np.mean(pre) - np.mean(post))
new_diffs = []


# Convert columns to NumPy arrays for faster operations
dates = df['date_integers'].to_numpy()
thresholds = np.random.choice(dates, 1000, replace=False)
counter = 0

# Loop through unique threshold values
for threshold in tqdm(thresholds):
    
    pre = []
    post = []

    for i, date in enumerate(df['date_integers']):
        if date > threshold:
            post.append(df.iloc[i,3])
        elif date < threshold:
            pre.append(df.iloc[i,3])
    
    new_diffs.append(abs(np.mean(pre) - np.mean(post)))
    # Compute Mann-Whitney U test
    try:
        u, p_value = stats.mannwhitneyu(pre, post)
    except:
        continue


    # Store results if significant
    if p_value < 0.05:
        counter +=1
        grouping = [1]*len(list(pre)) + [0]*len(list(post))
        values = list(pre) + list(post)
        eff_size = stats.pointbiserialr(grouping,values)[0]
        eff_sizes.append(eff_size)
        x.append(threshold)
        y.append(eff_size)


start_war = datetime(2022, 2, 1)
delta = start_war - reference_date
plt.figure(figsize=(10, 6))
hb = plt.hexbin(x, y, gridsize=50, cmap='Blues', mincnt=1)
plt.axvline(x=delta.days, color='r', linestyle='--', linewidth=2)
plt.colorbar(hb, label='Count')
plt.title('Hexbin Plot of Effect Sizes')
plt.xlabel('Days')
plt.ylabel('Effect Size')

plt.savefig('results/hexbin_eff_size.png')

count_higher = sum(1 for d in new_diffs if d >= diff)
p_value = count_higher / 1000

with open('results/statistical_tests_ratings.txt', 'a') as f:
    f.write(f'\n\n\nPERMUTATION\n\npermutation p: {p_value}\naverage diff: {np.mean(new_diffs)}\nsd: {np.std(new_diffs)}\n')
    f.write(f'results are significant {counter/1000} of the times\nmean effect size: {np.mean(eff_sizes)}')
    f.write(f'\nminimum: {min(eff_sizes)}\nmaximum: {max(eff_sizes)}\nstandard deviation: {stdev(eff_sizes)}')
    

#test two

dates_and_ratings = list(sorted(zip(df['date_integers'], df['rating']), key=lambda x:x[0]))
dates_and_ratings = pd.DataFrame(dates_and_ratings)
dates_and_ratings.columns = ['date', 'rating']

highlight_days = [5510, 2616]
window = 100
dates_and_ratings['smoothed'] = dates_and_ratings['rating'].rolling(window=window*2, center=True,min_periods=10).mean()  # Adjust window size as needed


evals = dates_and_ratings['rating'].to_numpy()
days = dates_and_ratings['date'].to_numpy()

algo = rpt.Pelt(model="l2").fit(evals)
result_pelt = algo.predict(pen=10)
algo = rpt.Binseg(model='l2').fit(evals)
result_binary = algo.predict(n_bkps=1)
algo = rpt.Dynp(model='l2', min_size=10, jump=10).fit(evals)
result_dyn = algo.predict(n_bkps=1)
algo = rpt.BottomUp(model='l2').fit(evals)
result_bottomup = algo.predict(n_bkps=1)
algo = rpt.Window(width=60, model='l2').fit(evals)
result_window = algo.predict(n_bkps=1)


results = {'pelt':result_pelt, 'binary':result_binary, 'dynamic programming':result_dyn,'bottomup':result_bottomup, 'sliding window':result_window}
for result in results:
    new_results = []
    for d in results[result][:-1]:
        day = days[d]
        date = reference_date + timedelta(days=int(day))
        new_results.append(date.strftime("%Y-%m-%d"))
    results[result] = new_results
        
with open('results/statistical_tests_ratings.txt', 'a') as f:
    f.write('\n\n\nTEST two\n')
    for result in results:
        f.write(f'\n{result}: {results[result]}')
       

cps = [result_pelt[0], result_binary[0], result_dyn[0], result_bottomup[0], result_window[0]]
# Plot
plt.figure(figsize=(12, 6))
plt.plot(dates_and_ratings['date'], dates_and_ratings['smoothed'], label="Smoothed Ratings", color='blue')
plt.axvspan(highlight_days[0] - window, highlight_days[0] + window, color='red', alpha=0.2, label="Invasion of Ukraine")
plt.axvspan(highlight_days[1] - window, highlight_days[1] + window, color='green', alpha=0.2, label="Annexation of Crimea")
first = True
for cp in cps:
    day = days[cp]
    if first:
        plt.axvline(day, color='red', linestyle='dashed', label="Changepoint")
        first = False
    else:
        plt.axvline(day, color='red', linestyle='dashed', ymin=0, ymax=1)
plt.legend()
plt.xlabel("Days")
plt.ylabel("Rating")
plt.title("Book Ratings Trend")
plt.savefig('results/ratings_trend.png')



  

#test four

x = days
y = evals
plt.scatter(x, y, label='Data')

slope, intercept, r_value, p_value, std_err = linregress(x, y)
regression_line = [slope * i + intercept for i in x]
plt.plot(x, regression_line, color='red', label='Regression Line')

# Add legend and labels
plt.legend()
plt.xlabel('number of days from January 1 2007')
plt.ylabel('sentiment')
plt.title('General trend of sentiment')

plt.savefig('results/regression.png')

with open('results/statistical_tests_ratings.txt', 'a') as f:
    f.write(f'\n\n\nTEST FOUR\n\nRegression line\nslope: {slope}\nintercept: {intercept}\np_value:{p_value}')
    
