#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:56:11 2025

@author: gabrielevezzani
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.stats import chi2_contingency

#clean dataset
df = pd.read_excel('data/dataset_processed.xlsx', index_col=0)
reference_date = datetime(2007, 1, 1)
start_war = datetime(2022, 2, 1)
delta = start_war - reference_date
threshold = delta.days

# descriptive stuff

works_to_author = {
    'Father_Sergius': 't',
    'The_Living_Corpse': 't',
    'A_Confession': 't',
    'Master_and_Man': 't',
    'White_Nights': 'd',
    'What_I_Believe': 't',
    'Notes_from_Underground': 'd',
    'The_Landlady': 'd',
    'The_Death_of_Ivan_Ilych': 't',
    'Childhood_Boyhood_Youth': 't',
    'The_Kreutzer_Sonata': 't',
    'Crime_and_Punishment': 'd',
    'Notes_from_Underground_White_Nights_The_Dream_of_a_Ridiculous_Man_and_Selections_from_The_House_of_the_Dead': 'd',
    'Demons': 'd',
    'The_Brothers_Karamazov': 'd',
    'Netochka_Nezvanova': 'd',
    'The_Crocodile': 'd',
    'Tolstoy_s_Short_Fiction': 't',
    'The_Power_Of_Darkness': 't',
    'Resurrection': 't',
    'Great_Short_Works_of_Leo_Tolstoy': 't',
    'What_Is_Art_': 't',
    'Bobok': 'd',
    'The_Adolescent': 'd',
    'Hadji_Mur_d': 't',
    'The_Idiot': 'd',
    'The_Insulted_and_Humiliated': 'd',
    'The_House_of_the_Dead': 'd',
    'The_Eternal_Husband': 'd',
    'What_Is_to_Be_Done_and_Life': 't',
    'The_Grand_Inquisitor': 'd',
    'The_Cossacks': 't',
    'Poor_Folk': 'd',
    'The_Light_That_Shines_in_the_Darkness': 't',
    'The_Best_Short_Stories_of_Fyodor_Dostoevsky': 'd',
    'Fables_and_Fairy_Tales': 't',
    'The_Double': 'd',
    'War_and_Peace': 't',
    'The_Forged_Coupon': 't',
    'The_Gambler': 'd',
    'The_Village_of_Stepanchikovo': 'd',
    'Anna_Karenina': 't',
    'The_Devil': 'd',
    'The_Sebastopol_Sketches': 't',
    'The_Kingdom_of_God_Is_Within_You': 't',
    'The_Dream_of_a_Ridiculous_Man': 'd'
}


pre = []
post = []

for i, date in enumerate(df['date_integers']):
    if date > threshold:
        post.append(works_to_author[df.iloc[i,4]])
    elif date < threshold:
        pre.append(works_to_author[df.iloc[i,4]])

# Create the contingency table
pre_d = pre.count('d')
pre_t = pre.count('t')
post_d = post.count('d')
post_t = post.count('t')

# Construct the contingency table (2x2)
contingency_table = pd.DataFrame({
    'Dostoevsky': [pre_d, post_d],
    'Tolstoy': [pre_t, post_t]
}, index=['Pre', 'Post'])

chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)

with open('results/revision.txt', 'w') as f:
    f.write(f"Chi2 Stat: {chi2_stat}\n")
    f.write(f"P-Value: {p_val}\n")
    f.write(f"Degrees of Freedom: {dof}\n")
    f.write(f"Expected Frequencies: \n{expected}\n\n")
    

df['group'] =  np.where(df['date_integers'] > threshold, 'post', 'pre')
contingency_table = df.groupby(['title', 'group']).size().unstack(fill_value=0)
contingency_table.columns = ['Post', 'Pre']
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Output results
print("Chi-Square Statistic:", chi2)
print("p-value:", p)
print("Degrees of Freedom:", dof)

expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
contingency_table.to_excel("results/books_contingency_table.xlsx")
expected_df.to_excel("results/books_expected_values.xlsx")


# Annexion of Crimea

#test one
reference_date = datetime(2007, 1, 1)
start_war = datetime(2014, 3, 1)
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
    grouping = [1]*len(list(pre)) + [0]*len(list(post))
    values = list(pre) + list(post)
    eff_size_test_one = stats.pointbiserialr(grouping,values)[0]

data = {'Group': ['pre'] * len(group1) + ['post'] * len(group2),
        'Values': group1 + group2}
sns.boxplot(x='Group', y='Values', data=data)
plt.savefig('results/boxplot_revision.png')


with open('results/revision.txt', 'a') as f:
    f.write('\n\n\nTEST ONE\n')
    if normality_check:
        f.write('t-test\n')
    else:
        f.write('mann-whitney\n')
    f.write(f'mean score pre war: {np.mean(pre)}, mean score post war: {np.mean(post)}\ndiff: {np.mean(pre)-np.mean(post)}')
    f.write(f't/u: {t_one}\neffect size: {eff_size_test_one}\np value: {p_value_test_one}')
    


# permutation

combined = pre+post
diff = np.mean(pre) - np.mean(post)
new_diffs = []

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
   
   
count_higher = sum(1 for d in new_diffs if d >= abs(diff))
p_value = count_higher / 1000
print(p_value)

with open('results/revision.txt', 'a') as f:
    f.write(f'\n\n\nPERMUTATION\np value: {p_value}\naverage diff: {np.mean(new_diffs)}\nSD: {np.std(new_diffs)}\n\n')



#keyword

import spacy 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from langdetect import detect
from nltk.corpus import words

english_words = set(words.words())


nlp = spacy.load('en_core_web_md')


def is_english(review):
    try:
        return detect(review) == 'en'  # 'en' stands for English
    except:
        return False

df_revs = pd.read_excel('data/dataset_RusLit.xlsx', index_col=0)

df_revs['combined_key'] = df_revs['link'] + '_' + df_revs['title']
matcher = pd.Series(df_revs['review'].values, index=df_revs['combined_key']).to_dict()
df['combined_key'] = df['link'] + '_' + df['title']
df['review'] = df['combined_key'].map(matcher)
df.drop(columns=['combined_key'], inplace=True)



reference_date = datetime(2007, 1, 1)
start_war = datetime(2022, 2, 1)
delta = start_war - reference_date
threshold = delta.days

pre = []
post = []

for i, date in tqdm(enumerate(df['date_integers'])):
    rev = df.iloc[i,-1]
    
    if not is_english(rev):
        continue

    rev = nlp(rev)
        
    if date > threshold:
        post.append(rev)
    elif date < threshold:
        pre.append(rev)
        



texts = pre + post
labels = [0] * len(pre) + [1] * len(post)  

cleaned = []
for t in texts:
    cleaned.append(' '.join([token.lemma_ for token in t if not token.is_stop and not token.is_punct and token.lemma_ in english_words]))

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(cleaned)

chi2_scores, p_values = chi2(X, labels)

feature_names = vectorizer.get_feature_names_out()
chi2_df = pd.DataFrame({"Feature": feature_names, "Chi2 Score": chi2_scores, "P-Value": p_values})
chi2_df = chi2_df.sort_values(by="Chi2 Score", ascending=False)


chi2_df.to_excel('results/words_chi2.xlsx')


words = chi2_df['Feature'].tolist()[:100]


with open('results/word_freq_first100.txt', 'w') as f:
    for word in words:
        n_pre = 0
        n_post = 0
        
        for i, text in enumerate(cleaned):
            if word not in text.split():
                continue
            
            if labels[i] == 0:
                n_pre += 1
            elif labels[i] == 1:
                n_post += 1
        
        f.write(f'word:{word}\npre: {n_pre} {n_pre/len(pre)}\npost: {n_post} {n_post / len(post)}\n\n')
    
#collocations

def get_contexts(target_word, reviews, window=20):
    contexts = []

    for review in reviews:
        words = [token.lemma_ for token in review if token.lemma_ in english_words]
        out = [token.text for token in review if token.lemma_ in english_words]
        if target_word in words:
            idx = words.index(target_word)
            start = max(0, idx - window)
            end = min(len(words), idx + window + 1)
            contexts.append(" ".join(out[start:end]))

    return contexts

# Example usage:
target = "brutality"


pre_contexts = get_contexts(target, pre)
post_contexts = get_contexts(target, post)

print("Pre-review contexts:")
print("\n\n".join(pre_contexts))  # Print first 10 examples

print("\nPost-review contexts:")
print("\n\n".join(post_contexts))  # Print first 10 examples



#more descriptive statistics 



def try_detect(x):
    try:
        return detect(x)
    except:
        return np.nan

all_revs = texts
lengths = [len(x) for x in all_revs]

with open('results/revision.txt', 'a') as f:
    f.write(f'average rev lenght: {np.mean(lengths)}, sd: {np.std(lengths)}')
    
languages = [try_detect(x) for x in df['review']]
lang_dist = {x:languages.count(x) for x in languages if x}

with open('results/revision.txt', 'a') as f:
    for x in lang_dist:
        f.write(f'num revs in {x}: {lang_dist[x]}\n')


