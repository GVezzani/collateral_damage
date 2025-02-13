#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:42:40 2024

@author: gabrielevezzani
"""

import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import nltk
import numpy as np

def date_to_integers(date, reference): 
    months = {'June':6, 'January':1, 'April':4, 'November':11, 'July':7, 
              'October':10, 'August':8, 'May':5, 'December':12, 'March':3, 'February':2, 'September':9}
    date = ''.join([l for l in date if l != ','])
    date = date.split()
    date_processed = datetime(int(date[2]), months[date[0]], int(date[1]))
    delta = date_processed - reference
    
    return delta.days
    
    

print('processing dataset')

df = pd.read_excel('data/dataset_RusLit.xlsx', index_col=0)

#process dates
years = {}
for date in df['date']:
    date = date.split()[-1]
    if date not in years:
        years[date] = 1
    else:
        years[date] += 1
        
min_year = (min([int(x) for x in years.keys()]))
reference_date = datetime(min_year, 1, 1)
date_integers = [date_to_integers(x, reference_date) for x in df['date']]
df['date_integers'] = date_integers

#create plot
years = dict(sorted(years.items(), key=lambda x:x[0]))
plt.bar(years.keys(), years.values())
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Reviews per year')
plt.xticks(rotation=45) 
plt.savefig('results/revs_per_year.png')


#last touches to the dataset
df.drop('review', axis=1, inplace=True)
df.dropna(subset=['valence'], inplace=True) 
df['rating'] = df.rating.apply(lambda x: int(x.split()[1]))
df.to_excel('data/processed_lan_df.xlsx')

