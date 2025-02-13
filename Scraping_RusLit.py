#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:56:18 2024

@author: gabrielevezzani
"""

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import pandas as pd
from tqdm import tqdm


def scrape_reviews(page, n, driver, wait):

    out = []

    driver.get(page)
    time.sleep(2)
    for _ in range(n):
        pushed = 'n'
        errors = 0
        while pushed == 'n':
            try:
                buttons = wait.until(EC.visibility_of_all_elements_located((By.CSS_SELECTOR, 'button[class="Button Button--secondary Button--small"]')))
                if len(buttons) > 1:
                    button = buttons[1]
                else:
                    button = buttons[0]
                driver.execute_script("arguments[0].click();", button)
                pushed = 'y'
            except:
                errors += 1
                if errors > 3:
                    break
                else:
                    continue
        
            
    revs = wait.until(EC.visibility_of_all_elements_located((By.TAG_NAME, 'article')))
    
    for r in revs:
        try:
            content = r.find_element(By.CSS_SELECTOR, 'section[class="ReviewCard__content"]')
            text = content.find_element(By.CSS_SELECTOR, 'section[class="ReviewText"]').text
            meta = r.find_element(By.CSS_SELECTOR, 'section[class="ReviewCard__row"]')
            div = meta.find_element(By.CSS_SELECTOR, 'div[class="ShelfStatus"]')
            rating = div.find_element(By.TAG_NAME, 'span').get_attribute('aria-label')
            d = meta.find_element(By.CSS_SELECTOR, 'span[class="Text Text__body3"]')
            date = d.text
            profile_section = r.find_element(By.TAG_NAME, 'div')
            profile_info = profile_section.find_element(By.CSS_SELECTOR, 'section[class="ReviewerProfile__info"]')
            l = profile_info.find_element(By.TAG_NAME, 'a')
            link = l.get_attribute('href')
            user_id = l.text
            out.append({'id':user_id, 'link':link, 'date':date, 'review':text, 'rating':rating})
        except:
            continue

    return out



print("Start driver")

driver = webdriver.Remote(
   command_executor='http://localhost:4444/wd/hub',
   desired_capabilities=DesiredCapabilities.CHROME)

wait = WebDriverWait(driver, 10)

print('login')

email = ''
pw = ''
driver.get('https://www.goodreads.com/user/sign_in')
time.sleep(3)
buttons = driver.find_elements(By.TAG_NAME, 'button')
my_button = buttons[3]
my_button.click()
time.sleep(3)
signin_email = driver.current_url
driver.get(signin_email)
e = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'input[type="email"]')))
e.send_keys(email)
p = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'input[type="password"]')))
p.send_keys(pw)
sig = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'input[id="signInSubmit"]')))
sig.submit()
time.sleep(3)
print(driver.current_url)

entered = input('entered? y/n')
while entered == 'n':
    driver.save_screenshot('screenshot.png')
    p = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'input[type="text"]')))
    my_input = input('write captcha')
    p.send_keys(my_input)
    p.send_keys(Keys.RETURN)
    print(driver.current_url)
    entered = input('entered?')
    

print('retriving book pages from lists "Best of Fyodor Dostoevsky" and "The Best of Tolstoy"')
      
author_pages = ['https://www.goodreads.com/list/show/22373.The_Best_of_Tolstoy','https://www.goodreads.com/list/show/5742.Best_of_Fyodor_Dostoevsky']
review_pages = {}

for page in author_pages:
    driver.get(page)
    titles = driver.find_elements(By.CSS_SELECTOR, 'a[class="bookTitle"]')
    for title in titles:
        try:
            book_page = title.get_attribute('href')
            book = book_page.split('/')[-1].split('.')[1]
            book_id = book_page.split('/')[-1].split('.')[0]
            review_page = f'https://www.goodreads.com/book/show/{book_id}/reviews'
            review_pages[book] = review_page
        except:
            continue
        
df = []

for book in tqdm(review_pages):
    revs = scrape_reviews(review_pages[book], 5, driver, wait)
    for rev in revs:
        rev['title'] = book
        df.append(rev)
    data = pd.DataFrame(df)
    data.to_excel('data/dataset_RusLit.xlsx')

driver.quit()