import requests
import lxml
from bs4 import BeautifulSoup
from xlwt import *

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

import collections
import io
import os
from PIL import Image

import hashlib
import re

# https://towardsdatascience.com/image-scraping-with-python-a96feda8af2d
# https://www.topcoder.com/thrive/articles/web-crawler-in-python

def main():
    workbook = Workbook(encoding = 'utf-8')
    table = workbook.add_sheet('data')
    table.write(0, 0, 'recipe_url')
    table.write(0, 1, 'recipe_name')
    table.write(0, 2, 'image_path')

    line = 1
    links = set()
    toCheck = collections.deque()

    folder_path = 'images'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    headers = {
        'User-Agent':
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'
    }
    
    # url = 'https://www.simplyrecipes.com/recipes-5090746'
    # url = 'https://www.simplyrecipes.com/fried-brussels-sprouts-recipe-6825672'
    url = 'https://www.simplyrecipes.com/'
    links.add(url)
    toCheck.append(url)

    while toCheck and line < 10000:
        url = toCheck.popleft()
        # print(url)
        try:
            f = requests.get(url, headers=headers)
        except Exception as e:
            print(f"ERROR - Could not access {url} - {e}")
        soup = BeautifulSoup(f.content, 'lxml')
       
        # if this is a recipe page
        try:
            type = soup.find('html')['id']
        except:
            continue

        if type == 'structuredProjectTemplate_1-0':
            # get title
            try:
                title = soup.find(class_='loc article-header').find(
                class_='comp heading'
                ).find(
                class_='heading__title'
                ).getText()
            except:
                continue

            # get image url
            try:
                image_url = soup.find(class_='loc main').find(
                    class_=
                    'comp mntl-article--two-column-right-rail right-rail sc-ad-container article--structured-project lifestyle-food-article mntl-article'
                ).find(class_='loc article-content').find(
                    class_=
                    'comp article-header__media primary-media figure-wrapper--article mntl-block'
                ).find(class_='figure__media js-figure-media figure__media--portrait').find(
                    class_='img-placeholder').find('img')['src']
            except:
                continue
            
            # download image and record file path
            try:
                image_content = requests.get(image_url).content
            except:
                continue
            try:
                image_file = io.BytesIO(image_content)
                image = Image.open(image_file).convert('RGB')
                file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
                with open(file_path, 'wb') as f:
                    image.save(f, "JPEG", quality=85)
                # print(f"SUCCESS - saved {image_url} - as {file_path}")
            except Exception as e:
                print(f"ERROR - Could not save {image_url} - {e}")
                continue
            
            # add to excel file
            table.write(line, 0, url)
            table.write(line, 1, title) # recipe names
            table.write(line, 2, file_path) # image paths
            line += 1

        for item in soup.findAll('a', attrs={'href': re.compile("^https://www.simplyrecipes.com/")}):
            link = item.get('href')
            # print(link)
            if link not in links:
                toCheck.append(link)
                links.add(link)
        
        if line % 10 == 0:
            workbook.save('foods.xls')
        if line % 250 == 0:
            print(f'{line} entries')

    workbook.save('foods.xls')

    # table.write(line, 0, links)


if __name__ == '__main__':
    main()