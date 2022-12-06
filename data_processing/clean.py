import pandas as pd
import re

def main():
    df = pd.read_csv('combined.csv', encoding='utf8')
    stopwords = ['the', 'a', 'an', 'best', 'our', 'favorite', 'delicious', 'yummy', 'my', 'ingredient', '3']
    for ind, row in df.iterrows():
        if not isinstance(row['caption'], str):
            break
        caption = row['caption'].lower()

        # remove punctuation
        caption = caption.replace('-', ' ')
        caption = re.sub(r'[^\w\s]', '', caption)

        words = caption.split()
        cleanList = [word for word in words if word not in stopwords]
        df.at[ind, 'caption'] = (' '.join(cleanList))
        
    df.to_csv('combined-clean.csv')


if __name__ == '__main__':
    main()
