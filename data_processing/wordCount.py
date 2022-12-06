import pandas as pd
import collections

def main():
    df = pd.read_csv(r'combined-clean.csv')
    wordCounts = collections.defaultdict(int)
    for ind, row in df.iterrows():
        caption = row['caption']
        if not isinstance(caption, str):
            break
        words = caption.split()
        for word in words:
            wordCounts[word] += 1
    wordCountDf = pd.DataFrame.from_dict(wordCounts, orient='index', columns=['count']).sort_values(by=['count'], ascending=False)
    wordCountDf.to_csv('word-counts.csv')



if __name__ == '__main__':
    main()