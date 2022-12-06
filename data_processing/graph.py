import matplotlib.pyplot as plt
import pandas as pd
import sys

def main(item):
    item0 = item.lower()
    df = pd.read_csv(f'{item0}-counts.csv')
    
    X = list(df.iloc[:, 0])[:20]
    Y = list(df.iloc[:, 1])[:20]
    
    # Plot the data using bar() method
    plt.bar(X, Y)
    plt.xticks(rotation=90)
    plt.title(f'Most Frequent {item}s')
    plt.xlabel(f'{item}')
    plt.ylabel("Frequency")
    
    plt.savefig(f'{item0}-counts.png', bbox_inches='tight')

if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        raise Exception('Add argument to count.')
    main(args[0])