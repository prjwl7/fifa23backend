import pandas as pd

url = 'https://drive.google.com/file/d/1ZDEEBvZIPGDI0hyxi7WPhcRxEi7r7AEm/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df = pd.read_csv(path ,dtype='object')
print(df.head())