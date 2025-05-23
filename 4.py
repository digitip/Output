import pandas as pd 
def find_s_algorithm(file_path):
    df=pd.read_csv(file_path)
    
    print("Columns in the dataset:",df.columns)
    positive_examples=df[df.iloc[:,-1].str.lower() =='yes']
    
    if positive_examples.empty:
        print("<No positive examples found in the dataset>")
        return
    
    hypothesis=positive_examples.iloc[0,:-1].copy()
    for index,row in positive_examples.iterrows():
        for i in range(len(hypothesis)):
            if hypothesis.iloc[i]!=row.iloc[i]:
                hypothesis.iloc[i]='?'
    print("<"+",".join([f"{col}:{val}"for col,val in zip (df.columns[:-1],hypothesis)])+">")
    
find_s_algorithm("tennis.csv")