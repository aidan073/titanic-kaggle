import pandas as pd

# get dataframe from csv
def get_df(data_path:str):
    return pd.read_csv(data_path)

# seperate labels from training df
def seperate_labels(df:pd.DataFrame):
    labels = list(df["Survived"])
    df.drop("Survived", axis=1, inplace=True)
    return labels

def get_prefix(ticket):
    lead = ticket.split(' ')[0][0]
    if lead.isalpha():
        return ticket.split(' ')[0]
    else:
        return 'NoPrefix'

def feature_engineering(df:pd.DataFrame):
    # get lastnames and family size
    df = df.assign(lastName=df["Name"].apply(lambda x: x.split(",")[0].strip().lower()))
    df.drop("Name", axis=1, inplace=True)
    df["familySize"] = df.groupby("lastName")["lastName"].transform("count")
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df["Cabin"] = df["Cabin"].fillna(df["Cabin"].mode()[0])
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # process tickets, referenced kaggle user "Oscar Takeshita"
    # df['Ticket'] = df['Ticket'].replace('LINE','LINE 0')
    # df['Ticket'] = df['Ticket'].apply(lambda x: x.replace('.','').replace('/','').lower())
    # df['Prefix'] = df['Ticket'].apply(lambda x: get_prefix(x))
    # df['TNumeric'] = df['Ticket'].apply(lambda x: int(x.split(' ')[-1])//1)
    # df['TNlen'] = df['TNumeric'].apply(lambda x : len(str(x)))
    # df['LeadingDigit'] = df['TNumeric'].apply(lambda x : int(str(x)[0]))
    # df['TGroup'] = df['Ticket'].apply(lambda x: str(int(x.split(' ')[-1])//10))
    # df = df.drop(columns=['Ticket','TNumeric','Pclass'])
    # df = df.join(pd.get_dummies(df[['Prefix', 'TGroup']], dtype=int))
    # df.head()  
    return df

# normalize features
def normalize():
    pass