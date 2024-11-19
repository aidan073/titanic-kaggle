import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# get dataframe from csv
def get_df(data_path:str):
    return pd.read_csv(data_path)

# seperate labels from training df
def seperate_labels(train_df:pd.DataFrame, test_df:pd.DataFrame):
    labels = list(train_df["Survived"])
    train_df.drop("Survived", axis=1, inplace=True)
    train_df.drop("PassengerId", axis=1, inplace=True)

    ids = list(test_df["PassengerId"])
    test_df.drop("PassengerId", axis=1, inplace=True)
    return labels, ids

def get_prefix(ticket):
    lead = ticket.split(" ")[0][0]
    if lead.isalpha():
        return ticket.split(" ")[0]
    else:
        return "NoPrefix"

def feature_engineering(df:pd.DataFrame):
    # get lastnames and family size
    df = df.assign(Title=df["Name"].apply(lambda x: x.split(" ")[1].strip().lower())) # get Title i.e. Mr Mrs
    df["HasCabin"] = df["Cabin"].notnull().astype(int) # bool for if sample has a cabin
    df["Deck"] = df["Cabin"].str[0] # get deck
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

    # get dummies for categoricals
    df = pd.get_dummies(df, columns=["Embarked"], prefix="Embarked", drop_first=False)
    df = pd.get_dummies(df, columns=["Deck"], prefix="Deck", drop_first=False)
    df = pd.get_dummies(df, columns=["Title"], prefix="Title", drop_first=False)

    # normalize
    scaler = MinMaxScaler()
    df["Fare"] = scaler.fit_transform(df["Fare"].fillna(df["Fare"].median()).values.reshape(-1, 1))
    df["FamilySize"] = scaler.fit_transform((df["SibSp"] + df["Parch"] + 1).values.reshape(-1,1))
    df["Age"] = scaler.fit_transform(df["Age"].fillna(df["Age"].mean()).values.reshape(-1,1))

    # remove un-needed rows
    df = df.drop(columns=["Name","Cabin","SibSp", "Parch"])

    # process tickets, referenced kaggle user "Oscar Takeshita"
    df["Ticket"] = df["Ticket"].replace("LINE","LINE 0")
    df["Ticket"] = df["Ticket"].apply(lambda x: x.replace(".","").replace("/","").lower())
    df["Prefix"] = df["Ticket"].apply(lambda x: get_prefix(x))
    df["TNumeric"] = df["Ticket"].apply(lambda x: int(x.split(" ")[-1])//1)
    df["TNlen"] = df["TNumeric"].apply(lambda x : len(str(x)))
    df["LeadingDigit"] = df["TNumeric"].apply(lambda x : int(str(x)[0]))
    df["TGroup"] = df["Ticket"].apply(lambda x: str(int(x.split(" ")[-1])//10))
    df = df.join(pd.get_dummies(df[["Prefix", "TGroup"]], dtype=int))
    scaler.fit_transform(df["TNlen"].values.reshape(-1,1))
    scaler.fit_transform(df["LeadingDigit"].values.reshape(-1,1))
    df = df.drop(columns=["Ticket","TNumeric","Pclass", "Prefix", "TGroup"])
    return df