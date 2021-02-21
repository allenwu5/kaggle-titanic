from collections import defaultdict

import pandas as pd

prefixs = defaultdict(int)

# https://www.kaggle.com/imoore/titanic-the-only-notebook-you-need-to-see/notebook
personal_titles = set(t.lower() for t in ["Mr.",
                                          #   "Miss.",  != Miss
                                          "Mrs.",

                                          ])


def extract_feature(X):

    # Int64Index: 1309 entries, 0 to 417
    # Data columns (total 12 columns):
    # #   Column       Non-Null Count  Dtype
    # ---  ------       --------------  -----
    # 0   PassengerId  1309 non-null   int64
    # 1   Survived     891 non-null    float64
    # 2   Pclass       1309 non-null   int64
    # 3   Name         1309 non-null   object
    # 4   Sex          1309 non-null   object
    # 5   Age          1046 non-null   float64
    # 6   SibSp        1309 non-null   int64
    # 7   Parch        1309 non-null   int64
    # 8   Ticket       1309 non-null   object
    # 9   Fare         1308 non-null   float64
    # 10  Cabin        295 non-null    object
    # 11  Embarked     1307 non-null   object

    # Name ...
    X['NameWordCount'] = X['Name'].apply(lambda x: len(x.split()))
    X['Personal Title'] = X['Name'].apply(extract_personal_title)

    # For numeric data, "Age" missing 20% records, "Fare" missing only 1 record.
    # Age of couple might be close
    #   Name, Age, Ticket
    #   Clark, Mr. Walter Miller	27	13508
    #   Clark, Mrs. Walter Miller (Virginia McDowell)	26	13508

    X['Age'] = X['Age'].fillna(X['Age'].median())
    X['Fare'] = X['Fare'].fillna(X['Fare'].median())

    # For categorical data, "Embarked" missing 2 records, "Cabin" missing 77% records.
    # Most of "Embarked" are "S"
    # Do NOT turn it into integer, which performs bad ...
    # X['Embarked'] = X['Embarked'].fillna('S')

    X['CabinInitChar'] = X['Cabin'].fillna(
        '').apply(extract_cabin_initial_char)
    X['CabinNumber'] = X['Cabin'].fillna('').apply(extract_cabin_number)

    X['TicketPrefix'] = X['Ticket'].fillna('').apply(extract_ticket_prefix)
    X['TicketNumber'] = X['Ticket'].fillna('').apply(extract_ticket_number)

    X = X.drop(columns=['Name', 'Ticket', 'Cabin'])
    return pd.get_dummies(X)


def get_dataset():
    train_df = pd.read_csv('titanic/train.csv')
    train_df['is_test'] = 0

    test_df = pd.read_csv('titanic/test.csv')
    test_df['is_test'] = 1

    _ = train_df.pop('PassengerId')
    test_passenger_ids = test_df.pop('PassengerId')
    return test_passenger_ids, pd.concat([train_df, test_df])


def extract_cabin_initial_char(s):
    # C85 -> C
    return s[0] if s else ""


def extract_cabin_number(s):
    # C85 -> 85
    s = s.split(' ')[0]
    if len(s) > 1:
        return int(s[1:])
    else:
        return -1


def extract_ticket_number(s):
    # A/5 21171 -> 21171
    tokens = s.split(' ')
    if len(tokens) > 1:
        return int(tokens[-1])
    else:
        try:
            return int(tokens[0])
        except:
            # print(f'Fail to extract ticket number for: "{s}"')
            return -1


def extract_ticket_prefix(s):
    # A/5 21171 -> a5
    s = s.replace('.', '')
    s = s.replace('/', '')
    tokens = s.split(' ')
    if len(tokens) > 1:
        prefix = ''.join(tokens[0:-1])
    else:
        prefix = tokens[0]
        try:
            int(prefix)
            prefix = ""
        except:
            pass

    prefix = prefix.lower()
    prefixs[prefix] += 1
    return prefix[0:1]


def extract_personal_title(s):
    # abbreviations: titles with personal names
    #   https://www.btb.termiumplus.gc.ca/tpv2guides/guides/wrtps/index-eng.html?lang=eng&lettr=indx_catlog_a&page=9NBnYuQ324Yc.html
    for token in s.lower().split(" "):
        if token in personal_titles:
            return token
    return ""
