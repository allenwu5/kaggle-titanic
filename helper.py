from collections import defaultdict

import pandas as pd

prefixs = defaultdict(int)

# https://www.kaggle.com/imoore/titanic-the-only-notebook-you-need-to-see/notebook
personal_titles = set(t.lower() for t in ["Dr.",
                                          "Jr.",
                                          "Mr.",
                                          "Mrs.",
                                          "Ms.",
                                          "Sr.",
                                          "Miss",
                                          "Ms",
                                          ])


def extract_feature(X):
    X['NameWordCount'] = X['Name'].apply(lambda x: len(x.split()))
    X['Personal Title'] = X['Name'].apply(extract_personal_title)

    # X['Embarked'] = X['Embarked'].fillna('C').map(
    #     {'S': 0, 'C': 1, 'Q': 2}).astype(int)

    X['Age'] = X['Age'].fillna(X['Age'].median())
    X['Fare'] = X['Fare'].fillna(X['Fare'].median())

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


def extract_personal_title(str):
    # abbreviations: titles with personal names
    #   https://www.btb.termiumplus.gc.ca/tpv2guides/guides/wrtps/index-eng.html?lang=eng&lettr=indx_catlog_a&page=9NBnYuQ324Yc.html
    for token in str.lower().split(" "):
        if token in personal_titles:
            return token
    return ""
