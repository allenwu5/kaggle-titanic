import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from helper import *


def main():

    test_passenger_ids, dataset = get_dataset()

    features = extract_feature(dataset)

    # Train
    X = features[features['is_test'] == 0]
    y = X.pop('Survived')

    clf = xgb.XGBClassifier(
        max_depth=5,
        gamma=0.9,
        # subsample=0.8,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=.3, random_state=42)

    clf = clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    val_score = clf.score(X_val, y_val)
    print(f'train: {train_score}, val: {val_score}')

    # Test
    X_test = features[features['is_test'] == 1]
    X_test.pop('Survived')
    y_test = clf.predict(X_test)

    output = pd.DataFrame(
        {'PassengerId': test_passenger_ids, 'Survived': y_test})
    output['Survived'] = output['Survived'].astype(int)

    output[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
