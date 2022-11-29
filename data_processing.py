import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df = pd.read_csv('data/raw_data.csv')

    # Dropping columns
    df = df.drop(["EmployeeCount", "EmployeeNumber", "Over18", "RelationshipSatisfaction", "StandardHours"], axis = 1)

    # Formatting columns
    df['Attrition'] = df['Attrition'].replace(["Yes", "No"], [1, 0])
    df['Gender'] = df['Gender'].replace(["Male", "Female"], [0, 1])
    df['OverTime'] = df['OverTime'].replace(["Yes", "No"], [1, 0])

    # One-Hot Encoding
    cat_features = []
    for col, value in df.iteritems():
        if value.dtype == 'object':
            cat_features.append(col)
        
    num_features = df.columns.difference(cat_features)

    # Concat the two dataframes together columnwise
    df = pd.concat([df[num_features], pd.get_dummies(df[cat_features])], axis=1)

    df.to_csv("data/data.csv", index=False)

    # Generating train/test files
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)