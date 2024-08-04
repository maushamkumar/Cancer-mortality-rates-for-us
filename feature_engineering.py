import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def bin_to_num(data):
    binnedinc = []
    for i in data["binnedinc"]:
        # remove the parentheses and brackets
        i = i.strip("()[]") 
        print(i)
        # split the string into a list after splitting by comma
        i = i.split(",")
        print(i)
        # convert the list to a tuple
        i = tuple(i) 
        print(i)
        # convert individual elements to float
        i = tuple(map(float, i)) 
        print(i)
        # convert the tuple to a list
        i = list(i) 
        print(i)
        # append the list to the binnedinc list
        binnedinc.append(i)
    data["binnedinc"] = binnedinc

    # make a new column lower and upper bound
    data["lower_bound"] = [i[0] for i in data["binnedinc"]]  # lower bound
    data["upper_bound"] = [i[1] for i in data["binnedinc"]]  # upper bound
    # and also median point
    data["median"] = (data["lower_bound"] + data["upper_bound"]) / 2
    # drop the binnedinc column
    data.drop("binnedinc", axis=1, inplace=True)
    return data


def cat_to_col(data):
    # make a new column by splitting the geography column
    data["county"] = [i.split(",")[0] for i in data["geography"]]
    data["state"] = [i.split(",")[1] for i in data["geography"]]
    # drop the geography column
    data.drop("geography", axis=1, inplace=True)
    return data


def one_hot_encoding(X):
    # select categorical columns
    categorical_columns = X.select_dtypes(include=["object"]).columns
    # one hot encode categorical columns
    one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    one_hot_encoded = one_hot_encoder.fit_transform(X[categorical_columns])
    # convert the one hot encoded array to a dataframe
    one_hot_encoded = pd.DataFrame(
        one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_columns)
    )
    # drop the categorical columns from the original dataframe
    X = X.drop(categorical_columns, axis=1)
    # concatenate the one hot encoded dataframe to the original dataframe
    X = pd.concat([X, one_hot_encoded], axis=1)
    return X
