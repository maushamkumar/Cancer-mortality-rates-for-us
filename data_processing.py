from sklearn.model_selection import train_test_split as tts


def find_constant_columns(dataframe):
    """
    This function takes in a dataframe and returns the columns that contain a single value.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe to be analyzed

    Returns:
    list: A list of columns that contain a single value
    """
    constant_columns = []
    for column in dataframe.columns:
        # Get unique values in the column
        unique_values = dataframe[column].unique()
        # check if the column contains only one unique value
        if len(unique_values) == 1:
            constant_columns.append(column)
    return constant_columns


def delete_constant_columns(dataframe, columns_to_delete):
    """
    This function takes in a dataframe and a list of columns to delete and deletes the columns that contain a single value.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe to be analyzed
    columns_to_delete (list): A list of columns to delete

    Returns:
    pandas.DataFrame: The dataframe with columns that contain a single value deleted
    """
    # Delete the specified columns
    dataframe = dataframe.drop(columns_to_delete, axis=1)
    return dataframe


def find_columns_with_few_values(dataframe, threshold):
    """
    This function takes in a dataframe and a threshold value as input and returns the columns that have less than the threshold number of unique values.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe to be analyzed
    threshold (int): The minimum number of unique values required for a column

    Returns:
    list: A list of columns that have less than the threshold number of unique values
    """
    few_values_columns = []
    for column in dataframe.columns:
        # Get the number of unique values in the column
        unique_values_count = len(dataframe[column].unique())
        # Check if the column has less than the threshold number of unique values
        if unique_values_count < threshold:
            few_values_columns.append(column)
    return few_values_columns




def find_duplicate_rows(dataframe):
    """
    This function takes in a dataframe as input and returns the rows that contain duplicate data.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe to be analyzed

    Returns:
    pandas.DataFrame: The dataframe containing duplicate rows
    """
    # Identify duplicate rows
    duplicate_rows = dataframe[dataframe.duplicated()]
    return duplicate_rows


def delete_duplicate_rows(dataframe):
    """
    This function takes in a dataframe as input and deletes the rows that contain duplicate data.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe to be analyzed

    Returns:
    pandas.DataFrame: The dataframe without duplicate rows
    """
    # Drop duplicate rows
    dataframe = dataframe.drop_duplicates(keep="first")
    return dataframe


def drop_and_fill(dataframe):
    # Get the columns with more than 50% missing values
    cols_to_drop = dataframe.columns[dataframe.isnull().mean() > 0.5]
    # Drop the columns
    dataframe = dataframe.drop(cols_to_drop, axis=1)
    # Fill the remaining missing values with the mean of the column
    dataframe = dataframe.fillna(dataframe.mean())
    return dataframe


def split_data(dataframe, target_column):
    """
    This function takes in a dataframe and a target column as input and splits the dataframe into a feature dataframe and a target dataframe.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe to be analyzed
    target_column (str): The name of the target column

    Returns:
    pandas.DataFrame: The dataframe containing the features
    pandas.DataFrame: The dataframe containing the target column
    """
    # Split the dataframe into a feature dataframe and a target dataframe
    X = dataframe.drop(target_column, axis=1)
    y = dataframe[target_column]
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=0)
    return (X_train, X_test, y_train, y_test)
