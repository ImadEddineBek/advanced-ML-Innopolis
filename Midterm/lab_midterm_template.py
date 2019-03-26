import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold


# In this assignment you are going to work with Pandas
# what is assessed is your ability to work with documentation
# overall you need to write 5 lines of code
# Go through the following code line by line and complete
# the functions fill_na and embed_categories

# Taks Description
# In current task you are going to do data pre-processing for KNN algorithm.
# KNN with euclidean distance metric is designed to work only with numerical features
# You are going to test KNN on modified Titanic dataset. You can get familiar with
# dataset header in the file titanic_modified.csv

# You are going to complete two types of pre-processing:
# 1. Eliminating Null (N/A) values
# 2. Mapping categorical features to binary indicator features


def load_data(path_to_csv, has_header=True):
    """
    Loads a csv file, the last column is assumed to be the output label
    All values are interpreted as strings, empty cells interpreted as empty
    strings

    returns: X - pandas DataFrame of shape (n,m) of input features
             Y - numpy array of output features of shape (n,)
    """
    if has_header:
        data = pd.read_csv(path_to_csv, header='infer')
    else:
        data = pd.read_csv(path_to_csv, header=None)
    X = data.loc[:, data.columns[:-1]]
    Y = data.loc[:, data.columns[-1]]
    return X, Y.values


def fill_na(data):
    """
    Iterates over columns of DataFrame X, any N/A values replaced by the most
    frequent element in the column
    :param X: DataFrame with input features
    :return: copy of the DataFrame with N/A replaced by the most frequent elements
    """
    ### Just to be safe, create a copy of current DataFrame
    X = data.copy()

    ### Hint: use the function mode from 'pandas.DataFrame.mode' 
    ### to find a the most frequent element value of each column
    ### pay attention to keys: axis, numeric_only, dropna

    moda_columns = None  # <<-- write code here

    for col_name in X.columns:  # current column name
        is_null = X[col_name].isnull()  # check for N/A values
        if is_null.any():
            ### Find the most frequent element value in the current column
            ### Hint: the value of moda_columns can be obtained as
            ### moda_columns[col_name]

            modal_value = X[col_name].mode().values[0]
            print(col_name, "modal value:", modal_value, end=" ")
            ### Replace N/A entries with most_common value
            ### Hint: slice DataFrame with X.loc[is_null, col_name]

            X.loc[is_null, col_name] = modal_value
            print("verification", X[col_name].isnull().any())

    ### Please visually check the first 10 rows of data
    ### Hint: try a comand from pandas 'DataFrame[column_name].values[0]'
    print(data.head(10))
    print(X.head(10))
    return X


def one_hot(data, dummy_columns=[]):
    """
    Replaces columns with categorical features by binary feature indicator columns
    >> print(X.info())
    You'll see something like that:
    Data columns (total 6 columns):
    Pclass      891 non-null int64
    Sex         891 non-null object
    Age         891 non-null float64
    SibSp       891 non-null int64
    Parch       891 non-null int64
    Embarked    891 non-null object
    dtypes: float64(1), int64(3), object(2)

    columns ['Sex', 'Embarked'] are cathegorical
    column ['Pclass'] is cathegorical as well. This is a passenger ticket class

    INPUT: 
    DataFrame with input features: cathegorical and numeric
    dummy_columns = column names which need to be transformed to dummies

    OUTPUT: 
    DataFrame with numeric feature indicators 
    """
    X = data.copy()
    # print(X.info())

    for col_name in dummy_columns:
        ### create a new DataFrame which represents a categorical column
        ### Hint: use a pd.get_dummies(data=None, prefix=None)

        dummy_df = pd.get_dummies(data=X[col_name], prefix=col_name)

        ### drop a categorical column from X
        ### Hint: use DataFrame.drop(labels=[None], inplaec=None, axis=None)

        X = X.drop(col_name, axis=1)

        ### join dummy columns with DataFrame
        ### Hint: use DataFrame.join([None])

        X = X.join(dummy_df)
    print(data.columns, X.columns)
    return X.values


def k_fold_validation(X, Y):
    kf = KFold(n_splits=50)
    fold_score = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        fold_score.append(f1_score(y_test, y_pred))
    return np.mean(fold_score)


def performance_test(X, Y):
    ### Test the performance with one hot categorical feature embeddings
    s1 = k_fold_validation(X, Y)
    print("Score for original features: ", s1)

    ### After checking the performance of this classification procedure
    ### apply a dimensionality reduction technique (PCA). It several
    ### purposes:
    ### 1. Implicit data normalization (PCA pre-processing)
    ### 2. Feature orthogonalization
    ### 3. Dimensionality reduction

    ### Apply Principal Component Analysis
    pca_components = X.shape[1] - 2  # reduce features dimension by 2
    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X)

    ### Test the performance with transformed features
    s2 = k_fold_validation(X_pca, Y)
    print("Score for transformed features: ", s2)
    ### You should observe gain around 1-3%
    print("Gain: ", s2 - s1)


X, Y = load_data("titanic_modified.csv")

X = fill_na(X)

X2 = one_hot(X, ['Sex', 'Embarked'])
X3 = one_hot(X, ['Sex', 'Embarked', 'Pclass'])

performance_test(X2, Y)
performance_test(X3, Y)

### MANDATORY! 
### write your observation and conclusion in the comment below
### observation:
# the conversion of the Pclass column to dummy variables improved the results
# specially after performing PCA, which is because the first representation of Pclass column
# Implied an order making the distance between 1-3 bigger than the distances 2-3 and 2-1
# Such order would be useful as a proxy for socio-economic status but eliminating it made the modal work better
# because in fact it remains a categorical variable (not nominal but ordinal hence there is an order)
# But the numbers 1 2 3 don't just imply an order but also a magnitude(distance) and such concept cannot be represented
# in an ordinal scale, hence making it a dummy variable both eliminated the order and the distance between values
# resulting in an improvement in the model.
