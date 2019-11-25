import pandas as pd
from sklearn.linear_model import LinearRegression

def bostonEval(Boston):
    # Convert to DataFrame
    boston = pd.DataFrame(Boston.data)
    boston.columns = Boston.feature_names
    # print(boston.head())
    # print(boston.columns)

    # Define the features from dataframe column
    features = boston.columns
    # Set variable 1 to the dataframe
    var1 = boston
    # Set Variable 2 to the target of initial dataset
    var2 = Boston.target
    # Define Linear Regression
    lr = LinearRegression()
    # Create Regression Model
    bosModel = lr.fit(var1, var2)

    # Create dataframe woth the features, contribution and absolute value of contribution to control for negitive being stronmgest contribution
    bosContributions = pd.DataFrame(zip(boston.columns, bosModel.coef_, abs(bosModel.coef_)),
                                    columns=['Feature', 'Contribution', 'Absolute'])
    # Sort based on absolute value
    bosContributionsSorted = bosContributions.sort_values('Absolute', ascending=False)
    # Print list sorted by absolutely value
    print(bosContributionsSorted)
    # print(bosContributionsSorted.iloc[0,0:2])
    print("The greatest contributor, positive or negitive was " + (str(bosContributionsSorted.iloc[0, 0:2])))
