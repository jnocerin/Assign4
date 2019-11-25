import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def winePlot(Wine):
    # Make DataFrame from Wine Dataset
    wine = pd.DataFrame(Wine.data)
    #Get info on Dataset
    print(wine.describe())
    # Set Variable 3 to the target of initial dataset
    var3 = Wine.target
    # Define the column from dataset feature names
    wine.columns = Wine.feature_names

    #print(wine)
    #print(Wine.feature_names)
    # Create variable equal to number of columns for looping
    numCol = int(len(wine.columns))
    #print(numCol)
    # Create Dataframe to hold squared values
    distframe = pd.DataFrame(index=range(2,13),columns=['Squared Distance'])
    # Loop to calculate kMeans and store in new dataframe (distFrame)
    for i in range(2, numCol):
        kms = KMeans(n_clusters=i, n_init=1, max_iter=177)
        kms.fit(wine)
        distframe.loc[i] = kms.inertia_
    #    plt.plot(i, kms.inertia_)  - Tried to plot in the loop at first

    #plot the dataframe of kMeans using elbow
    plt.plot(range(2,numCol), distframe['Squared Distance'])
    plt.show()
