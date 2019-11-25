from sklearn.datasets import load_boston
from sklearn.datasets import load_wine
import boston_work
import wine_work

# PART 1
# Find element with most influence on housing price in Boston ( nay be negative or positive influence.
# Use Linear regression and Boston Data Set

# Load Boston Data Set
Boston = load_boston()
# Call method to process
boston_work.bostonEval(Boston)


# PART 2
# Graph how squared distance decreases as clusters increase
# Use K-Means and Wine Data Set

# Load Dataset
Wine = load_wine()
# Call method to process
wine_work.winePlot(Wine)
