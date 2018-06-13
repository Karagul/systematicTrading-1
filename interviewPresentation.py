#################
PseudoCode Presentation:

import numpy as np
import pylab as plt
import networkx as nx
import tensorflow as tf
import tensorflow.contrib.slim as slim


################################
################################
# Finding the groups:
################################
################################

# Assume we have the data for 24 stocks.
# We want to find the cointegrated stocks to facilitate a modified pairs trading strategy.

# Let us assume that the 15-day moving average(MA) is an underlying stable trend for
#   these stock prices and that the prices fluctuate randomly around this trend.
# We will be using 90 data points, each corresponding to the 15-day MA on a specific day,
#   for each stock.

# I have chosen to use the 15-day MA since we are trading on a medium term
# and it will be more responsive to short/medium term changes in the market than say a 90-day MA.

# this is the imported data for our 24 stocks (the 15-day MA for 90 days).

numStocks = 24
data = tf.Variable( tf.zeros( numStocks, 90 ), tf.float32, name='data' )  # this shows the shape of data
data = import15MAData() # Get the actual data


# Randomly rearrange the data.
# This function must also keep the association of labels.
data = randomizeBySlice(data)

# This normalizes all the data points so that they are between 0 and 1.  
# Each row in the tensor sums to 1
data = softmax( data )


################################
# Now that our data is prepared, we can get the initial centroids for k-means clustering
################################

# We will now use k-means clustering to group the data into 4 groups.
#   - we are hope to group the stocks into cointegrated groups.
#   - we are assuming that we've determined the optimal number of groups for this data something
#       is 4.

k = 4 # the number of groups.

# This contains the centroids for the k clusters
centroids = tf.Variable( tf.zeros( k, 90 ))

# This is how many stocks we are randomly assigning to a group.
groupCutoff = numStocks / k

# Here we are generating the k centroids.
# We take an equal number of stocks for each group and average them to generate a centroid.
for i in range(k):
    cur_centroid = tf.Variable( tf.zeros(1, 90) )
    for j in range(groupCutoff):
        # adds a new stock to the current centroid.
        cur_centroid = tf.add( cur_centroid, tf.slice( data, [ (j*i) + j], [90] ) 
        
    tf.div(cur_centroid, groupCutoff) # averages the centroid so that it stays between (0,1)
    tf.scatter_update( centroids, indices=[i], updates=cur_centroid)


# Now we want to apply a random offset to all the centroids to ensure they are not
#   clustered too close together.
offset = tf.random_uniform( [k, 90], minval=0.5, maxval=1.5, tf.float32)
data = tf.multiply( data, offset ) # multiply the tensors element wise.



################################
# Now that we have our centroids,
# we can simply use k-means clustering to group the stocks.
################################

def assign_to_nearest( stocks, centroids ):
    # assign each stock to the nearest centroid.
    # This will use the square difference between each data point in the stock
    # and the corresponding data point in the centroid.
    # ...
    # nearest indices is a vector of the index of each centroid that each stock
    #   is closest to.
    return nearest_indices

def update_centroids( stocks, nearest_indices, centroids ):
    # Take all the stocks that are in a cluster
    # Get a new average for this cluster
    # Update the corresponding centroid with the new average.
    return new_centroids


################################
# Now we run the algorithm and get our k-clusters and centroids.
################################

nearest_indices = tf.Variable( tf.zeros( numStocks) )
numIterations = 1000
for i in range(numIterations):
    nearest_indices = assign_to_nearest(data)
    centroids = update_centroids( data, nearest_indices, centroids)


################################
################################
# Now we have our groups of stocks. 
# We can set up a trading strategy using this information.
################################
################################

# Our trading strategy is as follows:
# Use a generalization of pair trading to identify stocks that are deviating
# from the expected value of the group.
# Then before we generate a signal,
# We verify that the the trend we found within the group (using pair trading),
# is sufficiently inconsistant with the overall momentum of the group.
# This is because if the deviation is too consistant with the trend of the group,
# it could indicate that the stock price may not return to a value that is close
# to the original moving average.
# For example, this will allow us to avoid buy signals on a stock that suddenly dropped
# if the other stocks in the group have also dropped and the momentum is downward.
# ie: If we want to go short, we don't want the momentum to be upward and the
#   stock deviation to also be upward.  And vice versa with if we are going long.



# This gets the momentum value for a group using 
# a moving average crossover technique.
# Requires: MA1 is a smaller moving average than MA2
def getMomentum( MA1, MA2 ):
    # Here we get the most recent intersection between the 15-day and the 45-day MAs.
    cur_intersect = getMostRecentIntersect(MA1, MA2)
    # here we can get the derivatives of the of each MA after the intersection.
    # We have 4 possible cases for what each of the two derivatives could be.
    # From this we can extract whether or not the stock currently seems to be
    # in an upturn or a downturn.
    # We can also use the magnitudes of the derivatives to determine 
    # the magnitude of the momentum.
    return momentum


data15MA = data # This is the 15-day MA data from before.
data45MA = [array of prepared 45-day MA data]

# this will store the momentums of each group.
# In this case, assume the momentum will be a positive or negative float.
momentums = tf.Variable( tf.zeros( numStocks ), tf.float32 ) 
def setGroupMomentums( groups )
    for i in range(k): # k is the number of groups/clusters
        # here we get the 45-day MA of the group.
        # We do this by adding up all stocks in the group and then averaging
        group45MA = getGroupMA( groups[i], data45MA) # this is a 1-by-90 vector
        # Similarly, we get the 15-day MA of the group
        group15MA = getGroupMA( groups[i], data15MA)
        momentums[i] = getMomentum( group15MA, group45MA )
        


# Now we will outline functions that will determine the if an individual stock
# is deviating from its group and whether or not this deviation is inconsistant
# with the overall momentum of the group.

# This is the vector containing the price data for each stock at a given time
prices = [array of the price of each stock]


# This gets whether to buy, sell, or do nothing from a stocks performance
# relative to its group.
def getPairSignal( stock, group ):
    price = prices[stock]
    mu = getGroupMean(group) # this gets the mean value of the group
    stdDev = getGroupStdDev(group) # this gets the standard deviation of the group
    # We get the zscore and use it to determine the signal
    zscore = (price - mu) / stdDev
    
    # if we are more or less that one standard deviation away from the mean, do something.
    if zscore > 1:
        return "Sell short"
    elif zscore < -1:
        return "Buy long"
    elif abs(zscore) < 0.5:
        return "clear positions"
    return "hold"


# We define a threshold value.  If the magnitude of the momentum is greater than
#   this value and the corresponding signal from the pair trading modification
#   is consistant with the momentum, we will cancel the signal.
threshold = 5


# This returns the final signal for a specific stock at a given point in time.
def getFinalSignal( stock, group ):
    # Here we have the overall momentum of the group
    momentum = momentums[ group ]    
    pairSignal = getPairSignal( stock, group )
    finalSignal = "hold"
    
    if( abs(momentum) > threshold ):
        if(momentum <= 0): # If there is a downward trend in the group
            if(pairSignal == "Buy Long"):
                # this is when the price of a stock drops below the average,
                # however, as previously explained, due to the momentum of the group
                # it risks not rebounding in value
                finalSignal = "Do not buy"
                
            else:
                finalSignal = pairSignal
                
        else: # this is when the momentum is positive.  Ie. an upward swing.
            if(pairSignal == "Sell short"):
                # this is when the price of the stock is rising, however,
                # given the group trend, it is likely to stay at a higher value.
                # Therefore it would make no sense to sell short.
                finalSignal = "Do not sell short"
                
            else:
                finalSignal = pairSignal
                
    else: 
        # this is when the momentum is not greater than the threshold.
        # ie. It is acceptable to base signals just off of the modified
        #   pair trading strategy.
        finalSignal = pairSignal
    
    return finalSignal


# Here we call our functions to get all the actions this trading strategy recommends
#   for each stock.
def getAllActions( stocks, groups ):
    setGroupMomentums( groups ) # this calculates the momentums for each group
    
    # we will store all of our signals corresponding to each stock in this array.
    actions = []
    for i in range(len(stocks)):
        action[i] = getFinalSignal( stock[i], groups[ stock[i] ])
    
    return actions





################################
################################
# Backtesting:
################################
################################

- ensure that the number of clusters we use is optimal.
    - For example, the stock market is not bipartite with only 2 groups of stocks that
            are cointegrated. So using 2 groups would not be optimal.
    - Also, breaking into 4000 groups (roughly the number of companies on the TSX+TSXV)
        is also not useful since each stock will likely be placed in its own
        group and we will not be able to use group momentum or pair trading.
        
- Optimize the length of windows for the MAs.
    - Where the intersections are and the timeframes we use will greatly affect
        what we calculate the group momentum to be.
    - Optimizing this will improve the group momentum determination and allow
        us to produce better signals as a result.
        
- Optimize our momentum threshold value.
    - This will reduce the risk of performing an action based off of the mean reversion
        strategy that will not return to its expected value.
        
- Note: We must be careful to account for what data we use when optimizing and
    backtesting this strategy.  For example,
    tuning the strategy using market data from the 2008 financial crisis may lead to
    poor performance in today's market.




################################
################################
# Additional concerns I have with my strategy:
################################
################################

- It assumes that we are able to properly find cointegrated stocks.
    - There must be something to ensure that if a stock is placed in a group,
        it actually belongs there.  This likely would have to involve
        some sort of labelling of the data.    
    
- I have not implemented something to find the optimal number of groups.
    - clearly, the stock market is not bipartite with only two groups of stocks that
        are cointegrated.  
    - However, breaking into 4000 groups (roughly the number of companies on the TSX+TSXV)
        is also not useful since each stock will likely be placed in its own
        group and we will not be able to use group momentum or pair trading.
        
- My particular clustering strategy does not account for overlapping groups.
    - While it allows for stocks to be cointegrated with multiple others,
        this is only within its group.

- Issues may arise from having varied data.
    - For example, if we have some high volatility stocks and then some very
        low volitility stocks, we may find that we our threshold for momentum
        is suitable for one and not the other.

- My "pair trading" strategy with groups may be flawed.  One could perhaps assume that because
    the clusters are cointegrated, that the other companies are substitutes
    for the current stock and it does allow one to perform the inverse action
    on the other stocks in the group, therefore remaining market neutral. However,
    I think that this may not be the best approach for a few reasons:
    1. The fees for buying many stocks against only one may be detrimental.
    2. How do you choose which other stocks in the group to invest in? An then,
        if we are not investing in all of the other stocks, why do not just use normal
        pair trading?









