


Airline Twitter Sentiment
-------------------------
[https://www.figure-eight.com/data-for-everyone/](https://www.figure-eight.com/data-for-everyone/)

A sentiment analysis job about the problems of each major U.S. airline. Twitter data was scraped from February of 2015 and contributors were asked to first classify positive, negative, and neutral tweets, followed by categorizing negative reasons (such as “late flight” or “rude service”).

Rows:  14,640

|Column|Type|Description        |
|------| :---: |----------------|
|airline_sentiment|str|sentiment of tweet - positive, negative, neutral|
|negativereason|str|Reason for negative sentiment|
|airline|str|Name of airline|
|name|str|Name of twitter account|
|retweet_count|int|Number of times retweeted|
|text|str|Text of tweet|
|tweet_created|datetime|date and time of tweet|		
|tweet_id|int|unique id of the tweet|
|tweet_location|str|geographical location of tweet|		


Boston House Prices
-------------------
[https://archive.ics.uci.edu/ml/machine-learning-databases/housing/]( https://archive.ics.uci.edu/ml/machine-learning-databases/housing/)

Contains information collected by the U.S. Census Service regarding housing in the Boston, Massachusetts area.

Originally published by Harrison, D. and Rubinfeld, D.L. `Hedonic prices and the demand for clean air', J. Environ. Economics & Management, vol.5, 81-102, 1978.

Rows: 506  

|Column|Type|Description        |
|------| :---: |----------------|
|crim|float|per capita crime rate by town|
|zn|float|proportion of residential land zoned for lots over 25,000 sq.ft|
|indus|float|proportion of non-retail business acres per town|
|chas|int|Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)|
|nox|float|nitric oxides concentration (parts per 10 million)|
|rm|float|average number of rooms per dwelling|
|age|float|proportion of owner-occupied units built prior to 1940|
|dis|float|weighted distances to five Boston employment centres|
|rad|float|index of accessibility to radial highways|
|tax|float|full-value property-tax rate per $10,000|
|ptratio|float|pupil-teacher ratio by town|
|b|float|1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town|
|lstat|float|% lower status of the population|
|medv|float|median value of owner-occupied homes in $1000’s|


Iris Species
------------
[https://en.wikipedia.org/wiki/Iris_flower_data_set](https://en.wikipedia.org/wiki/Iris_flower_data_set)

Taxonomic classification of of iris plants based on flower measurements.
  
Originally published by Robert Fisher `The use of multiple measurements in taxonomic problems', Annals of Eugenics,  vol. 7, 179-188, 1936.

Rows: 150

|Column|Type|Description        |
|------| :---: |----------------|
|SepalLength|float|Length of sepal in cm.|
|SepalWidth|float|Width of sepal in cm.|
|PetalLength|float|Length of petal in cm.|
|PetalWidth|float|Width of petal in cm.|
|Species|str|Species of iris plant|
