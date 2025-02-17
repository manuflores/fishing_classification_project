## Fishing classification project

### Installation 

Assuming you have python3 you can run locally by running 
```
pip install numpy pandas matplotlib seaborn geopy scikit-learn xgboost
```

The main script can be run using: 

```
python classification.py --data_dir path_to_parquet
```

You can also run the code interactively in Google colab: 

### Design choices and analysis

We started using a very simple Logistic Regression model to get a baseline for the accuracy using only the boat's `course`, `lat` and `lon` and `speed`. We found that this simple model was close to being random, with an accuracy on the test set of $0.57$.

| Metric         | Class 0 | Class 1 | Macro Avg | Weighted Avg |
|---------------|---------|---------|-----------|--------------|
| **Precision** | 0.61    | 0.36    | 0.48      | 0.51         |
| **Recall**    | 0.84    | 0.14    | 0.49      | 0.57         |
| **F1-Score**  | 0.70    | 0.20    | 0.45      | 0.51         |
| **Support**   | 33802   | 21481   | 55283     | 55283        |


After making some simple data exploration we found that whether boats are classified as fishing or not was **based on purely local behavior**, and not necessarily on the boat speed and or relative position. Hence we decided to compute a series of features to extract this local behavior. These are some of the variables we calculated: 

* speed difference
* acceleration 
* speed and acceleration summary stats: mean, min, max, std
* relative displacement with respect to a previous timestep
* previous 10 values of course,speed,lat,lon

After these transformations we improved the accuracy of a Logistic Regression model to $0.85$, which we believed was a good starting point to increase model complexity to arrive at a better accuracy. 

We found that a Random Forest model was a good performer achieving accuracy of $0.96$, and an F1-score of $0.97$. 

We performed K-Fold cross validation to ensure that this model was not overfitting and found a . 

Using feature importance analysis we found that the Random Forest performed well by aggregating speed values of previous the timesteps. With this result, we predict that using an RNN or Transformer model that ingests data from a window of $k$ previous timesteps would be an even better predictive model.

