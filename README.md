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

We started using a very simple Logistic Regression model to get a baseline for the accuracy using only the boat's `course`, `lat` and `lon` and `speed`. We found that this simple model was close to being random: 

| Metric         | Class 0 | Class 1 | Accuracy | Macro Avg | Weighted Avg |
|---------------|---------|---------|----------|-----------|--------------|
| **Precision** | 0.61    | 0.36    | -        | 0.48      | 0.51         |
| **Recall**    | 0.84    | 0.14    | -        | 0.49      | 0.57         |
| **F1-Score**  | 0.70    | 0.20    | 0.57     | 0.45      | 0.51         |
| **Support**   | 33802   | 21481   | 55283    | 55283     | 55283        | 
