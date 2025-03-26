# Sales Forecasting - README

## Project Overview
This project focuses on forecasting sales using various machine learning and deep learning models. The dataset consists of historical sales data, store information, oil prices, and holiday events. The goal is to build predictive models that can help businesses make informed decisions based on future sales trends.

## Dataset
The following datasets were used in this project:
- **train.csv**: Contains historical sales data.
- **test.csv**: Data for which predictions need to be made.
- **stores.csv**: Store-related metadata.
- **oil.csv**: Daily oil price data.
- **holidays_events.csv**: Holiday and event data.
- **sample_submission.csv**: Sample format for submission.

## Data Preprocessing
Data preprocessing steps include:
- Handling missing values (interpolation, forward fill, median imputation).
- Converting date columns to datetime format.
- Merging datasets based on relevant keys.
- Creating time-based features (year, month, day, day of the week, etc.).
- Generating moving averages and lag features for time series forecasting.
- Encoding categorical variables using OneHotEncoding.
- Normalizing data for deep learning models.

## Exploratory Data Analysis (EDA)
Several visualizations were created to understand trends and relationships:
- Sales trends over time.
- Monthly sales seasonality.
- Top 10 stores by total sales.
- Sales distribution across product categories.
- Correlation heatmap of numerical features.
- Scatter plot of oil prices vs. sales.

## Model Training and Evaluation
The following models were trained and evaluated for sales forecasting:

### 1. **Random Forest Regressor**
- Trained using historical sales data.
- Evaluated using RMSE, MAE, and R2 Score.

### 2. **XGBoost Regressor**
- Gradient boosting-based model.
- Evaluated using RMSE, MAE, and R2 Score.

### 3. **Long Short-Term Memory (LSTM) Network**
- Deep learning model designed for time series forecasting.
- Data was scaled and reshaped for sequential learning.
- Trained using Adam optimizer and MSE loss function.
- Visualized actual vs. predicted sales trends.


## Model Performance
| Model            | RMSE  |
|-----------------|-------|
| Random Forest   | 198.588188 |
| XGBoost        | 304.161669 |
| LSTM           | 269.780841 |

## Results and Conclusion
- The model with the lowest RMSE was selected as the best-performing model.
- Feature engineering played a crucial role in improving model performance.
- The project demonstrated the effectiveness of different forecasting techniques, highlighting trade-offs between model interpretability and accuracy.

## Future Improvements
- Hyperparameter tuning for improved performance.
- Incorporating external economic indicators.
- Experimenting with other deep learning architectures like Transformer-based models.
- Deploying the model for real-time forecasting.

## Repository Structure
```
├── data/                   # Raw and processed datasets
├── notebooks/              # Jupyter notebooks for analysis
├── models/                 # Trained models
├── scripts/                # Python scripts for data processing & modeling
├── results/                # Visualizations and evaluation metrics
├── README.md               # Project documentation
```

## Requirements
To run the project, install the required dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the preprocessing script:
   ```bash
   python scripts/preprocessing.py
   ```
2. Train models:
   ```bash
   python scripts/train_models.py
   ```
3. Evaluate models:
   ```bash
   python scripts/evaluate_models.py
   ```
4. Generate forecasts:
   ```bash
   python scripts/forecast.py
   ```

## Author
[Nallapu Naveen]

## License
This project is licensed under the MIT License.

