# Forest Cover Type Classification - Task 3 Results

## Overview
Multi-class classification of forest cover types using cartographic features from the UCI Covertype dataset.

## Dataset
- **Samples**: 581,012 (50,000 sampled for training)
- **Features**: 54 (10 quantitative, 4 wilderness areas, 40 soil types)
- **Target**: 7 cover types (Spruce/Fir, Lodgepole Pine, Ponderosa Pine, Cottonwood/Willow, Aspen, Douglas-fir, Krummholz)
- **Train/Test Split**: 80/20 with stratification

## Key Findings

### 1. Exploratory Data Analysis
- **Data Quality**: No missing values ✓
- **Class Imbalance**: Lodgepole Pine (48.76%) and Spruce/Fir (36.46%) dominate; Cottonwood/Willow only 0.47%
- **Feature Importance**: Elevation (28-39%), Roadways Distance (8-12%), Fire Points Distance (7-11%)
- **Correlations**: Elevation-Slope (-0.24), Hillshade features (0.59-0.78)

### 2. Model Performance

| Model | Accuracy | F1-Score | Improvement |
|-------|----------|----------|------------|
| Random Forest (Baseline) | 83.50% | 0.8305 | - |
| Gradient Boosting (Baseline) | 88.64% | 0.8849 | +5.14% |
| Random Forest (Tuned) | 85.49% | 0.8513 | +1.99% |
| **Gradient Boosting (Tuned)** | **88.97%** | **0.8880** | **+0.33%** |

### 3. Best Model Performance (Gradient Boosting - Tuned)
**Hyperparameters:**
- n_estimators: 150
- max_depth: 10
- learning_rate: 0.1

**Per-Class Performance:**
- Spruce/Fir: 90% precision, 88% recall
- Lodgepole Pine: 89% precision, 92% recall
- Ponderosa Pine: 84% precision, 91% recall
- Cottonwood/Willow: 80% precision, 73% recall (challenging class)
- Aspen: 77% precision, 47% recall (rare, underrepresented)
- Douglas-fir: 84% precision, 67% recall
- Krummholz: 92% precision, 85% recall

### 4. Key Insights
1. **Gradient Boosting outperforms Random Forest** by 5.14% on baseline models
2. **Elevation is the dominant feature** explaining majority of forest type variation
3. **Hyperparameter tuning provides marginal improvements** (0.33% for GB, 1.99% for RF)
4. **Class imbalance affects minority classes** (Aspen, Cottonwood/Willow) - consider SMOTE/stratification for future work
5. **Models handle majority classes well** (Spruce/Fir, Lodgepole Pine 88-92% recall)

## Implementation Details
- **XGBoost Issue**: Native XGBoost failed (OpenMP library missing on macOS), replaced with sklearn's GradientBoostingClassifier
- **Feature Scaling**: StandardScaler applied to quantitative features only
- **Data Sampling**: Used 50,000 samples for faster training while maintaining stratification
- **Grid Search**: 3-fold cross-validation on 10,000-sample subset for hyperparameter optimization

## Recommendations
1. Address class imbalance using SMOTE or weighted loss functions for minority classes
2. Ensemble methods combining predictions of both models
3. Feature engineering: create interaction terms between elevation and distance metrics
4. Collect more samples for minority classes (Aspen, Cottonwood/Willow)
5. Consider deep learning approaches (neural networks) with class weights

## Conclusion
Successfully built and optimized forest cover type classification model achieving **88.97% accuracy** with Gradient Boosting. The model effectively captures patterns based on elevation and landscape metrics, with consistent performance across 7-class multi-class classification task.
