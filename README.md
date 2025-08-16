# Project California — Housing Price Prediction 🏠

This is my second Machine Learning project.  
I built a model to predict California housing prices using data from Kaggle.

## 🔧 Tech & Tools
- Python
- Pandas, NumPy
- Scikit-learn
- Random Forest Regressor (best model compared to Linear Regression & Decision Tree for this project)

## 📂 Files
- **Main.py** → main script with data processing, training & evaluation
- **requirements.txt** → dependencies
- **README.md** → project overview

## 📊 Model
- Tried Linear Regression, Decision Tree, Random Forest
- **Best model**: Random Forest Regressor

## 📁 Data
Dataset was taken from [Kaggle California Housing Dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices).  
(Not included here due to size — please download separately.)

## ▶️ Run the Project
```bash
pip install -r requirements.txt
python Main.py

```
## ✨ Results
- Random Forest performed best with the lowest RMSE
- Model + pipeline are saved for inference
