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
- **Main_old.py** → like main script but used for spliting training and testing data and to find best regression method for this project
  (Not used for model training)

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
- ![Model Comparison](<img width="1934" height="1349" alt="model_comparison" src="https://github.com/user-attachments/assets/c7fb009a-82b4-4763-bdfa-310909931721" />)
- ![Feature Importance](<img width="2970" height="1764" alt="feature_importance" src="https://github.com/user-attachments/assets/55e5b3cf-fc88-43c5-8c05-29cd47bfc12a" />)

