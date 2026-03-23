
README

MTH 9899 - Final Project
Authors: Lorenzo Senatori, Mat Zaga

FILES INCLUDED:
- main.py: The primary script to generate predictions.
- model.pkl: The saved, pre-trained ensemble model.
- Study Notebook.ipynb: The complete Jupyter notebook containing our EDA, feature engineering, model tuning, and evaluation. (All cells have been pre-run so outputs can be viewed without re-running).
- eval_2015.py: An auxiliary script used to compute the R^2 metrics for the 2015 holdout data.
- Copies: folder containing copy of Study Notebook, in case of issues in running it (see IMPORTANT NOTES)

ENVIRONMENT & DEPENDENCIES:
Please ensure the following packages are installed to replicate our exact environment (as noted in our white paper, different versions of pandas or scikit-learn may result in slight floating-point divergences during rolling window calculations):
- pandas
- scikit-learn
- xgboost

HOW TO RUN:
To run the main pipeline, run from terminal specifying the arguments and the mode. Pay attention to the folder the pickle file is in.


IMPORTANT NOTES:
- Runtime: main.py takes approximately 10 minutes to execute. This is due to the computational complexity of the rolling windows and cross-sectional normalizations for our intraday features.
- Memory / RAM: The dataset is quite large. If you are running the Jupyter Notebook on a machine with limited RAM, you may experience memory crashes if you attempt to run all cells at once. We have submitted the notebook with all outputs already generated for easy reviewing.
- Support: In case of any execution issues, missing documents, or bugs, please contact us immediately at lorenzo.senatori.baruchmfe@gmail.com and mat.zaga.baruchmfe@gmail.com, and we will respond right away.
