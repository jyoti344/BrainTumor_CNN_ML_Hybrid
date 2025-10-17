# BrainTumor_CNN_ML_Hybrid
Brain tumor detection using a hybrid approach:
 1. VGG16 extracts features from MRI images.
 2. PCA reduces dimensionality of the extracted features.
 3. SVC or Logistic Regression classifies the tumor type.
The project includes a Flask web app that allows uploading MRI images for real-time prediction with high accuracy.

## Prerequisites
Before running the project, make sure you have the following installed:
 1. [vscode](https://code.visualstudio.com/)
 2. [python](https://www.python.org/downloads/)

## Datasete
You can download the dataset here:
 1. [brain_tumor_MRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Set your dataset paths in the code:
~~~
    train_dir = 'C:/Users/Jyoti Prakash Dhala/OneDrive/Desktop/brain_tumer_dataset/Training'
    test_dir = 'C:/Users/Jyoti Prakash Dhala/OneDrive/Desktop/brain_tumer_dataset/Testing'
~~~
Make sure to replace these paths with your own dataset locations.
## Required Python Libraries
Install the following libraries before running the program:
 1. Flask
 2. tensorflow
 3. scikit-learn
 4. matplotlib
 5. numpy
 6. seaborn
 7. xgboost
 8. lightgbm
 9. joblib
 10. json
~~~
    pip install flask tensorflow scikit-learn matplotlib numpy seaborn xgboost lightgbm joblib
~~~
json is part of the standard Python library and does not need installation.

## Steps to Run the Project:
 1. Clone this repository:
 ~~~
  git clone https://github.com/jyoti344/BrainTumor_CNN_ML_Hybrid.git
  cd BrainTumor_CNN_ML_Hybrid
 ~~~
 2. Install the required libraries.
 3. Set the correct dataset paths in the training and testing scripts.
 3. Run the Flask web app:
  ~~~
  python app.py
  ~~~
 4. Open your browser and go to http://127.0.0.1:5000 to upload MRI images and get predictions.

## Other CNN Models
 In the directory other_cnn_model, you will find Jupyter notebooks (.ipynb) of other CNN architectures.
 1. You can compare the accuracy of the current VGG16 model with these models to evaluate performance.
 2. Use these notebooks to experiment and enhance your model.

## Project Workflow Diagram
 ~~~
                                        +----------------+
                                        |   Input MRI    |
                                        |   Images       |
                                        +-------+--------+
                                                |
                                                v
                                        +--------------------+
                                        | Feature Extraction |
                                        |      VGG16         |
                                        +--------+-----------+
                                                    |
                                                    v
                                        +----------------+
                                        | Dimensionality |
                                        |      Reduction |
                                        |       PCA      |
                                        +--------+-------+
                                                    |
                                                    v
                                    +-------------------------+
                                    | Classification Models   |
                                    | - SVC                   |
                                    | - Logistic Regression   |
                                    +-----------+-------------+
                                                |
                                                v
                                        +------------------+
                                        | Tumor Prediction |
                                        +------------------+

 ~~~

## web app interface:

![Flask App Screenshot](uploads\image.png)


