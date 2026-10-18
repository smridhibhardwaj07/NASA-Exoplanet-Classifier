# 🌌 NASA Exoplanet Classifier

### Can machine learning help us decide whether a possible exoplanet is actually worth investigating?

This project started with a simple idea: **take real astronomical observations and see whether a machine learning model can learn the difference between a confirmed exoplanet and a false positive.**

Using data from NASA's Kepler mission, I built a Random Forest classifier that analyzes properties of a planet candidate and its host star, then predicts whether the candidate is **CONFIRMED** or a **FALSE POSITIVE**.

The final result is not just a trained model. I also built an interactive Streamlit application so the model can actually be used instead of just sitting inside a Python file.

---

##  What the project does

The application takes nine astronomical measurements as input, including things such as:

- Orbital period
- Transit duration
- Transit depth
- Planet radius
- Equilibrium temperature
- Insolation flux
- Signal-to-noise ratio
- Stellar temperature
- Stellar radius

The Random Forest model uses these values to make a classification and also shows the probability associated with each class.

For example:

CONFIRMED       80%
FALSE POSITIVE  20%

So instead of simply saying *"yes" or "no"*, the application also gives an idea of how confident the model is about its prediction.

---

##  Why I chose this problem

Exoplanet discovery produces a huge amount of observational data, and not every interesting signal turns out to represent a real planet.

That made this a good machine learning problem to explore because it combines:

**real-world data + classification + scientific observations + model interpretation.**

I also wanted the project to go beyond simply training a model and printing an accuracy score.

---

##  How I built it

The project follows this pipeline:

NASA Kepler Data
       ↓
Data Cleaning
       ↓
Feature Selection
       ↓
Star-Based Train/Test Split
       ↓
Random Forest Classifier
       ↓
Model Evaluation
       ↓
Saved ML Model
       ↓
Streamlit Web Application

One important part of the project was using a **star-based split** rather than simply randomly splitting individual observations. This helps reduce the chance of related observations from the same star appearing on both sides of the train/test split.

---

##  How well did it perform?

On the test data, the model achieved approximately:

| Metric | Result |
|---|---:|
| Accuracy | **93%** |
| CONFIRMED F1-score | **0.89** |
| FALSE POSITIVE F1-score | **0.94** |
| Macro F1-score | **0.92** |

I also evaluated the model using a confusion matrix and a learning curve rather than relying on accuracy alone.

---

##  What did the model learn?

One of the more interesting parts of the project was looking at **feature importance**.

The Random Forest model found that some of the strongest contributors to its decisions were:

1. **Planet Radius**
2. **Model SNR**
3. **Orbital Period**
4. **Transit Duration**
5. **Insolation Flux**

This was useful because it gave me something more meaningful than just a prediction — I could actually investigate **which measurements the model was relying on most heavily**.

The Streamlit application displays these feature importances as a chart.

---

##  Turning the model into an application

After training the model, I saved it as a `.pkl` file and connected it to a Streamlit interface.

The application allows a user to:

1. Enter astronomical measurements
2. Run the trained model
3. See the predicted class
4. See prediction probabilities
5. Explore feature importance
6. View the model's performance metrics

This turned the project from a machine learning experiment into something that can actually be interacted with.

---

##  Tech Stack

**Language**
- Python

**Machine Learning**
- Scikit-learn
- Random Forest

**Data Processing**
- Pandas

**Model Storage**
- Joblib

**Web Application**
- Streamlit

**Dataset**
- NASA Kepler exoplanet observations

---

##  Project Structure

NASA-Exoplanet-Classifier/
│
├── model/
│   └── random_forest_model.pkl
│
├── app.py
├── predict.py
├── train.py
├── requirements.txt
└── README.md


## Tech Stack 

**train.py**  
Cleans the dataset, prepares the features, trains the Random Forest model, evaluates it, and saves the trained model.

**predict.py**  
Allows predictions to be made directly from the terminal.

**app.py**  
Runs the interactive Streamlit application.

**random_forest_model.pkl**  
Contains the trained machine learning model.

**requirements.txt**  
Lists the Python libraries required to run the project.

---

##  Running the project locally

Clone the repository and open the project folder.

Install the dependencies:
py -m pip install -r requirements.txt

Then start the application:
py -m streamlit run app.py

The application will open in your browser.

---

##  What I would improve next

This is a first version of the project, so there are several directions I would like to explore:

- Compare Random Forest with models such as SVM and Gradient Boosting
- Tune the model's hyperparameters
- Experiment with additional astronomical features
- Add model explainability beyond basic feature importance
- Deploy the application online
- Improve the handling of unseen or unusual input values

---

## What I learned from building it

The biggest takeaway from this project was that building an ML application is more than just calling a classifier.

I had to work through the entire process — **understanding the data, cleaning it, deciding what to predict, splitting the data carefully, training the model, evaluating it, saving it, and finally putting it behind an interface that someone else can use.**

That end-to-end process is what makes this project valuable to me.

---

##  Author 

**Smridhi Bhardwaj**  
Computer Science Engineering Student