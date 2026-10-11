# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img
# MAGIC     src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png"
# MAGIC     alt="Databricks Learning"
# MAGIC   >
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Lab - Getting Started with Databricks for Machine Learning
# MAGIC
# MAGIC ## Overview
# MAGIC In this lab, we will construct a comprehensive ML model pipeline using Databricks. Initially, we will train and monitor our model using MLflow. Subsequently, we will register the model and advance it to the next stage. In the latter part of the lab, we will use Databricks Model Serving to deploy the registered model. Following deployment, we will interact with the model via a REST endpoint.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this lab, you will be able to:
# MAGIC - Ingest data from a CSV file and save it as a Unity Catalog managed table
# MAGIC - Create a feature table and store it in Databricks Feature Store
# MAGIC - Train a model and track it with MLflow
# MAGIC - Register a model in Unity Catalog and manage its stage
# MAGIC - Serve the registered model using Databricks Model Serving

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## REQUIRED - SELECT A COMPUTE ENVIRONMENT
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Select Compute</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Before starting this notebook, select the required compute environment listed below.</p>
# MAGIC <ul style="margin: 12px 0 0 16px; color: #333;">
# MAGIC <li><strong>Serverless Compute, ML Environment Version 5 (ML v5)</strong>: How to select an environment version (<a href="https://docs.databricks.com/aws/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">GCP</a>)</li>
# MAGIC </ul>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"><strong>NOTE:</strong> This notebook was <strong>developed and tested using Serverless compute with the ML v5 environment</strong>. Other compute options may work but are not guaranteed to behave the same or support all features demonstrated.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC To select the **ML v5** environment:
# MAGIC
# MAGIC 1. Click the environment button <img src="https://files.training.databricks.com/binder/prod_main/get-started-with-databricks-for-machine-learning-en_us-3.0.2/images/20261006T181945Z/Includes/images/environment.png" style="height: 32px; vertical-align: middle;" alt="Environment panel button in the notebook right sidebar"> on the right sidebar
# MAGIC 2. Open the **Environment version** dropdown
# MAGIC 3. Select **ML v5**
# MAGIC
# MAGIC If you do not see **Serverless** available, contact your workspace administrator.

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Classroom Setup
# MAGIC
# MAGIC Ensure you have completed all tasks in the previous notebooks. The next cell will create some Python variables that will be used in this notebook.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-Lab

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Data Ingestion
# MAGIC
# MAGIC ### B1. Airbnb Sample Data
# MAGIC
# MAGIC The first step in this lab is to ingest data from `.csv` files and save them as Unity Catalog managed tables.
# MAGIC
# MAGIC - Find `airbnb-cleaned-mlflow.csv` located in the folder `sf-listings` in the `labuser_#####_#######.get_started_ml.myfiles` volume.
# MAGIC - Replace `<FILL_IN>` with the full path to this CSV file in the cell below.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">If you need help, the cell immediately following the cell below contains the solution.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

## Copy and paste the location of the airbnb dataset
file_path = '<FILL_IN>'

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 1: Set the Dataset Path
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC file_path = f'/Volumes/dbacademy_airbnb/v01/sf-listings/airbnb-cleaned-mlflow.csv'
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

df = spark.read.format(<FILL_IN>).option("header", "true").load(<FILL_IN>)
display(df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 2: Read the CSV into a DataFrame
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC df = spark.read.format("csv").option("header", "true").load(file_path)
# MAGIC display(df)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC Let's preprocess this dataset since the schema shows all variables being of time `string`.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType, StringType

## Specify columns that should be treated as categorical (e.g., integers in categorical context)
categorical_columns = ['neighbourhood_cleansed', 'zipcode', 'property_type', 'room_type', 'bed_type']
for col in categorical_columns:
    df = df.withColumn(col, df[col].cast(StringType()))

## Specify columns that should remain as floats for machine learning
numerical_columns = ['host_total_listings_count', 'latitude', 'longitude', 'accommodates', 'bathrooms', 
                 'bedrooms', 'beds', 'minimum_nights', 'number_of_reviews', 'review_scores_rating',
                 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
                 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'price']
for col in numerical_columns:
    df = df.withColumn(col, df[col].cast(FloatType()))

df = df.withColumn("airbnb_id", F.monotonically_increasing_id()).select(['airbnb_id'] + numerical_columns + categorical_columns)

## Check the schema to confirm data type changes
df.printSchema()

# COMMAND ----------

df.write.mode('overwrite').saveAsTable(f'{DA.catalog_name}.{DA.schema_name}.airbnb_lab')

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Feature Engineering
# MAGIC
# MAGIC Next, using PySpark, create a DataFrame called `feature_df` that is the feature table. Recall that the feature table must contain a primary key and does not contain the target variable, which is `price` in our case.

# COMMAND ----------

feature_df = df.select(<FILL_IN>)

## Find rooms with a score of at least 6.0 and 80 reviews
feature_df = feature_df.filter((<FILL_IN>) & (<FILL_IN>))
display(feature_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 3: Build the Feature DataFrame
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC feature_df = df.select(['airbnb_id'] + numerical_columns)
# MAGIC
# MAGIC ## Find rooms with a score of at least 6.0 and 80 reviews
# MAGIC feature_df = feature_df.filter((df.review_scores_rating >= 6.0) & (df.number_of_reviews >= 80))
# MAGIC display(feature_df)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### C1. Write to Databricks Feature Store
# MAGIC
# MAGIC Write the features to Databricks Feature Store. Remember, we do not include our target variable.

# COMMAND ----------

## Write feature_df to Databricks Feature Store.
from databricks.feature_engineering import FeatureEngineeringClient

fs = FeatureEngineeringClient()

feature_df = feature_df.drop(<FILL_IN>)

fs.create_table(
    name=f"{DA.catalog_name}.{DA.schema_name}.airbnb_features",
    primary_keys = <FILL_IN>,
    df = <FILL_IN>,
    description = "This is the airbnb feature table",
    tags = {"source": "bronze", "format": "delta"}
    )

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 4: Write to Databricks Feature Store
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC ## Write feature_df to Databricks Feature Store.
# MAGIC from databricks.feature_engineering import FeatureEngineeringClient
# MAGIC
# MAGIC fs = FeatureEngineeringClient()
# MAGIC
# MAGIC feature_df = feature_df.drop("price")
# MAGIC
# MAGIC fs.create_table(
# MAGIC     name=f"{DA.catalog_name}.{DA.schema_name}.airbnb_features",
# MAGIC     primary_keys = ['airbnb_id'],
# MAGIC     df = feature_df,
# MAGIC     description = "This is the airbnb feature table",
# MAGIC     tags = {"source": "bronze", "format": "delta"}
# MAGIC     )
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Train a Model
# MAGIC
# MAGIC To summarize what you have accomplished so far:
# MAGIC 1. You have created a table that is a snapshot of the original dataset (Airbnb CSV file) called `airbnb_lab`.
# MAGIC 2. You have created a feature table and stored it in Databricks Feature Store called `airbnb_features`.
# MAGIC
# MAGIC Next, we will simulate the process of reading in these Unity Catalog managed tables and training a model. We will train a machine learning model and register it to Unity Catalog.

# COMMAND ----------

from sklearn.model_selection import train_test_split
## Read in the feature table airbnb_features from Unity Catalog using PySpark and store it as training_df
prediction_df = spark.read.format('delta').table(f'{DA.catalog_name}.{DA.schema_name}.<FILL_IN>').select(<FILL_IN>)
features_df = spark.read.format('delta').table(f'{DA.catalog_name}.{DA.schema_name}.<FILL_IN>')

## Join these two dataframes on airbnb_id
training_df = prediction_df.join(<FILL_IN>, on='airbnb_id').toPandas()

## Perform train-test split
X = training_df.drop(columns = [<FILL_IN>])
y = training_df[<FILL_IN>]

X_train, X_test, y_train, y_test = train_test_split(<FILL_IN>, <FILL_IN>, test_size=0.2, random_state=42)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 5: Load, Join, and Split the Data
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.model_selection import train_test_split
# MAGIC ## Read in the feature table airbnb_features from Unity Catalog using PySpark and store it as training_df
# MAGIC prediction_df = spark.read.format('delta').table(f'{DA.catalog_name}.{DA.schema_name}.airbnb_lab').select('airbnb_id','price')
# MAGIC features_df = spark.read.format('delta').table(f'{DA.catalog_name}.{DA.schema_name}.airbnb_features')
# MAGIC
# MAGIC ## Join these two dataframes on airbnb_id
# MAGIC training_df = prediction_df.join(features_df, on='airbnb_id').drop('airbnb_id')
# MAGIC training_pdf = training_df.toPandas()
# MAGIC
# MAGIC ## Perform train-test split
# MAGIC X = training_pdf.drop(columns = ['price'])
# MAGIC y = training_pdf['price']
# MAGIC
# MAGIC X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### D1. Model Tracking and Management with MLflow
# MAGIC
# MAGIC Next, you will use MLflow to track the training of your model.

# COMMAND ----------

username = spark.sql("SELECT current_user()").first()["current_user()"]

# COMMAND ----------

import mlflow
## Set the path for mlflow experiment
mlflow.set_experiment(f"/Users/{username}/<FILL_IN>")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 6: Set the MLflow Experiment Path
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC ## Set the path for mlflow experiment
# MAGIC mlflow.set_experiment(f"/Users/{username}/model-serving-experiment")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from mlflow.models.signature import infer_signature
## Start the MLflow run
with mlflow.start_run(run_name=<FILL_IN>) as run:
    ## Initialize the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=<FILL_IN>, random_state=42)

    ## Fit the model on the training data
    rf_classifier.fit(<FILL_IN>, <FILL_IN>)

    ## Make predictions on the test data
    y_pred = rf_classifier.predict(<FILL_IN>)

    ## Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(log_input_examples=<FILL_IN>, silent=True)
    ## Calculate F1 score with 'macro' averaging for multiclass
    mlflow.log_metric("test_f1", f1_score(<FILL_IN>, <FILL_IN>, average="macro"))
    ## mlflow.log_metric("test_f1", f1_score(y_test, y_pred))

    mlflow.sklearn.log_model(
        rf_classifier,
        artifact_path="model-artifacts",
        input_example=X_train[:3],
        signature=infer_signature(<FILL_IN>, <FILL_IN>),
    )

    model_uri = f"runs:/{run.info.run_id}/model-artifacts"

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 7: Train and Track the Model with MLflow
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.ensemble import RandomForestClassifier
# MAGIC from sklearn.metrics import f1_score
# MAGIC from mlflow.models.signature import infer_signature
# MAGIC ## Start the MLflow run
# MAGIC with mlflow.start_run(run_name="model-serving-run") as run:
# MAGIC     ## Initialize the Random Forest classifier
# MAGIC     rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# MAGIC
# MAGIC     ## Fit the model on the training data
# MAGIC     rf_classifier.fit(X_train, y_train)
# MAGIC
# MAGIC     ## Make predictions on the test data
# MAGIC     y_pred = rf_classifier.predict(X_test)
# MAGIC
# MAGIC     ## Enable automatic logging of input samples, metrics, parameters, and models
# MAGIC     mlflow.sklearn.autolog(log_input_examples=True, silent=True)
# MAGIC     ## Calculate F1 score with 'macro' averaging for multiclass
# MAGIC     mlflow.log_metric("test_f1", f1_score(y_test, y_pred, average="macro"))
# MAGIC     ## mlflow.log_metric("test_f1", f1_score(y_test, y_pred))
# MAGIC
# MAGIC     mlflow.sklearn.log_model(
# MAGIC         rf_classifier,
# MAGIC         artifact_path="model-artifacts",
# MAGIC         input_example=X_train[:3],
# MAGIC         signature=infer_signature(X_train, y_train),
# MAGIC     )
# MAGIC
# MAGIC     model_uri = f"runs:/{run.info.run_id}/model-artifacts"
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### D2. Register the Model
# MAGIC
# MAGIC Now, let's register the trained model in the model registry:
# MAGIC
# MAGIC 1. Use the logged model from the previous step.
# MAGIC 2. Provide a name and description for the model.
# MAGIC 3. Register the model to Unity Catalog.

# COMMAND ----------

## Modify the registry uri to point to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

sanitized_username = username.replace('.', '-').replace('@', '-')
model_name = f"gs_db_ml_LAB_{sanitized_username}"

## Register the model in the model registry
registered_model = mlflow.register_model(model_uri=model_uri, name=f'{DA.catalog_name}.{DA.schema_name}.{model_name}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### D3. Manage Model Stages
# MAGIC
# MAGIC As the model is registered into Model Registry, we can manage its stage using the UI or the API. In this lab, we will use the API to transition the registered model to the `staging` stage.

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
client = MlflowClient()

## Transition the model to the "lab-staging" stage
client.<FILL_IN>(
    name = <FILL_IN>,
    version = <FILL_IN>,
    stage = <FILL_IN>
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 8: Assign a Model Alias
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.tracking.client import MlflowClient
# MAGIC
# MAGIC ## Initialize an MLflow Client
# MAGIC client = MlflowClient()
# MAGIC
# MAGIC ## Assign a "staging" alias to model version 1
# MAGIC client.set_registered_model_alias(
# MAGIC     name= registered_model.name,  # The registered model name
# MAGIC     alias="lab-staging",  # The alias representing the staging environment
# MAGIC     version=registered_model.version  # The version of the model you want to move to "staging"
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Databricks Model Serving
# MAGIC
# MAGIC ### E1. Set Up Model Serving
# MAGIC
# MAGIC We can create Model Serving endpoints with the Databricks SDK or the Databricks UI. An endpoint can serve any registered model in the **Model Registry**.
# MAGIC
# MAGIC In order to keep it simple, in this lab, we are going to use the Model Serving UI for creating, managing, and using the Model Serving endpoints. We can create model serving endpoints with the **Serving** page UI or directly from the registered **Models** page.
# MAGIC
# MAGIC Let's go through the steps of creating a model serving endpoint on the Models page:
# MAGIC
# MAGIC 1. In the left navigation bar, right-click **Models** and select **Open Link in New Tab**.
# MAGIC 2. Select **Owned by me**.
# MAGIC 3. Select the model you want to serve under the **Name** column (the model we created earlier is called "gs_db_ml_LAB_your-username"). This will take you to Catalog Explorer.
# MAGIC 4. Click **Serve this model** in the top-right corner. This will take you to the **Serving endpoints** screen.
# MAGIC 5. In **General**, enter a name for the endpoint. Note that this name is used in the URL for the endpoint.
# MAGIC 6. There are several configurations under **Served entities** that we will not discuss here. Leave **Entity**, **Compute type**, and **Compute scale-out** at their default values. If you are using Databricks Free Edition, ensure **Scale to zero** is selected.
# MAGIC 7. Click **Create** in the lower-right corner.
# MAGIC
# MAGIC The endpoint will take about 12 minutes to provision.

# COMMAND ----------

# MAGIC %md
# MAGIC ### E2. Query the Serving Endpoint
# MAGIC
# MAGIC Let's use the deployed model for real-time inference. Here’s a step-by-step guide for querying an endpoint in Databricks Model Serving:
# MAGIC
# MAGIC 1. Go to the **Serving** endpoints page and select the endpoint you want to query.
# MAGIC 2. Click the **Use** button in the top-right corner.
# MAGIC 3. There are 4 methods for querying an endpoint: **browser**, **CURL**, **Python**, and **SQL**. For now, let's use the easiest method — querying right in the **browser** window. In this method, we need to provide the input parameters in JSON format. Recall that we registered an example with MLflow, which appears automatically when selecting **browser**.
# MAGIC 4. Click **Send request**.
# MAGIC 5. The **Response** field on the right panel will show the result of the inference.

# COMMAND ----------

# MAGIC %md
# MAGIC ## F. Conclusion
# MAGIC
# MAGIC In this lab, we explored the full potential of the Databricks Data Intelligence Platform for machine learning tasks. From data ingestion to model deployment, we covered essential steps such as data preparation, model training, tracking, registration, and serving. By utilizing MLflow for model tracking and management, and Model Serving for deployment, we demonstrated how Databricks offers a seamless experience.
# MAGIC
# MAGIC Key takeaways:
# MAGIC - Raw CSV data can be ingested and persisted as a governed Unity Catalog managed table
# MAGIC - A feature table is a Unity Catalog managed table with a primary key, stored in Databricks Feature Store
# MAGIC - MLflow tracks model training and lets you register the model to Unity Catalog
# MAGIC - Model aliases let you manage a registered model across lifecycle stages
# MAGIC - Databricks Model Serving deploys a registered model as a real-time REST endpoint

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>