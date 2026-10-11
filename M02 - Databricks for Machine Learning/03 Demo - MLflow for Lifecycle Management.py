# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img
# MAGIC     src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png"
# MAGIC     alt="Databricks Learning"
# MAGIC   >
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Demo - MLflow for Lifecycle Management
# MAGIC
# MAGIC ## Overview
# MAGIC In this demo, we will explore the model development lifecycle using MLflow. You’ll train a model, log artifacts, metrics, and parameters using MLflow, and then register the model in Unity Catalog for production readiness.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this demo, you will be able to:
# MAGIC - Train and log models using MLflow
# MAGIC - Track experiments and view runs using MLflow UI
# MAGIC - Register a model in Unity Catalog
# MAGIC - Use model version aliases like `staging` and `dev`

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### A1. Run the Classroom Setup
# MAGIC
# MAGIC To get into the lesson, we first need to build some data assets and define configuration variables required for this demonstration.
# MAGIC
# MAGIC This Model Serving Endpoint will be available to everyone and deployed once. Once it is created, you will see the message `"Endpoint 'firstName-lastName-endpoint' already exists."` when trying to create it again. This is to limit resource deployment within this workspace. The first creation will take between 5 and 10 minutes.
# MAGIC
# MAGIC **Do not delete or modify this endpoint in any way.**
# MAGIC We will cover how to create the model serving endpoint later in this demo.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-mlflow

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Datasets Location: {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## B. Introduction to MLflow for Lifecycle Management
# MAGIC
# MAGIC **MLflow** is an open-source platform, created by Databricks, for managing the end-to-end machine learning lifecycle. From preparing raw data to deploying models, Databricks handles every stage of the ML lifecycle.
# MAGIC
# MAGIC <style>
# MAGIC @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
# MAGIC .mlf2{font-family:'DM Sans','Segoe UI',Arial,sans-serif;max-width:1120px;margin:18px auto;color:#1B3139;}
# MAGIC .mlf2 *{box-sizing:border-box;}
# MAGIC .mlf2-grid{display:flex;gap:30px;align-items:center;flex-wrap:wrap;justify-content:center;}
# MAGIC .mlf2-left{flex:1 1 340px;min-width:290px;display:flex;flex-direction:column;gap:12px;}
# MAGIC .mlf2-lead{font-size:14px;font-weight:700;color:#5A6F77;letter-spacing:.3px;}
# MAGIC .comp{background:#fff;border-radius:12px;border-left:5px solid var(--acc);box-shadow:0 4px 12px rgba(27,49,57,0.08);padding:14px 16px;}
# MAGIC .comp-t{font-size:15px;font-weight:800;color:var(--acc);margin-bottom:3px;}
# MAGIC .comp-d{font-size:13px;color:#1B3139;line-height:1.45;}
# MAGIC .mlf2-right{flex:1 1 460px;min-width:320px;display:flex;justify-content:center;}
# MAGIC </style>
# MAGIC <div class="mlf2">
# MAGIC <div class="mlf2-grid">
# MAGIC <div class="mlf2-left"><div class="mlf2-lead">MLflow has four main components:</div><div class="comp" style="--acc:#4299E0;"><div class="comp-t">Model Tracking</div><div class="comp-d">Record and query experiments &mdash; code, data, configuration, and results.</div></div><div class="comp" style="--acc:#FFAB00;"><div class="comp-t">Projects</div><div class="comp-d">Package ML code so it reproduces and runs consistently on any platform.</div></div><div class="comp" style="--acc:#00A972;"><div class="comp-t">Models</div><div class="comp-d">Store models in a standard format compatible with many deployment tools.</div></div><div class="comp" style="--acc:#FF5F46;"><div class="comp-t">Unity Catalog Model Registry</div><div class="comp-d">Centralized, governed model lifecycle &mdash; versioning, approvals, and deployment.</div></div></div>
# MAGIC <div class="mlf2-right"><svg width="100%" viewBox="0 0 730 520" role="img" style="font-family:'DM Sans','Segoe UI',Arial,sans-serif;max-width:540px;">
# MAGIC   <title>The MLflow machine learning lifecycle</title>
# MAGIC   <desc>A clockwise cycle: Raw Data, then Data Preparation, then Training, then Deployment, and back to Raw Data, all managed with MLflow.</desc>
# MAGIC   <defs>
# MAGIC     <marker id="mlf-ar" markerUnits="userSpaceOnUse" markerWidth="17" markerHeight="17" refX="4" refY="8.5" orient="auto"><path d="M1 1 L15 8.5 L1 16 Z" fill="#1B5162"/></marker>
# MAGIC     <filter id="mlf-sh" x="-30%" y="-40%" width="160%" height="180%"><feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#1B3139" flood-opacity="0.18"/></filter>
# MAGIC     <filter id="mlf-shC" x="-25%" y="-25%" width="150%" height="150%"><feDropShadow dx="0" dy="7" stdDeviation="14" flood-color="#1B3139" flood-opacity="0.13"/></filter>
# MAGIC   </defs>
# MAGIC   <ellipse cx="365" cy="250" rx="232" ry="178" fill="none" stroke="#EEEDE9" stroke-width="11"/>
# MAGIC   <path d="M140.9 203.9 A 232 178 0 0 1 235.3 102.4" fill="none" stroke="#1B5162" stroke-width="11" stroke-linecap="round" marker-end="url(#mlf-ar)"/>
# MAGIC   <path d="M494.7 102.4 A 232 178 0 0 1 589.1 203.9" fill="none" stroke="#1B5162" stroke-width="11" stroke-linecap="round" marker-end="url(#mlf-ar)"/>
# MAGIC   <path d="M589.1 296.1 A 232 178 0 0 1 466.7 410.0" fill="none" stroke="#1B5162" stroke-width="11" stroke-linecap="round" marker-end="url(#mlf-ar)"/>
# MAGIC   <path d="M263.3 410.0 A 232 178 0 0 1 140.9 296.1" fill="none" stroke="#1B5162" stroke-width="11" stroke-linecap="round" marker-end="url(#mlf-ar)"/>
# MAGIC   <circle cx="365" cy="250" r="132" fill="#FFFFFF" stroke="#EEEDE9" stroke-width="1.5" filter="url(#mlf-shC)"/>
# MAGIC   <svg x="265" y="213" width="200" height="73" viewBox="0 0 109 40"><path d="M0 31.0316V15.5024H3.54258V17.4699C4.43636 15.8756 6.38278 15.045 8.13589 15.045C10.178 15.045 11.9636 15.9713 12.7943 17.7895C14.0096 15.7474 15.8278 15.045 17.8373 15.045C20.6469 15.045 23.3263 16.8325 23.3263 20.9493V31.0316H19.7531V21.5541C19.7531 19.7359 18.8268 18.3636 16.7522 18.3636C14.8057 18.3636 13.5292 19.8947 13.5292 21.8086V31.0297H9.89282V21.5541C9.89282 19.7684 8.99522 18.3732 6.88995 18.3732C4.91292 18.3732 3.66699 19.8412 3.66699 21.8182V31.0392Z" fill="#333333"/><path d="M27.8546 31.0316V7.92917H31.5579V31.0316Z" fill="#333333"/><path d="M30.0708 39.4871C30.9033 39.7187 31.6517 39.8699 33.2402 39.8699C36.1933 39.8699 39.6765 38.2048 40.5933 33.5311L44.3789 14.7923H50.0076L50.6947 11.6746H45.0086L45.7741 7.95023C46.3598 5.05837 47.9598 3.59234 50.5282 3.59234C51.1962 3.59234 51.0086 3.64975 51.6038 3.76267L52.4268 0.570327C51.6344 0.333006 50.9244 0.191379 49.378 0.191379C47.7454 0.166831 46.1514 0.688934 44.8497 1.67463C43.4086 2.78851 42.4574 4.42487 42.023 6.53779L40.9569 11.6746H35.9234L35.5119 14.7943H40.3349L36.8593 32.1206C36.4765 34.0861 35.3588 36.4364 32.1512 36.4364C31.4239 36.4364 31.688 36.3809 31.0297 36.2737Z" fill="#0194E2"/><path d="M53.3416 30.9053H49.6402L54.7139 7.59616H58.4153Z" fill="#0194E2"/><path d="M71.8067 16.4766C68.5762 14.2161 64.1778 14.6606 61.4649 17.5216C58.7519 20.3826 58.5416 24.7984 60.9703 27.9043L63.3952 26.1244C62.1915 24.6312 61.9471 22.5815 62.7658 20.8471C63.5845 19.1127 65.3224 17.9987 67.2402 17.979V19.8737Z" fill="#43C9ED"/><path d="M62.6179 29.4717C65.8484 31.7322 70.2468 31.2877 72.9597 28.4267C75.6727 25.5657 75.883 21.1499 73.4543 18.044L71.0294 19.8239C72.2331 21.3171 72.4775 23.3668 71.6588 25.1012C70.8401 26.8356 69.1022 27.9496 67.1844 27.9693V26.0746Z" fill="#0194E2"/><path d="M78.0919 15.4928H82.1359L82.9588 26.1053L88.7177 15.4928L92.5569 15.5483L94.0651 26.1053L99.1387 15.4928L102.84 15.5483L95.1617 31.0412H91.4603L89.6765 19.9349L83.7818 31.0412H79.9426Z" fill="#0194E2"/><path d="M105.072 15.7684H104.306V15.5024H106.151V15.7741H105.386V18.0172H105.072Z" fill="#0194E2"/><path d="M106.614 15.5024H106.997L107.479 16.8421C107.541 17.0143 107.598 17.1904 107.657 17.3665H107.675C107.734 17.1904 107.788 17.0143 107.847 16.8421L108.325 15.5024H108.708V18.0172H108.41V16.6297C108.41 16.4096 108.434 16.1072 108.45 15.8832H108.434L108.243 16.4574L107.768 17.7608H107.56L107.079 16.4593L106.888 15.8852H106.873C106.89 16.1091 106.915 16.4115 106.915 16.6316V18.0191H106.624Z" fill="#0194E2"/></svg>
# MAGIC   <rect x="20.0" y="224.0" width="138" height="52" rx="15" fill="#FF5F46" filter="url(#mlf-sh)"/>
# MAGIC   <circle cx="47.0" cy="250.0" r="14" fill="#FFFFFF"/>
# MAGIC   <text x="47.0" y="255.0" text-anchor="middle" font-size="16" font-weight="800" fill="#FF5F46">1</text>
# MAGIC   <text x="104.0" y="256.0" text-anchor="middle" font-size="17" font-weight="800" fill="#FFFFFF">Raw Data</text>
# MAGIC   <rect x="256.0" y="45.0" width="218" height="52" rx="15" fill="#FF5F46" filter="url(#mlf-sh)"/>
# MAGIC   <circle cx="283.0" cy="71.0" r="14" fill="#FFFFFF"/>
# MAGIC   <text x="283.0" y="76.0" text-anchor="middle" font-size="16" font-weight="800" fill="#FF5F46">2</text>
# MAGIC   <text x="380.0" y="77.0" text-anchor="middle" font-size="17" font-weight="800" fill="#FFFFFF">Data Preparation</text>
# MAGIC   <rect x="572.0" y="224.0" width="138" height="52" rx="15" fill="#FF5F46" filter="url(#mlf-sh)"/>
# MAGIC   <circle cx="599.0" cy="250.0" r="14" fill="#FFFFFF"/>
# MAGIC   <text x="599.0" y="255.0" text-anchor="middle" font-size="16" font-weight="800" fill="#FF5F46">3</text>
# MAGIC   <text x="656.0" y="256.0" text-anchor="middle" font-size="17" font-weight="800" fill="#FFFFFF">Training</text>
# MAGIC   <rect x="286.0" y="403.0" width="158" height="52" rx="15" fill="#FF5F46" filter="url(#mlf-sh)"/>
# MAGIC   <circle cx="313.0" cy="429.0" r="14" fill="#FFFFFF"/>
# MAGIC   <text x="313.0" y="434.0" text-anchor="middle" font-size="16" font-weight="800" fill="#FF5F46">4</text>
# MAGIC   <text x="380.0" y="435.0" text-anchor="middle" font-size="17" font-weight="800" fill="#FFFFFF">Deployment</text></svg></div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Data Preparation for MLflow
# MAGIC
# MAGIC For this first part, we will read in a customer table, create a feature table, and register that feature table to Feature Engineering in Unity Catalog. This is all performed in the background and was covered in **02 Demo - Exploratory Data Analysis (EDA) and Feature Engineering**. Next, we will separate out our features from the target variable and perform a train-test split.
# MAGIC
# MAGIC Note: We are working with a pandas DataFrame called `training_df`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### C1. Load the Feature Table
# MAGIC
# MAGIC Run the next cell to load the feature table as a pandas DataFrame.

# COMMAND ----------

# Load the feature table
training_df = spark.read.format('delta').table('wine_quality_features').toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### C2. Train-Test Split
# MAGIC
# MAGIC The next cell prepares the data for training:
# MAGIC
# MAGIC 1. Define a mapping to convert the `pHCategory` labels to numeric values.
# MAGIC 2. Apply the mapping to the `pHCategory` column.
# MAGIC 3. Store the features as `X` and the target variable (`quality`) as `y`.
# MAGIC 4. Split the data into training and testing sets.

# COMMAND ----------

from sklearn.model_selection import train_test_split

# Define the mapping
mapping = {'Low': 0.0, 'Average': 1.0, 'High': 2.0}

# Apply the mapping to the 'pHCategory' column
training_df['pHCategory'] = training_df['pHCategory'].map(mapping)
# Use the training dataset to store variables X, the features, and y, the target variable.
X = training_df.drop(columns = ["quality", "wine_id"])
y = training_df["quality"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC Run the next cell to view the prepared training DataFrame.

# COMMAND ----------

display(training_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## D. Training and Model Tracking/Management with MLflow
# MAGIC
# MAGIC Tracking your experiments is made easier with the **Databricks Experiments UI**, and **MLflow Model Tracking** simplifies monitoring and managing machine learning model development.
# MAGIC
# MAGIC <style>
# MAGIC @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
# MAGIC .mt{font-family:'DM Sans','Segoe UI',Arial,sans-serif;max-width:1080px;margin:16px auto;color:#1B3139;}
# MAGIC .mt *{box-sizing:border-box;}
# MAGIC .mt-lead{font-size:13px;font-weight:800;letter-spacing:.6px;text-transform:uppercase;color:#5A6F77;margin:22px 0 12px;}
# MAGIC .mt-row{display:flex;gap:16px;flex-wrap:wrap;}
# MAGIC .mt-card{flex:1 1 260px;min-width:240px;background:#fff;border-radius:14px;border-top:5px solid var(--acc);box-shadow:0 6px 14px rgba(27,49,57,0.09);padding:18px 18px 16px;}
# MAGIC .mt-badge{width:46px;height:46px;border-radius:12px;background:var(--tint);color:var(--acc);display:inline-flex;align-items:center;justify-content:center;margin-bottom:12px;}
# MAGIC .mt-badge svg{width:26px;height:26px;}
# MAGIC .mt-t{font-size:16px;font-weight:800;color:var(--acc);margin-bottom:5px;line-height:1.2;}
# MAGIC .mt-d{font-size:13.5px;color:#1B3139;line-height:1.5;}
# MAGIC .mt-mcard{flex:1 1 320px;min-width:280px;display:flex;gap:14px;align-items:flex-start;background:#fff;border-radius:14px;border-left:5px solid var(--acc);box-shadow:0 6px 14px rgba(27,49,57,0.09);padding:18px;}
# MAGIC .mt-mcard .mt-badge{flex:0 0 46px;margin-bottom:0;}
# MAGIC .mt-mtxt .mt-t{margin-bottom:3px;}
# MAGIC .mt-panel{background:#F8F9FC;border:1px solid #EEEDE9;border-radius:14px;padding:20px;margin-top:12px;}
# MAGIC .mt-panel-h{font-size:15px;font-weight:800;color:#1B3139;text-align:center;margin-bottom:14px;}
# MAGIC .mt-panel-h b{color:#FF5F46;}
# MAGIC .mt-chips{display:flex;flex-wrap:wrap;gap:10px;justify-content:center;}
# MAGIC .mt-chip{background:#fff;color:var(--acc);border:2px solid var(--acc);font-size:13.5px;font-weight:700;border-radius:22px;padding:7px 16px;}
# MAGIC .mt-take{display:flex;gap:12px;align-items:center;background:#FFF6F4;border:1px solid #FADAD3;border-left:5px solid #FF5F46;border-radius:12px;padding:15px 18px;margin-top:20px;box-shadow:0 4px 10px rgba(27,49,57,0.06);}
# MAGIC .mt-take-i{flex:0 0 30px;width:30px;height:30px;border-radius:8px;background:#FF5F46;color:#fff;display:inline-flex;align-items:center;justify-content:center;font-size:17px;font-weight:800;}
# MAGIC .mt-take p{margin:0;font-size:13.5px;line-height:1.5;color:#1B3139;}
# MAGIC .mt-take b{color:#1B3139;}
# MAGIC .mt-docs{text-align:right;margin-top:14px;}
# MAGIC .mt-docs a{font-size:12.5px;font-weight:700;color:#2272B4;text-decoration:none;}
# MAGIC </style>
# MAGIC <div class="mt">
# MAGIC <div class="mt-lead">Databricks Experiments UI</div>
# MAGIC <div class="mt-row"><div class="mt-card" style="--acc:#4299E0;--tint:rgba(66,153,224,0.12);"><span class="mt-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/></svg></span><div class="mt-t">Metadata Visibility</div><div class="mt-d">View each experiment&#8217;s name, owner, creation date, and last modification at a glance.</div></div><div class="mt-card" style="--acc:#00A972;--tint:rgba(0,169,114,0.12);"><span class="mt-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 5h18l-7 8v6l-4-2v-4L3 5Z"/></svg></span><div class="mt-t">Experiment Management</div><div class="mt-d">Filter experiments and search by name or status to keep everything organized.</div></div><div class="mt-card" style="--acc:#FFAB00;--tint:rgba(255,171,0,0.14);"><span class="mt-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3 3 8l9 5 9-5-9-5Z"/><path d="M3 13l9 5 9-5"/></svg></span><div class="mt-t">Integrated Tracking</div><div class="mt-d">Track both traditional and AutoML models together in one unified platform.</div></div></div>
# MAGIC <div class="mt-lead">MLflow Model Tracking &mdash; two ways to log</div>
# MAGIC <div class="mt-row"><div class="mt-mcard" style="--acc:#1B5162;--tint:rgba(27,81,98,0.12);"><span class="mt-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 20h4L19 9l-4-4L4 16v4Z"/><path d="M14 6l4 4"/></svg></span><div class="mt-mtxt"><div class="mt-t">Manual Logging</div><div class="mt-d">Detailed, custom tracking &mdash; you choose exactly what to record.</div></div></div><div class="mt-mcard" style="--acc:#FF5F46;--tint:rgba(255,95,70,0.12);"><span class="mt-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z"/></svg></span><div class="mt-mtxt"><div class="mt-t">Auto-Logging</div><div class="mt-d">Effortless monitoring with a single line of code.</div></div></div></div>
# MAGIC <div class="mt-panel"><div class="mt-panel-h">With just <b>one line of code</b>, keep tabs on every part of ML development</div><div class="mt-chips"><span class="mt-chip" style="--acc:#4299E0;">Parameters</span><span class="mt-chip" style="--acc:#00A972;">Metrics</span><span class="mt-chip" style="--acc:#FFAB00;">Data lineage</span><span class="mt-chip" style="--acc:#FF5F46;">Models</span><span class="mt-chip" style="--acc:#1B5162;">Environment</span></div></div>
# MAGIC <div class="mt-take"><span class="mt-take-i">&#10003;</span><p>All tracked information is captured in a <b>standardized format</b>, ensuring <b>reproducibility</b> throughout the development lifecycle.</p></div>
# MAGIC <div class="mt-docs"><a href="https://docs.databricks.com/en/mlflow/tracking.html" target="_blank">MLflow tracking docs &#8599;</a></div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### D1. Train and Track the Model
# MAGIC
# MAGIC Now that we have our training set ready to go, the next step is to train a model using the `sklearn` library. We will build a random forest classification model, tracking the F1-score for a single run. We will initiate the tracking before creating the model using `mlflow.start_run()` as the context manager. Within this manager we will:
# MAGIC
# MAGIC 1. Initialize the random forest classifier
# MAGIC 2. Fit the model
# MAGIC 3. Make a prediction using our test set
# MAGIC 4. Log the F1-score metric as `test_f1`
# MAGIC 5. Capture the artifacts for model tracking and management using the flavor `mlflow.sklearn`. **Flavor** in this context simply means that MLflow will package our scikit-learn model in a consistent and standardized way. If we wished to use a different ML library, we would use a different *flavor*.
# MAGIC 6. Register the model to Unity Catalog.

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# set the path for mlflow experiment
mlflow.set_experiment(f"/Users/{DA.username}/get-started-with-ml-flow-experiment")

with mlflow.start_run(run_name = 'get-started-with-ml-flow-run') as run:
    # Initialize the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the model on the training data
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = rf_classifier.predict(X_test)

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples = True,
        silent = True
    )
    # Calculate F1 score with 'macro' averaging for multiclass
    mlflow.log_metric("test_f1", f1_score(y_test, y_pred, average='macro'))
    # mlflow.log_metric("test_f1", f1_score(y_test, y_pred))

    model_info = mlflow.sklearn.log_model(
        rf_classifier,
        name = "model-artifacts",
        input_example=X_train[:3],
        signature=infer_signature(X_train, y_train)
    )

    model_uri = model_info.model_uri

# COMMAND ----------

# MAGIC %md
# MAGIC ### D2. Unity Catalog Model Registry
# MAGIC
# MAGIC After tracking an ML experiment, the final model is typically preserved for future use, and this is where the **Unity Catalog Model Registry** comes into play.
# MAGIC
# MAGIC It serves as a collaborative and centralized hub for managing models.
# MAGIC
# MAGIC Here are some key functionalities:
# MAGIC
# MAGIC - version your ML artifacts
# MAGIC
# MAGIC - facilitate additional experimentation
# MAGIC
# MAGIC - integrate with approval and governance Lakeflow Jobs
# MAGIC
# MAGIC - audit any logs
# MAGIC
# MAGIC - automate with CI/CD integration
# MAGIC
# MAGIC Easily version your ML models and associated artifacts while testing to enable a smooth transition from one model aliases to the next. All of this can be done through supportive CI/CD integration.
# MAGIC
# MAGIC Run the next cell to register the model in Unity Catalog.

# COMMAND ----------

# Modify the registry uri to point to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Define the model name 
model_name = f"{DA.catalog_name}.{DA.schema_name}.my_model"

# Register the model in the model registry
registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### D3. Set a Model Alias
# MAGIC
# MAGIC Finally, we are going to add an alias for this version of the model.

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

# Initialize an MLflow Client
client = MlflowClient()

# Assign a "dev" alias to model version 1
client.set_registered_model_alias(
    name= registered_model.name,  # The registered model name
    alias="dev",  # The alias representing the dev environment
    version=registered_model.version  # The version of the model you want to move to "dev"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### D4. Verify in the Databricks UI
# MAGIC
# MAGIC Navigate to your model in Catalog Explorer, if desired. You should see the registered model with version 1 and the `dev` alias.

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Prepare for an Upcoming Demo
# MAGIC
# MAGIC **We will discuss the code below in a future demo.**
# MAGIC
# MAGIC Run the next cell to deploy the model to Databricks Model Serving. We will discuss this in a future demo of the course, but the deployment will take about 10-12 minutes, so we are deploying now to save some time.

# COMMAND ----------

from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
endpoint_name = f"{DA.username}-model"
endpoint_name = endpoint_name.replace('.', '-').replace('@', '-')

# Check if the endpoint already exists
try:
    # Attempt to get the endpoint
    existing_endpoint = client.get_endpoint(endpoint_name)
    print(f"Endpoint '{endpoint_name}' already exists.")
except Exception as e:
    # If not found, create the endpoint
    if "RESOURCE_DOES_NOT_EXIST" in str(e):
        print(f"Creating a new endpoint: {endpoint_name}")
        endpoint = client.create_endpoint(
            name=endpoint_name,
            config={
                "served_entities": [
                    {
                        "name": "my-model",
                        "entity_name": registered_model.name,
                        "entity_version": registered_model.version,
                        "workload_size": "Small",
                        "scale_to_zero_enabled": True
                    }
                ],
                "traffic_config": {
                    "routes": [
                        {
                            "served_model_name": "my-model",
                            "traffic_percentage": 100
                        }
                    ]
                }
            }
        )
    else:
        print(f"An error occurred: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## F. Conclusion
# MAGIC
# MAGIC In this demo, we covered how to track, manage, and stage models using **MLflow** on Databricks.
# MAGIC
# MAGIC Key takeaways:
# MAGIC - MLflow tracks experiments, including parameters, metrics, artifacts, and models
# MAGIC - The Databricks Experiments UI lets you view and manage runs in one place
# MAGIC - Registering a model in Unity Catalog provides centralized governance and versioning
# MAGIC - Model version aliases like `dev` support managing models across lifecycle stages

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>