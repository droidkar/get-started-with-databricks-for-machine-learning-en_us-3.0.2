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
# MAGIC # Demo - Genie Code for ML
# MAGIC
# MAGIC ## Overview
# MAGIC In this demo, we will explore how **Databricks Genie Code (Agent Mode)** can help automate common data science workflows in Databricks, including data exploration, feature preparation, model training, evaluation, and **MLflow** tracking.
# MAGIC
# MAGIC Unlike AutoML, which automatically trains and selects models using a predefined workflow, **Agent Mode allows you to guide the workflow using natural language prompts**, while still keeping full transparency into the code that is generated and executed.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this demo, you will be able to:
# MAGIC - **Use Databricks Genie Code (Agent Mode) to build ML workflows**: generate code for an end-to-end experimentation workflow using natural language prompts, and execute multi-step tasks such as EDA, training, evaluation, and tracking
# MAGIC - **Train baseline models for different prediction tasks**: classification (predict whether a wine is high quality) and regression (predict the wine quality score)
# MAGIC - **Track and compare results using MLflow Experiments**: view metrics, parameters, and artifacts in the Experiments UI, and identify a baseline "champion" model based on evaluation metrics
# MAGIC - **Register the best model to Unity Catalog** for governance and future deployment

# COMMAND ----------

# DBTITLE 1,REQUIRED - SELECT A COMPUTE ENVIRONMENT
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
# MAGIC Run the following cell to configure the environment for this lesson.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-Agent_mode

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Introduction to Genie Code for ML

# COMMAND ----------

# MAGIC %md
# MAGIC ### B1. What is Genie Code?
# MAGIC
# MAGIC **Genie Code** is a conversational AI agent embedded in the Databricks UI that helps you build data science workflows by:
# MAGIC
# MAGIC - Understanding your dataset context (tables, schemas, metadata)
# MAGIC - Generating executable notebook code
# MAGIC - Running cells with your approval
# MAGIC - Reading outputs and iteratively fixing errors
# MAGIC - Logging experiments and results
# MAGIC
# MAGIC The old way → “AutoML runs a pipeline for me”
# MAGIC
# MAGIC The new way → “Genie Code writes and runs the pipeline with me - transparently.”

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B2. Genie Code Use-Cases
# MAGIC
# MAGIC Where does **Agent Mode** help most?
# MAGIC
# MAGIC <style>
# MAGIC @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
# MAGIC .guc{font-family:'DM Sans','Segoe UI',Arial,sans-serif;max-width:1000px;margin:16px auto;color:#1B3139;display:flex;gap:22px;flex-wrap:wrap;}
# MAGIC .guc *{box-sizing:border-box;}
# MAGIC .guc-card{flex:1 1 380px;min-width:300px;background:#fff;border-radius:16px;border-top:6px solid var(--acc);box-shadow:0 6px 16px rgba(27,49,57,0.09);padding:22px 22px 20px;cursor:pointer;transition:transform .2s ease,box-shadow .2s ease;}
# MAGIC .guc-card:hover{transform:translateY(-4px);box-shadow:0 16px 30px rgba(27,49,57,0.16);}
# MAGIC .guc-head{display:flex;align-items:center;gap:12px;margin-bottom:16px;}
# MAGIC .guc-num{flex:0 0 34px;width:34px;height:34px;border-radius:50%;background:var(--acc);color:#fff;font-size:17px;font-weight:800;display:inline-flex;align-items:center;justify-content:center;}
# MAGIC .guc-head h4{margin:0;font-size:17px;font-weight:800;color:#1B3139;line-height:1.25;}
# MAGIC .guc-qlab,.guc-clab{font-size:11px;font-weight:800;letter-spacing:.7px;text-transform:uppercase;color:#5A6F77;margin-bottom:7px;}
# MAGIC .guc-clab{margin-top:18px;}
# MAGIC .guc-quote{background:var(--tint);border-left:4px solid var(--acc);border-radius:8px;padding:12px 14px;font-size:15px;font-style:italic;font-weight:600;color:var(--acc);line-height:1.4;}
# MAGIC .guc-list{list-style:none;margin:0;padding:0;}
# MAGIC .guc-list li{position:relative;padding-left:26px;font-size:14px;color:#1B3139;line-height:1.4;margin-bottom:10px;}
# MAGIC .guc-list li:last-child{margin-bottom:0;}
# MAGIC .guc-list li::before{content:'';position:absolute;left:0;top:2px;width:17px;height:17px;border-radius:50%;background:var(--acc);}
# MAGIC .guc-list li::after{content:'';position:absolute;left:5px;top:6px;width:4px;height:7px;border:solid #fff;border-width:0 2px 2px 0;transform:rotate(45deg);}
# MAGIC </style>
# MAGIC <div class="guc"><div class="guc-card" style="--acc:#4299E0;--tint:rgba(66,153,224,0.10);"><div class="guc-head"><span class="guc-num">1</span><h4>Quickly verify the predictive power of a dataset</h4></div><div class="guc-qlab">Question</div><div class="guc-quote">&ldquo;Can this dataset be used to predict customer churn?&rdquo;</div><div class="guc-clab">Genie Code can</div><ul class="guc-list"><li>Explore the data</li><li>Generate features</li><li>Train a baseline model</li><li>Show evaluation metrics</li></ul></div><div class="guc-card" style="--acc:#00A972;--tint:rgba(0,169,114,0.10);"><div class="guc-head"><span class="guc-num">2</span><h4>Get a baseline model to guide project direction</h4></div><div class="guc-qlab">Question</div><div class="guc-quote">&ldquo;What baseline should I start with, and what benchmark should I aim to beat?&rdquo;</div><div class="guc-clab">Genie Code can</div><ul class="guc-list"><li>Try multiple algorithms</li><li>Log runs to MLflow</li><li>Compare performance</li><li>Provide a transparent starting point</li></ul></div></div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div class="gca"><style>
# MAGIC @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
# MAGIC .gca{font-family:'DM Sans','Segoe UI',Arial,sans-serif;max-width:1080px;margin:16px auto;color:#1B3139;}
# MAGIC .gca *{box-sizing:border-box;}
# MAGIC .gca-hero{background:linear-gradient(135deg,#1B5162 0%,#12414F 100%);border-radius:16px;padding:22px 26px;margin-bottom:22px;box-shadow:0 8px 20px rgba(27,49,57,0.16);}
# MAGIC .gca-hero h3{margin:0;font-size:23px;font-weight:800;color:#fff;letter-spacing:.2px;}
# MAGIC .gca-hero p{margin:7px 0 0;font-size:14.5px;color:#CFE0E5;line-height:1.5;}
# MAGIC .gca-flow{display:flex;align-items:stretch;gap:0;flex-wrap:wrap;justify-content:center;}
# MAGIC .gca-node{flex:1 1 200px;min-width:190px;background:#fff;border-radius:14px;border-top:6px solid var(--acc);box-shadow:0 6px 14px rgba(27,49,57,0.09);padding:20px 16px;text-align:center;}
# MAGIC .gca-node.wide{flex:1.5 1 270px;min-width:250px;}
# MAGIC .gca-nt{font-size:17px;font-weight:800;color:var(--acc);margin-bottom:4px;}
# MAGIC .gca-ns{font-size:13.5px;color:#5A6F77;line-height:1.45;}
# MAGIC .gca-chips{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-top:12px;}
# MAGIC .gca-chips span{background:var(--tint);color:var(--acc);font-size:12.5px;font-weight:700;border-radius:8px;padding:6px 12px;}
# MAGIC .gca-arw{flex:0 0 54px;display:flex;align-items:center;justify-content:center;}
# MAGIC .gca-arw span{width:38px;height:38px;border-radius:50%;background:#EDF1F3;color:#1B5162;display:inline-flex;align-items:center;justify-content:center;font-size:20px;font-weight:800;box-shadow:0 2px 7px rgba(27,49,57,0.12);}
# MAGIC .gca-take{margin-top:22px;display:flex;gap:12px;align-items:flex-start;background:#F8F9FC;border:1px solid #E4E9F2;border-left:5px solid #1B5162;border-radius:12px;padding:16px 18px;box-shadow:0 4px 10px rgba(27,49,57,0.05);}
# MAGIC .gca-take-b{flex:0 0 auto;font-size:12px;font-weight:800;letter-spacing:.5px;text-transform:uppercase;color:#fff;background:#1B5162;border-radius:7px;padding:5px 11px;margin-top:1px;}
# MAGIC .gca-take p{margin:0;font-size:14px;line-height:1.6;color:#1B3139;}
# MAGIC .gca-take b{color:#1B5162;}
# MAGIC </style><div class="gca-hero"><h3>Genie Code Architecture</h3><p>You chat in natural language; the control plane generates code that runs on the compute plane.</p></div><div class="gca-flow"><div class="gca-node" style="--acc:#4299E0;"><div class="gca-nt">Users</div><div class="gca-ns">Chat interface &mdash; ask in natural language</div></div><div class="gca-arw"><span>&rarr;</span></div><div class="gca-node wide" style="--acc:#FF5F46;--tint:rgba(255,95,70,0.10);"><div class="gca-nt">Control Plane</div><div class="gca-ns">generates the code</div><div class="gca-chips"><span>AI Services</span><span>Metadata</span><span>LLM</span></div></div><div class="gca-arw"><span>&rarr;</span></div><div class="gca-node" style="--acc:#00A972;"><div class="gca-nt">Compute Plane</div><div class="gca-ns">Spark Cluster / Runtime &mdash; runs the generated code</div></div></div><div class="gca-take"><span class="gca-take-b">Flow</span><p>Natural-language request &rarr; <b>Control Plane</b> (AI Services + Metadata + LLM) generates code &rarr; <b>Compute Plane</b> runs it on Spark, and results return to the user.</p></div></div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B4. Genie Code vs Traditional AutoML
# MAGIC
# MAGIC Both automate model building &mdash; here&#8217;s how their approaches differ.
# MAGIC
# MAGIC <div class="b4"><style>
# MAGIC @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
# MAGIC .b4{font-family:'DM Sans','Segoe UI',Arial,sans-serif;max-width:1000px;margin:16px auto;color:#1B3139;}
# MAGIC .b4 *{box-sizing:border-box;}
# MAGIC .b4-grid{display:flex;gap:0;align-items:stretch;flex-wrap:wrap;justify-content:center;}
# MAGIC .b4-card{flex:1 1 360px;min-width:290px;background:#fff;border-radius:16px;border-top:6px solid var(--acc);box-shadow:0 6px 16px rgba(27,49,57,0.09);padding:22px 22px 20px;cursor:pointer;transition:transform .2s ease,box-shadow .2s ease;}
# MAGIC .b4-card:hover{transform:translateY(-4px);box-shadow:0 16px 30px rgba(27,49,57,0.16);}
# MAGIC .b4-h{display:flex;align-items:center;gap:12px;margin-bottom:16px;}
# MAGIC .b4-badge{flex:0 0 42px;width:42px;height:42px;border-radius:12px;background:var(--tint);color:var(--acc);display:inline-flex;align-items:center;justify-content:center;}
# MAGIC .b4-badge svg{width:24px;height:24px;}
# MAGIC .b4-h .t{font-size:18px;font-weight:800;color:var(--acc);line-height:1.15;}
# MAGIC .b4-h .s{font-size:12px;font-weight:700;color:#5A6F77;letter-spacing:.3px;}
# MAGIC .b4-list{list-style:none;margin:0;padding:0;}
# MAGIC .b4-list li{position:relative;padding-left:28px;font-size:14px;color:#1B3139;line-height:1.45;margin-bottom:11px;}
# MAGIC .b4-list li:last-child{margin-bottom:0;}
# MAGIC .b4-check li::before{content:'';position:absolute;left:0;top:1px;width:18px;height:18px;border-radius:50%;background:var(--acc);}
# MAGIC .b4-check li::after{content:'';position:absolute;left:5px;top:5px;width:4px;height:7px;border:solid #fff;border-width:0 2px 2px 0;transform:rotate(45deg);}
# MAGIC .b4-dot li::before{content:'';position:absolute;left:4px;top:6px;width:8px;height:8px;border-radius:50%;background:var(--acc);}
# MAGIC .b4-vs{flex:0 0 60px;display:flex;align-items:center;justify-content:center;}
# MAGIC .b4-vs span{width:44px;height:44px;border-radius:50%;background:#1B3139;color:#fff;font-size:14px;font-weight:800;letter-spacing:.5px;display:inline-flex;align-items:center;justify-content:center;box-shadow:0 4px 10px rgba(27,49,57,0.22);}
# MAGIC .b4-outcome{display:flex;gap:12px;align-items:center;background:#FFF6F4;border:1px solid #FADAD3;border-left:5px solid #FF5F46;border-radius:12px;padding:15px 18px;margin-top:22px;box-shadow:0 4px 10px rgba(27,49,57,0.06);}
# MAGIC .b4-outcome .lab{flex:0 0 auto;font-size:12px;font-weight:800;letter-spacing:.5px;text-transform:uppercase;color:#fff;background:#FF5F46;border-radius:7px;padding:5px 11px;}
# MAGIC .b4-outcome p{margin:0;font-size:14px;line-height:1.5;color:#1B3139;}
# MAGIC .b4-outcome b{color:#FF5F46;}
# MAGIC </style><div class="b4-grid"><div class="b4-card" style="--acc:#FF5F46;--tint:rgba(255,95,70,0.12);"><div class="b4-h"><span class="b4-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 5h16v11H9l-5 4V5z"/></svg></span><div><div class="t">Genie Code</div><div class="s">Interactive &amp; transparent</div></div></div><ul class="b4-list b4-check"><li>Transparent &mdash; see all generated code</li><li>Conversational &mdash; guide with prompts</li><li>Interactive &mdash; iterate step-by-step</li><li>Educational &mdash; learn from the code</li><li>Editable notebooks</li></ul></div><div class="b4-vs"><span>VS</span></div><div class="b4-card" style="--acc:#618794;--tint:rgba(97,135,148,0.14);"><div class="b4-h"><span class="b4-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3.2"/><path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.2 5.2l2.1 2.1M16.7 16.7l2.1 2.1M18.8 5.2l-2.1 2.1M7.3 16.7l-2.1 2.1"/></svg></span><div><div class="t">Traditional AutoML</div><div class="s">Wizard-driven pipeline</div></div></div><ul class="b4-list b4-dot"><li>Wizard-based pipeline</li><li>Configuration-driven</li><li>Batch-style execution</li><li>Results-focused</li><li>Less focus on code understanding</li></ul></div></div><div class="b4-outcome"><span class="lab">Outcome</span><p><b>Genie Code</b> keeps the automation, but adds <b>control</b> and <b>learning</b>.</p></div></div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Warning</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Genie Code can generate and execute code in your notebook (with your permission). While it has guardrails to prevent dangerous actions, you should only use it with code and data you trust.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### B5. Understanding Databricks Genie Code (Agent Mode)
# MAGIC
# MAGIC Databricks Genie Code is an AI-powered assistant embedded into the Databricks workspace. It can help you write code, understand data, debug errors, and explain results.
# MAGIC
# MAGIC **Agent Mode** is a specialized feature that enables the Assistant to perform multi-step data science tasks such as:
# MAGIC
# MAGIC - Understanding the dataset and generating an approach
# MAGIC - Building features for ML workflows
# MAGIC - Training one or more models
# MAGIC - Evaluating models using common metrics
# MAGIC - Logging runs and artifacts to **MLflow**
# MAGIC - Helping identify a baseline "champion" model
# MAGIC
# MAGIC Agent Mode is especially useful when you want an AutoML-like experience, but with:
# MAGIC
# MAGIC - **More control** over the workflow
# MAGIC - **More transparency** into the generated code
# MAGIC - The ability to customize the workflow using natural language

# COMMAND ----------

# MAGIC %md
# MAGIC ### B6. How to Access Agent Mode
# MAGIC
# MAGIC To use Agent Mode in Databricks:
# MAGIC
# MAGIC 1. Open the **Genie Code** panel on the right side of the notebook.
# MAGIC
# MAGIC    <img src="https://files.training.databricks.com/binder/prod_main/get-started-with-databricks-for-machine-learning-en_us-3.0.2/images/20261006T181945Z/Includes/images/03_Genie_code_intro.png"
# MAGIC         alt="The Genie Code panel in the notebook"
# MAGIC         title="The Genie Code panel in the notebook"
# MAGIC         width="300"/>
# MAGIC
# MAGIC 2. You will see the **Genie Code** panel like this- 
# MAGIC
# MAGIC    <p align="center">
# MAGIC      <img src="https://files.training.databricks.com/binder/prod_main/get-started-with-databricks-for-machine-learning-en_us-3.0.2/images/20261006T181945Z/Includes/images/genie_code_chat.png"
# MAGIC           alt="The Agent option in the Genie Code mode drop-down"
# MAGIC           title="The Agent option in the Genie Code mode drop-down"
# MAGIC           width="600"/>
# MAGIC    </p>
# MAGIC
# MAGIC 3. Use prompts like:
# MAGIC    - "Analyze the dataset and suggest features"
# MAGIC    - "Train baseline classification models and log to MLflow"
# MAGIC    - "Pick the best run based on F1-score"
# MAGIC
# MAGIC **Genie Code** generates executable code directly in the notebook context, which you can run, inspect, and modify.

# COMMAND ----------

# MAGIC %md
# MAGIC ### B7. Best Practices for Agent Mode Prompts
# MAGIC
# MAGIC To get reliable and repeatable results from Agent Mode, write prompts that clearly specify:
# MAGIC
# MAGIC - **Dataset name**
# MAGIC - **Target column**
# MAGIC - **Task type** (classification or regression)
# MAGIC - **Columns to exclude** (such as identifiers)
# MAGIC - **Train/test split instructions**
# MAGIC - **Evaluation metrics**
# MAGIC - **Where to log results** (MLflow experiment name)
# MAGIC
# MAGIC **Example:**
# MAGIC > "Train a classification model using `wine_quality_table`. Create label `is_high_quality`.
# MAGIC > Exclude `wine_id`. Split train/test with seed 42.
# MAGIC > Evaluate using F1-score and ROC-AUC. Log runs to MLflow."

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B8. Important Considerations
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Always Review Generated Code</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Agent Mode can accelerate development, but you remain responsible for reviewing the generated code and verifying correctness before running it.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Troubleshooting the Genie Code Panel</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">If you do not see the Genie Code panel, try the following:</p>
# MAGIC <ul style="margin: 8px 0 0 18px; color: #333;">
# MAGIC <li>Confirm you are using a supported Databricks Runtime version</li>
# MAGIC <li>Ensure Databricks Genie Code is enabled in your workspace</li>
# MAGIC <li>Refresh your browser or reopen the notebook</li>
# MAGIC </ul>
# MAGIC <p style="margin: 10px 0 0 0; color: #333;"><strong>Note:</strong> Your instructor may have already configured the required workspace settings for this course.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Comprehensive Exploratory Data Analysis
# MAGIC
# MAGIC Genie Code can perform sophisticated EDA tasks through natural language instructions. This section guides you through comprehensive data exploration.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### C1. Statistical Analysis and Data Profiling
# MAGIC
# MAGIC Use Genie Code to generate comprehensive statistical summaries and identify data quality issues.
# MAGIC
# MAGIC 1. Select this cell to ensure any code cells Genie Code generates will be placed immediately below this cell.
# MAGIC 2. Click the **Copy** button below and paste the prompt into Genie Code's input box.
# MAGIC
# MAGIC <div class="code-block-light" data-language="text">
# MAGIC Perform comprehensive EDA on @wine_quality_table. I want to understand column statistics, data distributions, missing values, and potential data quality issues. Think like a data scientist and provide insights.
# MAGIC
# MAGIC Place any generated cells immediately below the currently selected cell.
# MAGIC </div>
# MAGIC
# MAGIC 3. Genie Code will propose one or more code blocks to run.
# MAGIC     - When prompted, click **Allow** to let Genie Code execute the code.
# MAGIC     - You may see the approval buttons like **Allow** / **Decline** (as shown in the screenshot).
# MAGIC
# MAGIC     ![Genie Code asking for approval to run a generated cell](https://files.training.databricks.com/binder/prod_main/get-started-with-databricks-for-machine-learning-en_us-3.0.2/images/20261006T181945Z/Includes/images/03_running_cell.png "Genie Code asking for approval to run a generated cell")
# MAGIC
# MAGIC     - If you see the dropdown that says **Ask every time**, you can keep it as-is (recommended for learners), or change it based on your preference.
# MAGIC 4. Review the generated code in the notebook cells and run/modify it if needed.
# MAGIC
# MAGIC **Note:** The suggested code will print its output in the next cell after this one.
# MAGIC
# MAGIC <style id="code-block-light-css">
# MAGIC .code-block-light {
# MAGIC     position: relative;
# MAGIC     margin: 16px 0;
# MAGIC     background: #f8f9fa;
# MAGIC     border: 1px solid #dee2e6;
# MAGIC     border-radius: 8px;
# MAGIC     padding: 16px 20px;
# MAGIC     padding-right: 80px;
# MAGIC     font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
# MAGIC     font-size: 0.95em;
# MAGIC     line-height: 1.5;
# MAGIC     color: #333;
# MAGIC     white-space: pre-wrap;
# MAGIC }
# MAGIC </style>
# MAGIC
# MAGIC <script>
# MAGIC (function(){
# MAGIC document.querySelectorAll('.code-block-light').forEach(function(block) {
# MAGIC     if (block.getAttribute('data-processed')) return;
# MAGIC     block.setAttribute('data-processed', 'true');
# MAGIC     var code = block.textContent.trim();
# MAGIC     var btn = document.createElement('button');
# MAGIC     btn.textContent = 'Copy';
# MAGIC     btn.style.cssText = 'position:absolute;top:8px;right:8px;padding:4px 12px;font-size:12px;background:#fff;color:#333;border:1px solid #dee2e6;border-radius:4px;cursor:pointer;font-weight:600;';
# MAGIC     btn.onmouseover = function(){ this.style.background='#e9ecef'; };
# MAGIC     btn.onmouseout = function(){ this.style.background='#fff'; };
# MAGIC     btn.onclick = function() {
# MAGIC         var t = document.createElement('textarea');
# MAGIC         t.value = code;
# MAGIC         document.body.appendChild(t);
# MAGIC         t.select();
# MAGIC         document.execCommand('copy');
# MAGIC         document.body.removeChild(t);
# MAGIC         this.textContent = 'Copied!';
# MAGIC         var b = this;
# MAGIC         setTimeout(function(){ b.textContent='Copy'; },2000);
# MAGIC     };
# MAGIC     block.style.position = 'relative';
# MAGIC     block.insertBefore(btn, block.firstChild);
# MAGIC });
# MAGIC })();
# MAGIC </script>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### C2. Data Visualization and Pattern Discovery
# MAGIC
# MAGIC Leverage Genie Code's visualization capabilities to uncover patterns and relationships in your data.
# MAGIC
# MAGIC Use the same process as above to execute the next prompt:
# MAGIC
# MAGIC 1. Select this cell to ensure any code cells Genie Code generates will be placed immediately below this cell.
# MAGIC 2. Click the **Copy** button below and paste the prompt into Genie Code's input box.
# MAGIC
# MAGIC <div class="code-block-light" data-language="text">
# MAGIC Generate correlation analysis and heatmaps to identify relationships between numeric columns.
# MAGIC
# MAGIC Place any generated cells immediately below the currently selected cell.
# MAGIC </div>
# MAGIC
# MAGIC <style id="code-block-light-css">
# MAGIC .code-block-light {
# MAGIC     position: relative;
# MAGIC     margin: 16px 0;
# MAGIC     background: #f8f9fa;
# MAGIC     border: 1px solid #dee2e6;
# MAGIC     border-radius: 8px;
# MAGIC     padding: 16px 20px;
# MAGIC     padding-right: 80px;
# MAGIC     font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
# MAGIC     font-size: 0.95em;
# MAGIC     line-height: 1.5;
# MAGIC     color: #333;
# MAGIC     white-space: pre-wrap;
# MAGIC }
# MAGIC </style>
# MAGIC
# MAGIC <script>
# MAGIC (function(){
# MAGIC document.querySelectorAll('.code-block-light').forEach(function(block) {
# MAGIC     if (block.getAttribute('data-processed')) return;
# MAGIC     block.setAttribute('data-processed', 'true');
# MAGIC     var code = block.textContent.trim();
# MAGIC     var btn = document.createElement('button');
# MAGIC     btn.textContent = 'Copy';
# MAGIC     btn.style.cssText = 'position:absolute;top:8px;right:8px;padding:4px 12px;font-size:12px;background:#fff;color:#333;border:1px solid #dee2e6;border-radius:4px;cursor:pointer;font-weight:600;';
# MAGIC     btn.onmouseover = function(){ this.style.background='#e9ecef'; };
# MAGIC     btn.onmouseout = function(){ this.style.background='#fff'; };
# MAGIC     btn.onclick = function() {
# MAGIC         var t = document.createElement('textarea');
# MAGIC         t.value = code;
# MAGIC         document.body.appendChild(t);
# MAGIC         t.select();
# MAGIC         document.execCommand('copy');
# MAGIC         document.body.removeChild(t);
# MAGIC         this.textContent = 'Copied!';
# MAGIC         var b = this;
# MAGIC         setTimeout(function(){ b.textContent='Copy'; },2000);
# MAGIC     };
# MAGIC     block.style.position = 'relative';
# MAGIC     block.insertBefore(btn, block.firstChild);
# MAGIC });
# MAGIC })();
# MAGIC </script>

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Machine Learning Model Development
# MAGIC
# MAGIC Now we'll use Genie Code to build and evaluate machine learning models based on our exploratory analysis.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### D1. Data Preparation and Feature Engineering
# MAGIC
# MAGIC Use Genie Code to prepare your data for machine learning, including feature engineering and data preprocessing.
# MAGIC
# MAGIC Use the same process as above to execute the next prompt:
# MAGIC
# MAGIC 1. Select this cell to ensure any code cells Genie Code generates will be placed immediately below this cell.
# MAGIC 2. Click the **Copy** button below and paste the prompt into Genie Code's input box.
# MAGIC
# MAGIC <div class="code-block-light" data-language="text">
# MAGIC Identify which features most influence wine quality, guiding process improvements and ingredient selection.
# MAGIC
# MAGIC Place any generated cells immediately below the currently selected cell.
# MAGIC </div>
# MAGIC
# MAGIC <style id="code-block-light-css">
# MAGIC .code-block-light {
# MAGIC     position: relative;
# MAGIC     margin: 16px 0;
# MAGIC     background: #f8f9fa;
# MAGIC     border: 1px solid #dee2e6;
# MAGIC     border-radius: 8px;
# MAGIC     padding: 16px 20px;
# MAGIC     padding-right: 80px;
# MAGIC     font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
# MAGIC     font-size: 0.95em;
# MAGIC     line-height: 1.5;
# MAGIC     color: #333;
# MAGIC     white-space: pre-wrap;
# MAGIC }
# MAGIC </style>
# MAGIC
# MAGIC <script>
# MAGIC (function(){
# MAGIC document.querySelectorAll('.code-block-light').forEach(function(block) {
# MAGIC     if (block.getAttribute('data-processed')) return;
# MAGIC     block.setAttribute('data-processed', 'true');
# MAGIC     var code = block.textContent.trim();
# MAGIC     var btn = document.createElement('button');
# MAGIC     btn.textContent = 'Copy';
# MAGIC     btn.style.cssText = 'position:absolute;top:8px;right:8px;padding:4px 12px;font-size:12px;background:#fff;color:#333;border:1px solid #dee2e6;border-radius:4px;cursor:pointer;font-weight:600;';
# MAGIC     btn.onmouseover = function(){ this.style.background='#e9ecef'; };
# MAGIC     btn.onmouseout = function(){ this.style.background='#fff'; };
# MAGIC     btn.onclick = function() {
# MAGIC         var t = document.createElement('textarea');
# MAGIC         t.value = code;
# MAGIC         document.body.appendChild(t);
# MAGIC         t.select();
# MAGIC         document.execCommand('copy');
# MAGIC         document.body.removeChild(t);
# MAGIC         this.textContent = 'Copied!';
# MAGIC         var b = this;
# MAGIC         setTimeout(function(){ b.textContent='Copy'; },2000);
# MAGIC     };
# MAGIC     block.style.position = 'relative';
# MAGIC     block.insertBefore(btn, block.firstChild);
# MAGIC });
# MAGIC })();
# MAGIC </script>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### D2. Train and Compare Models
# MAGIC
# MAGIC Use the same process as above to execute the next prompt:
# MAGIC
# MAGIC 1. Select this cell to ensure any code cells Genie Code generates will be placed immediately below this cell.
# MAGIC 2. Click the **Copy** button below and paste the prompt into Genie Code's input box.
# MAGIC
# MAGIC <div class="code-block-light" data-language="text">
# MAGIC Train multiple machine learning models on our prepared dataset to predict wine quality using the most important features. Compare different algorithms (e.g., Random Forest, Gradient Boosting, Linear models) and evaluate them using appropriate metrics. Show me model performance comparisons and feature importance.
# MAGIC
# MAGIC Place any generated cells immediately below the currently selected cell.
# MAGIC </div>
# MAGIC
# MAGIC <style id="code-block-light-css">
# MAGIC .code-block-light {
# MAGIC     position: relative;
# MAGIC     margin: 16px 0;
# MAGIC     background: #f8f9fa;
# MAGIC     border: 1px solid #dee2e6;
# MAGIC     border-radius: 8px;
# MAGIC     padding: 16px 20px;
# MAGIC     padding-right: 80px;
# MAGIC     font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
# MAGIC     font-size: 0.95em;
# MAGIC     line-height: 1.5;
# MAGIC     color: #333;
# MAGIC     white-space: pre-wrap;
# MAGIC }
# MAGIC </style>
# MAGIC
# MAGIC <script>
# MAGIC (function(){
# MAGIC document.querySelectorAll('.code-block-light').forEach(function(block) {
# MAGIC     if (block.getAttribute('data-processed')) return;
# MAGIC     block.setAttribute('data-processed', 'true');
# MAGIC     var code = block.textContent.trim();
# MAGIC     var btn = document.createElement('button');
# MAGIC     btn.textContent = 'Copy';
# MAGIC     btn.style.cssText = 'position:absolute;top:8px;right:8px;padding:4px 12px;font-size:12px;background:#fff;color:#333;border:1px solid #dee2e6;border-radius:4px;cursor:pointer;font-weight:600;';
# MAGIC     btn.onmouseover = function(){ this.style.background='#e9ecef'; };
# MAGIC     btn.onmouseout = function(){ this.style.background='#fff'; };
# MAGIC     btn.onclick = function() {
# MAGIC         var t = document.createElement('textarea');
# MAGIC         t.value = code;
# MAGIC         document.body.appendChild(t);
# MAGIC         t.select();
# MAGIC         document.execCommand('copy');
# MAGIC         document.body.removeChild(t);
# MAGIC         this.textContent = 'Copied!';
# MAGIC         var b = this;
# MAGIC         setTimeout(function(){ b.textContent='Copy'; },2000);
# MAGIC     };
# MAGIC     block.style.position = 'relative';
# MAGIC     block.insertBefore(btn, block.firstChild);
# MAGIC });
# MAGIC })();
# MAGIC </script>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### D3. Model Optimization and Hyperparameter Tuning
# MAGIC
# MAGIC Please note: The code that Genie Code produces may take more than 10 minutes to execute. If you do not have time to accomplish this, please skip this part of the demo.
# MAGIC
# MAGIC Use Genie Code to optimize your best-performing model through hyperparameter tuning.
# MAGIC
# MAGIC Use the same process as above to execute the next prompt:
# MAGIC
# MAGIC 1. Select this cell to ensure any code cells Genie Code generates will be placed immediately below this cell.
# MAGIC 2. Click the **Copy** button below and paste the prompt into Genie Code's input box.
# MAGIC
# MAGIC <div class="code-block-light" data-language="text">
# MAGIC Perform hyperparameter tuning on the best-performing model from our comparison. Use techniques like grid search or random search to optimize model performance. Show me the improvement in metrics after tuning.
# MAGIC
# MAGIC Place any generated cells immediately below the currently selected cell.
# MAGIC </div>
# MAGIC
# MAGIC <style id="code-block-light-css">
# MAGIC .code-block-light {
# MAGIC     position: relative;
# MAGIC     margin: 16px 0;
# MAGIC     background: #f8f9fa;
# MAGIC     border: 1px solid #dee2e6;
# MAGIC     border-radius: 8px;
# MAGIC     padding: 16px 20px;
# MAGIC     padding-right: 80px;
# MAGIC     font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
# MAGIC     font-size: 0.95em;
# MAGIC     line-height: 1.5;
# MAGIC     color: #333;
# MAGIC     white-space: pre-wrap;
# MAGIC }
# MAGIC </style>
# MAGIC
# MAGIC <script>
# MAGIC (function(){
# MAGIC document.querySelectorAll('.code-block-light').forEach(function(block) {
# MAGIC     if (block.getAttribute('data-processed')) return;
# MAGIC     block.setAttribute('data-processed', 'true');
# MAGIC     var code = block.textContent.trim();
# MAGIC     var btn = document.createElement('button');
# MAGIC     btn.textContent = 'Copy';
# MAGIC     btn.style.cssText = 'position:absolute;top:8px;right:8px;padding:4px 12px;font-size:12px;background:#fff;color:#333;border:1px solid #dee2e6;border-radius:4px;cursor:pointer;font-weight:600;';
# MAGIC     btn.onmouseover = function(){ this.style.background='#e9ecef'; };
# MAGIC     btn.onmouseout = function(){ this.style.background='#fff'; };
# MAGIC     btn.onclick = function() {
# MAGIC         var t = document.createElement('textarea');
# MAGIC         t.value = code;
# MAGIC         document.body.appendChild(t);
# MAGIC         t.select();
# MAGIC         document.execCommand('copy');
# MAGIC         document.body.removeChild(t);
# MAGIC         this.textContent = 'Copied!';
# MAGIC         var b = this;
# MAGIC         setTimeout(function(){ b.textContent='Copy'; },2000);
# MAGIC     };
# MAGIC     block.style.position = 'relative';
# MAGIC     block.insertBefore(btn, block.firstChild);
# MAGIC });
# MAGIC })();
# MAGIC </script>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## E. Make an MLflow Run and Register the Model in Unity Catalog
# MAGIC
# MAGIC In this section, we will register our best model to **Unity Catalog**.
# MAGIC
# MAGIC ### E1. Why Register Models?
# MAGIC
# MAGIC Registering models in Unity Catalog provides:
# MAGIC
# MAGIC - Centralized governance and discoverability
# MAGIC - Access control and auditing
# MAGIC - A versioned, reusable model artifact (`ModelName` + `Version`)
# MAGIC - A clear path to deployment using **Model Serving**
# MAGIC
# MAGIC Use the same process as above to execute the next prompt:
# MAGIC
# MAGIC 1. Select this cell to ensure any code cells Genie Code generates will be placed immediately below this cell.
# MAGIC 2. Click the **Copy** button below and paste the prompt into Genie Code's input box.
# MAGIC
# MAGIC <div class="code-block-light" data-language="text">
# MAGIC Please run the best model using MLflow and register the model in Unity Catalog using the current default catalog and schema. Name the model "genie_model". Then, set the alias of this model to "genie-generated".
# MAGIC
# MAGIC Place any generated cells immediately below the currently selected cell.
# MAGIC </div>
# MAGIC
# MAGIC <style id="code-block-light-css">
# MAGIC .code-block-light {
# MAGIC     position: relative;
# MAGIC     margin: 16px 0;
# MAGIC     background: #f8f9fa;
# MAGIC     border: 1px solid #dee2e6;
# MAGIC     border-radius: 8px;
# MAGIC     padding: 16px 20px;
# MAGIC     padding-right: 80px;
# MAGIC     font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
# MAGIC     font-size: 0.95em;
# MAGIC     line-height: 1.5;
# MAGIC     color: #333;
# MAGIC     white-space: pre-wrap;
# MAGIC }
# MAGIC </style>
# MAGIC
# MAGIC <script>
# MAGIC (function(){
# MAGIC document.querySelectorAll('.code-block-light').forEach(function(block) {
# MAGIC     if (block.getAttribute('data-processed')) return;
# MAGIC     block.setAttribute('data-processed', 'true');
# MAGIC     var code = block.textContent.trim();
# MAGIC     var btn = document.createElement('button');
# MAGIC     btn.textContent = 'Copy';
# MAGIC     btn.style.cssText = 'position:absolute;top:8px;right:8px;padding:4px 12px;font-size:12px;background:#fff;color:#333;border:1px solid #dee2e6;border-radius:4px;cursor:pointer;font-weight:600;';
# MAGIC     btn.onmouseover = function(){ this.style.background='#e9ecef'; };
# MAGIC     btn.onmouseout = function(){ this.style.background='#fff'; };
# MAGIC     btn.onclick = function() {
# MAGIC         var t = document.createElement('textarea');
# MAGIC         t.value = code;
# MAGIC         document.body.appendChild(t);
# MAGIC         t.select();
# MAGIC         document.execCommand('copy');
# MAGIC         document.body.removeChild(t);
# MAGIC         this.textContent = 'Copied!';
# MAGIC         var b = this;
# MAGIC         setTimeout(function(){ b.textContent='Copy'; },2000);
# MAGIC     };
# MAGIC     block.style.position = 'relative';
# MAGIC     block.insertBefore(btn, block.firstChild);
# MAGIC });
# MAGIC })();
# MAGIC </script>

# COMMAND ----------

# MAGIC %md
# MAGIC ### E2. Verify in the Databricks UI
# MAGIC
# MAGIC To confirm that your model was successfully registered in Unity Catalog:
# MAGIC
# MAGIC 1. In the left sidebar, click **Catalog**.
# MAGIC 2. Open the catalog and schema created during classroom setup.
# MAGIC 3. Select the **Models** tab.
# MAGIC
# MAGIC The model is now ready for deployment using **Databricks Model Serving**, which we will talk about shortly.

# COMMAND ----------

# MAGIC %md
# MAGIC ## F. Model Interpretation and Business Insights
# MAGIC
# MAGIC The final step involves interpreting your machine learning results and extracting actionable business insights.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### F1. Business Recommendations and Next Steps
# MAGIC
# MAGIC Use the same process as above to execute the next prompt:
# MAGIC
# MAGIC 1. Select this cell to ensure any code cells Genie Code generates will be placed immediately below this cell.
# MAGIC 2. Click the **Copy** button below and paste the prompt into Genie Code's input box.
# MAGIC
# MAGIC <div class="code-block-light" data-language="text">
# MAGIC Based on our EDA and machine learning results, provide business recommendations and actionable insights. What should stakeholders know about the patterns we discovered? What are the next steps for deploying or improving this model?
# MAGIC
# MAGIC Place any generated cells immediately below the currently selected cell.
# MAGIC </div>
# MAGIC
# MAGIC <style id="code-block-light-css">
# MAGIC .code-block-light {
# MAGIC     position: relative;
# MAGIC     margin: 16px 0;
# MAGIC     background: #f8f9fa;
# MAGIC     border: 1px solid #dee2e6;
# MAGIC     border-radius: 8px;
# MAGIC     padding: 16px 20px;
# MAGIC     padding-right: 80px;
# MAGIC     font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
# MAGIC     font-size: 0.95em;
# MAGIC     line-height: 1.5;
# MAGIC     color: #333;
# MAGIC     white-space: pre-wrap;
# MAGIC }
# MAGIC </style>
# MAGIC
# MAGIC <script>
# MAGIC (function(){
# MAGIC document.querySelectorAll('.code-block-light').forEach(function(block) {
# MAGIC     if (block.getAttribute('data-processed')) return;
# MAGIC     block.setAttribute('data-processed', 'true');
# MAGIC     var code = block.textContent.trim();
# MAGIC     var btn = document.createElement('button');
# MAGIC     btn.textContent = 'Copy';
# MAGIC     btn.style.cssText = 'position:absolute;top:8px;right:8px;padding:4px 12px;font-size:12px;background:#fff;color:#333;border:1px solid #dee2e6;border-radius:4px;cursor:pointer;font-weight:600;';
# MAGIC     btn.onmouseover = function(){ this.style.background='#e9ecef'; };
# MAGIC     btn.onmouseout = function(){ this.style.background='#fff'; };
# MAGIC     btn.onclick = function() {
# MAGIC         var t = document.createElement('textarea');
# MAGIC         t.value = code;
# MAGIC         document.body.appendChild(t);
# MAGIC         t.select();
# MAGIC         document.execCommand('copy');
# MAGIC         document.body.removeChild(t);
# MAGIC         this.textContent = 'Copied!';
# MAGIC         var b = this;
# MAGIC         setTimeout(function(){ b.textContent='Copy'; },2000);
# MAGIC     };
# MAGIC     block.style.position = 'relative';
# MAGIC     block.insertBefore(btn, block.firstChild);
# MAGIC });
# MAGIC })();
# MAGIC </script>

# COMMAND ----------

# MAGIC %md
# MAGIC ## G. Conclusion
# MAGIC
# MAGIC In this demo, you learned how to use **Databricks Genie Code (Agent Mode)** to accelerate common data science workflows in Databricks. By completing this demo, you were able to use Agent Mode prompts to generate ML workflows and register champion models to Unity Catalog.
# MAGIC
# MAGIC Key takeaways:
# MAGIC - **Agent Mode** provides an AutoML-like experience with more control and transparency
# MAGIC - You can use natural language prompts to generate end-to-end ML workflows
# MAGIC - Agent Mode integrates seamlessly with **MLflow** for experiment tracking
# MAGIC - Trained models can be registered in **Unity Catalog** for governance and deployment
# MAGIC
# MAGIC In the next lesson, you will explore **Databricks Model Serving** and learn how to deploy registered models for real-time inference using serving endpoints. This will allow you to serve models trained in this course, whether built manually or generated using Genie Code.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>