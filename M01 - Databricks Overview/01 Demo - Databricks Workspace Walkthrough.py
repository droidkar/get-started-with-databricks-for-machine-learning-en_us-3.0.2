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
# MAGIC # Demo - Databricks Workspace Walkthrough
# MAGIC
# MAGIC ## Overview
# MAGIC In this short, follow-along demo, you will get oriented in your Databricks workspace and run your first cells. You will confirm your compute environment, run the classroom setup, tour the workspace sidebar, find data in the catalog, and practice notebook basics. By the end, you will be ready to dive into the lessons that follow.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this demo, you will be able to:
# MAGIC - Confirm your compute environment and run the classroom setup
# MAGIC - Run SQL and Python cells in a notebook
# MAGIC - Navigate the workspace sidebar
# MAGIC - Browse Unity Catalog and query a sample table from a notebook
# MAGIC - Add cells, switch languages, and use Genie Code

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
# MAGIC ### A1. Confirm Your Compute
# MAGIC
# MAGIC Look at the **top-right corner of this notebook** to confirm your compute is selected.
# MAGIC
# MAGIC 1. You should see **Serverless** selected by default.
# MAGIC 2. If a different cluster is selected, click the dropdown and choose **Serverless**.
# MAGIC
# MAGIC With Databricks **serverless compute**, you don't need to start or wait for a cluster — it attaches automatically when you run a cell.

# COMMAND ----------

# MAGIC %md
# MAGIC ### A2. Run the Classroom Setup
# MAGIC
# MAGIC The cell below runs the classroom setup script, which creates your personal catalog and schema for the course. Click the **Run** button at the top-left of the cell (or press **Shift + Enter** while the cell is selected).

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-1

# COMMAND ----------

# MAGIC %md
# MAGIC ### A3. Confirm Your Environment
# MAGIC
# MAGIC The classroom setup has created your working catalog and schema. Run the cell below to print your username, catalog, and schema — you will use these throughout the course.

# COMMAND ----------

print(f"Username: {spark.sql('SELECT current_user()').first()[0]}")
print(f"Catalog:  {DA.catalog_name}")
print(f"Schema:   {DA.schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Run Your First Cell
# MAGIC
# MAGIC Now run a SQL cell to confirm everything is working. The cell below greets you and shows the current date and time.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   'Hello Databricks Learner!' AS greeting,
# MAGIC   current_date() AS today,
# MAGIC   current_timestamp() AS right_now;

# COMMAND ----------

# MAGIC %md
# MAGIC You should see a one-row result with your greeting, today's date, and the current time. **You just ran your first SQL query in Databricks.**

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## C. Tour the Workspace Sidebar
# MAGIC
# MAGIC The left sidebar is your home base in Databricks. Hover over the **menu icon** at the top-left to expand it.
# MAGIC
# MAGIC The items you'll use most often in this course:
# MAGIC
# MAGIC | Sidebar item | What it's for |
# MAGIC |---|---|
# MAGIC | **+ New** | Create a notebook, query, dashboard, job, or pipeline |
# MAGIC | **Workspace** | Browse and organize your own notebooks and folders |
# MAGIC | **Catalog** | Browse governed data — catalogs, schemas, tables, and volumes |
# MAGIC | **Compute** | View and configure compute resources |
# MAGIC | **Search** (top bar) | Find any notebook, table, or asset by name |
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC Try it: click the <strong>Databricks logo</strong> in the top-left to go to the workspace homepage. You'll see <strong>Suggested</strong>, <strong>Favorites</strong>, and <strong>What's New</strong> tabs. Then click your browser's back button (or use <strong>Recents</strong> in the sidebar) to return to this notebook.
# MAGIC </p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Find Data in the Catalog
# MAGIC
# MAGIC All governed data in Databricks lives in **Unity Catalog**, organized in a three-level hierarchy: **catalog → schema → table**. Let's find a sample table that's available in every Databricks workspace.

# COMMAND ----------

# MAGIC %md
# MAGIC ### D1. Browse to the Table
# MAGIC
# MAGIC 1. In the left sidebar, click **Catalog** to open Catalog Explorer.
# MAGIC 2. Find and expand the **`samples`** catalog.
# MAGIC 3. Expand the **`nyctaxi`** schema.
# MAGIC 4. Click the **`trips`** table.
# MAGIC 5. On the right panel, look at the **Overview** tab to see the column list, then click the **Sample Data** tab to preview a few rows.
# MAGIC
# MAGIC You're looking at NYC taxi trip data: pickup and dropoff times, fare amounts, distances, and zip codes.

# COMMAND ----------

# MAGIC %md
# MAGIC ### D2. Query the Table from This Notebook
# MAGIC
# MAGIC Run the cell below to query the table you just previewed. You should see the same data you saw in the **Sample Data** tab.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM samples.nyctaxi.trips
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC You're now seeing the same data you previewed in Catalog Explorer, just queried directly in the notebook. This is the loop you'll use throughout the course: **find data in the catalog, then work with it in a notebook.**

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### D3. Knowledge Check
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #4299E0); color: white; padding: 12px 18px; cursor: pointer; font-weight: 600; font-size: 12pt; border-radius: 8px; user-select: none;">
# MAGIC How is data organized in Unity Catalog?
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 16px 20px; background: #F9F7F4; font-size: 12pt; line-height: 1.7; color: #333;">
# MAGIC
# MAGIC <p style="margin-top: 10px;">Unity Catalog organizes governed data in a three-level hierarchy: <strong>catalog → schema → table</strong>. In the table you just queried, <code>samples</code> is the catalog, <code>nyctaxi</code> is the schema, and <code>trips</code> is the table — which is why the full name in SQL is <code>samples.nyctaxi.trips</code>. Your own work in this course uses the personal catalog and schema created by the classroom setup.</p>
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Notebook Basics: Cells, Languages, and Genie Code
# MAGIC
# MAGIC Databricks notebooks support multiple languages in the same notebook. The cell below is Python. It counts the rows in the sample table and prints the result.

# COMMAND ----------

trip_count = spark.sql("SELECT COUNT(*) AS total FROM samples.nyctaxi.trips").collect()[0]["total"]
print(f"There are {trip_count:,} trips in the sample dataset.")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### E1. Add a New Cell and Switch Languages
# MAGIC
# MAGIC Try adding a new cell to this notebook:
# MAGIC
# MAGIC 1. Hover between this cell and the next one. A **+ Code** button appears. Click it to add a new cell below.
# MAGIC 2. In the new cell, set the language to **SQL** using the dropdown at the top-right of the cell (or type `%sql` on the first line).
# MAGIC 3. Click the **Copy** button below and paste the query into your new cell.
# MAGIC 4. Press **Shift + Enter** to run it.
# MAGIC
# MAGIC <div class="code-block-light" data-language="text">
# MAGIC SELECT MAX(fare_amount) AS biggest_fare FROM samples.nyctaxi.trips
# MAGIC </div>
# MAGIC
# MAGIC You should see a one-row result showing the largest fare in the sample dataset.
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
# MAGIC ### E2. Try Genie Code
# MAGIC
# MAGIC **Genie Code** can write, explain, and debug code for you. There are two ways to open it:
# MAGIC
# MAGIC 1. Click the **Genie icon** in the right sidebar to open the chat panel.
# MAGIC 2. Press **Cmd + I** (Mac) or **Ctrl + I** (Windows) inside a cell to invoke the in-cell Genie Code prompt.
# MAGIC
# MAGIC Click the **Copy** button below and paste the prompt into Genie Code:
# MAGIC
# MAGIC <div class="code-block-light" data-language="text">
# MAGIC Write a SQL query that counts trips by pickup zip code from samples.nyctaxi.trips, sorted by trip count descending.
# MAGIC </div>
# MAGIC
# MAGIC Genie Code should respond with a working query that you can insert into a cell and run.
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
# MAGIC ## F. Conclusion
# MAGIC
# MAGIC You're now oriented in Databricks. You confirmed your compute environment, ran the classroom setup, and used the notebook to run both SQL and Python — the same loop of finding data in the catalog and working with it in a notebook that you'll use throughout this course.
# MAGIC
# MAGIC Key takeaways:
# MAGIC - Serverless compute attaches automatically when you run a cell — no cluster to start or manage
# MAGIC - The classroom setup creates your personal catalog and schema for the course
# MAGIC - Unity Catalog organizes governed data as **catalog → schema → table**
# MAGIC - Notebooks support multiple languages (SQL, Python) side by side
# MAGIC - Genie Code can write, explain, and debug code from a natural-language prompt

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC