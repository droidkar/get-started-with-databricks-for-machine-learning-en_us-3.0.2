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
# MAGIC # Lecture - Introduction to Machine Learning on Databricks
# MAGIC
# MAGIC ## Overview
# MAGIC This lecture introduces machine learning on Databricks. You will learn about the roles that collaborate across the machine learning lifecycle, the processes that make up that lifecycle, and how Databricks supports model monitoring.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this lecture, you will be able to:
# MAGIC
# MAGIC 1. Identify the roles involved in the machine learning lifecycle
# MAGIC 2. Describe the processes that make up the machine learning lifecycle
# MAGIC 3. Explain how Databricks supports model monitoring

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left: 4px solid #FF5F46; background: #FFF6F4; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #FF5F46; font-size: 15pt;">Important Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #0b2026;">This course will run in Databricks Free Edition. However, please note that, in the course notebooks, we will be deploying a model to Databricks Model Serving. Databricks Free Edition has caps on the number of model serving endpoints you can deploy. If you run into errors in any of the notebooks that deploy this infrastructure, please double-check that you have not hit these caps.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## The machine learning lifecycle on Databricks
# MAGIC
# MAGIC End to end on Databricks: from data preparation to scheduled retraining, governed in Unity Catalog and tracked with MLflow, across Data Scientist, ML Engineer, and Data Engineer roles. Click any stage to see the role and what happens there.
# MAGIC
# MAGIC <style>
# MAGIC @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
# MAGIC .mll{font-family:'DM Sans','Segoe UI',Arial,sans-serif;max-width:1400px;margin:16px auto;color:#1B3139;}
# MAGIC .mll *{box-sizing:border-box;}
# MAGIC .mll-legend{display:flex;gap:24px;margin:4px 2px 18px;font-size:15px;font-weight:700;}
# MAGIC .mll-legend span{display:inline-flex;align-items:center;gap:8px;}
# MAGIC .mll-legend i{width:14px;height:14px;border-radius:50%;display:inline-block;}
# MAGIC .mll-row{display:flex;align-items:center;gap:0;overflow-x:auto;padding:6px 2px 8px;}
# MAGIC .flip{flex:0 0 172px;width:172px;height:216px;perspective:1100px;cursor:pointer;}
# MAGIC .flip-in{position:relative;width:100%;height:100%;transition:transform .55s cubic-bezier(.2,.7,.3,1);transform-style:preserve-3d;}
# MAGIC .flip.flipped .flip-in{transform:rotateY(180deg);}
# MAGIC .face{position:absolute;inset:0;-webkit-backface-visibility:hidden;backface-visibility:hidden;border-radius:12px;background:#fff;box-shadow:0 6px 12px rgba(27,49,57,0.10);overflow:hidden;}
# MAGIC .front{display:flex;flex-direction:column;align-items:center;text-align:center;padding:16px 12px 12px;}
# MAGIC .acc{position:absolute;top:0;left:0;right:0;height:8px;}
# MAGIC .num{width:34px;height:34px;border-radius:50%;color:#fff;font-size:16px;font-weight:800;display:flex;align-items:center;justify-content:center;margin:12px 0 8px;}
# MAGIC .ttl{font-size:15.5px;font-weight:700;line-height:1.22;}
# MAGIC .cap{font-size:12.5px;color:#5A6F77;margin-top:8px;line-height:1.38;}
# MAGIC .rtag{margin-top:auto;font-size:10.5px;font-weight:700;letter-spacing:.6px;}
# MAGIC .hint{font-size:9px;color:#90A5B1;margin-top:6px;font-weight:600;}
# MAGIC .back{transform:rotateY(180deg);color:#fff;padding:14px 14px;text-align:left;display:flex;flex-direction:column;}
# MAGIC .back h4{margin:0 0 7px;font-size:13.5px;font-weight:800;letter-spacing:.3px;}
# MAGIC .back p{margin:0;font-size:11.7px;line-height:1.42;}
# MAGIC .back .h2{margin-top:auto;font-size:9px;opacity:.85;font-weight:600;}
# MAGIC .arw{flex:0 0 22px;text-align:center;color:#618794;font-size:22px;line-height:1;}
# MAGIC .loop{position:relative;height:60px;margin:2px 0 0;}
# MAGIC .loop-u{position:absolute;left:8%;right:8%;top:0;height:38px;border:2.5px dashed #98102A;border-top:none;border-radius:0 0 16px 16px;}
# MAGIC .loop-ah{position:absolute;left:calc(8% - 7px);top:-9px;width:0;height:0;border-left:8px solid transparent;border-right:8px solid transparent;border-bottom:12px solid #98102A;}
# MAGIC .loop-lb{position:absolute;left:50%;top:24px;transform:translateX(-50%);background:#FFF6F4;color:#98102A;font-weight:700;font-size:13.5px;padding:3px 15px;border-radius:14px;white-space:nowrap;}
# MAGIC .found{margin-top:16px;background:#F8F9FC;border:1px solid #DCE0E2;border-radius:16px;box-shadow:0 6px 12px rgba(27,49,57,0.10);padding:18px 22px;text-align:center;}
# MAGIC .found-lbl{font-size:12.5px;font-weight:700;letter-spacing:1.5px;color:#5A6F77;}
# MAGIC .found-chips{display:flex;gap:16px;justify-content:center;flex-wrap:wrap;margin-top:14px;}
# MAGIC .chip{display:inline-flex;align-items:center;gap:8px;background:#fff;border:1px solid #EEEDE9;border-radius:22px;padding:8px 18px;font-size:14.5px;font-weight:700;box-shadow:0 2px 4px rgba(27,49,57,0.08);}
# MAGIC .chip .mlf i{font-style:italic;color:#0194E2;}
# MAGIC </style>
# MAGIC <div class="mll">
# MAGIC <div class="mll-legend"><span><i style="background:#4299E0;"></i>Data Scientist</span><span><i style="background:#FF5F46;"></i>ML Engineer</span><span><i style="background:#00A972;"></i>Data Engineer</span></div>
# MAGIC <div class="mll-row">
# MAGIC <div class="flip"><div class="flip-in"><div class="face front"><div class="acc" style="background:#00A972;"></div><div class="num" style="background:#00A972;">1</div><div class="ttl">Data Preparation</div><div class="cap">ETL · EDA · feature engineering (pandas API)</div><div class="rtag" style="color:#00A972;">DATA ENGINEER</div><div class="hint">click for detail &#8635;</div></div><div class="face back" style="background:#00A972;"><h4>1. Data Preparation</h4><p>Owned by the <b>Data Engineer</b>. Ingest raw sources, run exploratory analysis, and engineer features with the pandas API so data is clean, consistent, and analysis-ready for modeling.</p><div class="h2">click to flip back</div></div></div></div>
# MAGIC <div class="arw">&#8250;</div>
# MAGIC <div class="flip"><div class="flip-in"><div class="face front"><div class="acc" style="background:#4299E0;"></div><div class="num" style="background:#4299E0;">2</div><div class="ttl">Build Baseline Model</div><div class="cap">Genie Code + MLflow autologging</div><div class="rtag" style="color:#4299E0;">DATA SCIENTIST</div><div class="hint">click for detail &#8635;</div></div><div class="face back" style="background:#4299E0;"><h4>2. Build Baseline Model</h4><p>Owned by the <b>Data Scientist</b>. Prompt Genie Code to scaffold a baseline model; MLflow autologging captures parameters, metrics, and artifacts for every run automatically.</p><div class="h2">click to flip back</div></div></div></div>
# MAGIC <div class="arw">&#8250;</div>
# MAGIC <div class="flip"><div class="flip-in"><div class="face front"><div class="acc" style="background:#FF5F46;"></div><div class="num" style="background:#FF5F46;">3</div><div class="ttl">Automated Testing</div><div class="cap">Lakeflow Jobs + Slack alerts</div><div class="rtag" style="color:#FF5F46;">ML ENGINEER</div><div class="hint">click for detail &#8635;</div></div><div class="face back" style="background:#FF5F46;"><h4>3. Automated Testing</h4><p>Owned by the <b>ML Engineer</b>. Testing Lakeflow Jobs fire on new runs and post Slack notifications, so promising candidates are reviewed quickly and consistently.</p><div class="h2">click to flip back</div></div></div></div>
# MAGIC <div class="arw">&#8250;</div>
# MAGIC <div class="flip"><div class="flip-in"><div class="face front"><div class="acc" style="background:#4299E0;"></div><div class="num" style="background:#4299E0;">4</div><div class="ttl">Promote to Registry</div><div class="cap">Unity Catalog + @staging alias</div><div class="rtag" style="color:#4299E0;">DATA SCIENTIST</div><div class="hint">click for detail &#8635;</div></div><div class="face back" style="background:#4299E0;"><h4>4. Promote to Registry</h4><p>Owned by the <b>Data Scientist</b>. Register the best run in the Unity Catalog Model Registry, annotate it, and request a <b>@staging</b> alias for governed promotion.</p><div class="h2">click to flip back</div></div></div></div>
# MAGIC <div class="arw">&#8250;</div>
# MAGIC <div class="flip"><div class="flip-in"><div class="face front"><div class="acc" style="background:#FF5F46;"></div><div class="num" style="background:#FF5F46;">5</div><div class="ttl">Model Testing &amp; Approval</div><div class="cap">Schema · accuracy · docs</div><div class="rtag" style="color:#FF5F46;">ML ENGINEER</div><div class="hint">click for detail &#8635;</div></div><div class="face back" style="background:#FF5F46;"><h4>5. Model Testing & Approval</h4><p>Owned by the <b>ML Engineer</b>. Automated checks (schema, accuracy, demographic slices, docs &amp; artifacts) decide whether to <b>approve</b> or <b>reject</b> the staging request.</p><div class="h2">click to flip back</div></div></div></div>
# MAGIC <div class="arw">&#8250;</div>
# MAGIC <div class="flip"><div class="flip-in"><div class="face front"><div class="acc" style="background:#FF5F46;"></div><div class="num" style="background:#FF5F46;">6</div><div class="ttl">Serve Inference</div><div class="cap">Batch or real-time HTTP</div><div class="rtag" style="color:#FF5F46;">ML ENGINEER</div><div class="hint">click for detail &#8635;</div></div><div class="face back" style="background:#FF5F46;"><h4>6. Serve Inference</h4><p>Owned by the <b>ML Engineer</b>. Deploy the approved model to Databricks Model Serving for batch scoring or low-latency real-time HTTP inference.</p><div class="h2">click to flip back</div></div></div></div>
# MAGIC <div class="arw">&#8250;</div>
# MAGIC <div class="flip"><div class="flip-in"><div class="face front"><div class="acc" style="background:#00A972;"></div><div class="num" style="background:#00A972;">7</div><div class="ttl">Scheduled Retrain</div><div class="cap">Monthly Lakeflow Job</div><div class="rtag" style="color:#00A972;">DATA ENGINEER</div><div class="hint">click for detail &#8635;</div></div><div class="face back" style="background:#00A972;"><h4>7. Scheduled Retrain</h4><p>Owned by the <b>Data Engineer</b>. A monthly Lakeflow Job retrains on fresh data, closing the loop back to data preparation and keeping the model current.</p><div class="h2">click to flip back</div></div></div></div>
# MAGIC </div>
# MAGIC <div class="loop"><div class="loop-u"></div><div class="loop-ah"></div><div class="loop-lb">retrain monthly &#8594; back to data preparation</div></div>
# MAGIC <div class="found">
# MAGIC <div class="found-lbl">GOVERNED ON THE DATABRICKS DATA + AI PLATFORM</div>
# MAGIC <div class="found-chips">
# MAGIC <span class="chip"><svg width="24" height="24" viewBox="-5.315 -5.0 210.0 210.0" style="flex:0 0 auto;"><path d="M51.5066 84.0254H19.5312V115.974H51.5066V84.0254Z" fill="#FABFBA"/><path d="M21.6914 129.215L35.5186 153.105L49.3457 129.215H21.6914Z" fill="#FF5F46"/><path d="M21.6914 70.9293H49.2017L35.3745 47.0396L21.6914 70.9293Z" fill="#FF5F46"/><path d="M61.7338 133.82L45.7461 161.451L73.4004 177.426L89.3881 149.794L61.7338 133.82Z" fill="#FABFBA"/><path d="M73.4004 22.7178L45.7461 38.6923L61.7338 66.3238L89.3881 50.3493L73.4004 22.7178Z" fill="#FABFBA"/><path d="M99.7571 72.3687L75.8477 86.1844V113.816L99.7571 127.632L123.667 113.816V86.1844L99.7571 72.3687Z" fill="#FF5F46"/><path d="M85.9297 182.031H113.44L99.6128 158.142L85.9297 182.031Z" fill="#FF5F46"/><path d="M85.9297 17.9688L99.7568 41.8585L113.584 17.9688H85.9297Z" fill="#FF5F46"/><path d="M125.972 22.7178L109.984 50.3493L137.639 66.3238L153.626 38.6923L125.972 22.7178Z" fill="#FABFBA"/><path d="M137.639 133.82L109.984 149.794L125.972 177.426L153.626 161.451L137.639 133.82Z" fill="#FABFBA"/><path d="M179.839 84.0254H147.863V115.974H179.839V84.0254Z" fill="#FABFBA"/><path d="M150.168 70.9293H177.678L163.851 47.0396L150.168 70.9293Z" fill="#FF5F46"/><path d="M150.168 129.215L163.995 153.105L177.822 129.215H150.168Z" fill="#FF5F46"/></svg>Unity Catalog</span>
# MAGIC <span class="chip mlf">ml<i>flow</i></span>
# MAGIC <span class="chip">Lakeflow Jobs</span>
# MAGIC <span class="chip">Model Serving</span>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC <script>
# MAGIC (function(){var cs=document.querySelectorAll('.mll .flip');cs.forEach(function(c){c.addEventListener('click',function(){c.classList.toggle('flipped');});});})();
# MAGIC </script>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## B. Processes
# MAGIC
# MAGIC The machine learning lifecycle is made up of the following processes:
# MAGIC
# MAGIC - **Data Preparation**: clean and organize raw data for ML.
# MAGIC - **EDA**: summarize and visualize data to guide modeling.
# MAGIC - **Feature Engineering**: modify variables to improve models.
# MAGIC - **Model Training**: fit model to minimize errors.
# MAGIC - **Validation**: test model on holdout data.
# MAGIC - **Deployment**: move validated models to production; Databricks suggests deploying code for safer automation.
# MAGIC - **Monitoring**: track models for issues like drift.
# MAGIC
# MAGIC Databricks supports the entire ML lifecycle. This course covers EDA, feature engineering, training, validation, and deployment.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1B5162; background: #F8F9FC; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #1B5162; font-size: 15pt;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #0b2026;">This is a beginning/introductory level course. For more, please see more advanced courses.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Machine Learning Lifecycle
# MAGIC
# MAGIC Let’s first look at the role of Data Scientist. Starting from data preparation and featurization, using custom code, our data scientist is responsible for managing the pipeline. This includes EDA, baseline model testing, and other tasks that take place in the development environment.
# MAGIC
# MAGIC Next, let’s look at our ML Engineer. The tasks of setting up webhooks and automated model testing typically belong to them. MLflow, which we will discuss later, is used for model tracking, and Unity Catalog will be our model registry. For scheduling model retraining jobs, we use Databricks Lakeflow Jobs, which will also be discussed later.
# MAGIC
# MAGIC A data engineer can then load the registered model and use it for batch or real-time inference, in addition to monitoring deployed models.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## D. Model Monitoring on Databricks
# MAGIC
# MAGIC Let’s close out this introduction to machine learning with what is usually the last step in the overall process: model monitoring.
# MAGIC
# MAGIC Model monitoring is a key component of the ML process. Databricks supports both batch and streaming datasets and uses a data-centric approach to monitor model performance. As a machine learning practitioner, paying close attention to profile and drift metrics will be an important part of the monitoring process.
# MAGIC
# MAGIC Databricks offers a data-centric approach to model monitoring with auto-generated dashboards that can be used to check the overall health of your deployed models. Data Profile monitors include the following features:
# MAGIC
# MAGIC - Incrementally processes data in UC tables
# MAGIC - Calculates profile metrics and drift metrics
# MAGIC - Supports custom metrics as SQL expressions
# MAGIC - Auto-generates DBSQL dashboard for tracking drift, performance, and quality metrics over time.
# MAGIC - For MLOps, use in conjunction with inference tables to monitor models
# MAGIC
# MAGIC <div style="border-left: 4px solid #1B5162; background: #F8F9FC; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #1B5162; font-size: 15pt;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #0b2026;">In this course, we only introduce you to monitoring. More information is available in the <a href="https://docs.databricks.com/aws/en/data-quality-monitoring/data-profiling/" style="color: #2272B4; text-decoration: underline;">data profiling documentation (AWS)</a>.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Conclusion
# MAGIC
# MAGIC In this lecture, you were introduced to machine learning on Databricks: the roles that collaborate across the machine learning lifecycle, the processes that make up that lifecycle from data preparation through monitoring, and how Databricks supports model monitoring with profile and drift metrics and auto-generated dashboards. The lessons that follow put these ideas into practice, covering EDA, feature engineering, training, validation, and deployment.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>