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
# MAGIC # Demo - Databricks Model Serving
# MAGIC
# MAGIC ## Overview
# MAGIC In this lesson, we will focus on how to serve a registered model using **Databricks Model Serving** for real-time inferencing. We’ll also introduce **Databricks Lakeflow Jobs** as a way to automate ML pipelines.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this demo, you will be able to:
# MAGIC - Serve a registered model using the Databricks Model Serving UI
# MAGIC - Query a serving endpoint from the UI
# MAGIC - Explore endpoint metrics and logs using the built-in dashboard
# MAGIC - Describe how Databricks Lakeflow Jobs automate ML tasks
# MAGIC - Create and inspect a Lakeflow job that runs a notebook

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

# MAGIC %md-sandbox
# MAGIC ## A. Classroom Setup
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Prerequisites</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">This demo builds on the models created in <strong>03 Demo - MLflow for Lifecycle Management</strong> and <strong>04 Demo - Genie Code for ML</strong>. Ensure you have completed those demos before starting this one.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### A1. Run the Classroom Setup
# MAGIC
# MAGIC Run the following cell to create the Python variables that will be used in this notebook.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-Model_Serving

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Introduction to Databricks Model Serving

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B1. Model Serving is Difficult
# MAGIC
# MAGIC The truth is the majority of models never make it to production. We analyzed six key industries to better understand these trends by looking at the ratio of logged-to-registered models. What did we find? Our three most efficient industries put 25% of their models into production.
# MAGIC
# MAGIC <div class="ms"><style>
# MAGIC @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
# MAGIC .ms{font-family:'DM Sans','Segoe UI',Arial,sans-serif;max-width:1080px;margin:18px auto;color:#1B3139;display:flex;gap:20px;flex-wrap:wrap;justify-content:center;}
# MAGIC .ms *{box-sizing:border-box;}
# MAGIC .ms-card{flex:1 1 300px;min-width:270px;background:#fff;border-radius:16px;border-top:6px solid var(--acc);box-shadow:0 6px 16px rgba(27,49,57,0.09);padding:26px 22px 22px;display:flex;flex-direction:column;align-items:center;text-align:center;transition:transform .2s ease,box-shadow .2s ease;}
# MAGIC .ms-card:hover{transform:translateY(-4px);box-shadow:0 16px 30px rgba(27,49,57,0.16);}
# MAGIC .ms-badge{position:relative;width:72px;height:72px;border-radius:50%;background:var(--tint);color:var(--acc);display:inline-flex;align-items:center;justify-content:center;margin-bottom:16px;}
# MAGIC .ms-badge svg{width:38px;height:38px;}
# MAGIC .ms-alert{position:absolute;top:-2px;right:-2px;width:24px;height:24px;border-radius:50%;background:#FF3621;color:#fff;font-size:15px;font-weight:800;display:inline-flex;align-items:center;justify-content:center;border:3px solid #fff;box-shadow:0 2px 5px rgba(27,49,57,0.20);}
# MAGIC .ms-t{font-size:16.5px;font-weight:800;color:#1B3139;line-height:1.3;margin-bottom:12px;}
# MAGIC .ms-d{font-size:14px;color:#5A6F77;line-height:1.55;}
# MAGIC .ms-d p{margin:0 0 9px;}
# MAGIC .ms-d p:last-child{margin-bottom:0;}
# MAGIC </style><div class="ms-card" style="--acc:#FF5F46;--tint:rgba(255,95,70,0.12);"><span class="ms-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="6.2" rx="1.6"/><rect x="3" y="13.8" width="18" height="6.2" rx="1.6"/><circle cx="7" cy="7.1" r="0.6" fill="currentColor" stroke="none"/><circle cx="7" cy="16.9" r="0.6" fill="currentColor" stroke="none"/><line x1="11" y1="7.1" x2="17.5" y2="7.1"/><line x1="11" y1="16.9" x2="17.5" y2="16.9"/></svg><span class="ms-alert">!</span></span><div class="ms-t">ML infrastructure is hard</div><div class="ms-d"><p>Real-time ML systems require fast, scalable serving infrastructure &mdash; costly to build and maintain.</p></div></div><div class="ms-card" style="--acc:#FFAB00;--tint:rgba(255,171,0,0.16);"><span class="ms-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10.4 3.6a1.9 1.9 0 013.7 0c0 .6-.3 1.1-.3 1.6 0 .5.4.8.9.8h2.4c.6 0 1 .5 1 1v2.4c0 .5.3.9.8.9.5 0 1-.3 1.6-.3a1.9 1.9 0 010 3.7c-.6 0-1.1-.3-1.6-.3-.5 0-.8.4-.8.9v3.5c0 .6-.5 1-1 1h-3.5c-.5 0-.9-.4-.9-.9 0-.5.3-1 .3-1.6a1.9 1.9 0 00-3.7 0c0 .6.3 1.1.3 1.6 0 .5-.4.9-.9.9H4.6c-.6 0-1-.4-1-1v-3.5c0-.5.4-.8.9-.8.5 0 1 .3 1.6.3a1.9 1.9 0 000-3.7c-.6 0-1.1.3-1.6.3-.5 0-.9-.4-.9-.9V8c0-.5.4-1 1-1H8c.5 0 .8-.3.8-.8 0-.5-.3-1-.3-1.6z"/></svg><span class="ms-alert">!</span></span><div class="ms-t">Deploying real-time models needs disparate tools</div><div class="ms-d"><p>Data teams use diverse tools to develop models.</p><p>Separate platforms for data, ML, and Serving add complexity and cost.</p></div></div><div class="ms-card" style="--acc:#4299E0;--tint:rgba(66,153,224,0.12);"><span class="ms-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 4 2 9l10 5 10-5-10-5Z"/><path d="M6 11.5v4.2c0 1.3 2.7 2.8 6 2.8s6-1.5 6-2.8v-4.2"/><line x1="22" y1="9" x2="22" y2="14.5"/></svg><span class="ms-alert">!</span></span><div class="ms-t">Operating production ML requires expert resources</div><div class="ms-d"><p>Steep learning curve of deployment tools.</p><p>Deployment is bottlenecked by limited engineering resources, capping the ability to scale AI.</p></div></div></div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B2. Databricks Model Serving
# MAGIC
# MAGIC Databricks Model Serving addresses these challenges by providing a unified, production-ready path from a registered model to a live serving endpoint.
# MAGIC
# MAGIC <div class="ms2"><style>
# MAGIC @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
# MAGIC .ms2{font-family:'DM Sans','Segoe UI',Arial,sans-serif;max-width:1080px;margin:18px auto;color:#1B3139;display:flex;gap:20px;flex-wrap:wrap;justify-content:center;}
# MAGIC .ms2 *{box-sizing:border-box;}
# MAGIC .ms2-card{flex:1 1 300px;min-width:270px;background:#fff;border-radius:16px;border-top:6px solid var(--acc);box-shadow:0 6px 16px rgba(27,49,57,0.09);padding:26px 22px 22px;display:flex;flex-direction:column;align-items:center;text-align:center;transition:transform .2s ease,box-shadow .2s ease;}
# MAGIC .ms2-card:hover{transform:translateY(-4px);box-shadow:0 16px 30px rgba(27,49,57,0.16);}
# MAGIC .ms2-badge{position:relative;width:72px;height:72px;border-radius:50%;background:var(--tint);color:var(--acc);display:inline-flex;align-items:center;justify-content:center;margin-bottom:16px;}
# MAGIC .ms2-badge svg{width:38px;height:38px;}
# MAGIC .ms2-good{position:absolute;top:-2px;right:-2px;width:24px;height:24px;border-radius:50%;background:#00A972;color:#fff;font-size:13px;font-weight:800;display:inline-flex;align-items:center;justify-content:center;border:3px solid #fff;box-shadow:0 2px 5px rgba(27,49,57,0.20);}
# MAGIC .ms2-t{font-size:16.5px;font-weight:800;color:#1B3139;line-height:1.3;margin-bottom:12px;}
# MAGIC .ms2-d{font-size:14px;color:#5A6F77;line-height:1.55;}
# MAGIC .ms2-d p{margin:0 0 9px;}
# MAGIC .ms2-d p:last-child{margin-bottom:0;}
# MAGIC .ms2-d b{color:#1B3139;font-weight:700;}
# MAGIC </style><div class="ms2-card" style="--acc:#FF5F46;--tint:rgba(255,95,70,0.12);"><span class="ms2-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l7 2.8v5.2c0 4.4-3 7.4-7 8.9-4-1.5-7-4.5-7-8.9V5.8L12 3Z"/><path d="M8.8 11.8l2.2 2.2 4.2-4.4"/></svg><span class="ms2-good">&#10003;</span></span><div class="ms2-t">Production-Grade Serving</div><div class="ms2-d"><p><b>Highly available</b>, low latency, scalable serving that works for small and large Lakeflow Jobs.</p></div></div><div class="ms2-card" style="--acc:#FFAB00;--tint:rgba(255,171,0,0.16);"><span class="ms2-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z"/></svg><span class="ms2-good">&#10003;</span></span><div class="ms2-t">Accelerate deployments with Lakehouse-Unified Serving</div><div class="ms2-d"><p>Automatic feature lookups, monitoring, and unified governance that <b>automates deployment</b> and reduces errors.</p></div></div><div class="ms2-card" style="--acc:#4299E0;--tint:rgba(66,153,224,0.12);"><span class="ms2-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 20 14.5 9.5"/><path d="M13.5 8.5l2 2"/><path d="M17.5 2.5l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8z"/></svg><span class="ms2-good">&#10003;</span></span><div class="ms2-t">Simplified Deployment</div><div class="ms2-d"><p>Simple and flexible deployment through <b>UI</b> or <b>API</b>.</p></div></div></div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div class="sm"><style>@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');.sm{font-family:'DM Sans','Segoe UI',Arial,sans-serif;max-width:1120px;margin:16px auto;color:#1B3139;}.sm *{box-sizing:border-box;}.sm-hero{background:linear-gradient(135deg,#1B5162 0%,#12414F 100%);border-radius:16px;padding:22px 26px;margin-bottom:22px;box-shadow:0 8px 20px rgba(27,49,57,0.16);}.sm-hero h3{margin:0;font-size:23px;font-weight:800;color:#fff;}.sm-hero p{margin:7px 0 0;font-size:14.5px;color:#CFE0E5;line-height:1.5;}.sm-flow{display:flex;align-items:stretch;flex-wrap:wrap;justify-content:center;margin-bottom:18px;}.sm-node{flex:1 1 200px;min-width:180px;background:#fff;border-radius:14px;border-top:5px solid var(--acc);box-shadow:0 6px 14px rgba(27,49,57,0.09);padding:16px 14px;text-align:center;}.sm-nbadge{display:inline-flex;align-items:center;justify-content:center;width:40px;height:40px;border-radius:11px;background:var(--tint);margin-bottom:8px;}.sm-nbadge svg{width:23px;height:23px;}.sm-nt{font-size:15px;font-weight:800;color:#1B3139;line-height:1.25;}.sm-ns{font-size:12.5px;color:#5A6F77;margin-top:3px;}.sm-arw{flex:0 0 50px;display:flex;align-items:center;justify-content:center;}.sm-arw span{width:36px;height:36px;border-radius:50%;background:#EDF1F3;color:#1B5162;display:inline-flex;align-items:center;justify-content:center;font-size:19px;font-weight:800;box-shadow:0 2px 7px rgba(27,49,57,0.12);}.sm-div{font-size:12px;font-weight:800;letter-spacing:.8px;text-transform:uppercase;color:#5A6F77;text-align:center;margin:16px 0 12px;}.sm-tabs{display:flex;gap:10px;flex-wrap:wrap;justify-content:center;margin-bottom:16px;}.sm-tab{display:inline-flex;align-items:center;gap:8px;font-family:inherit;font-size:14.5px;font-weight:700;color:var(--acc);background:#fff;border:2px solid var(--acc);border-radius:26px;padding:9px 18px;cursor:pointer;transition:all .18s ease;}.sm-tab:hover{background:var(--tint);}.sm-tab .sm-ti{display:inline-flex;}.sm-tab .sm-ti svg{width:18px;height:18px;}.sm-tab.active{background:var(--acc);color:#fff;box-shadow:0 6px 14px rgba(27,49,57,0.16);}.sm-panels{position:relative;}.sm-panel{display:none;background:#fff;border-radius:16px;border-top:6px solid var(--acc);box-shadow:0 6px 16px rgba(27,49,57,0.10);padding:24px 26px;}.sm-panel.show{display:block;animation:smf .25s ease;}@keyframes smf{from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:none;}}.sm-phead{display:flex;align-items:center;gap:14px;margin-bottom:14px;}.sm-pbadge{flex:0 0 54px;width:54px;height:54px;border-radius:14px;background:var(--tint);color:var(--acc);display:inline-flex;align-items:center;justify-content:center;}.sm-pbadge svg{width:30px;height:30px;}.sm-pt{font-size:20px;font-weight:800;color:#1B3139;line-height:1.15;}.sm-plat{display:inline-block;margin-top:5px;font-size:12px;font-weight:700;color:var(--acc);background:var(--tint);border-radius:20px;padding:3px 12px;}.sm-pul{list-style:none;margin:0;padding:0;display:flex;flex-wrap:wrap;gap:10px 22px;}.sm-pul li{position:relative;padding-left:24px;font-size:14.5px;color:#1B3139;line-height:1.5;flex:1 1 240px;}.sm-pul li::before{content:'';position:absolute;left:0;top:2px;width:16px;height:16px;border-radius:50%;background:var(--acc);}.sm-pul li::after{content:'';position:absolute;left:4.5px;top:5px;width:4px;height:7px;border:solid #fff;border-width:0 2px 2px 0;transform:rotate(45deg);}</style><div class="sm-hero"><h3>Model Serving Modes</h3><p>Trained models register in Unity Catalog, then serve in one of four modes &mdash; pick a mode below to see how it works.</p></div><div class="sm-flow"><div class="sm-node" style="--acc:#618794;--tint:rgba(97,135,148,0.10);"><span class="sm-nbadge" style="color:#618794;"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6"/><path d="M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3"/></svg></span><div class="sm-nt">Delta Lake / Feature Engineering</div><div class="sm-ns">training data</div></div><div class="sm-arw"><span>&rarr;</span></div><div class="sm-node" style="--acc:#4299E0;--tint:rgba(97,135,148,0.10);"><span class="sm-nbadge" style="color:#4299E0;"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1"/></svg></span><div class="sm-nt">Model training</div><div class="sm-ns">+ Agent Bricks</div></div><div class="sm-arw"><span>&rarr;</span></div><div class="sm-node" style="--acc:#FF5F46;--tint:rgba(97,135,148,0.10);"><span class="sm-nbadge" style="color:#FF5F46;"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 4h11l5 5v11a1 1 0 01-1 1H4a1 1 0 01-1-1V5a1 1 0 011-1Z"/><path d="M14 4v5h5"/><path d="M8 13l2.5 2.5L16 10"/></svg></span><div class="sm-nt">Model Registry</div><div class="sm-ns">in Unity Catalog</div></div></div><div class="sm-div">Four serving modes &mdash; click to explore</div><div class="sm-tabs"><button class="sm-tab active" data-i="0" style="--acc:#4299E0;--tint:rgba(66,153,224,0.12);"><span class="sm-ti"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3 3 8l9 5 9-5-9-5Z"/><path d="M3 13l9 5 9-5"/></svg></span>Batch</button><button class="sm-tab" data-i="1" style="--acc:#00A972;--tint:rgba(0,169,114,0.12);"><span class="sm-ti"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2 12h4l2.5-8 4 15 2.5-7H22"/></svg></span>Streaming</button><button class="sm-tab" data-i="2" style="--acc:#FF5F46;--tint:rgba(255,95,70,0.12);"><span class="sm-ti"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z"/></svg></span>Real Time</button><button class="sm-tab" data-i="3" style="--acc:#FFAB00;--tint:rgba(255,171,0,0.16);"><span class="sm-ti"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="1.6"/><path d="M10 7V4M14 7V4M10 20v-3M14 20v-3M7 10H4M7 14H4M20 10h-3M20 14h-3"/></svg></span>Embedded (Edge)</button></div><div class="sm-panels"><div class="sm-panel show" data-i="0" style="--acc:#4299E0;--tint:rgba(66,153,224,0.12);"><div class="sm-phead"><span class="sm-pbadge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3 3 8l9 5 9-5-9-5Z"/><path d="M3 13l9 5 9-5"/></svg></span><div><div class="sm-pt">Batch</div><span class="sm-plat">Latency: High latency</span></div></div><ul class="sm-pul"><li>Databases or object storage</li><li>Fast retrieval of stored predictions</li></ul></div><div class="sm-panel" data-i="1" style="--acc:#00A972;--tint:rgba(0,169,114,0.12);"><div class="sm-phead"><span class="sm-pbadge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2 12h4l2.5-8 4 15 2.5-7H22"/></svg></span><div><div class="sm-pt">Streaming</div><span class="sm-plat">Latency: Moderate latency</span></div></div><ul class="sm-pul"><li>Stream processing</li><li>Moderately fast scoring on new data</li></ul></div><div class="sm-panel" data-i="2" style="--acc:#FF5F46;--tint:rgba(255,95,70,0.12);"><div class="sm-phead"><span class="sm-pbadge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z"/></svg></span><div><div class="sm-pt">Real Time</div><span class="sm-plat">Latency: Low latency (ms)</span></div></div><ul class="sm-pul"><li>Low-latency scoring</li><li>High availability</li><li>REST via containers / K8s</li></ul></div><div class="sm-panel" data-i="3" style="--acc:#FFAB00;--tint:rgba(255,171,0,0.16);"><div class="sm-phead"><span class="sm-pbadge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="1.6"/><path d="M10 7V4M14 7V4M10 20v-3M14 20v-3M7 10H4M7 14H4M20 10h-3M20 14h-3"/></svg></span><div><div class="sm-pt">Embedded (Edge)</div><span class="sm-plat">Latency: On-device</span></div></div><ul class="sm-pul"><li>Special-case deployments</li><li>Limited cloud connectivity</li></ul></div></div><script>(function(){var tabs=document.querySelectorAll('.sm-tab');var panels=document.querySelectorAll('.sm-panel');function activate(i){tabs.forEach(function(t){t.classList.toggle('active',t.getAttribute('data-i')===i);});panels.forEach(function(p){p.classList.toggle('show',p.getAttribute('data-i')===i);});}tabs.forEach(function(t){t.addEventListener('click',function(){activate(t.getAttribute('data-i'));});});})();</script></div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### B4. Model Serving on Databricks
# MAGIC
# MAGIC Since Databricks builds on top of MLflow, it inherits key components like model artifacts and tracking. Databricks takes this further with automation and by integrating platform features like production-ready evaluation strategies and inference tables. Serverless deployment delivers low latency and high throughput in a secure environment governed by Unity Catalog.
# MAGIC
# MAGIC <div class="b4b"><style>@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');.b4b{font-family:'DM Sans','Segoe UI',Arial,sans-serif;max-width:1080px;margin:18px auto;color:#1B3139;display:flex;gap:16px;flex-wrap:wrap;justify-content:center;}.b4b *{box-sizing:border-box;}.fc{flex:1 1 230px;min-width:210px;height:222px;perspective:1200px;cursor:pointer;}.fc-inner{position:relative;width:100%;height:100%;transition:transform .55s cubic-bezier(.4,.2,.2,1);transform-style:preserve-3d;}.fc.flipped .fc-inner{transform:rotateY(180deg);}.fc-face{position:absolute;inset:0;backface-visibility:hidden;-webkit-backface-visibility:hidden;border-radius:16px;padding:22px 18px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;box-shadow:0 6px 16px rgba(27,49,57,0.10);}.fc-front{background:#fff;border-top:6px solid var(--acc);}.fc-badge{width:60px;height:60px;border-radius:16px;background:var(--acc);color:#fff;display:inline-flex;align-items:center;justify-content:center;margin-bottom:14px;box-shadow:0 4px 12px rgba(27,49,57,0.16);}.fc-badge svg{width:32px;height:32px;}.fc-t{font-size:16.5px;font-weight:800;color:#1B3139;line-height:1.25;}.fc-hint{position:absolute;bottom:12px;left:0;right:0;font-size:11px;font-weight:700;letter-spacing:.4px;text-transform:uppercase;color:var(--acc);opacity:.75;}.fc-back{background:var(--acc);transform:rotateY(180deg);}.fc-d{font-size:14px;font-weight:600;color:#fff;line-height:1.55;}.fc-hint-b{color:#fff;opacity:.85;}.fc:hover .fc-face{box-shadow:0 14px 28px rgba(27,49,57,0.18);}</style><div class="fc" data-i="0" style="--acc:#00A972;--tint:rgba(0,0,0,0);"><div class="fc-inner"><div class="fc-face fc-front"><span class="fc-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9.5 14.5 14.5 9.5"/><path d="M10.5 6.5 12.3 4.7a4 4 0 015.7 5.7l-1.8 1.8"/><path d="M13.5 17.5l-1.8 1.8a4 4 0 01-5.7-5.7l1.8-1.8"/></svg></span><div class="fc-t">Databricks Integrated</div><div class="fc-hint">Click to flip &#8635;</div></div><div class="fc-face fc-back"><div class="fc-d">Automatic feature / vector lookups, monitoring, and unified governance.</div><div class="fc-hint fc-hint-b">&#8617; Back</div></div></div></div><div class="fc" data-i="1" style="--acc:#4299E0;--tint:rgba(0,0,0,0);"><div class="fc-inner"><div class="fc-face fc-front"><span class="fc-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 18a4 4 0 01-.5-8 6 6 0 0111.5 1.5A3.5 3.5 0 0117 18H7Z"/></svg></span><div class="fc-t">Simplified, Serverless Deployment</div><div class="fc-hint">Click to flip &#8635;</div></div><div class="fc-face fc-back"><div class="fc-d">Deploy any model type (CPU / GPU) with automated container builds, low latency (p50 &lt;10ms), and high throughput (&gt;25K QPS).</div><div class="fc-hint fc-hint-b">&#8617; Back</div></div></div></div><div class="fc" data-i="2" style="--acc:#FF5F46;--tint:rgba(0,0,0,0);"><div class="fc-inner"><div class="fc-face fc-front"><span class="fc-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 3h6"/><path d="M10 3v5.5L4.8 17.5A2 2 0 006.5 20.5h11a2 2 0 001.7-3L14 8.5V3"/><path d="M7.6 15h8.8"/></svg></span><div class="fc-t">Production Ready</div><div class="fc-hint">Click to flip &#8635;</div></div><div class="fc-face fc-back"><div class="fc-d">Online evaluation strategies such as A/B testing &mdash; serve multiple models to a single endpoint.</div><div class="fc-hint fc-hint-b">&#8617; Back</div></div></div></div><div class="fc" data-i="3" style="--acc:#FFAB00;--tint:rgba(0,0,0,0);"><div class="fc-inner"><div class="fc-face fc-front"><span class="fc-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 10h18M9 5v14"/></svg></span><div class="fc-t">Inference Tables</div><div class="fc-hint">Click to flip &#8635;</div></div><div class="fc-face fc-back"><div class="fc-d">Log each request &amp; response for monitoring, retraining, debugging, and more.</div><div class="fc-hint fc-hint-b">&#8617; Back</div></div></div></div><script>(function(){document.querySelectorAll('.b4b .fc').forEach(function(c){c.addEventListener('click',function(){c.classList.toggle('flipped');});});})();</script></div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Serve a Model with Databricks Model Serving
# MAGIC
# MAGIC We can create Model Serving endpoints with the Databricks Machine Learning API or the Databricks Machine Learning UI. An endpoint can serve any registered Python MLflow model in the **Unity Catalog Model Registry**.
# MAGIC
# MAGIC To keep it simple, in this demo we are going to use the Model Serving UI for creating, managing, and using Model Serving endpoints. We can create model serving endpoints with the **Serving** page UI or directly from the registered **Models** page.

# COMMAND ----------

# MAGIC %md
# MAGIC ### C1. Set Up Model Serving
# MAGIC
# MAGIC Let's go through the steps of creating a model serving endpoint on the Models page. **You will not actually create the endpoint.**
# MAGIC
# MAGIC 1. Right-click **Models** in the left navigation, and select **Open Link in New Tab**.
# MAGIC 2. Select **Owned by me**.
# MAGIC 3. Select the model you want to serve under the **Name** column. Notice this will take you to Catalog Explorer.
# MAGIC 4. Click the **Serve this model** button in the top-right corner. This will take you to the **Serving endpoints** page.
# MAGIC 5. In **General**, enter a name of the form **firstName-lastName-endpoint**.
# MAGIC 6. There are several configurations under **Served entities** that we will not discuss here. Leave **Entity**, **Compute type**, and **Compute scale-out** at their default values. You can select **Scale to zero** for this lesson as well. This feature can save costs by allowing the endpoint compute to shut down if there are no requests after a period of time. Databricks Free Edition requires this to be checked.
# MAGIC 7. **Do not click Create** at the bottom right. The above instructions are only for demonstration purposes. **You do not need to provision an endpoint because we already did so in the last demo.**
# MAGIC     - If you happen to accidentally create an endpoint, navigate to the left sidebar and click **Serving**. Then click the endpoint you provisioned, click the three-dot menu at the top right, and select **Delete**. Again, **do not provision an endpoint.**

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### C2. Create a Model Serving Endpoint with Python
# MAGIC
# MAGIC In the last demo, we used code very similar to the following to create a Databricks Model Serving endpoint. This code is shown for reference and is not meant to be run here.
# MAGIC
# MAGIC <div class="cc"><style>@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');.cc{font-family:'DM Sans','Segoe UI',Arial,sans-serif;max-width:900px;margin:16px auto;border-radius:14px;overflow:hidden;box-shadow:0 10px 26px rgba(27,49,57,0.20);}.cc *{box-sizing:border-box;}.cc-bar{display:flex;align-items:center;gap:10px;background:#14313A;padding:12px 18px;}.cc-dots{display:flex;gap:7px;}.cc-dots i{width:12px;height:12px;border-radius:50%;display:inline-block;}.cc-dots i:nth-child(1){background:#FF5F46;}.cc-dots i:nth-child(2){background:#FFAB00;}.cc-dots i:nth-child(3){background:#00A972;}.cc-lang{color:#CFE0E5;font-size:13px;font-weight:700;letter-spacing:.4px;}.cc-note{margin-left:auto;color:#7E97A1;font-size:12px;font-weight:600;}.cc-body{background:#0F252B;color:#E6EDEF;font-family:'JetBrains Mono','SFMono-Regular',Menlo,Consolas,monospace;font-size:14px;line-height:1.7;padding:18px 0;overflow-x:auto;counter-reset:ln;}.cc-body .cl{display:block;white-space:pre;padding:0 20px 0 3.6em;position:relative;}.cc-body .cl:hover{background:rgba(255,255,255,0.04);}.cc-body .cl::before{counter-increment:ln;content:counter(ln);position:absolute;left:0;width:2.7em;text-align:right;color:#48626D;user-select:none;}.cc-body .k{color:#FF8A73;font-weight:600;}.cc-body .s{color:#8FD9A8;}.cc-body .num{color:#FFC46B;}.cc-body .c{color:#6E8B99;font-style:italic;}.cc-body .fn{color:#6FB6F0;}.cc-body .o{color:#9FB0B8;}</style><div class="cc-bar"><span class="cc-dots"><i></i><i></i><i></i></span><span class="cc-lang">Python</span><span class="cc-note">reference only &mdash; not meant to run here</span></div><div class="cc-body"><span class="cl"><span class="k">from</span> mlflow<span class="o">.</span>deployments <span class="k">import</span> get_deploy_client</span><span class="cl"></span><span class="cl">client <span class="o">=</span> <span class="fn">get_deploy_client</span><span class="o">(</span><span class="s">&quot;databricks&quot;</span><span class="o">)</span></span><span class="cl">endpoint_name <span class="o">=</span> <span class="s">f&quot;{username}-model&quot;</span></span><span class="cl">endpoint_name <span class="o">=</span> endpoint_name<span class="o">.</span><span class="fn">replace</span><span class="o">(</span><span class="s">&quot;@databricks.com&quot;</span><span class="o">,</span> <span class="s">&quot;&quot;</span><span class="o">)</span><span class="o">.</span><span class="fn">replace</span><span class="o">(</span><span class="s">&#x27;.&#x27;</span><span class="o">,</span> <span class="s">&#x27;-&#x27;</span><span class="o">)</span></span><span class="cl"></span><span class="cl"><span class="c"># Check if the endpoint already exists</span></span><span class="cl"><span class="k">try</span><span class="o">:</span></span><span class="cl">    <span class="c"># Attempt to get the endpoint</span></span><span class="cl">    existing_endpoint <span class="o">=</span> client<span class="o">.</span><span class="fn">get_endpoint</span><span class="o">(</span>endpoint_name<span class="o">)</span></span><span class="cl">    <span class="fn">print</span><span class="o">(</span><span class="s">f&quot;Endpoint &#x27;{endpoint_name}&#x27; already exists.&quot;</span><span class="o">)</span></span><span class="cl"><span class="k">except</span> Exception <span class="k">as</span> e<span class="o">:</span></span><span class="cl">    <span class="c"># If not found, create the endpoint</span></span><span class="cl">    <span class="k">if</span> <span class="s">&quot;RESOURCE_DOES_NOT_EXIST&quot;</span> <span class="k">in</span> <span class="fn">str</span><span class="o">(</span>e<span class="o">)</span><span class="o">:</span></span><span class="cl">        <span class="fn">print</span><span class="o">(</span><span class="s">f&quot;Creating a new endpoint: {endpoint_name}&quot;</span><span class="o">)</span></span><span class="cl">        endpoint <span class="o">=</span> client<span class="o">.</span><span class="fn">create_endpoint</span><span class="o">(</span></span><span class="cl">            name<span class="o">=</span>endpoint_name<span class="o">,</span></span><span class="cl">            config<span class="o">=</span><span class="o">{</span></span><span class="cl">                <span class="s">&quot;served_entities&quot;</span><span class="o">:</span> <span class="o">[</span></span><span class="cl">                    <span class="o">{</span></span><span class="cl">                        <span class="s">&quot;name&quot;</span><span class="o">:</span> <span class="s">&quot;my-model&quot;</span><span class="o">,</span></span><span class="cl">                        <span class="s">&quot;entity_name&quot;</span><span class="o">:</span> model_name<span class="o">,</span></span><span class="cl">                        <span class="s">&quot;entity_version&quot;</span><span class="o">:</span> <span class="s">&quot;1&quot;</span><span class="o">,</span></span><span class="cl">                        <span class="s">&quot;workload_size&quot;</span><span class="o">:</span> <span class="s">&quot;Small&quot;</span><span class="o">,</span></span><span class="cl">                        <span class="s">&quot;scale_to_zero_enabled&quot;</span><span class="o">:</span> <span class="k">True</span></span><span class="cl">                    <span class="o">}</span></span><span class="cl">                <span class="o">]</span><span class="o">,</span></span><span class="cl">                <span class="s">&quot;traffic_config&quot;</span><span class="o">:</span> <span class="o">{</span></span><span class="cl">                    <span class="s">&quot;routes&quot;</span><span class="o">:</span> <span class="o">[</span></span><span class="cl">                        <span class="o">{</span></span><span class="cl">                            <span class="s">&quot;served_model_name&quot;</span><span class="o">:</span> <span class="s">&quot;my-model&quot;</span><span class="o">,</span></span><span class="cl">                            <span class="s">&quot;traffic_percentage&quot;</span><span class="o">:</span> <span class="num">100</span></span><span class="cl">                        <span class="o">}</span></span><span class="cl">                    <span class="o">]</span></span><span class="cl">                <span class="o">}</span></span><span class="cl">            <span class="o">}</span></span><span class="cl">        <span class="o">)</span></span><span class="cl">    <span class="k">else</span><span class="o">:</span></span><span class="cl">        <span class="fn">print</span><span class="o">(</span><span class="s">f&quot;An error occurred: {e}&quot;</span><span class="o">)</span></span></div></div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### C3. Query the Serving Endpoint
# MAGIC
# MAGIC Let's use the deployed model for real-time inference. Here’s a step-by-step guide for querying an endpoint in Databricks Model Serving:
# MAGIC
# MAGIC 1. Right-click **Serving** in the left navigation and select **Open Link in New Tab**.
# MAGIC 2. Select the endpoint you want to query. The one we created in the last demo is named "your-user-name-model". Please note that it is possible that the endpoint has not finished the setup process, which takes 10–12 minutes from the time we ran the last demo.
# MAGIC 3. Click the **Use** button in the top-right corner.
# MAGIC 4. There are 4 methods for querying an endpoint: **browser**, **CURL**, **Python**, and **SQL**. For now, let's use the easiest method — querying right in the **browser** window. In this method, we need to provide the input parameters in JSON format. Since we used `mlflow.sklearn.autolog()` with `log_input_examples = True`, we registered an example with MLflow, which appears automatically when selecting **browser**.
# MAGIC 5. Click **Send request**.
# MAGIC 6. The **Response** field on the right panel will show the result of the inference.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Since we enabled "Scale to zero", it is possible that our request takes longer than normal. Scale to zero is not recommended for production use because of the possibility of this delay.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Introduction to ML with Lakeflow Jobs

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### D1. Databricks Lakeflow Overview
# MAGIC
# MAGIC Let&#8217;s look at how Lakeflow supports the end-to-end ML pipeline within the Databricks ecosystem.
# MAGIC
# MAGIC <div class="lf"><style>@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');.lf{font-family:'DM Sans','Segoe UI',Arial,sans-serif;max-width:1080px;margin:16px auto;color:#1B3139;}.lf *{box-sizing:border-box;}.lf-kick{font-size:12px;font-weight:800;letter-spacing:.8px;text-transform:uppercase;color:#5A6F77;text-align:center;margin-bottom:14px;}.lf-flow{display:flex;align-items:flex-start;justify-content:center;flex-wrap:wrap;}.lf-card{flex:1 1 240px;min-width:230px;background:#fff;border-radius:16px;border-top:6px solid var(--acc);box-shadow:0 6px 16px rgba(27,49,57,0.09);padding:20px 20px 14px;cursor:pointer;transition:transform .2s ease,box-shadow .2s ease;}.lf-card:hover{transform:translateY(-3px);box-shadow:0 14px 28px rgba(27,49,57,0.15);}.lf-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;}.lf-badge{width:50px;height:50px;border-radius:14px;background:var(--tint);color:var(--acc);display:inline-flex;align-items:center;justify-content:center;}.lf-badge svg{width:28px;height:28px;}.lf-step{width:28px;height:28px;border-radius:50%;background:var(--acc);color:#fff;font-size:15px;font-weight:800;display:inline-flex;align-items:center;justify-content:center;}.lf-name{font-size:16px;font-weight:800;color:#1B3139;line-height:1.25;}.lf-role{display:inline-block;margin-top:7px;font-size:12px;font-weight:700;color:var(--acc);background:var(--tint);border-radius:20px;padding:3px 12px;}.lf-detail{max-height:0;overflow:hidden;opacity:0;transition:max-height .38s ease,opacity .3s ease,margin-top .3s ease;}.lf-card.open .lf-detail{max-height:220px;opacity:1;margin-top:12px;}.lf-detail p{margin:0;font-size:13.5px;color:#5A6F77;line-height:1.55;}.lf-toggle{margin-top:12px;font-size:11.5px;font-weight:800;letter-spacing:.4px;text-transform:uppercase;color:var(--acc);display:flex;align-items:center;gap:5px;}.lf-chev{display:inline-block;transition:transform .3s ease;font-size:14px;}.lf-card.open .lf-chev{transform:rotate(180deg);}.lf-arw{flex:0 0 46px;display:flex;align-items:center;justify-content:center;align-self:center;margin-top:34px;}.lf-arw span{width:34px;height:34px;border-radius:50%;background:#EDF1F3;color:#1B5162;display:inline-flex;align-items:center;justify-content:center;font-size:18px;font-weight:800;box-shadow:0 2px 6px rgba(27,49,57,0.12);}.lf-found{margin-top:22px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;background:#F8F9FC;border:1px solid #E4E9F2;border-left:5px solid #1B5162;border-radius:12px;padding:16px 18px;box-shadow:0 4px 10px rgba(27,49,57,0.05);}.lf-found p{margin:0;font-size:13.5px;line-height:1.55;color:#1B3139;flex:1 1 320px;}.lf-found b{color:#1B5162;}.lf-chips{display:flex;gap:8px;flex-wrap:wrap;}.lf-chips span{background:#fff;border:1.5px solid #1B5162;color:#1B5162;font-size:12px;font-weight:700;border-radius:20px;padding:5px 12px;}</style><div class="lf-kick">The Lakeflow pipeline &mdash; click a stage to expand</div><div class="lf-flow"><div class="lf-card" data-i="1" style="--acc:#4299E0;--tint:rgba(66,153,224,0.12);"><div class="lf-top"><span class="lf-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3v9"/><path d="M8 9l4 4 4-4"/><path d="M4 15v3a2 2 0 002 2h12a2 2 0 002-2v-3"/></svg></span><span class="lf-step">1</span></div><div class="lf-name">Lakeflow Connect</div><div class="lf-role">Ingest</div><div class="lf-detail"><p>High-performance ingestion connectors bring raw data into the Lakehouse &mdash; critical for building robust feature sets.</p></div><div class="lf-toggle">Details <span class="lf-chev">&#8964;</span></div></div><div class="lf-arw"><span>&rarr;</span></div><div class="lf-card" data-i="2" style="--acc:#00A972;--tint:rgba(0,169,114,0.12);"><div class="lf-top"><span class="lf-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 5h18l-7 8v6l-4-2v-4L3 5Z"/></svg></span><span class="lf-step">2</span></div><div class="lf-name">Lakeflow pipelines</div><div class="lf-role">Transform &middot; ETL</div><div class="lf-detail"><p>Manages feature pipelines with reliable, scalable, declarative ETL.</p></div><div class="lf-toggle">Details <span class="lf-chev">&#8964;</span></div></div><div class="lf-arw"><span>&rarr;</span></div><div class="lf-card" data-i="3" style="--acc:#FF5F46;--tint:rgba(255,95,70,0.12);"><div class="lf-top"><span class="lf-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3.2"/><path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.2 5.2l2.1 2.1M16.7 16.7l2.1 2.1M18.8 5.2l-2.1 2.1M7.3 16.7l-2.1 2.1"/></svg></span><span class="lf-step">3</span></div><div class="lf-name">Lakeflow Jobs</div><div class="lf-role">Orchestrate</div><div class="lf-detail"><p>Automates feature engineering, triggers model training, and orchestrates model deployments.</p></div><div class="lf-toggle">Details <span class="lf-chev">&#8964;</span></div></div></div><div class="lf-found"><p>Built on <b>Apache Spark</b> and <b>Structured Streaming</b> for performance and scale, with <b>Unity Catalog</b> ensuring secure access and governance across your data, models, and assets.</p><div class="lf-chips"><span>Apache Spark</span><span>Structured Streaming</span><span>Unity Catalog</span></div></div><script>(function(){document.querySelectorAll('.lf .lf-card').forEach(function(c){c.addEventListener('click',function(){c.classList.toggle('open');});});})();</script></div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### D2. Databricks Lakeflow Jobs Features
# MAGIC
# MAGIC Lakeflow Jobs offer a powerful and flexible way to automate ML pipelines across any cloud.
# MAGIC
# MAGIC <div class="lj"><style>@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');.lj{font-family:'DM Sans','Segoe UI',Arial,sans-serif;max-width:1080px;margin:18px auto;color:#1B3139;display:flex;gap:20px;flex-wrap:wrap;align-items:stretch;justify-content:center;}.lj *{box-sizing:border-box;}.lj-card{flex:1 1 300px;min-width:270px;background:#fff;border-radius:16px;border-top:6px solid var(--acc);box-shadow:0 6px 16px rgba(27,49,57,0.09);padding:24px 22px;display:flex;flex-direction:column;transition:transform .2s ease,box-shadow .2s ease;}.lj-card:hover{transform:translateY(-4px);box-shadow:0 16px 30px rgba(27,49,57,0.16);}.lj-badge{width:60px;height:60px;border-radius:16px;background:var(--tint);color:var(--acc);display:inline-flex;align-items:center;justify-content:center;margin-bottom:16px;}.lj-badge svg{width:32px;height:32px;}.lj-t{font-size:17.5px;font-weight:800;color:var(--acc);line-height:1.25;margin-bottom:9px;}.lj-d{font-size:14px;color:#5A6F77;line-height:1.55;}.lj-chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px;}.lj-chips span{background:var(--tint);color:var(--acc);font-size:12px;font-weight:700;border-radius:8px;padding:5px 11px;}</style><div class="lj-card" style="--acc:#FF5F46;--tint:rgba(255,95,70,0.12);"><span class="lj-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="3.5" width="7" height="5" rx="1.2"/><rect x="14" y="3.5" width="7" height="5" rx="1.2"/><rect x="8.5" y="15" width="7" height="5" rx="1.2"/><path d="M6.5 8.5v3a2 2 0 002 2h3.5M17.5 8.5v3a2 2 0 01-2 2H12"/></svg></span><div class="lj-t">Orchestrate Anything, Anywhere</div><div class="lj-d">Run diverse Lakeflow Jobs across the full data &amp; AI lifecycle, on any cloud.<div class="lj-chips"><span>Notebooks</span><span>Lakeflow pipelines</span><span>SQL queries</span><span>ML training &amp; deployment</span></div></div></div><div class="lj-card" style="--acc:#4299E0;--tint:rgba(66,153,224,0.12);"><span class="lj-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 18a4 4 0 01-.5-8 6 6 0 0111.5 1.5A3.5 3.5 0 0117 18H8"/><path d="M9.3 14.2l1.7 1.7 3.4-3.6"/></svg></span><div class="lj-t">Fully Managed</div><div class="lj-d">Remove operational overhead with a fully managed orchestration service &mdash; focus on your jobs, not your infrastructure.</div></div><div class="lj-card" style="--acc:#00A972;--tint:rgba(0,169,114,0.12);"><span class="lj-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5.5 3.2l13.8 6.6-6 2.1-2.1 6z"/><path d="M13.5 13.5 19 19"/></svg></span><div class="lj-t">Simplified Authoring</div><div class="lj-d">An easy point-and-click authoring experience for all your data teams &mdash; not just those with specialized skills.</div></div></div>
# MAGIC
# MAGIC With the **Orchestrate Anything, Anywhere** capability, we can automate key ML tasks like running notebooks for data processing, training models, refreshing feature tables, and more &mdash; across clouds and regions.
# MAGIC
# MAGIC It&rsquo;s a **fully managed** orchestration layer, so you don&rsquo;t need to worry about infrastructure. Just define your ML tasks, and Lakeflow Jobs will run them reliably.
# MAGIC
# MAGIC And it&rsquo;s **simple to use**, with an intuitive UI for both beginners and advanced users. Even complex pipelines involving CI/CD for model promotion can be built directly into the workflow system.
# MAGIC
# MAGIC This lets teams focus more on building and deploying models &mdash; and less on managing how they&rsquo;re run.

# COMMAND ----------

# MAGIC %md
# MAGIC ### D3. Create a Job
# MAGIC
# MAGIC Now, let's create a Lakeflow job to run this notebook:
# MAGIC
# MAGIC 1. Start by navigating to **Jobs & Pipelines** on the left sidebar.
# MAGIC 2. You have 2 options to proceed:
# MAGIC     - Option 1: Navigate to the **Job Runs** tab, then click **Create Job** at the top-right corner.
# MAGIC     - Option 2: In the **Jobs and Pipelines** section, click the **Create** dropdown at the top-right corner, and select **Job** from the options.
# MAGIC 3. Click **Notebook**.
# MAGIC 4. In the presented menu, enter a name under **Task Name**. For example, you can use **test_task_name**.
# MAGIC 5. Under **Type**, ensure that **Notebook** is selected. Also, make sure **Workspace** is selected under **Source**. Both should be selected by default.
# MAGIC 6. Next, we will attach this notebook to the job. Go to **Path**. Click the drop-down menu. You will be presented with a menu that will allow you to navigate to this notebook. By default, you will already be in your home folder. Navigate to the course materials folder. Select `05 Demo - Databricks Model Serving`. Click **Confirm** at the bottom right of the **Select Notebook** menu.
# MAGIC 7. Under **Compute**, you'll see **Serverless** is automatically selected.
# MAGIC 8. You’ll see additional options, such as **dependent libraries**, **parameters**, **notifications**, **retries**, and **Metric threshold**. You can leave these options unpopulated for this lesson.
# MAGIC 9. Click **Create Task**. A message will appear at the top right indicating your task was successfully created.
# MAGIC 10. In the right menu, there are various options. For example, you can add tags, set job parameters, and configure schedules for runs. We'll leave these alone for this lesson.
# MAGIC 11. Finally, to give the job a name, double-click the title at the top left and enter an appropriate name. For example, `Workflow1`.
# MAGIC 12. When you are ready to run the job, click **Run Now** at the top right. A note will appear at the top right indicating a run has been initiated.
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Note</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Please make sure the Lakeflow Pipelines Editor is turned <strong>ON</strong>.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### D4. Inspect Your Run
# MAGIC
# MAGIC After your job completes its run, you can inspect it:
# MAGIC
# MAGIC 1. Select **Jobs & Pipelines** again on the sidebar menu.
# MAGIC 2. Click on the name of the job you just created.
# MAGIC 3. There are a few ways to view the notebook you just ran. You can do any of the following and it will take you to the same location:
# MAGIC     - Click **go to the latest successful run**.
# MAGIC     - Click the date under **Start time**.
# MAGIC     - Click the green bar (meaning a successful run) displayed within the diagram. If the bar is red, that means your job failed, but you can still inspect the notebook that ran. This option will also display the workflow configuration you set up previously.
# MAGIC 4. This will take you to a static copy of the notebook. You cannot edit it, but you can view the outputs of each cell. Confirm that all cells have run successfully.
# MAGIC
# MAGIC Of course, this is a simple example of creating a job with a single notebook. In practice, we use the various options mentioned earlier to tie together complex pipelines.

# COMMAND ----------

# MAGIC %md
# MAGIC ## E. Conclusion
# MAGIC
# MAGIC In this demo, you learned how to use **Databricks Model Serving** for real-time inference with a registered model. You also explored **Databricks Lakeflow Jobs** as a way to schedule and automate ML jobs.
# MAGIC
# MAGIC Key takeaways:
# MAGIC - Databricks Model Serving turns a registered model into a real-time inference endpoint from the UI or the API
# MAGIC - Endpoints can be served in four modes: batch, streaming, real-time, and edge
# MAGIC - Enabling **Scale to zero** saves cost but can add latency, so it is not recommended for production
# MAGIC - Databricks Lakeflow Jobs provide a fully managed way to orchestrate and automate ML tasks across clouds

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>