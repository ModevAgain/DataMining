# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 10:14:37 2018

@author: Manuel
"""

DFevents = pd.read_sql_table("events", diskEngine, columns=["device_id", "event_id"])
DFbrands = pd.read_sql_table("phone_brand_device_model", diskEngine)
DFappevents = pd.read_sql_table("app_events", diskEngine)
DFapplabels = pd.read_sql_table("app_labels", diskEngine)
DFlabelcat = pd.read_sql_table("label_categories", diskEngine)
DFtrain = pd.read_sql_table("gender_age_train", diskEngine)

resultBrands = pd.merge(DFtrain, DFbrands, on='device_id')
resultEvents = pd.merge(resultBrands, DFevents, on='device_id')
resultAppEvents = pd.merge(resultEvents, DFappevents, on='event_id')
resultLabels = pd.merge(resultAppEvents, DFapplabels, on='app_id')
result = pd.merge(resultLabels, DFlabelcat, on='label_id')
print(result[['device_id','gender','age','agegroup','phone_brand','device_model','category']].head(100000))
