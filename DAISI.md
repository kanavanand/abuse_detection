# Multilingual-Indian Language-Abuse-detection

[![N|Solid](https://media-exp1.licdn.com/dms/image/C560BAQGwSoU6jT3zHA/company-logo_200_200/0/1649031376821?e=2147483647&v=beta&t=wwuPa26cNdV3NmRlxrmOL62Tg5AGEvgpu64-r6jfTbM)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

The Multilingual-Indian Language-Abuse-detection daisi will help you to detect any abuse happening in different languages. The model is trained on millions of rows having abusive content across 15 different languages.  Below are the languages supported 
- English
- Hindi
- Punjabi
- Telugu
- Marathi 
- Tamil 
- Malayalam 
- Bengali
- Kannada 
- Odia
- Gujarati
- Haryanvi
- Bhojpuri
- Rajasthani
- Assamese

## About Training data

Model is trained on the millions of data points gathered from twitter and then we run a language detection algorithm to get teh languages. These rows were further annotated manually where 1 stands for abuse and 0 stands for not abuse. 
The distribution of target variables is 
- Abusive rows 92%
- Non Abusive rows 8%


### Training Model

Steps used for training text models
- Preprocessing text
- Converting it into vectorizer using TFIDF vectorization
- Training LGB model 


Model gives 91% roc and latency is ~10 ms