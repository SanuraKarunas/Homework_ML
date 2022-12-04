import os
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import json
import csv

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
df_train.drop_duplicates(subset=df_train.columns.difference(['selling_price']), keep='first', inplace=True)
df_train.reset_index(drop=True, inplace=True)
df_train['engine'] = df_train['engine'].str.replace('CC', '')
df_train['max_power'] = df_train['max_power'].str.replace('bhp', '')
df_train['mileage'] = np.where(df_train['mileage'].str.contains('km/kg'),
                               pd.to_numeric(df_train['mileage'].str.replace('km/kg', ''), errors='coerce') * 1.4,
                               pd.to_numeric(df_train['mileage'].str.replace('kmpl', ''), errors='coerce'))

df_train['engine'] = pd.to_numeric(df_train['engine'], errors='coerce')
df_train['max_power'] = pd.to_numeric(df_train['max_power'], errors='coerce')
df_train.drop('torque', axis=1, inplace=True)

cols = ['mileage', 'engine', 'max_power', 'seats']
for col in cols:
    m = df_train[col].mean()
    df_train[col].fillna(m, inplace=True)

df_train['engine'] = df_train['engine'].astype(int)
df_train['seats'] = df_train['seats'].astype(int)

y_train = df_train['selling_price']
X_train_cat = df_train.drop(['selling_price', 'name'], axis=1)
X_train_cat['seats'] = X_train_cat['seats'].astype('object')
X_dum = pd.get_dummies(X_train_cat, drop_first=True)

gs_dum_ridge = GridSearchCV(estimator=Ridge(), param_grid={"alpha": np.arange(0.01, 10.01, 0.5)}, scoring='r2', cv=10)


def csv_to_items(x):
    file = open(x, "r")
    data = list(csv.DictReader(file, delimiter=","))
    file.close()

    return data


def similar_columns(a, b):
    b = b.drop(['selling_price', 'name'], axis=1)

    b['seats'] = b['seats'].astype('object')
    b = pd.get_dummies(b, drop_first=True)

    missing_cols = set(a.columns) - set(b.columns)
    for c in missing_cols:
        b[c] = 0
    missing_cols2 = set(b.columns) - set(a.columns)
    for c in missing_cols2:
        b.drop(labels=c, axis=1)
    b = b.reindex(columns=a.columns, fill_value=0)

    return b


def trained_model(wow, y):
    gs_dum_ridge.fit(wow, y)
    model_dum_ridge = gs_dum_ridge.best_estimator_
    model_dum_ridge.fit(wow, y)

    return model_dum_ridge


trained = trained_model(X_dum, y_train)


def y_format(a, b):
    a['engine'] = a['engine'].str.replace('CC', '')
    a['max_power'] = a['max_power'].str.replace('bhp', '')
    a['mileage'] = np.where(a['mileage'].str.contains('km/kg'),
                            pd.to_numeric(a['mileage'].str.replace('km/kg', ''), errors='coerce') * 1.4,
                            pd.to_numeric(a['mileage'].str.replace('kmpl', ''), errors='coerce'))

    a['engine'] = pd.to_numeric(a['engine'], errors='coerce')
    a['max_power'] = pd.to_numeric(a['max_power'], errors='coerce')
    a.drop('torque', axis=1, inplace=True)

    for col in ['mileage', 'engine', 'max_power', 'seats']:
        m = b[col].mean()
        a[col].fillna(m, inplace=True)

    a['engine'] = pd.to_numeric(a['engine'], downcast='integer')
    a['seats'] = pd.to_numeric(a['seats'], downcast='integer')
    a['year'] = pd.to_numeric(a['year'], downcast='integer')
    a['selling_price'] = pd.to_numeric(a['selling_price'], downcast='integer')
    a['km_driven'] = pd.to_numeric(a['km_driven'], downcast='integer')

    return a


class Items(BaseModel):
    objects: List[Item]


def predict_item(item: Item) -> float:
    item_pd = pd.DataFrame(item, index=[0])
    item_pd = y_format(item_pd, df_train)
    wow_predict = similar_columns(X_dum, item_pd)
    prediction = trained.predict(wow_predict)

    return prediction[0]


def csv_to_items(CSV_file: UploadFile = File(...)) -> Items:
    file = open(CSV_file, "r")
    data = list(csv.DictReader(file, delimiter=","))
    file.close()

    return data


def predict_items(items: Items):
    items_pd1 = pd.DataFrame.from_records(items)
    items_pd = items_pd1.copy()
    items_pd = y_format(items_pd, df_train)

    wow_items = similar_columns(X_dum, items_pd)
    prediction = trained.predict(wow_items)
    items_pd1['prediction'] = prediction

    return items_pd1


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})


@app.post("/predict_item")
async def predict_JSON(JSON_file: UploadFile = File(...)):
    data = json.load(JSON_file.file)

    return {'result': predict_item(data)}


@app.post("/predict_items")
async def predict_CSV(name: str = Form(...), CSV_file: UploadFile = File(...)):
    file_name = '_'.join(name.split()) + '.csv'
    save_path = f'static/predicted/{file_name}'

    with open(save_path, "wb+") as file_object:
        file_object.write(CSV_file.file.read())

    items = csv_to_items(save_path)
    items_predicted = predict_items(items)

    items_predicted.to_csv(save_path, encoding='utf-8')

    return {'result': f'На странице /predicted ждет файличек {file_name}'}


@app.get("/predicted")
async def get_lib(request: Request):
    predictions = os.listdir('static/predicted')
    return templates.TemplateResponse('predicted.html',
                                      {"request": request,
                                       "predictions": predictions})
