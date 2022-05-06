import numpy as np
import random
import requests
import json
import time
from datetime import datetime

def generate_amount(nb_alea):
    if (nb_alea == 1):
        it = np.random.choice([0, 1], p=[0.7, 0.3])
        if (it == 1):    
            return np.random.normal(20,5)
        else:
            return np.random.normal(200,100)
    else :
        return np.random.normal(100,50)
    return

def generate_old_balance(nb_alea):
    if (nb_alea == 1):
        it = np.random.choice([0, 1], p=[0.5, 0.5])
        if (it == 1):    
            return np.random.normal(10000,5000)
        else:
            return np.random.normal(200000,100000)
    else :
        return np.random.normal(100000,50000)
    return

def generate_income(nb_alea):
    if (nb_alea == 1):
        it = np.random.choice([0, 1], p=[0.7, 0.3])
        if (it == 1):    
            return np.random.normal(300,800)
        else:
            return np.random.normal(2000,1000)
    else :
        return np.random.normal(2000,1000)
    return

def generate_credit(nb_alea):
    if (nb_alea == 1):
        return np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
    else :
        return np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
    return

def generate_marital_status(nb_alea):
    ms = [0, 1]
    if (nb_alea == 1):
        return np.random.choice(ms, p=[0.7, 0.3])
    else :
        return np.random.choice(ms, p=[0.5, 0.5])

def generate_children(nb_alea):
    nb_child = [0, 1, 2, 3, 4] #4 ou +
    if (nb_alea == 1):
        return np.random.choice(nb_child, p=[0.4, 0.1, 0.1, 0.25, 0.15])
    else :
        return np.random.choice(nb_child, p=[0.35, 0.30, 0.2, 0.1, 0.05])

def generate_month(nb_alea):
    month = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    if (nb_alea == 1):
        return np.random.choice(month, p=[0.11, 0.06, 0.06, 0.06, 0.06, 0.06, 0.11, 0.11, 0.06, 0.06, 0.12, 0.13])
    else :
        return np.random.choice(month)

def generate_day_of_week(nb_alea):
    week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    if (nb_alea == 1):
        return np.random.choice(week,p=[0.15, 0.10, 0.10, 0.10, 0.15, 0.2, 0.2])
    else :
        return np.random.choice(week,p=[0.14, 0.14, 0.14, 0.14, 0.14, 0.16, 0.14])
    

def generate_category_recipient(nb_alea):
    category_recipient = ["food_store", "restaurant", "dress_store", "tech_store", "big_purchase"]
    if (nb_alea == 1):
        return np.random.choice(category_recipient, p=[0.37, 0.07, 0.22, 0.22, 0.12])
    else :
        return np.random.choice(category_recipient, p=[0.45, 0.15, 0.2, 0.15, 0.05])
    
    
def generate_transaction_type(nb_alea): 
    if (nb_alea == 1):
        return np.random.choice(["online","onsite"],p=[0.42, 0.58])
    else :
        return np.random.choice(["online","onsite"],p=[0.3, 0.7])
    


def generate_transaction_method(nb_alea, transact_type):
    transact_method = ["phone", "card_with_contact", "card_without_contact", "other"]
    if ((nb_alea == 1) and (transact_type == "onsite")):
        return np.random.choice(transact_method, p=[0.07, 0.48, 0.38, 0.07])
    elif (transact_type == "onsite"):
        return np.random.choice(transact_method, p=[0.05, 0.55, 0.35, 0.05])
    else:
        return "internet_payement"


def generate_foreign_transaction(nb_alea):
    if (nb_alea == 1):
        if (random.random() < 0.2):
            return 1
        else:
            return 0
    else :
        if (random.random() < 0.1):
            return 1
        else:
            return 0


def generate_is_fraud(nb_alea):
    if (nb_alea == 1):
        return 1
    else :
        return 0
    
from datetime import datetime

# datetime object containing current date and time
def generate_date(): 
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")

#10K effectué

nb_row = 2
a = 1001000

for i in range(nb_row):
    
    json_dict = {}
    
    nb_alea = random.randint(1, 10)
    
    json_dict["id"] = i + a
    json_dict["date"] = generate_date()
    
    json_dict["amount"] = generate_amount(nb_alea)
    
    json_dict["old_balance"] = generate_old_balance(nb_alea)
    json_dict["income"] = generate_income(nb_alea)
    json_dict["credit"] = generate_credit(nb_alea)
    json_dict["marital_status"] = generate_marital_status(nb_alea)
    json_dict["children"] = generate_children(nb_alea)
    
    json_dict["month"] = generate_month(nb_alea)
    json_dict["day_of_week"] = generate_day_of_week(nb_alea)
    
    json_dict["category_recipient"] = generate_category_recipient(nb_alea)  
    
    json_dict["transaction_type"] = generate_transaction_type(nb_alea)
    json_dict["transaction_method"] = generate_transaction_method(nb_alea, json_dict["transaction_type"])
    
    json_dict["foreign_transaction"] = generate_foreign_transaction(nb_alea)

    #json_dict["isFraud"] = generate_is_fraud(nb_alea)
    
    print("C'est le JSON dict numéro : ", i)
    print(json_dict)
    
    #Obtain AAD Token
    url = "https://login.microsoftonline.com/253bb4e8-8e40-4739-9b26-c2d5d2ea2d04/oauth2/token"
    
    payload={'grant_type': 'client_credentials',
    'client_id': '16927c99-1ce9-460e-9c9c-e2b693a53315',
    'client_secret': 'wGw7Q~A3syAI5sP2UKH5e-zd.nvmVC6GKLsVk',
    'resource': 'https://eventhubs.azure.net'}
    
    headers = {
      'Content-Type': 'application/x-www-form-urlencoded',
      'Cookie': 'fpc=AtYw0mDu1vlPg3bhIy1YMobw1pLaAQAAAKONstkOAAAA; stsservicecookie=estsfd; x-ms-gateway-slice=estsfd'
    }
    
    response = requests.request("GET", url, headers=headers, data=payload)
    
    access_token = json.loads(response.text)["access_token"]
    
    #hub = "eventhubspark"
    hub = "realtimefdhub" 
    
    #Send data to event hub
    url = "https://ProjectSparkEventHub.servicebus.windows.net/" + hub + "/messages?partitionId=1"
    
    headers = {
      'Content-Type': 'application/atom+xml;type=entry;charset=utf-8',
      'Authorization': 'Bearer' + ' ' + access_token
    }
    
    response = requests.request("POST", url, headers=headers, data=str(json_dict))
    
    print("successfully transferred !!!")

    time.sleep(3)

    