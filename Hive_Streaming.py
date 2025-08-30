# Install dependencies before running:
# !apt-get install openjdk-11 -y
# !pip install findspark pyspark paho-mqtt pandas joblib faker azure-storage-blob

import os
import findspark                                                                                                                                             # type: ignore
findspark.init()

import json
import random
import threading
import time
from queue import Queue
from datetime import datetime

import pandas as pd

import paho.mqtt.client as mqtt                                                                                                                                              # type: ignore
from faker import Faker                                                                                                                                                    # type: ignore

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MQTT_Spark_Streaming")

from pyspark.sql import SparkSession, Row                                                                                                                                                    # type: ignore
from pyspark.sql.types import *                                                                                                                                                    # type: ignore
from pyspark.sql.functions import udf                                                                                                                                                    # type: ignore
from pyspark.sql.types import StringType                                                                                                                                                    # type: ignore

from azure.storage.blob import BlobServiceClient                                                                                                                                                    # type: ignore

import joblib

# ---------------------------
# Setup Spark Session
# ---------------------------
os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ['SPARK_HOME'] = "/content/spark-3.5.1-bin-hadoop3"

spark = SparkSession.builder \
    .appName("HiveMQ_IoT_ML_Streaming") \
    .getOrCreate()

# ---------------------------
# MQTT Config & Queues
# ---------------------------
BROKER = "863b2bea6a5246238f2ae57eac2dc400.s1.eu.hivemq.cloud"
PORT = 8883
USERNAME = "Bharathnath"
PASSWORD = "#Bharath123"
TOPIC = "iot/failure"

fake = Faker()
iot_queue = Queue()

DeviceType_list = ["Anesthesia Machine","CT Scanner","Defibrillator","Dialysis Machine",
                  "ECG Monitor","Infusion Pump","Patient Ventilator","Ultrasound Machine"]
DeviceName_list = ["Alaris GH","Baxter AK 96","Baxter Flo-Gard","Datex Ohmeda S5","Drager Fabius Trio",
                  "Drager V500","Fresenius 4008","GE Aisys","GE Logiq E9","GE MAC 2000","GE Revolution",
                  "Hamilton G5","HeartStart FRx","Lifepak 20","NxStage System One","Philips EPIQ",
                  "Philips HeartStrart","Philips Ingenuity","Phillips PageWriter","Puritan Bennett 980",
                  "Siemens Acuson","Siemens S2000","Smiths Medfusion","Zoll R Series"]
ClimateControl_list = ["Yes","No"]
Location_list = [
    "Hospital A - Central Region","Hospital A - East Region","Hospital A - North Region","Hospital A - South Region","Hospital A - West Region",
    "Hospital B - Central Region","Hospital B - East Region","Hospital B - North Region","Hospital B - South Region","Hospital B - West Region",
    "Hospital C - Central Region","Hospital C - East Region","Hospital C - North Region","Hospital C - South Region","Hospital C - West Region",
    "Hospital D - Central Region","Hospital D - East Region","Hospital D - North Region","Hospital D - South Region","Hospital D - West Region",
    "Hospital E - Central Region","Hospital E - East Region","Hospital E - North Region","Hospital E - South Region","Hospital E - West Region",
    "Hospital F - Central Region","Hospital F - East Region","Hospital F - North Region","Hospital F - South Region","Hospital F - West Region",
    "Hospital G - Central Region","Hospital G - East Region","Hospital G - North Region","Hospital G - South Region","Hospital G - West Region",
    "Hospital H - Central Region","Hospital H - East Region","Hospital H - North Region","Hospital H - South Region","Hospital H - West Region"
]

# Azure Blob Storage connection info (from you)
AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=livedataset;AccountKey=BTzGWwYZckqB+3BCgI7mUiS2t7QDbGxUNWqXuU2Ifih1YaCZdUjGz2J8kKUcRUzk1ShyAQ8p36Ab+ASt4quwiQ==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "data"
LOCAL_CSV_FILENAME = "predictions_output.csv"

blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
blob_container_client = blob_service_client.get_container_client(CONTAINER_NAME)


# Load your ML pipeline
MODEL_PATH = "/content/xgboost_pipeline.pkl"
ml_pipeline = joblib.load(MODEL_PATH)
logger.info(f"Loaded ML pipeline from {MODEL_PATH}")

# ---------------------------
# MQTT Publisher (simulate 100 records and then stop)
# ---------------------------
def publish_simulated():
    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()
    client.connect(BROKER, PORT)
    client.loop_start()
    count = 0
    max_records = 100
    while count < max_records:
        record = {
            "DeviceType": random.choice(DeviceType_list),
            "DeviceName": random.choice(DeviceName_list),
            "RuntimeHours": round(random.uniform(102.32, 9999.85),2),
            "TemperatureC": round(random.uniform(16.07, 40),2),
            "PressureKPa": round(random.uniform(90,120),2),
            "VibrationMM_S": round(random.uniform(0,1),3),
            "CurrentDrawA": round(random.uniform(0.1,1.5),3),
            "SignalNoiseLevel": round(random.uniform(0,5),2),
            "ClimateControl": random.choice(ClimateControl_list),
            "HumidityPercent": round(random.uniform(20,70),2),
            "Location": random.choice(Location_list),
            "OperationalCycles": random.randint(5,11887),
            "UserInteractionsPerDay": round(random.uniform(0,26.4),2),
            "LastServiceDate": fake.date_between(start_date="-2y", end_date="today").strftime("%d-%m-%Y"),
            "ApproxDeviceAgeYears": round(random.uniform(0.1,35.89),2),
            "NumRepairs": random.randint(0,19),
            "ErrorLogsCount": random.randint(0,22)
        }
        client.publish(TOPIC, json.dumps(record))
        count +=1
        time.sleep(0.1)
    client.loop_stop()
    client.disconnect()
    logger.info("Finished publishing 100 simulated records")

# ---------------------------
# MQTT subscriber to push messages into in-memory queue
# ---------------------------
def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode())
    iot_queue.put(data)

def mqtt_subscribe():
    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()
    client.on_message = on_message
    client.connect(BROKER, PORT)
    client.subscribe(TOPIC)
    client.loop_forever()

# ---------------------------
# Spark Schema with your data columns
# ---------------------------
schema = StructType([ # type: ignore
    StructField("DeviceType", StringType(), True),                                                                                                                                                   # type: ignore
    StructField("DeviceName", StringType(), True),                                                                                                                                                     # type: ignore
    StructField("RuntimeHours", DoubleType(), True),                                                                                                                                                    # type: ignore# type: ignore
    StructField("TemperatureC", DoubleType(), True),                                                                                                                                                    # type: ignore
    StructField("PressureKPa", DoubleType(), True),                                                                                                                                                    # type: ignore
    StructField("VibrationMM_S", DoubleType(), True),                                                                                                                                                    # type: ignore
    StructField("CurrentDrawA", DoubleType(), True),                                                                                                                                                    # type: ignore
    StructField("SignalNoiseLevel", DoubleType(), True),                                                                                                                                                    # type: ignore
    StructField("ClimateControl", StringType(), True),                                                                                                                                                    # type: ignore
    StructField("HumidityPercent", DoubleType(), True),                                                                                                                                                    # type: ignore
    StructField("Location", StringType(), True),                                                                                                                                                    # type: ignore
    StructField("OperationalCycles", IntegerType(), True),                                                                                                                                                    # type: ignore
    StructField("UserInteractionsPerDay", DoubleType(), True),                                                                                                                                                    # type: ignore
    StructField("LastServiceDate", StringType(), True),                                                                                                                                                    # type: ignore
    StructField("ApproxDeviceAgeYears", DoubleType(), True),                                                                                                                                                    # type: ignore
    StructField("NumRepairs", IntegerType(), True),                                                                                                                                                     # type: ignore# type: ignore# type: ignore
    StructField("ErrorLogsCount", IntegerType(), True)                                                                                                                                                    # type: ignore
])

# ---------------------------
# UDF to apply sklearn pipeline prediction
# ---------------------------
def predict_udf(*cols):
    try:
        features = pd.DataFrame([cols], columns=[f.name for f in schema])
        prediction = ml_pipeline.predict(features)[0]
        return str(prediction)
    except Exception as e:
        return "Error"

predict = udf(predict_udf, StringType())

# ---------------------------
# Save batch records to CSV and upload to Azure Blob Storage
# ---------------------------
def save_and_upload(records, filename=LOCAL_CSV_FILENAME):
    import csv

    if not records:
        return

    file_exists = os.path.exists(filename)
    keys = records[0].keys()

    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerows(records)

    blob_name = f"predictions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, "rb") as data:
        blob_container_client.upload_blob(blob_name, data, overwrite=True)

    logger.info(f"Uploaded file '{blob_name}' to Azure Blob Storage.")

# ---------------------------
# Threads for simultaed publishing and subscribing
# ---------------------------
threading.Thread(target=publish_simulated, daemon=True).start()
threading.Thread(target=mqtt_subscribe, daemon=True).start()

# ---------------------------
# Process streaming queue and predict using Spark & ML pipeline
# ---------------------------
batch_size = 100
batch_records = []

while True:
    if not iot_queue.empty():
        data = iot_queue.get()
        row = Row(**data)
        df = spark.createDataFrame([row], schema)
        df = df.withColumn("PredictedFailureRisk", predict(*df.columns))
        results = df.collect()
        for r in results:
            rec = r.asDict()
            print(rec)
            batch_records.append(rec)
        if len(batch_records) >= batch_size:
            save_and_upload(batch_records)
            batch_records.clear()
