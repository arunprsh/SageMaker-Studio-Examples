
from sklearn.neighbors import KNeighborsClassifier
from os.path import join
from io import BytesIO
import pandas as pd
import numpy as np
import argparse
import logging
import pickle
import time
import json
import sys
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

if 'SAGEMAKER_METRICS_DIRECTORY' in os.environ:
    log_file_handler = logging.FileHandler(join(os.environ['SAGEMAKER_METRICS_DIRECTORY'], "metrics.json"))
    logger.addHandler(log_file_handler)
   
    
def model_fn(model_dir):
    print('[-------------- INSIDE MODEL FN --------------]')
    print(f'MODEL DIR: {model_dir}')
    model = pickle.load(open(os.path.join(model_dir, 'model'), 'rb'))
    return model


def input_fn(request_body, request_content_type):
    print('[-------------- INSIDE INPUT FN --------------]')
    print(f'REQUEST BODY: {request_body}')
    print(f'REQUEST CONTENT TYPE: {request_content_type}')
    if request_content_type == 'application/x-npy':
        stream = BytesIO(request_body)
        return np.load(stream)
    else:
        raise ValueError('Content type must be application/x-npy')


def predict_fn(input_data, model):
    print('[-------------- INSIDE PREDICT FN --------------]')
    print(f'INPUT DATA: {input_data}')
    print(f'MODEL: {model}')
    X = input_data.reshape(1, -1)
    prediction = model.predict(X)
    return prediction


def output_fn(prediction, content_type):
    print('[-------------- INSIDE OUTPUT FN --------------]')
    print(f'PREDICTION: {prediction}')
    print(f'CONTENT TYPE: {content_type}')
    if content_type == 'application/x-npy':
        buffer = BytesIO()
        np.save(buffer, prediction)
        return buffer.getvalue()
    else:
        raise ValueError('Accept header must be application/x-npy')


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    # hyperparameters
    parser.add_argument('--nneighbors', type=int, default=5)
    args = parser.parse_args()
    
    # ------------------------- YOUR MODEL TRAINING LOGIC STARTS HERE -------------------------
    # Load data from the location specified by args.train (In this case, an S3 bucket)
    print("------- [STARTING TRAINING] -------")
    train_df = pd.read_csv(os.path.join(args.train, 'train.csv'), names=['class', 'mass', 'width', 'height', 'color_score'])
    train_df.head()
    X_train = train_df[['mass', 'width', 'height', 'color_score']]
    y_train = train_df['class']
    knn = KNeighborsClassifier(n_neighbors=args.nneighbors)
    knn.fit(X_train, y_train)
    # Save the trained Model inside the Container
    pickle.dump(knn, open(os.path.join(args.model_dir, 'model'), 'wb'))
    print("------- [TRAINING COMPLETE!] -------")
    
    print("------- [STARTING EVALUATION] -------")
    test_df = pd.read_csv(os.path.join(args.test, 'test.csv'), names=['class', 'mass', 'width', 'height', 'color_score'])
    X_test = train_df[['mass', 'width', 'height', 'color_score']]
    y_test = train_df['class']
    acc = knn.score(X_test, y_test)
    print('Accuracy = {:.4f}%'.format(acc * 100))
    logger.info('Test Accuracy: {:.4f}%;\n'.format(acc * 100))
    print("------- [EVALUATION DONE!] -------")

if __name__ == '__main__':
    train()
