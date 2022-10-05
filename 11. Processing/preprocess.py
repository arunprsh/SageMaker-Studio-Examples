from pyspark.ml.feature import StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler
from pyspark.sql.types import StructField, StructType, StringType, DoubleType, IntegerType
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

import argparse
import time
import sys
import os

# https://spark.apache.org/docs/latest/ml-features.html


def main():
    # --------------------------------- CONSTRUCT ---------------------------------
    spark = SparkSession.builder.appName("PySparkJob").getOrCreate()
    
    parser = argparse.ArgumentParser(description="app inputs and outputs")
    parser.add_argument("--s3_input_bucket", type=str, help="s3 input bucket")
    parser.add_argument("--s3_input_key_prefix", type=str, help="s3 input key prefix")
    parser.add_argument("--s3_output_bucket", type=str, help="s3 output bucket")
    parser.add_argument("--s3_output_key_prefix", type=str, help="s3 output key prefix")
    args = parser.parse_args()
    
    # This is needed to save RDDs which is the only way to write nested Dataframes into CSV format
    spark.sparkContext._jsc.hadoopConfiguration().set("mapred.output.committer.class",
                                                      "org.apache.hadoop.mapred.FileOutputCommitter")
    # --------------------------------- CONSTRUCT ---------------------------------
    
    schema = StructType([StructField('id', IntegerType(), True), 
                     StructField('name', StringType(), True),
                     StructField('age', IntegerType(), True),
                     StructField('sex', StringType(), True),
                     StructField('weight', DoubleType(), True),
                     StructField('eye_color', StringType(), True)
                    ])
    
    # Downloading the data from S3 into a Dataframe
    # without header (IMPORTANT)
    df = spark.read.csv(('s3a://' + os.path.join(args.s3_input_bucket, 
                                                 args.s3_input_key_prefix,
                                                 'raw.csv')), header=False, schema=schema)
    sex_indexer = StringIndexer(inputCol='sex', outputCol='indexed_sex')
    sex_encoder = OneHotEncoder(inputCol='indexed_sex', outputCol='sex_vector')
    eye_color_indexer = StringIndexer(inputCol='eye_color', outputCol='indexed_eye_color')
    eye_color_encoder = OneHotEncoder(inputCol='indexed_eye_color', outputCol='eye_color_vector')
    assembler = VectorAssembler(inputCols=['age',
                                           'weight',
                                           'sex_vector',
                                           'eye_color_vector'], 
                                outputCol='features')
    scaler = MinMaxScaler(inputCol='features', outputCol='scaledFeatures')
    
    # The pipeline comprises of the steps added above
    pipeline = Pipeline(stages=[sex_indexer, sex_encoder, eye_color_indexer, eye_color_encoder, assembler, scaler])

    # This step trains the feature transformers
    model = pipeline.fit(df)

    # This step transforms the dataset with information obtained from the previous fit
    df = model.transform(df)
    
    age_udf = udf(lambda x: x[0].item(), DoubleType())
    weight_udf = udf(lambda x: x[1].item(), DoubleType())
    sex_udf = udf(lambda x: x[2].item(), DoubleType())
    blue_eye_udf = udf(lambda x: x[3].item(), DoubleType())
    black_eye_udf = udf(lambda x: x[3].item(), DoubleType())
    
    df = df.select(age_udf('scaledFeatures').alias('age'), 
               weight_udf('scaledFeatures').alias('weight'),
               sex_udf('scaledFeatures').alias('sex'),
               blue_eye_udf('scaledFeatures').alias('is_blue_eye'),
               black_eye_udf('scaledFeatures').alias('is_black_eye'),
              )
    df.show()
    
    df.write.format('csv') \
        .option('header', True) \
        .mode('overwrite') \
        .option('sep', ',') \
        .save('s3a://' + os.path.join(args.s3_output_bucket, args.s3_output_key_prefix))
    
    
if __name__ == '__main__':
    main()
