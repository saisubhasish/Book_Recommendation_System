import os,sys 
import numpy as np
import pandas as pd
from bookRecommender import utils
from typing import Optional
from bookRecommender.logger import logging
from bookRecommender.exception import BookRecommenderException
from bookRecommender.entity import artifact_entity,config_entity



class DataValidation:


    def __init__(self,
                    data_validation_config:config_entity.DataValidationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.validation_error=dict()
        except Exception as e:
            raise BookRecommenderException(e, sys)

    

    def drop_missing_values_columns(self,df:pd.DataFrame,report_key_name:str)->Optional[pd.DataFrame]:
        """
        This function will drop column which contains missing value more than specified threshold

        df : Accepts a pandas dataframe
        =========================================================================================
        returns Pandas Dataframe if atleast a single column is available after missing columns drop else None
        """
        try:
            
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isna().sum()/df.shape[0]
            # Selecting column name which contains null
            logging.info(f"selecting column name which contains null above to {threshold}")
            drop_column_names = null_report[null_report>threshold].index

            logging.info(f"Columns to drop: {list(drop_column_names)}")
            self.validation_error[report_key_name]=list(drop_column_names)
            df.drop(list(drop_column_names),axis=1,inplace=True)

            # Return None if no columns left
            if len(df.columns)==0:
                return None
            return df
        except Exception as e:
            raise BookRecommenderException(e, sys)

    def is_required_columns_exists(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str)->bool:
        """
        This function checks if required columns exists or not by comparing current df with base df and returns
        output as True and False
        """
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns = []
            for base_column in base_columns:    
                if base_column not in current_columns:
                    logging.info(f"Column: [{base_column} is not available.]")
                    missing_columns.append(base_column)

            # Return False if there are missing columns in current df other wise True
            if len(missing_columns)>0:
                self.validation_error[report_key_name]=missing_columns
                return False    
            return True
            
        except Exception as e:
            raise BookRecommenderException(e, sys)

    def data_drift(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str):
        try:
            drift_report=dict()

            base_columns = base_df.columns
            current_columns = current_df.columns

            for base_column in base_columns:
                base_data,current_data = base_df[base_column],current_df[base_column]
                # Null hypothesis : Both column data has same distribution
                
                logging.info(f"Checking Data Types of '{base_column}': {base_data.dtype}, {current_data.dtype} ")
                
                if base_df[base_column].dtype == current_df[base_column].dtype:
                    drift_report[base_column] = {"Same data type": True}
                else:
                    drift_report[base_column] = {"Same data type": False}

                logging.info(f"Checking number of classes in in {base_column} column\n: {base_df[base_column].value_counts(), current_df[base_column].value_counts()}")
                if len(base_df[base_column].value_counts()) == len(current_df[base_column].value_counts()):
                    drift_report[base_column] = {"Column has equal number of classes": True}
                else:
                    drift_report[base_column] = {"Column has equal number of classes": False} 

            self.validation_error[report_key_name]=drift_report
            
        except Exception as e:
            raise BookRecommenderException(e, sys)

    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact:
        try:
            logging.info("Reading base dataframe")
            books_base_df = pd.read_csv(self.data_validation_config.books_base_file_path, sep=';', encoding='latin-1', error_bad_lines = False)
            users_base_df = pd.read_csv(self.data_validation_config.users_base_file_path, sep=';', encoding='latin-1', error_bad_lines = False)
            ratings_base_df = pd.read_csv(self.data_validation_config.ratings_base_file_path, sep=';', encoding='latin-1', error_bad_lines = False)

            logging.info("Drop null values colums from base df")
            books_base_df=self.drop_missing_values_columns(df=books_base_df,report_key_name="missing_values_within_books_base_dataset")
            users_base_df=self.drop_missing_values_columns(df=users_base_df,report_key_name="missing_values_within_users_base_dataset")
            ratings_base_df=self.drop_missing_values_columns(df=ratings_base_df,report_key_name="missing_values_within_ratings_base_dataset")

            logging.info("Reading books dataframe")
            books_df = pd.read_csv(self.data_ingestion_artifact.books_file_path)
            logging.info("Reading users dataframe")
            users_df = pd.read_csv(self.data_ingestion_artifact.users_file_path)
            logging.info("Reading ratings dataframe")
            ratings_df = pd.read_csv(self.data_ingestion_artifact.ratings_file_path)

            logging.info("Drop null values colums from books df")
            books_df = self.drop_missing_values_columns(df=books_df,report_key_name="missing_values_within_train_dataset")
            logging.info("Drop null values colums from users df")
            users_df = self.drop_missing_values_columns(df=users_df,report_key_name="missing_values_within_test_dataset")
            logging.info("Drop null values colums from ratings df")
            ratings_df = self.drop_missing_values_columns(df=ratings_df,report_key_name="missing_values_within_test_dataset")

            logging.info("Is all required columns present in train df")
            books_df_columns_status = self.is_required_columns_exists(base_df=books_base_df, current_df=books_df,report_key_name="missing_columns_within_books_dataset")
            logging.info("Is all required columns present in test df")
            users_df_columns_status = self.is_required_columns_exists(base_df=users_base_df, current_df=users_df,report_key_name="missing_columns_within_users_dataset")
            logging.info("Is all required columns present in test df")
            ratings_df_columns_status = self.is_required_columns_exists(base_df=ratings_base_df, current_df=ratings_df,report_key_name="missing_columns_within_ratings_dataset")

            if books_df_columns_status:     # If True
                logging.info("As all column are available in books df hence detecting data drift in books dataframe")
                self.data_drift(base_df=books_base_df, current_df=books_df,report_key_name="data_drift_within_books_dataset")
            if users_df_columns_status:     # If True
                logging.info("As all column are available in users df hence detecting data drift users dataframe")
                self.data_drift(base_df=users_base_df, current_df=users_df,report_key_name="data_drift_within_users_dataset")
            if ratings_df_columns_status:     # If True
                logging.info("As all column are available in ratings df hence detecting data drift ratings dataframe")
                self.data_drift(base_df=ratings_base_df, current_df=ratings_df,report_key_name="data_drift_within_ratings_dataset")


            logging.info("create dataset directory folder if not available for validated train file and test file")
            # Create dataset directory folder if not available
            dataset_dir = os.path.dirname(self.data_validation_config.books_file_path)
            os.makedirs(dataset_dir,exist_ok=True)

            logging.info("Saving validated train df and test df to dataset folder")
            # Saving validated train df and test df to dataset folder
            books_df.to_csv(path_or_buf=self.data_validation_config.books_file_path,index=False,header=True)
            users_df.to_csv(path_or_buf=self.data_validation_config.users_file_path,index=False,header=True)
            ratings_df.to_csv(path_or_buf=self.data_validation_config.ratings_file_path,index=False,header=True)    

            # Write the report
            logging.info("Writing report in yaml file")
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path,
            data=self.validation_error)   # valiadtion_error: drop columns, missing columns, drift report

            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path, 
            books_file_path=self.data_validation_config.books_file_path, users_file_path=self.data_validation_config.users_file_path, 
            ratings_file_path=self.data_validation_config.ratings_file_path)
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise BookRecommenderException(e, sys)