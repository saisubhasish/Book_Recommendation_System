import os,sys
import pandas as pd 
import numpy as np

from bookRecommender import utils
from bookRecommender.logger import logging
from bookRecommender.exception import BookRecommenderException
from bookRecommender.entity import config_entity, artifact_entity



class DataIngestion:
    def __init__(self, data_ingestion_config:config_entity.DataIngestionConfig):
        '''
        Storing the input to a variable to use in pipeline
        '''
        try:
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise BookRecommenderException(e, sys)

    def initiate_data_ingestion(self)->artifact_entity.DataIngestionArtifact:
        """
        This function takes Input: Database name and collection name
        and returns output: feature store file, train file and test file
        """
        try:
            logging.info(f"Exporting collection data as pandas dataframe")
            # Exporting collection data as pandas dataframe
            books:pd.DataFrame  = pd.read_csv('D:/FSDS-iNeuron/10.Projects-DS/Book_Recommendation_System/books_data/books.csv', sep=';', encoding='latin-1', error_bad_lines = False)
            users:pd.DataFrame  = pd.read_csv('D:/FSDS-iNeuron/10.Projects-DS/Book_Recommendation_System/books_data/users.csv', sep=';', encoding='latin-1', error_bad_lines = False)
            ratings:pd.DataFrame  = pd.read_csv('D:/FSDS-iNeuron/10.Projects-DS/Book_Recommendation_System/books_data/ratings.csv', sep=';', encoding='latin-1', error_bad_lines = False)

            logging.info("Save data in feature store")
            # Save data in feature store
            logging.info("Create book folder if not available")
            #Create book folder if not available
            books_file_dir = os.path.dirname(self.data_ingestion_config.books_file_path)
            os.makedirs(books_file_dir,exist_ok=True)

            logging.info("Create user folder if not available")
            #Create user folder if not available
            users_file_dir = os.path.dirname(self.data_ingestion_config.users_file_path)
            os.makedirs(users_file_dir,exist_ok=True)

            logging.info("Create ratings folder if not available")
            #Create ratings folder if not available
            ratings_file_dir = os.path.dirname(self.data_ingestion_config.ratings_file_path)
            os.makedirs(ratings_file_dir,exist_ok=True)


            logging.info("Save book to feature store folder")
            # Save book to feature store folder
            books.to_csv(path_or_buf=self.data_ingestion_config.books_file_path,index=False,header=True)

            logging.info("Save users to feature store folder")
            # Save users to feature store folder
            users.to_csv(path_or_buf=self.data_ingestion_config.users_file_path,index=False,header=True)

            logging.info("Save ratings to feature store folder")
            # Save ratings to feature store folder
            ratings.to_csv(path_or_buf=self.data_ingestion_config.ratings_file_path,index=False,header=True)
            
            
            # Prepare artifact  

            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                books_file_path=self.data_ingestion_config.books_file_path,
                users_file_path=self.data_ingestion_config.users_file_path, 
                ratings_file_path=self.data_ingestion_config.ratings_file_path)

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact


        except Exception as e:
            raise BookRecommenderException(error_message=e, error_detail=sys)