import os, sys
from datetime import datetime
from bookRecommender.logger import logging
from bookRecommender.exception import BookRecommenderException

BOOKS_FILE_NAME = 'book.csv'
USERS_FILE_NAME = 'user.csv'
RATINGS_FILE_NAME = 'ratings.csv'
POPULAR_FILE_NAME = "popular.csv"
MODEL_FILE_NAME = "model.pkl"


class TrainingPipelineConfig:
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")

        except Exception as e:
            raise BookRecommenderException(e, sys)

class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        try:
            self.database_name="bookRecommender"
            self.books_collection_name="books"
            self.users_collection_name="users"
            self.ratings_collection_name="ratings"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir , "data_ingestion")
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir,"feature_store")
            self.books_file_path = os.path.join(self.feature_store_file_path, BOOKS_FILE_NAME)
            self.users_file_path = os.path.join(self.feature_store_file_path, USERS_FILE_NAME)
            self.ratings_file_path = os.path.join(self.feature_store_file_path, RATINGS_FILE_NAME)

        except Exception as e:
            raise BookRecommenderException(e, sys)

    def to_dict(self,)->dict:
        """
        To convert and return the output as dict : data_ingestion_config
        """ 
        try:
            return self.__dict__

        except Exception  as e:
            raise BookRecommenderException(e,sys) 

class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir , "data_validation")
            self.report_file_path=os.path.join(self.data_validation_dir, "report.yaml")
            self.books_file_path = os.path.join(self.data_validation_dir,"dataset",BOOKS_FILE_NAME)
            self.users_file_path = os.path.join(self.data_validation_dir,"dataset",USERS_FILE_NAME)
            self.ratings_file_path = os.path.join(self.data_validation_dir,"dataset",RATINGS_FILE_NAME)
            self.missing_threshold:float = 0.2
            self.books_base_file_path = os.path.join("books.csv")
            self.users_base_file_path = os.path.join("users.csv")
            self.ratings_base_file_path = os.path.join("ratings.csv")

        except Exception as e:
            raise BookRecommenderException(e, sys)

class DataTransformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir , "data_transformation")
            self.transformed_pivot_table_file_path =  os.path.join(self.data_transformation_dir,"transformed",BOOKS_FILE_NAME.replace("csv","npz"))
            self.popular_data_file_path = os.path.join(self.data_transformation_dir,"transformed",POPULAR_FILE_NAME)

        except Exception as e:
            raise BookRecommenderException(e, sys)

class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir , "model_trainer")
            self.model_path = os.path.join(self.model_trainer_dir,"model",MODEL_FILE_NAME)

        except Exception as e:
            raise BookRecommenderException(e, sys)

class ModelEvaluationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.change_threshold = 0.01


class ModelPusherConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.model_pusher_dir = os.path.join(training_pipeline_config.artifact_dir , "model_pusher")
            # Saving models outside of artifact dir to save model in each run
            self.saved_model_dir = os.path.join("saved_models")
            self.pusher_model_dir = os.path.join(self.model_pusher_dir,"saved_models")
            self.pusher_model_path = os.path.join(self.pusher_model_dir,MODEL_FILE_NAME)
            self.pusher_target_encoder_path = os.path.join(self.pusher_model_dir,TARGET_ENCODER_OBJECT_FILE_NAME)
            self.knn_imputer_object_path = os.path.join(self.pusher_model_dir,KNN_IMPUTER_OBJECT_FILE_NAME)

        except Exception as e:
            raise BookRecommenderException(e, sys)