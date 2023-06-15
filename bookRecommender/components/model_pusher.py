import os, sys
import pandas as pd
from bookRecommender.logger import logging
from bookRecommender.exception import BookRecommenderException
from bookRecommender.predictor import ModelResolver
from bookRecommender.entity.config_entity import ModelPusherConfig
from bookRecommender.utils import save_object, load_object, load_numpy_array_data
from bookRecommender.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ModelPusherArtifact, DataIngestionArtifact

class ModelPusher:

    def __init__(self,model_pusher_config:ModelPusherConfig,
        data_transformation_artifact:DataTransformationArtifact,
        data_ingestion_artifact:DataIngestionArtifact,
        model_trainer_artifact:ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20} Model Pusher {'<<'*20}")
            self.model_pusher_config=model_pusher_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)

        except Exception as e:
            raise BookRecommenderException(e, sys)

    def initiate_model_pusher(self)->ModelPusherArtifact:
        try:
            # Load object 
            logging.info("Loading model and files")
            similarity_score = load_object(file_path=self.model_trainer_artifact.model_path)
            books_file = self.data_ingestion_artifact.books_file_path
            books_df = pd.read_csv(books_file)
            popular_file = self.data_transformation_artifact.popular_data_file_path
            popular_df = pd.read_csv(popular_file)
            pivot_table = self.data_transformation_artifact.transformed_pivot_table_file_path
            pt = load_numpy_array_data(pivot_table)

            # Model pusher dir
            logging.info("Saving model into model pusher directory")
            save_object(file_path=self.model_pusher_config.books_path, obj=books_df)
            save_object(file_path=self.model_pusher_config.popular_df_path, obj=popular_df)
            save_object(file_path=self.model_pusher_config.pivot_table_path, obj=pt)
            save_object(file_path= self.model_pusher_config.similarity_score_model_path, obj=similarity_score)

            # Getting or fetching the directory location to save latest model in different directory in each run
            logging.info("Saving model in saved model dir")
            model_path = self.model_resolver.get_latest_save_model_path()
            books_df_path = self.model_resolver.get_latest_save_books_file_path()
            popular_df_path = self.model_resolver.get_latest_save_popular_df_path()
            pivot_table_path = self.model_resolver.get_latest_save_pivot_table_path()

            # Saved model dir outside artifact to use in prediction pipeline
            logging.info('Saving model outside of artifact directory')
            save_object(file_path=model_path, obj=similarity_score)
            save_object(file_path=books_df_path, obj=books_df)
            save_object(file_path=popular_df_path, obj=popular_df)
            save_object(file_path=pivot_table_path, obj=pt)

            model_pusher_artifact = ModelPusherArtifact(pusher_model_dir=self.model_pusher_config.pusher_model_dir, 
                                                        saved_model_dir=self.model_pusher_config.saved_model_dir)
            logging.info(f"Model pusher artifact : {model_pusher_artifact}")

            return model_pusher_artifact

        except Exception as e:
            raise BookRecommenderException(e, sys)
