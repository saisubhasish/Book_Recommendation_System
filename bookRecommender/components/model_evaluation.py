import os, sys
import pandas as pd
import numpy as np

from bookRecommender import utils
from bookRecommender.logger import logging
from bookRecommender.utils import load_object
from bookRecommender.predictor import ModelResolver
from bookRecommender.exception import BookRecommenderException
from bookRecommender.entity import config_entity, artifact_entity
from bookRecommender.components.data_transformation import DataTransformation
 

class ModelEvaluation:

    def __init__(self,
        model_eval_config:config_entity.ModelEvaluationConfig,
        data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
        data_transformation_artifact:artifact_entity.DataTransformationArtifact,
        model_trainer_artifact:artifact_entity.ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20}  Model Evaluation {'<<'*20}")
            logging.info("___________________________________________________________________________________________________________")
            self.model_eval_config=model_eval_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.data_transformation= DataTransformation(data_transformation_config=config_entity.DataTransformationConfig, data_validation_artifact=artifact_entity.DataValidationArtifact)
            self.model_resolver = ModelResolver()

        except Exception as e:
            raise BookRecommenderException(e, sys)


    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            # If saved model folder has model then we will compare which model is best
            # Trained model from artifact folder or the model from saved model folder
            logging.info("___________________________________________________________________________________________________________")
            logging.info("If saved model folder has model then we will compare which model is best, "
            "Trained model from artifact folder or the model from saved model folder")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path==None:                                 # If there is no saved_models then we will accept the currnt model
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                improved_accuracy=None)                                           
                logging.info(f"Model evaluation artifact: {model_eval_artifact}")
                return model_eval_artifact

            logging.info(f"Accepting the current model")
            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
            improved_accuracy=None)
            
            # Improved accuracy
            logging.info(f"Model eval artifact: {model_eval_artifact}")
            return model_eval_artifact
            
        except Exception as e:
            raise BookRecommenderException(e,sys)
