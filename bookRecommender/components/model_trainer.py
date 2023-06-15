import os,sys 

from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity


from bookRecommender import utils
from bookRecommender.logger import logging
from bookRecommender.exception import BookRecommenderException
from bookRecommender.entity import artifact_entity,config_entity


class ModelTrainer:

    def __init__(self, model_trainer_config:config_entity.ModelTrainerConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact

        except Exception as e:
            raise BookRecommenderException(e, sys)

    def train_model(self,arr):
        """
        Model training
        """
        try:
            similarity_scores =  cosine_similarity(arr)
            return similarity_scores

        except Exception as e:
            raise BookRecommenderException(e, sys)

    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        """
        Preparing dataset
        """
        try:
            logging.info("Loading train and test array.")
            books_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_pivot_table_file_path)

            # Finding the distance of each point to other --> Similarity Score
            logging.info("Finding the distance of each point to other --> Similarity Score")
            similarity_scores = cosine_similarity(books_arr)
            print(similarity_scores)
            
            # Saving trained model if it passes using utils
            logging.info("Saving mode object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=similarity_scores)

            # Prepare artifact
            logging.info("Prepare the artifact")
            model_trainer_artifact  = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
            
        except Exception as e:
            raise BookRecommenderException(e, sys)

