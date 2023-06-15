import sys
from bookRecommender.entity import config_entity
from bookRecommender.components.data_ingestion import DataIngestion
from bookRecommender.exception import BookRecommenderException



if __name__ == '__main__':
    try:
        training_pipeline_config = config_entity.TrainingPipelineConfig()

        #data ingestion         
        data_ingestion_config  = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        print(data_ingestion_config.to_dict())
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

    except Exception as e:
        raise BookRecommenderException(error_message=e, error_detail=sys)