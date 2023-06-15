from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    books_file_path:str
    users_file_path:str 
    ratings_file_path:str

@dataclass
class DataValidationArtifact:
    report_file_path:str
    books_file_path:str 
    users_file_path:str
    ratings_file_path:str

@dataclass
class DataTransformationArtifact:
    transformed_pivot_table_file_path:str
    popular_data_file_path:str

@dataclass
class ModelTrainerArtifact:
    model_path:str 

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    improved_accuracy:float

@dataclass
class ModelPusherArtifact:
    pusher_model_dir:str 
    saved_model_dir:str