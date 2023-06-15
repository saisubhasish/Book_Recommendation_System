import os,sys 
import numpy as np
import pandas as pd

from typing import Optional

from bookRecommender import utils
from bookRecommender.entity import artifact_entity,config_entity
from bookRecommender.exception import BookRecommenderException
from bookRecommender.logger import logging




class DataTransformation:

    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                    data_validation_artifact:artifact_entity.DataValidationArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config=data_transformation_config
            self.data_validation_artifact=data_validation_artifact
        except Exception as e:
            raise BookRecommenderException(e, sys)

    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:
        try:
            # Reading training and testing file
            logging.info("Reading data file")
            books_df = pd.read_csv(self.data_validation_artifact.books_file_path)
            users_df = pd.read_csv(self.data_validation_artifact.users_file_path)
            ratings_df = pd.read_csv(self.data_validation_artifact.ratings_file_path)
            
            logging.info("Preparing data for Popularity based recommendation system")
            logging.info("We will consider the highest rating 50 books for recommendation which got minimum 250 votes")
            logging.info("Merging ratings and books table on the top of 'ISBN' column")
            # We will consider the highest rating 50 books for recommendation which got minimum 250 votes
            # Merging ratings and books table on the top of 'ISBN' column
            ratings_with_name = ratings_df.merge(books_df, on='ISBN')        
            
            logging.info("There are books having multiple 'ISBN' number so gouping books on 'Book-Title' column ")
            # There are books having multiple 'ISBN' number so gouping books on 'Book-Title' column
            ratings_with_name.groupby('Book-Title').count()

            logging.info("Making dataframe as per book rating")
            num_rating_df = ratings_with_name.groupby('Book-Title').count()[['Book-Rating']].reset_index()
            logging.info("Changing name from book rating to num rating")
            num_rating_df.rename(columns={'Book-Rating':'num_ratings'}, inplace=True)

            print(f"maximum num rating: {num_rating_df['num_ratings'].max()}")

            # Getting average ratings of books
            logging.info("Getting average ratings of books")
            avg_rating_df = ratings_with_name.groupby('Book-Title').mean()[['Book-Rating']].reset_index()
            avg_rating_df.rename(columns={'Book-Rating':'avg_rating'}, inplace=True)

            # Merging num_rating_df and avg_rating_df
            logging.info("Merging num_rating_df and avg_rating_df")
            popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')

            # Filtering the books with voting greater than 250
            logging.info("Filtering the books with voting greater than 250")
            popular_df = popular_df[popular_df['num_ratings']>=250].sort_values('avg_rating', ascending=False).head(50)

            # Merging dataframe on top of 'Book-Title ' to get required columns
            logging.info("Merging dataframe on top of 'Book-Title ' to get required columns")
            popular_df = popular_df.merge(books_df, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']]

            # Colaberative filtering Based Recommended Syatem
            logging.info("Colaberative filtering Based Recommended Syatem")

            # Users rated more than 200 books
            logging.info("Users rated more than 200 books")
            x = ratings_with_name.groupby('User-ID').count()['Book-Rating']>200

            # saving the IDs of those users
            logging.info("saving the IDs of those users")
            padhe_likhe_users = x[x].index

            # Getting the dataframe using corredponding IDs
            logging.info("Getting the dataframe using corredponding IDs")
            filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]

            # Books with rating more than 50 times
            logging.info("Books with rating more than 50 times")
            y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
            famous_books = y[y].index

            # Getting final dataframe
            logging.info("Getting final datafrane")
            final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

            # Getting final table, with user with 200 or more votes and books with more than 50 votes
            # preparing pivot table
            logging.info("Getting final table, with user with 200 or more votes and books with more than 50 votes")
            pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
            
            # Filling NAN value with 0
            pt.fillna(0, inplace=True)
            logging.info(f"Shape of the data: {pt.shape}")
            print(f"Shape of the data: {pt.shape}")

            # Save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_file_path, array=pt)

            # Preparing Artifact
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transformed_file_path = self.data_transformation_config.transformed_file_path)

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
            
        except Exception as e:
            raise BookRecommenderException(e, sys)
