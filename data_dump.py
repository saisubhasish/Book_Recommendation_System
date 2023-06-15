import json
import pymongo
import pandas as pd
from bookRecommender.config import mongo_client

BOOKS_DATA_FILE_PATH="D:/FSDS-iNeuron/10.Projects-DS/Book_Recommendation_System/books_data/books.csv"
USERS_DATA_FILE_PATH="D:/FSDS-iNeuron/10.Projects-DS/Book_Recommendation_System/books_data/users.csv"
RATINGS_DATA_FILE_PATH="D:/FSDS-iNeuron/10.Projects-DS/Book_Recommendation_System/books_data/ratings.csv"

DATABASE_NAME = "bookRecommender"
BOOKS_COLLECTION_NAME = "books"
USERS_COLLECTION_NAME="users"
RATINGS_COLLECTION_NAME="ratings"


if __name__=="__main__":
    books = pd.read_csv(BOOKS_DATA_FILE_PATH)
    users = pd.read_csv(USERS_DATA_FILE_PATH)
    ratings = pd.read_csv(RATINGS_DATA_FILE_PATH)

    print(f"Rows and columns: {books.shape}")
    print(f"Rows and columns: {users.shape}")
    print(f"Rows and columns: {ratings.shape}")

    #Convert dataframe to json so that we can dump these record in mongo db
    books.reset_index(drop=True,inplace=True)
    users.reset_index(drop=True,inplace=True)
    ratings.reset_index(drop=True,inplace=True)

    # Each record will represent one row
    json_record_books = list(json.loads(books.T.to_json()).values())
    print(json_record_books[1])
    # Each record will represent one row
    json_record_users = list(json.loads(users.T.to_json()).values())
    print(json_record_users[1])
    # Each record will represent one row
    json_record_ratings = list(json.loads(ratings.T.to_json()).values())
    print(json_record_ratings[1])


    #insert converted json record to mongo db
    mongo_client[DATABASE_NAME][BOOKS_COLLECTION_NAME].insert_many(json_record_books)
    mongo_client[DATABASE_NAME][USERS_COLLECTION_NAME].insert_many(json_record_users)
    mongo_client[DATABASE_NAME][RATINGS_COLLECTION_NAME].insert_many(json_record_ratings)
