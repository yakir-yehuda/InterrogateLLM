import os
import csv
import pandas as pd

# https://www.kaggle.com/datasets/saurabhbagchi/books-dataset


def create_books_dataset(save_path):
    books_data_path = './datasets/books_data/books.csv'

    # Open the input CSV file
    with open(books_data_path, 'r', encoding='iso-8859-1') as input_file:
        # Create a CSV reader object
        reader = csv.reader(input_file)

        expected_column_count = None

        # Create a list to store the rows with separated columns
        rows_with_columns = []

        # Iterate over each line in the input CSV file
        for idx, line in enumerate(reader):
            # Separate the line into columns
            columns = line[0].split(';')

            if idx == 0:
                expected_column_count = len(columns)

            # Append the columns as a new row in the list
            if len(columns) == expected_column_count:
                rows_with_columns.append(columns)

    # Open the output CSV file
    with open(save_path, 'w', newline='') as output_file:
        # Create a CSV writer object
        writer = csv.writer(output_file)

        # Write each row with separated columns to the output CSV file
        for row in rows_with_columns:
            writer.writerow(row)


if __name__ == '__main__':
    dataset_save_path = './datasets/books_data/books_filtered.csv'
    if not os.path.isfile(dataset_save_path):
        print('Create books all dataset...')
        create_books_dataset(dataset_save_path)
        print('Dataset books all created.')
    else:
        print('The dataset books all already exists')
