import pandas as pd
import subprocess
import zipfile
import os 
import io
import sqlite3

## create a db name
database_name = 'db/house_prices.db' 

#Configuration to kaggle
competition_name = 'house-prices-advanced-regression-techniques'


## Install kaggle if not installed
try:
    import kaggle
    print('Kaggle is already installed')
except ImportError:
    print('Kaggle is not installed, installing now...')
    subprocess.run(['pip', 'install', 'kaggle'])
    print("Kaggle has been successfully installed")


## Downloading the data
# I've storedd the .kaggle/kaggle.json need to set the KAGGLE_CONFIG_DIR environment variable

if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
    print('Please check if the kaggle.json file is stored in the correct directory, not found!!')
else:
    print('Downloading the data from Kaggle...')


download_file = ['kaggle', 'competitions', 'download', '-c', competition_name]
try:
    subprocess.run(download_file, check=True)
    print("Data download complete.")
except subprocess.CalledProcessError as e:
    print(f"Error downloading data: {e}")
    print("Make sure you have accepted the competition rules on the Kaggle website.")
    exit()


# Connect to the database
conn = sqlite3.connect(database_name)
cursor = conn.cursor()

# Open the zippd file
zip_file = f"{competition_name}.zip"
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_files = zip_ref.namelist()
    
    for file_name in zip_files:
        if file_name.endswith('.csv'):  
            table_name = file_name.replace('.csv', '')  
            print(f"Loading {file_name} into table '{table_name}'...")
            
            try:
                # Read the CSV file directly from the ZIP into a pandas DataFrame
                with zip_ref.open(file_name) as csv_file:
                    df = pd.read_csv(io.BytesIO(csv_file.read()))
                
                # Load the DataFrame into the SQLite database
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                print(f"Data loaded into '{table_name}' table.")
            except Exception as e:
                print(f"Error loading data: {e}")

# Close the database connection
conn.close()
print("Finished loading data to the database.")