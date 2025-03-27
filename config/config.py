import yaml

with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# Access database path
db_path = config['paths']['database'] 

# Get ordinal mappings
qual_mapping = config['features']['ordinal_mappings']['ExterQual']

# Access hyperparameters
learning_rates = config['model']['hyperparameters']['learning_rate']