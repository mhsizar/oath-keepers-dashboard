import json
import pandas as pd
from pandas import json_normalize

# Load the JSON file
with open('data/messages.json', 'r', encoding='utf8') as f:
    data = json.load(f)

# Normalize the top level JSON structure
df = json_normalize(data)

def normalize_nested_json(df, column_name):
    '''Normalize the nested JSON structure in the specified column of the dataframe'''
    records = []
    for record in df[column_name].dropna():
        if isinstance(record, str):
            try:
                json_record = json.loads(record.replace("'", "\""))
                if isinstance(json_record, list):
                    records.extend(json_record)
                else:
                    records.append(json_record)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for column {column_name}: {e}")
        elif isinstance(record, list):
            records.extend(record)
        elif isinstance(record, dict):
            records.append(record)
    if records:
        return json_normalize(records)
    else:
        return pd.DataFrame()
   
#Iterate over the columns in the dataframe and normalize the nested JSON structures
nested_columns = [col for col in df.columns if any(isinstance(i, (dict, list)) for i in df[col].dropna())]
for column in nested_columns:
    normalized_df = normalize_nested_json(df, column)
    if not normalized_df.empty:
        normalized_df = normalized_df.add_prefix(f"{column}.")
        df = df.drop(columns=[column])
        df = pd.concat([df, normalized_df], axis=1)   
# Save the normalized dataframe to CSV
df.to_csv('data/messages_normalized.csv', index=False)

# Display the normalized dataframe
print(df.head())
            