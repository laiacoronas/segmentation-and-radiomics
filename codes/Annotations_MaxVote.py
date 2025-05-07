# -*- coding: utf-8 -*-


#%% Environment
# Import libraries
import pandas as pd
import numpy as np
import os

os.chdir(os.path.dirname(os.path.realpath('__file__')))
df = pd.read_excel("data/sample/MetadatabyAnnotation.xlsx")

#%% Max vote all _value
# Step 1: Identify the columns ending with '_value' and the len_mal_details column
value_columns = [col for col in df.columns if col.endswith('_value')]
value_columns.append('len_mal_details')  # Don't forget 'len_mal_details'

# Step 2: Group by 'patient_id' and 'nodule_id'
grouped = df.groupby(['patient_id', 'nodule_id'])

# Step 3: Apply Max-Voting for each group
def max_voting(group):
    result = {}
    
    # Apply Max-Voting for each relevant column
    for col in value_columns:
        # Get the mode values, and if there's a tie, select the max
        most_frequent_values = group[col].mode()
        if len(most_frequent_values) > 1:
            result[col] = most_frequent_values.max()  # Select the highest value in case of a tie
        else:
            result[col] = most_frequent_values.iloc[0]  # Otherwise, select the most frequent value
    
    return pd.Series(result)

# Step 4: Apply the function and reset index
final_df = grouped.apply(max_voting).reset_index()



#%% Diagnosis value
# Step 1: Count how many annotations per group have Malignancy_value > 3
diagnosis_flags = (
    df[df['Malignancy_value'] > 3]
    .groupby(['patient_id', 'nodule_id'])
    .size()
    .reset_index(name='count_malignant')
)

# Step 2: Set Diagnosis_value = 1 if there are two or more such annotations
diagnosis_flags['Diagnosis_value'] = (diagnosis_flags['count_malignant'] >= 2).astype(int)

# Step 3: Merge with final_df
final_df = final_df.merge(
    diagnosis_flags[['patient_id', 'nodule_id', 'Diagnosis_value']],
    on=['patient_id', 'nodule_id'],
    how='left'
)

# Step 4: Fill NaN values with 0 (i.e., cases with <2 annotations >3)
final_df['Diagnosis_value'] = final_df['Diagnosis_value'].fillna(0).astype(int)


#%% add variables not used for maxvoting but we want to keep

# List of variables to bring from the first observation
first_vars = [
    'seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm',
    'bboxLowX', 'bboxLowY', 'bboxLowZ', 'bboxHighX', 'bboxHighY', 'bboxHighZ'
]

# Keep patient_id and nodule_id in the selection
first_obs = (
    df.groupby(['patient_id', 'nodule_id'])
    .first()
    .reset_index()[['patient_id', 'nodule_id'] + first_vars]
)

# Merge with final_df
final_df = final_df.merge(first_obs, on=['patient_id', 'nodule_id'], how='left')

# Step 2: Reorder columns so the new ones are right after 'nodule_id'
# Get the desired insert position
nodule_idx = final_df.columns.get_loc('nodule_id') + 1

# Columns to insert
insert_cols = [
    'seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm',
    'bboxLowX', 'bboxLowY', 'bboxLowZ', 'bboxHighX', 'bboxHighY', 'bboxHighZ'
]

# Rebuild column order
other_cols = [col for col in final_df.columns if col not in insert_cols]
new_order = (
    other_cols[:nodule_idx] +
    insert_cols +
    other_cols[nodule_idx:]
)

# Apply the new column order
final_df = final_df[new_order]

# Get current column list
cols = list(final_df.columns)

# Remove Diagnosis_value temporarily
cols.remove('Diagnosis_value')

# Find index of Malignancy_value
mal_idx = cols.index('Malignancy_value')

# Insert Diagnosis_value just before Malignancy_value
cols.insert(mal_idx, 'Diagnosis_value')

# Reorder final_df columns
final_df = final_df[cols]

#%% add categorizations
# Define all mappings, including Diagnosis
mappings = {
    'Subtlety': {
        1: 'Extremely Subtle', 2: 'Moderately Subtle', 3: 'Fairly Subtle',
        4: 'Moderately Obvious', 5: 'Obvious'
    },
    'InternalStructure': {
        1: 'Soft Tissue', 2: 'Fluid', 3: 'Fat', 4: 'Air'
    },
    'Calcification': {
        1: 'Popcorn', 2: 'Laminated', 3: 'Solid', 4: 'Non-central',
        5: 'Central', 6: 'Absent'
    },
    'Sphericity': {
        1: 'Linear', 2: 'Ovoid/Linear', 3: 'Ovoid',
        4: 'Ovoid/Round', 5: 'Round'
    },
    'Margin': {
        1: 'Poorly Defined', 2: 'Near Poorly Defined', 3: 'Medium Margin',
        4: 'Near Sharp', 5: 'Sharp'
    },
    'Lobulation': {
        1: 'No Lobulation', 2: 'Nearly No Lobulation', 3: 'Medium Lobulation',
        4: 'Near Marked Lobulation', 5: 'Marked Lobulation'
    },
    'Spiculation': {
        1: 'No Spiculation', 2: 'Nearly No Spiculation', 3: 'Medium Spiculation',
        4: 'Near Marked Spiculation', 5: 'Marked Spiculation'
    },
    'Texture': {
        1: 'Non-Solid/GGO', 2: 'Non-Solid/Mixed', 3: 'Part Solid/Mixed',
        4: 'Solid/Mixed', 5: 'Solid'
    },
    'Malignancy': {
        1: 'Highly Unlikely', 2: 'Moderately Unlikely', 3: 'Indeterminate',
        4: 'Moderately Suspicious', 5: 'Highly Suspicious'
    },
    'Diagnosis': {
        0: 'Benign',
        1: 'Malign'
    }
}

# Loop over all mappings
for feature, mapping in mappings.items():
    value_col = f'{feature}_value'
    label_col = f'{feature}_label'

    # Only proceed if value_col exists in final_df
    if value_col in final_df.columns:
        # Create the label column using the mapping
        final_df[label_col] = final_df[value_col].map(mapping)

        # Reorder: insert label column just before value column
        cols = list(final_df.columns)
        if label_col in cols:
            cols.remove(label_col)
        val_idx = cols.index(value_col)
        cols.insert(val_idx, label_col)
        final_df = final_df[cols]

#%% Export annotations
final_df.to_excel('Annotations_MaxVote.xlsx', index=False)
