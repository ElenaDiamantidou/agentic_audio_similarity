import pandas as pd


def explode_dict_columns(df, keys_to_explode=None):
    """
    Explode dictionary columns into separate columns in the DataFrame.

    Parameters:
    - df: pandas DataFrame containing dictionary columns
    - keys_to_explode: dict or None. Dictionary mapping column names to lists of keys to explode.
                      If None, all keys in each dictionary column will be exploded.
                      Example: {'meta': ['artist', 'year'], 'features': ['tempo', 'key']}

    Returns:
    - DataFrame with dictionary columns exploded into separate columns
    """
    # Create a copy of the input DataFrame
    result_df = df.copy()

    # Find columns containing dictionaries
    dict_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column contains dictionaries (check first non-null value)
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(sample, dict):
                dict_columns.append(col)

    # Process each dictionary column
    for col in dict_columns:
        # Skip if column contains only nulls
        if df[col].isna().all():
            continue

        # Determine which keys to explode for this column
        specific_keys = keys_to_explode.get(col) if keys_to_explode else None

        if specific_keys:
            # Only extract specified keys
            for key in specific_keys:
                result_df[f"{col}_{key}"] = df[col].apply(
                    lambda x: x.get(key) if isinstance(x, dict) else None
                )
        else:
            # Extract all keys using apply with pd.Series
            dict_df = df[col].apply(pd.Series).add_prefix(f"{col}_")
            result_df = pd.concat([result_df, dict_df], axis=1)

        # Remove original dictionary column
        result_df.drop(col, axis=1, inplace=True)

    return result_df
