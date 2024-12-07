import pandas as pd
from datetime import timedelta

def aggregate_member_data(df):
    # Ensure datetime conversion
    df['Smsdelivered_dt'] = pd.to_datetime(df['Smsdelivered_dt'])
    df['Optout_dt'] = pd.to_datetime(df['Optout_dt'], errors='coerce')

    # Group by Subscriberkey
    aggregated = df.groupby('Subscriberkey').apply(lambda group: pd.Series({
        'Cnt_mobile': group['Mobile'].nunique(),
        'Days_since_first_sms': (df['Smsdelivered_dt'].max() - group['Smsdelivered_dt'].min()).days,
        'Days_btwn_first_last_sms': (group['Smsdelivered_dt'].max() - group['Smsdelivered_dt'].min()).days,
        'Cnt_optdown': group['Opted_down'].sum(),
        'Opted_out': int(group['Optout_dt'].notna().any()),
        'Cnt_sms': len(group),
        'Cnt_sms_2_day': len(group[group['Smsdelivered_dt'] >= group['Smsdelivered_dt'].max() - timedelta(days=2)]),
        'Cnt_sms_1_week': len(group[group['Smsdelivered_dt'] >= group['Smsdelivered_dt'].max() - timedelta(weeks=1)]),
        'Cnt_sms_1_month': len(group[group['Smsdelivered_dt'] >= group['Smsdelivered_dt'].max() - timedelta(days=30)]),
        'Cnt_sms_2_month': len(group[group['Smsdelivered_dt'] >= group['Smsdelivered_dt'].max() - timedelta(days=60)]),
        'Cnt_sms_6_month': len(group[group['Smsdelivered_dt'] >= group['Smsdelivered_dt'].max() - timedelta(days=180)]),
        'Cnt_sms_1_year': len(group[group['Smsdelivered_dt'] >= group['Smsdelivered_dt'].max() - timedelta(days=365)]),
        'Avg_length_sms_2_day': group[group['Smsdelivered_dt'] >= group['Smsdelivered_dt'].max() - timedelta(days=2)]['Message_len'].mean(),
        'Avg_length_sms_1_week': group[group['Smsdelivered_dt'] >= group['Smsdelivered_dt'].max() - timedelta(weeks=1)]['Message_len'].mean(),
        'Avg_length_sms_1_month': group[group['Smsdelivered_dt'] >= group['Smsdelivered_dt'].max() - timedelta(days=30)]['Message_len'].mean(),
        'Avg_length_sms_2_month': group[group['Smsdelivered_dt'] >= group['Smsdelivered_dt'].max() - timedelta(days=60)]['Message_len'].mean(),
        'Avg_length_sms_6_month': group[group['Smsdelivered_dt'] >= group['Smsdelivered_dt'].max() - timedelta(days=180)]['Message_len'].mean(),
        'Avg_length_sms_1_year': group[group['Smsdelivered_dt'] >= group['Smsdelivered_dt'].max() - timedelta(days=365)]['Message_len'].mean(),
    }))

    return aggregated.reset_index()

# Example usage
# Assuming df is your DataFrame
# result = aggregate_member_data(df)

import pandas as pd
import numpy as np

def aggregate_member_data_optimized(df):
    # Ensure datetime conversion
    df['Smsdelivered_dt'] = pd.to_datetime(df['Smsdelivered_dt'])
    df['Optout_dt'] = pd.to_datetime(df['Optout_dt'], errors='coerce')

    # Aggregate static member-level statistics
    aggregated = df.groupby('Subscriberkey').agg(
        Cnt_mobile=('Mobile', 'nunique'),
        First_sms=('Smsdelivered_dt', 'min'),
        Last_sms=('Smsdelivered_dt', 'max'),
        Cnt_optdown=('Opted_down', 'sum'),
        Opted_out=('Optout_dt', lambda x: int(x.notna().any())),
        Cnt_sms=('Smsdelivered_dt', 'count')
    ).reset_index()

    # Precompute reusable values for efficiency
    aggregated['Days_since_first_sms'] = (df['Smsdelivered_dt'].max() - aggregated['First_sms']).dt.days
    aggregated['Days_btwn_first_last_sms'] = (aggregated['Last_sms'] - aggregated['First_sms']).dt.days

    # Calculate days between the last two SMS
    last_two_sms = df.groupby('Subscriberkey')['Smsdelivered_dt'].apply(
        lambda x: (x.sort_values().iloc[-1] - x.sort_values().iloc[-2]).days if len(x) > 1 else 0
    )
    aggregated['Days_btwn_last_two_sms'] = last_two_sms.values

    # Rolling time window features
    max_date = df['Smsdelivered_dt'].max()

    def compute_rolling_features(window):
        threshold = max_date - pd.Timedelta(days=window)
        filtered = df[df['Smsdelivered_dt'] >= threshold]
        counts = filtered.groupby('Subscriberkey')['Smsdelivered_dt'].count()
        avg_lengths = filtered.groupby('Subscriberkey')['Message_len'].mean()
        return counts, avg_lengths

    windows = [2, 7, 30, 60, 180]
    for window in windows:
        counts, avg_lengths = compute_rolling_features(window)
        aggregated[f'Cnt_sms_{window}_day'] = aggregated['Subscriberkey'].map(counts).fillna(0)
        aggregated[f'Avg_length_sms_{window}_day'] = aggregated['Subscriberkey'].map(avg_lengths).fillna(0)

    # Drop intermediate columns no longer needed
    aggregated.drop(columns=['First_sms', 'Last_sms'], inplace=True)

    return aggregated




import pandas as pd
import numpy as np
from datetime import timedelta
from dask.dataframe import from_pandas

def aggregate_member_data_optimized(df, chunk_size=5_000_000):
    # Ensure datetime conversion
    df['Smsdelivered_dt'] = pd.to_datetime(df['Smsdelivered_dt'])
    df['Optout_dt'] = pd.to_datetime(df['Optout_dt'], errors='coerce')

    # Sort to simplify processing
    df = df.sort_values(['Subscriberkey', 'Smsdelivered_dt'])

    # Initialize final results list
    aggregated_results = []

    # Process data in chunks
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]

        # Groupby for chunk and calculate all static features
        grouped = chunk.groupby('Subscriberkey')
        agg = grouped.agg(
            Cnt_mobile=('Mobile', 'nunique'),
            First_sms=('Smsdelivered_dt', 'min'),
            Last_sms=('Smsdelivered_dt', 'max'),
            Cnt_optdown=('Opted_down', 'sum'),
            Opted_out=('Optout_dt', lambda x: int(x.notna().any())),
            Cnt_sms=('Smsdelivered_dt', 'count')
        )

        # Calculate additional static features
        agg['Days_since_first_sms'] = (df['Smsdelivered_dt'].max() - agg['First_sms']).dt.days
        agg['Days_btwn_first_last_sms'] = (agg['Last_sms'] - agg['First_sms']).dt.days

        # Compute days between last two SMS
        agg['Days_btwn_last_two_sms'] = grouped['Smsdelivered_dt'].apply(
            lambda x: (x.iloc[-1] - x.iloc[-2]).days if len(x) > 1 else 0
        )

        # Rolling time window features
        max_date = df['Smsdelivered_dt'].max()
        for window in [2, 7, 30, 60, 180]:
            threshold = max_date - timedelta(days=window)
            filtered = chunk[chunk['Smsdelivered_dt'] >= threshold]

            counts = filtered.groupby('Subscriberkey')['Smsdelivered_dt'].count()
            avg_lengths = filtered.groupby('Subscriberkey')['Message_len'].mean()

            agg[f'Cnt_sms_{window}_day'] = counts
            agg[f'Avg_length_sms_{window}_day'] = avg_lengths

        # Append to final results
        aggregated_results.append(agg.fillna(0))

    # Concatenate all chunks
    final_aggregated = pd.concat(aggregated_results)

    # Reset index for the final DataFrame
    final_aggregated.reset_index(inplace=True)

    return final_aggregated

