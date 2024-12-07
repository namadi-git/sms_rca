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
