import pandas as pd

# Load the dataframe
df = pd.read_csv('data/제주특별자치도관광객안전사고_2021.csv')

# Display the first few rows and column info to understand the data
print(df.head())
print(df.info())

# Replace '2021' with '2025' in RPTP_NO
df['RPTP_NO'] = df['RPTP_NO'].str.replace('2021', '2025', regex=False)

# Replace '2021' with '2025' in DCLR_YMD and DSPT_YMD
# Convert to string first, replace, then convert back to int (or keep as is if there are NaNs, but here they seem to be non-null int64)
df['DCLR_YMD'] = df['DCLR_YMD'].astype(str).str.replace('2021', '2025', regex=False).astype('int64')
df['DSPT_YMD'] = df['DSPT_YMD'].astype(str).str.replace('2021', '2025', regex=False).astype('int64')
df['DCLR_YR'] = df['DCLR_YR'].astype(str).str.replace('2021', '2025', regex=False).astype('int64')

# Verify the changes
print(df[['RPTP_NO', 'DCLR_YMD', 'DSPT_YMD','DCLR_YR']].head())

# Save the modified dataframe to a new CSV file
output_filename = 'data/Jeju_Safety_Accidents_2025.csv'
df.to_csv(output_filename, index=False)