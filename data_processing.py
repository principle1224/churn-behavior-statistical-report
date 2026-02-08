# Read in the customer data
import pandas as pd
import numpy as np

def load_and_process_data():
    """첫 번째 노트북 코드를 그대로 함수화"""
    
    customers = pd.read_csv('Data/maven_music_customers.csv')

    # Read in the listening history
    listening_history = pd.read_excel('Data/maven_music_listening_history.xlsx')

    # Hint: Where might you find listening history data beyond the ID's?
    ## Check the other sheets in the Excel spreadsheet

    # Read in the audio data
    audio = pd.read_excel('Data/maven_music_listening_history.xlsx', sheet_name=1)

    # Read in the session data
    sessions = pd.read_excel('Data/maven_music_listening_history.xlsx', sheet_name=2)

    # Convert objects to numeric and datetime fields
    customers['Member Since'] = pd.to_datetime(customers['Member Since'])
    customers['Subscription Rate'] = pd.to_numeric(customers['Subscription Rate'].str.replace('$', ''))
    customers['Cancellation Date'] = pd.to_datetime(customers['Cancellation Date'])

    # It looks like the $2.99 rate is for Basic plan, so fill missing Subscription Plan values with 'Basic'
    customers['Subscription Plan'] = customers['Subscription Plan'].fillna('Basic (Ads)')

    # Let's change to numeric to make our life easier later
    customers['Discount?'] = np.where(customers['Discount?']=='Yes', 1, 0)

    # Fix the 99.99 typo
    customers.iloc[15, 5] = 9.99

    # Pop and Pop Music should be mapped to the same value
    audio.Genre = np.where(audio.Genre == 'Pop Music', 'Pop', audio.Genre)

    # Create a 'Cancelled' column
    customers['Cancelled'] = np.where(customers['Cancellation Date'].notna(), 1, 0)

    # Create an updated Email column without the Email: portion
    customers['Email'] = customers.Email.str[6:]

    # Split the ID in the audio data so the column can be joined with other tables
    audio_clean = pd.DataFrame(audio.ID.str.split('-').to_list()).rename(columns={0:'Type', 1:'Audio ID'})

    # Add the new fields to the original audio table
    audio_all = pd.concat([audio_clean, audio], axis=1)

    # Change Audio ID to an int type instead of an object
    audio_all['Audio ID'] = audio_all['Audio ID'].astype('int')

    # Try the merge again
    df = listening_history.merge(audio_all, how='left', on='Audio ID')

    # Calculate the number of listening sessions for each customers
    number_of_sessions = df.groupby('Customer ID')['Session ID'].nunique().rename('Number of Sessions').to_frame().reset_index()

    # Group it by customer
    genres = pd.concat([df['Customer ID'], pd.get_dummies(df.Genre)], axis=1).groupby('Customer ID').sum().reset_index()

    # Add a column for total songs / podcasts listened to
    total_audio = listening_history.groupby('Customer ID')['Audio ID'].count().rename('Total Audio').to_frame().reset_index()

    # Create a master audio table to calculate percentages
    df_audio = genres.merge(total_audio, how='left', on='Customer ID')

    # Create a dataframe ready for modeling
    model_df = customers[['Customer ID', 'Cancelled', 'Discount?']]

    # Add it to the modeling dataframe
    model_df = model_df.merge(number_of_sessions, how='left', on='Customer ID')

    # Percent pop
    model_df['Percent Pop'] = df_audio.Pop / df_audio['Total Audio'] * 100

    # Percent podcasts
    model_df['Percent Podcasts'] = ((df_audio['Comedy'] + df_audio['True Crime']) / df_audio['Total Audio']) * 100

    return customers, listening_history, audio_all, sessions, df, model_df, genres
