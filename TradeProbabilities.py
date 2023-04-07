import pandas as pd
import matplotlib.pyplot as plt 
import yfinance as yf
import random
import numpy as np

class TradeProbabilities:
    def __init__(self, ticker, start_date, end_date, interval_type, period, pattern_type = None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval_type = interval_type
        self.period = period
        self.pattern_type = pattern_type

        self.df = yf.download(ticker,start_date, end_date,interval=interval_type)
        self.df.dropna(inplace=True)
        self.df.reset_index(inplace=True)

        if any(type_ in interval_type for type_ in ['wk', 'mo']):
            self.df = self.df[:-1].copy()
            self.df.sort_values(by='Date',ascending = False, inplace=True)
        elif any(type_ in interval_type for type_ in ['d']):
            self.df.sort_values(by='Date',ascending = False, inplace=True)
        elif any(type_ in interval_type for type_ in ['h']):
            self.df = self.df[:-1].copy()
            self.df = self.df.rename(columns={'index':'Date'})
            self.df.sort_values(by='Date',ascending = False, inplace=True)
        elif any(type_ in interval_type for type_ in ['m',]):
            self.df = self.df[:-1].copy()
            self.df = self.df.rename(columns={'Datetime':'Date'})
            self.df.sort_values(by='Date',ascending = False, inplace=True)
        
        self.df.reset_index(inplace=True)

        
    def get_up_days(self):

        number_of_days = []
        high_price = []

        for i in range(len(self.df)-1):
            j = 1
            curr_high = self.df['High'][i]

            try:
                yesterday_high = self.df['High'][i+j]
                while curr_high >= yesterday_high:
                    j += 1
                    curr_high = yesterday_high
                    yesterday_high = self.df['High'][i+j]
            except KeyError:
                pass

            number_of_days.append(j-1)
            high_price.append(curr_high)

        number_of_days =  number_of_days + [np.nan] 
        high_price = high_price + [np.nan] 

        self.df['number_of_up_days'] = number_of_days
        self.df['high_price'] = high_price
                

    def get_down_days(self):
        number_of_days = []
        low_price = []

        for i in range(len(self.df)-1):
            j = 1
            curr_low = self.df['Low'][i]

            try:
                yesterday_low = self.df['Low'][i+j]
                while curr_low <= yesterday_low:
                    j += 1
                    curr_low = yesterday_low
                    yesterday_low = self.df['Low'][i+j]
            except KeyError:
                pass

            number_of_days.append(j-1)
            low_price.append(curr_low)

        number_of_days =  number_of_days + [np.nan] 
        low_price = low_price + [np.nan] 

        self.df['number_of_down_days'] = number_of_days
        self.df['low_price'] = low_price

    def get_number_of_days(self):
        self.get_up_days()
        self.get_down_days()
        self.df['number_of_days'] = self.df['number_of_up_days'] - self.df['number_of_down_days']
        self.df.dropna(inplace=True)
        self.df = self.df.sort_values('Date', ascending = True)
        self.df.reset_index(inplace=True, drop=True)

    # def get_lowest_price_next_n_days( data, date_var, n):
    #     df_subset = data[data['Date']>date_var].copy()
    #     df_subset = df_subset.head(n)
    #     max_ = df_subset['High'].max()
    #     min_ = df_subset['Low'].min()

    #     return max_, min_

    def get_lowest_price_next_n_days(self, date_var):
        df_subset = self.df[self.df['Date']>date_var].copy()
        df_subset = df_subset.head(self.period)
        max_ = df_subset['High'].max()
        min_ = df_subset['Low'].min()

        return max_, min_
    
    def get_lowest_price_in_n_days(self):
        for i, row in self.df.iterrows():
            value_ = self.df['Open'].tail(1).item()
            
            max_, min_ = self.get_lowest_price_next_n_days( row['Date'])
            self.df.loc[i, 'Max_Next_N_Days'] = ((max_ - row['Open'])/row['Open'])*value_
            self.df.loc[i, 'Min_Next_N_Days'] = ((min_ - row['Open'])/row['Open'])*value_
            self.df.loc[i, 'Agg_Next_N_Days'] = self.df.loc[i, 'Max_Next_N_Days']  + self.df.loc[i, 'Min_Next_N_Days']
        self.df['N_Days'] = self.period

    def get_pattern(self):
        keyColumns = [
                    'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                    'N_Days',
                    'number_of_up_days', 'number_of_down_days',  'number_of_days',
                    'Max_Next_N_Days', 'Min_Next_N_Days','Agg_Next_N_Days' 
                    ]
        self.get_number_of_days()
        self.get_lowest_price_in_n_days()
        self.df = self.df[keyColumns].copy()
        self.df['number_of_up_days_lag'] = self.df['number_of_up_days'].shift(1)
        self.df['number_of_down_days_lag'] = self.df['number_of_down_days'].shift(1)
        self.df['number_of_days_lag'] = self.df['number_of_days'].shift(1)

        self.df['Pattern_Current'] = 'Days: ' + self.df['number_of_days'].astype(int).astype(str) +  ' Down: '  + self.df['number_of_down_days'].astype(int).astype(str) + ' Up: '  + self.df['number_of_up_days'].astype(int).astype(str)

        self.df['Pattern_Lag'] = self.df['Pattern_Current'].shift(1)

    def get_pattern_analytics(self, pattern_type = None):
        if pattern_type == None:
            pattern_type = self.df['Pattern_Current'].tail(1).item()

        pattern_summary = self.df.groupby(['Pattern_Lag','Pattern_Current'
                                    ]).count()[['Date']].sort_values('Date', ascending = False).reset_index()
        pattern_analytics = pattern_summary[pattern_summary['Pattern_Lag']==pattern_type].copy()

        pattern_analytics.columns = ['Current Pattern', 'Predicted Pattern', 'Observations']
        pattern_analytics['Probability'] = pattern_analytics['Observations']/pattern_analytics['Observations'].sum()
        pattern_analytics.reset_index(inplace=True, drop = True)
        self.pattern_analytics = pattern_analytics
        return self.pattern_analytics

    def get_pattern_summary(self):
        summary_df = self.df.groupby('Pattern_Current').describe()[['N_Days','Max_Next_N_Days','Min_Next_N_Days','Agg_Next_N_Days']]
        keepCols = [('N_Days',  'count'),

                    ('Max_Next_N_Days',  'mean'),
                    ('Max_Next_N_Days',   '50%'),

                    ('Min_Next_N_Days',  'mean'),
                    ('Min_Next_N_Days',   '50%'),

                    ('Agg_Next_N_Days',  'mean'),
                    ('Agg_Next_N_Days',   '50%'),
                ]
        summary_df = summary_df[keepCols].sort_values(('N_Days',  'count'), ascending = False)
        self.pattern_summary = summary_df
        return self.pattern_summary

    def get_pattern_expectations(self):
        pattern_analytics = self.pattern_analytics.round(4)
        current_pattern = pattern_analytics['Current Pattern'][0]
        predicted_pattern = pattern_analytics['Predicted Pattern'][0]
        obs_pattern = pattern_analytics['Observations'].sum().round(0)
        prob_pattern = pattern_analytics['Probability'][0].round(4)
        value_ = self.df['Open'].tail(1).item()

        data = {'Date' : self.df['Date'].tail(1).item(),
                'Data': self.interval_type,
            'Number of Days Forward': self.period,
            'Current Pattern': current_pattern,
            'Predicted Pattern': predicted_pattern,
            'Probability': prob_pattern.round(4),
            'Obervations': obs_pattern }

        summary_analytics = self.pattern_summary.reset_index()
        summary_analytics = summary_analytics[summary_analytics['Pattern_Current']==predicted_pattern].reset_index()
        summary_analytics = summary_analytics.round(2)
        summary_analytics_table = summary_analytics.copy()

        del summary_analytics['index']
        del summary_analytics['Pattern_Current']
        summary_analytics.columns = [
                                    'Expected Obs',
                                    'High_Mean', 'High_Median',
                                    'Low_Mean', 'Low_Median',
                                    'Net_Mean', 'Net_Median']

        summary_analytics['Current Price'] = value_

        summary_analytics['Expected High Mean'] = value_ + summary_analytics['High_Mean']
        summary_analytics['Expected High Median'] = value_ + summary_analytics['High_Median']

        summary_analytics['Expected Low Mean'] = value_ + summary_analytics['Low_Mean']
        summary_analytics['Expected Low Median'] = value_ + summary_analytics['Low_Median']

        summary_analytics['Expected Net Mean'] = value_ + summary_analytics['Net_Mean']
        summary_analytics['Expected Net Median'] = value_ + summary_analytics['Net_Median']

        PatternExpectations = summary_analytics.iloc[0].to_dict()
        data.update(PatternExpectations)
        self.pattern_expectations = data
        return self.pattern_expectations