import pandas as pd
class DataCollection :
    """
        Define the list of nasdaq consituent for a certain year or even day (for futur work)
        Data : From WRDS -> Compustats and 1990 to 2023
    """
    def __init__(self, file_path)-> None:
        self.data = pd.read_csv(file_path)
        
    def list_nasdaq_constituents_by_year(self, year: int = 2023) -> list:
        """
        List all NASDAQ constituents for a given year based on the dataset, with cleaned ticker symbols.

        Parameters:
        data (DataFrame): The DataFrame containing NASDAQ constituents data.
        year (int): The year for which to list the constituents.

        Returns:
        list: A list of tickers that were NASDAQ constituents in the given year, with no suffixes.
        """
        # Convert 'from' and 'thru' columns to datetime
        self.data['from'] = pd.to_datetime(self.data['from'])
        self.data['thru'] = pd.to_datetime(self.data['thru'])

        # Filter data for the specified year
        constituents = self.data[(self.data['from'].dt.year <= year) & (self.data['thru'].isna() | (self.data['thru'].dt.year >= year))]

        # Clean the ticker symbols and get the unique list
        tickers = constituents['co_tic'].unique().tolist()
        return tickers
    

    def create_nasdaq_constituents_daily(self) -> list:
        """
        Create a dictionary where each key is a date and the value is a list of tickers that were NASDAQ 100 constituents on that date.

        Parameters:
        data (DataFrame): The DataFrame containing NASDAQ constituents data.

        Returns:
        dict: A dictionary with dates as keys and lists of tickers as values.
        """
        # Convert 'from' and 'thru' columns to datetime
        self.data['from'] = pd.to_datetime(self.data['from'])
        self.data['thru'] = pd.to_datetime(self.data['thru']).fillna(pd.Timestamp('today'))

        # Initialize an empty dictionary
        constituents_dict = {}

        # Iterate over each row in the DataFrame
        for index, row in self.data.iterrows():
            # Generate a date range for each row
            date_range = pd.date_range(start=row['from'], end=row['thru'])

            for date in date_range:
                # Convert date to string format for dictionary key
                date_str = date.strftime('%Y-%m-%d')
                
                # Add the ticker to the corresponding date in the dictionary
                if date_str in constituents_dict:
                    constituents_dict[date_str].add(row['co_tic'])
                else:
                    constituents_dict[date_str] = {row['co_tic']}

        # Convert sets to lists for each date
        for date in constituents_dict:
            constituents_dict[date] = list(constituents_dict[date])

        return constituents_dict


