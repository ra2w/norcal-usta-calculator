import re
import numpy as np
from collections import deque
import pandas as pd
import warnings
import logging


# Create or get the logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)  # Set lower bound to debug to capture all messages with handlers

# Create file handler which logs even debug messages
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)  # Adjust as needed to capture all levels of logs in file

# Create console handler with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)  # Only errors and above to the console

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


NUM_DYNAMIC_RATINGS = 3
class Player:
    def __init__(self, name, player_id):
        self.name = name
        if isinstance(player_id, float):
             self.player_id = str(int(player_id))
        else:
             self.player_id = str(player_id)

        self.on_year = None
        self.self_rate_match_count = 0

        self.ratings_by_year = {}
        self.match_rating_hist = deque([np.nan]*NUM_DYNAMIC_RATINGS)
        self.dynamic = deque([np.nan]*NUM_DYNAMIC_RATINGS, maxlen=NUM_DYNAMIC_RATINGS)
        self.matches_dict = {}
        self.df_rating_changes = None


    def bootstrap_initial_rating(self, match_rating):
        self.dynamic = deque([match_rating]*NUM_DYNAMIC_RATINGS, maxlen=NUM_DYNAMIC_RATINGS)
        return
    
    def add_new_match_rating(self, match_rating, row):
        if (self.player_id == '163908'):
            breakpoint
       
        self.match_rating_hist.append(match_rating)

        if self.is_self_rated(row['Year']) and self.count_ratings() < 3:
            if self.count_ratings() == 2:
                new_dynamic = np.nanmean(np.concatenate([np.array([match_rating]),self.get_all_dynamic()]))
                self.dynamic = deque([np.nan]*NUM_DYNAMIC_RATINGS, maxlen=NUM_DYNAMIC_RATINGS)
                self.dynamic.append(new_dynamic)
            else:
                # self.count_rating = 0 or 1
                self.dynamic.append(match_rating)
            self.self_rate_match_count += 1
        else:       
            new_dynamic = np.nanmean(np.concatenate([np.array([match_rating]),self.get_all_dynamic()]))
            # Append new rating, oldest rating is automatically dropped if exceeds maxlen
            self.dynamic.append(new_dynamic)

        return self.get_all_dynamic()

    def save_match_row(self, row):
        if row is not None:
            self.matches_dict[row['Match Date']] = {'row':row.to_dict()}


    def count_ratings(self):
        return self.self_rate_match_count
    
    def rating_to_str(self, rating):
        if rating is None:
            return None
        return str(rating['rating_value'])+rating['rating_type']
    
    def rating_from_str(self, rating_str):
        # Updated pattern to handle lower and uppercase letters and optional second ratings more robustly
        pattern = r'(\d+\.\d+)([A-Za-z]?)/?(\d+\.\d+)?([A-Za-z]?)'
        match = re.search(pattern, rating_str)
        
        if match:
            # Determine if there's a second rating and if the type should be taken from the first or second part
            if match.group(3):  # If there is a second rating
                value = float(match.group(3))
                type_char = match.group(2) if match.group(2) else match.group(4)
            else:
                value = float(match.group(1))
                type_char = match.group(2)

            # Handle case where type might be empty
            if not type_char:
                logging.error("Rating type is empty", rating_str, value, type_char)
                type_char = 'Unknown'  # Default or handle it as you see fit

            return {'rating_value':value, 'rating_type': type_char}
        
        logging.error("No match found for:", rating_str)

        return None  # Return None if the pattern does not match


    def get_rating_for_year(self, year):
        year = str(year)
        if year in self.ratings_by_year:
            # check if self.rating_by_year[year] is a list
            if isinstance(self.ratings_by_year[year], list):
                # if it is a list, return the last element
                rating = self.ratings_by_year[year][-1]
            else:
                rating = self.ratings_by_year[year]

            return rating
        return None

    def add_rating_to_year(self, year, rating_str):

        if not isinstance(year, str):
            year = str(year)

        if rating_str is None or not isinstance(rating_str, str):
                return None

        rating = self.rating_from_str(rating_str)
        if rating is None:
            logging.error(f"Invalid rating format for {self.name}: {rating_str}")
            return
        
        #append rating to self.ratings_by_year
        if year not in self.ratings_by_year:
            self.ratings_by_year[year] = [rating]
        else:
            self.ratings_by_year[year] = self.ratings_by_year[year] + [rating]


    def dynamic_to_rating_str(self, dynamic):
        n = np.round(dynamic, 2)       # Round the input to two decimal places.
        n = n * 200                    # Scale by multiplying by 200.
        n = np.floor(n / 100) / 2 + 0.5  # Divide by 100, apply floor, rescale, and adjust upwards by 0.5.

        return str(np.round(n,2))

    def record_rating_change(self, new_year):
        if str(int(new_year)) not in self.ratings_by_year:
                # TODO FIX THIS
                breakpoint()
                logging.error("Need to refetch rating for", self.name, self.player_id,new_year)
                self.add_rating_to_year(new_year, '4.0C')
        try:
            prev_year_rating_str = self.rating_to_str(self.get_rating_for_year(self.on_year))
        except Exception as e:
            logging.error(e)
            breakpoint()

        estimated_dynamic = self.get_latest_dynamic(self.on_year)
        cur_year_rating_str = self.rating_to_str(self.get_rating_for_year(str(int(new_year))))

        # compute the estimated rating str
        # we take the estimated dynamic rating and and compute in the following way:
        # if rating > 4.50 then estimated_rating_str = "5.0C"
        # if 
        estimated_rating_str = self.dynamic_to_rating_str(estimated_dynamic)

        new_record_dict = {
            'player_id': self.player_id,
            'prev_year': self.on_year,
            'cur_year': new_year,
            'prev_year_rating': prev_year_rating_str,
            'cur_year_rating': cur_year_rating_str,
            'estimated_dynamic': estimated_dynamic,
            'Correct': estimated_rating_str in cur_year_rating_str
        }


        # if df_rating_changes is None, create it
        # if it exists, append the new rating change
        if self.df_rating_changes is not None:
            self.df_rating_changes = pd.concat([self.df_rating_changes, pd.DataFrame([new_record_dict])], ignore_index=True)
        else:
            # Create the DataFrame if it does not exist
            self.df_rating_changes = pd.DataFrame([new_record_dict])
            
        # 2015 one-time adjustment
        # Since 2014 is the first year in our records. The start ratings for 2014 are approximate
        # Therefore, in 2015 we adjust the ratings based on upgrades/downgrade to be more accurate where possible
        if new_year == 2015:
            if estimated_rating_str not in cur_year_rating_str:
                rating = self.get_rating_for_year(str(int(new_year)))
                self.bootstrap_initial_rating(rating['rating_value']-0.5+0.10)
                logging.info("Adjusting rating for", self.name, self.player_id, "from", estimated_rating_str, "to", self.dynamic_to_rating_str(self.get_latest_dynamic(self.on_year)))
                

    def set_year(self, year, rating_str):

        self.add_rating_to_year(year, rating_str)

        if self.on_year is not None and year != self.on_year:
            # we have a new year
            # if the new year rating is a self rating, we need to erase the previous history

            if (rating := self.get_rating_for_year(str(int(year)))) is not None:
                if rating['rating_type'] == 's':
                    # erase previous history if new year rating is a 'C' rating
                    self.dynamic = deque([np.nan]*NUM_DYNAMIC_RATINGS, maxlen=NUM_DYNAMIC_RATINGS)
                    self.self_rate_match_count = 0

            self.record_rating_change(year)
        
        # if self is not self rated and self.dynamic is all nan, we need to bootstrap the initial rating
        if not self.is_self_rated(year) and all(np.isnan(self.dynamic)):
            self.bootstrap_initial_rating(self.get_base_est_rating(year))
        
        self.on_year = year
        return
    
    def get_latest_dynamic(self, year):
        if self.on_year != year:
            breakpoint()
        assert self.on_year == year

        # return the latest dynamic rating
        return self.dynamic[-1]
        #return np.nanmean(self.match_r)
    

    def get_all_dynamic(self):
        # return an np.array of all dynamic ratings
        return np.array(self.dynamic)
    
    def get_avg_self_rate_dynamic(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m = np.nanmean(self.dynamic)
            return m

    def get_base_est_rating(self, year):
        est_dynamic = np.nan
        year = str(year)
        if (rating:= self.get_rating_for_year(year)) is not None:
            rating_type = rating['rating_type']
            rating_value = rating['rating_value']

            if rating_type == 'C' or rating_type == 'b' or rating_type == 'E' or rating_type == 'm' or rating_type == 'T':
                if float(rating_value) < 4.5:
                    est_dynamic = rating_value-0.25
                else:
                    est_dynamic = rating_value-0.30
            elif rating_type == 'D':
                est_dynamic = rating_value-0.25
            elif rating_type == 'A':
                est_dynamic = rating_value-0.02
            elif rating_type == 's':
                return np.nan
            else:
                logging.error("Unknown rating type:", rating_type)
                est_dynamic = rating_value-0.25
        return est_dynamic
    
    def is_self_rated(self, year):
        year = str(year)
        rating = self.get_rating_for_year(year)
        if rating is not None and rating['rating_type'] == 's':
            return True
        
        return False


    def __str__(self):
        return f"{self.name}: {self.ratings_by_year}"
    
    def to_dict_all(self):
        def replace_nan_with_none(value):
            """
            Recursively replace NaN values with None in nested dictionaries and lists.
            """
            if isinstance(value, dict):
                return {k: replace_nan_with_none(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [replace_nan_with_none(item) for item in value]
            elif isinstance(value, np.ndarray):
                 # Properly handle numpy arrays by applying a list comprehension
                return [None if pd.isna(v) else v for v in value]
            elif pd.isna(value):
                return None
            else:
                return value
            
        return {
        'name': self.name,
        'player_id': self.player_id,
        'on_year': self.on_year,
        'ratings_by_year': self.ratings_by_year,
        'match_rating_hist': [replace_nan_with_none(x) for x in self.match_rating_hist],
        'dynamic': [replace_nan_with_none(x) for x in self.dynamic],
        'matches_dict': {date: replace_nan_with_none(match) for date, match in self.matches_dict.items()},
        'df_rating_changes': replace_nan_with_none(self.df_rating_changes.to_dict()) if self.df_rating_changes is not None else None
        }
    
    @classmethod
    def from_dict_all(cls, data):
        def replace_none_with_nan(value):
            """
            Recursively replace None values with NaN in nested dictionaries and lists.
            """
            if isinstance(value, dict):
                return {k: replace_none_with_nan(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [replace_none_with_nan(item) for item in value]
            elif value is None:
                return np.nan
            else:
                return value

        player = cls(data['name'], data['player_id'])
        player.on_year = data['on_year']
        player.ratings_by_year = replace_none_with_nan(data['ratings_by_year'])
        player.match_rating_hist = [replace_none_with_nan(x) for x in data['match_rating_hist']]
        player.dynamic = [replace_none_with_nan(x) for x in data['dynamic']]
        player.matches_dict = {date: replace_none_with_nan(match) for date, match in data['matches_dict'].items()}
        
        if data['df_rating_changes'] is not None:
            player.df_rating_changes = pd.DataFrame.from_dict(replace_none_with_nan(data['df_rating_changes']))
        else:
            player.df_rating_changes = None

        return player
