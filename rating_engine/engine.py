#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Norcal USTA Ratings Calculator

This utility reverse engineers Norcal USTA player ratings from match data. It processes
matches sequentially from a specified start year, updating each player's rating based
on their performance in each match. By the end of its run, every player's rating is
updated to reflect their performance up to their last match.

Input:
- DataFrame titled 'all_matches_df_{start_year}'.
  This DataFrame contains detailed records of all tennis matches from the specified
  year within the Norcal section, excluding specific match types such as combo,
  mixed doubles, and 55 and over leagues.

Processing:
This code equentially processes matches starting from a designated year, updating each player's 
ratings based on their performance in each match. By the end of its run, every player's rating 
is updated to reflect their performance up to their last match.

Output:
- JSON file named 'player_db_processed_{start_year}.json', containing comprehensive
  updated ratings and match information for all players covered by the engine.

Usage:
python rate_engine.py --start_year YEAR --max_iterations ITERATIONS
  --start_year: Start year for collecting match data. Default is 2019.
  --max_iterations: Limit the number of matches processed. Default is -1 (entire dataset).

Next Steps:
After running this utility, you can view the estimated current ratings for any given
player by running:
    streamlit run app.py

This app loads the processed JSON data and displays it through an interactive web
interface, allowing users to explore player ratings and historical performance data
effectively.

This file is part of the Norcal USTA Ratings Calculator project.

Author: Ramu Arunachalam
Email: rar204@gmail.com
Created: July 2024
"""

__author__ = "Ramu Arunachalam"
__email__ = "rar204@gmail.com"
__version__ = "0.95.0"

# MIT License
#
# Copyright (c) 2024 Ramu Arunachalam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import streamlit as st
# Standard library imports
import json
import os
import sys
from typing import Dict

# Third-party imports
import click
import ijson
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# Rich library imports
from rich.console import Console
from rich.layout import Layout
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Local application imports
from player import Player, NUM_DYNAMIC_RATINGS


console = Console()

data = {
    0.000000: -0.479900,
    0.076900: -0.432800,
    0.142800: -0.384600,
    0.200000: -0.337600,
    0.250000: -0.290300,
    0.277700: -0.261200,
    0.294100: -0.242700,
    0.315700: -0.216500,
    0.333300: -0.195300,
    0.350000: -0.173100,
    0.352900: -0.169600,
    0.368400: -0.147300,
    0.380900: -0.128500,
    0.388800: -0.114600,
    0.400000: -0.099600,
    0.409000: -0.092400,
    0.411700: -0.089900,
    0.416600: -0.087000,
    0.421000: -0.081600,
    0.428500: -0.075900,
    0.434700: -0.069800,
    0.437500: -0.065500,
    0.440000: -0.064100,
    0.444400: -0.060200,
    0.450000: -0.054600,
    0.454500: -0.050000,
    0.458300: -0.046500,
    0.461500: -0.043200,
    0.470500: -0.032500,
    0.473600: -0.030100,
    0.476100: -0.027200,
    0.478200: -0.024500,
    0.481400: -0.021300,
    0.500000: -0.000100,
    0.521700: 0.025000,
    0.523800: 0.026700,
    0.526300: 0.029700,
    0.529400: 0.033000,
    0.538400: 0.043000,
    0.541600: 0.046300,
    0.545400: 0.049600,
    0.550000: 0.054000,
    0.555500: 0.059000,
    0.560000: 0.063600,
    0.562500: 0.066600,
    0.565200: 0.068700,
    0.571400: 0.075600,
    0.578900: 0.082000,
    0.583300: 0.085700,
    0.588200: 0.089200,
    0.590900: 0.092100,
    0.600000: 0.099800,
    0.611100: 0.118000,
    0.619000: 0.128800,
    0.631500: 0.147400,
    0.647000: 0.168400,
    0.650000: 0.172800,
    0.666600: 0.195200,
    0.684200: 0.217000,
    0.705800: 0.242500,
    0.722200: 0.261900,
    0.750000: 0.290100,
    0.800000: 0.337500,
    0.857100: 0.384900,
    0.923000: 0.432300,
    1.000000: 0.479400
}


def compute_raw_performance_gap(win_ratio):

    return win_ratio-0.5

    keys = sorted(data.keys())
    # Find the position where to insert the ratio so that the list remains sorted
    for i in range(1, len(keys)):
        if keys[i] >= win_ratio:
            lower_key = keys[i-1]
            upper_key = keys[i]
            
            # Linear interpolation formula:
            # value = value_lower + (ratio - key_lower) * ((value_upper - value_lower) / (key_upper - key_lower))
            interpolated_value = (data[lower_key] +
                                  ((win_ratio - lower_key) / (upper_key - lower_key)) *
                                  (data[upper_key] - data[lower_key]))
            return interpolated_value
    
    # If the ratio is above the highest key, return the value associated with the highest key
    return data[keys[-1]]

def compute_adj_raw_performance_gap(raw_perf_gap, current_rating_gap):
    if raw_perf_gap >= 0:
        if raw_perf_gap > 0.385:
            return max(raw_perf_gap, current_rating_gap)
        elif current_rating_gap > 0.385:
            return raw_perf_gap*current_rating_gap/0.385

    else:
        # raw_perf_gap < 0
        if raw_perf_gap < -0.385:
            return min(raw_perf_gap, current_rating_gap)
        elif current_rating_gap < -0.385:
            return raw_perf_gap*current_rating_gap/(-0.385)

    return raw_perf_gap

def print_dataframe(df):
    """Prints a pandas DataFrame using the Rich library to format as a table."""
    table = Table(show_header=True, header_style="bold magenta")

    # Add DataFrame columns as table columns
    for column in df.columns:
        table.add_column(column, style="dim")

    # Add DataFrame rows to table
    for _, row in df.iterrows():
        table.add_row(*[str(item) for item in row.values])

    console.print(table)

def calculate_win_ratio(score_string):
    games_won = 0
    total_games = 0

    # Remove any retirement indicator to proceed with normal score processing
    clean_score_string = score_string.replace(' RE', '')
    # Split the score string into individual sets
    sets = clean_score_string.split(',')

    for set_score in sets:
        if '-' in set_score:
            games = set_score.split('-')
            try:
                # Convert scores to integers
                games_won_team = int(games[0])
                games_lost_team = int(games[1])

                # Add the current set's games to totals
                games_won += games_won_team
                total_games += games_won_team + games_lost_team
            except ValueError:
                # Handle the case where the score might not be a valid integer
                print("Invalid score found:", set_score, score_string)
                continue

    return games_won/total_games

def process_normal_match(row, details=None):
    w1_r = row['w1_dynamic']
    w2_r = row['w2_dynamic']
    l1_r = row['l1_dynamic']
    l2_r = row['l2_dynamic']

    rating_gap = w1_r + (w2_r if not pd.isna(w2_r) else 0) - l1_r - (l2_r if not pd.isna(l2_r) else 0)
    if not pd.isna(w2_r):
        # Doubles
        if pd.isna(l2_r):
            breakpoint()
        rating_gap = rating_gap / 2

    break_p = False
    break_pid = '197576'
    if (break_p and (row['w1_id'] == break_pid or 
                     (not pd.isna(row['w2_id']) and row['w2_id'] == break_pid) or 
                     (row['l1_id'] == break_pid) or 
                     (not pd.isna(row['l2_id']) and row['l2_id'] == break_pid))):
        breakpoint()

    #row_as_df = row.to_frame().transpose()
    #if (row['w1_id'] == '98136' and not pd.isna(row['w2_id']) and row['w2_id'] == '124331'):
    #    breakpoint()

    raw_perf_gap = compute_raw_performance_gap(row['Win Ratio'])
    adj_perf_gap = compute_adj_raw_performance_gap(compute_raw_performance_gap(row['Win Ratio']), rating_gap)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        adj = np.nanmean([row['l1_dynamic'], row['l2_dynamic']]) + adj_perf_gap - np.nanmean([row['w1_dynamic'], row['w2_dynamic']])
    
    row['raw_perf_gap'] = adj_perf_gap
    row['adj'] = adj

    row['w1_new_match'] = (adj / 1) + row['w1_dynamic']
    row['w2_new_match'] = (adj / 1) + row['w2_dynamic'] if not pd.isna(row['w2_dynamic']) else np.nan
    row['l1_new_match'] = row['l1_dynamic'] - (adj / 1)
    row['l2_new_match'] = row['l2_dynamic'] - (adj / 1) if not pd.isna(row['l2_dynamic']) else np.nan

    num_partners = 1 if 'Singles' in row['Line'] else 2

    if details is not None:
        details['rating_gap'] = rating_gap
        details['perf_gap'] = raw_perf_gap
        details['gap_adjustment'] = adj_perf_gap - raw_perf_gap
        details['adj_perf_gap'] = adj_perf_gap
        details['w1_dynamic'] = w1_r
        details['w2_dynamic'] = w2_r
        details['avg_winning_team'] = np.nanmean([row['w1_dynamic'], row['w2_dynamic']])
        details['avg_losing_team'] = np.nanmean([row['l1_dynamic'], row['l2_dynamic']])
                                                 
        details['l1_dynamic'] = l1_r
        details['l2_dynamic'] = l2_r
        details['adj'] = adj
        details['w1_adj'] = adj
        details['w2_adj'] = adj
        details['l1_adj'] = -adj
        details['l2_adj'] = -adj

        details['w1_new_match'] = row['w1_new_match']
        details['w2_new_match'] = row['w2_new_match']
        details['l1_new_match'] = row['l1_new_match']
        details['l2_new_match'] = row['l2_new_match']
        details['num_partners'] = num_partners
        
    if (row['l1_new_match'] < 1.0 or row['l2_new_match'] < 1.0 or row['w1_new_match'] < 1.0 or row['w2_new_match'] < 1.0):
        breakpoint()
    return row

# Pull from player_db to row
def pull_ratings_data(row, player_db):
    # Extract dynamic ratings and ids for players and opponents
    for role in ['w1', 'w2', 'l1', 'l2']:
        player_id = row[f'{role}_id']
        if pd.notna(player_id):
            player = player_db.get(player_id)
            if player:
                row[f'{role}_dynamic'] = player.get_latest_dynamic(row['Year'])
                row[f'{role}_rating'] = player.rating_to_str(player.get_rating_for_year(row['Year']))
            else:
                row[f'{role}_dynamic'] = np.nan
                row[f'{role}_rating'] = np.nan
        else:
            row[f'{role}_dynamic'] = np.nan
            row[f'{role}_rating'] = np.nan
    return row

def get_self_rated_and_partner_prefix(self_rated_col):
    return ('w','w') if 'w' in self_rated_col else ('l','l')

def get_opponent_prefix(self_rated_col):
    return 'l' if 'w' in self_rated_col else 'w'

def get_self_rated_and_partner_numbers(self_rated_col):
    # '1' or '2' from 'w1' or 'w2', etc.
    return (self_rated_col[-1], (int(self_rated_col[-1]) % 2) + 1)  # Toggle between '1' and '2'

def get_self_rated_and_partner_labels(self_rated_col):
    sr_prefix, partner_prefix = get_self_rated_and_partner_prefix(self_rated_col)
    sr_number, partner_number = get_self_rated_and_partner_numbers(self_rated_col)
    return f"{sr_prefix}{sr_number}", f"{partner_prefix}{partner_number}"

def get_opponent_labels(self_rated_col):
    o_prefix = get_opponent_prefix(self_rated_col)
    return (f"{o_prefix}1", f"{o_prefix}2")

def get_non_self_player_dynamic(row, self_rated_col):
    o1_label, o2_label = get_opponent_labels(self_rated_col)
    o1_dynamic = row[f"{o1_label}_dynamic"]
    o2_dynamic = row[f"{o2_label}_dynamic"]
    _, partner_label = get_self_rated_and_partner_labels(self_rated_col)
    partner_dynamic = row[f"{partner_label}_dynamic"]

    return (partner_dynamic, o1_dynamic, o2_dynamic)

def calculate_self_rate_dynamic(o1_dynamic, o2_dynamic, partner_dynamic, raw_perf_gap, num_self_ratings, sr_avg_dynamic, num_partners, details=None):
    """
    Calculates the dynamic rating for a self-rated player.
    - o1_dynamic, o2_dynamic: Dynamic ratings of the opponents.
    - partner_dynamic: Dynamic rating of the partner.
    - raw_perf_gap: Performance gap to adjust ratings.
    - sp: Player object containing rating count and methods.
    - num_partners: Number of partners (usually 1 or 2).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)


        if num_self_ratings in (0, 1):
            # Calculate when there are fewer than 2 existing ratings
            avg_opponents = np.nanmean([o1_dynamic, o2_dynamic])
            adjusted_partner_dynamic = 0 if np.isnan(partner_dynamic) else partner_dynamic
            sr_dynamic = avg_opponents * num_partners + raw_perf_gap*num_partners - adjusted_partner_dynamic

            if details is not None:
                details.update({
                'first_two_ratings':True,
                'num_partners':num_partners,
                'o1_dynamic': o1_dynamic,
                'o2_dynamic': o2_dynamic,
                'avg_opponents': avg_opponents,
                'avg_self_rated_team': np.nan,
                'partner_dynamic': adjusted_partner_dynamic,
                'perf_gap': raw_perf_gap,
                'adj':raw_perf_gap*num_partners,
                'sr_dynamic': sr_dynamic,
                })
                

        elif num_self_ratings == 2:
            # Adjust rating when there are exactly 2 existing ratings
            avg_opponents = np.nanmean([o1_dynamic, o2_dynamic])
            avg_sr_team = np.nanmean([sr_avg_dynamic, partner_dynamic])
            rating_gap = avg_sr_team - avg_opponents
            adj_perf_gap = compute_adj_raw_performance_gap(raw_perf_gap, rating_gap)
            delta = adj_perf_gap - rating_gap
            sr_dynamic = delta + sr_avg_dynamic

            if details is not None:
                details.update({
                    'first_two_ratings':False,
                    'num_partners':num_partners,
                    'o1_dynamic': o1_dynamic,
                    'o2_dynamic': o2_dynamic,
                    'avg_opponents': avg_opponents,
                    'avg_self_rated_team': avg_sr_team,
                    'partner_dynamic': partner_dynamic,
                    'rating_gap': rating_gap,
                    'perf_gap': raw_perf_gap,
                    'gap_adjustment' : adj_perf_gap - raw_perf_gap,
                    'adj_perf_gap': adj_perf_gap,
                    'adj':delta,
                    'sr_dynamic': sr_dynamic,
     
                })


        else:
            # Handle other unexpected cases of rating counts
            raise ValueError(f"Error: self-rated player has an unexpected number of ratings: {num_self_ratings}")


    return sr_dynamic


def process_self_rated_match(row, player_db, self_rated_col):
    """
    Process a tennis match for a self-rated player and update dynamic ratings.
    
    Args:
        row : Data row containing match and player details.
        player_db (dict): Database of player stats and historical ratings.
        self_rated_col (str): Column identifier for the self-rated player.

    Returns:
        Updated row with new dynamic ratings.
    """

    # Assign labels for updating row entries
    sr_label, partner_label = get_self_rated_and_partner_labels(self_rated_col)
    o1_label, o2_label = get_opponent_labels(self_rated_col)

    # Calculate the performance gap based on match result
    raw_perf_gap = row['raw_perf_gap'] if 'w' in self_rated_col else -row['raw_perf_gap']
    num_partners = 1 if 'Singles' in row['Line'] else 2
    
    # Retrieve dynamics of other involved players
    partner_dynamic, o1_dynamic, o2_dynamic = get_non_self_player_dynamic(row, self_rated_col)
    
    # Fetch self-rated player's previous ratings and count
    sp_id = row[f"{sr_label}_id"]
    sp = player_db[sp_id]
    num_self_ratings = sp.count_ratings()
    sr_avg_dynamic = sp.get_avg_self_rate_dynamic()

    # Calculate new dynamic rating for self-rated player
    sr_dynamic = calculate_self_rate_dynamic(
        o1_dynamic, o2_dynamic, partner_dynamic, raw_perf_gap, 
        num_self_ratings, sr_avg_dynamic, num_partners)

    # Update the dynamic ratings in the row
    row[f"{sr_label}_new_match"] = sr_dynamic
    row[f"{partner_label}_new_match"] = partner_dynamic
    row[f"{o1_label}_new_match"] = o1_dynamic
    row[f"{o2_label}_new_match"] = o2_dynamic

    return row

# Update player_db from row
def update_player_ratings(row, player_db, player_ids):

    dynamic_keys = [f'{p.rsplit("_id", 1)[0]}_new_match' for p in player_ids]

    for pid_key, dyn_key in zip(player_ids, dynamic_keys):
        player_id = row[pid_key]
        if pd.notna(player_id) and player_id in player_db:

            new_match_rating = row[dyn_key]
            #if new_dynamic_rating > 5.15:
            #    breakpoint()
            if not pd.isna(new_match_rating):
                player_db[player_id].add_new_match_rating(new_match_rating, row)

def process_one_match(row, player_db):
    self_rated_count = 0
    self_rated_col = None
    
    row['self_rated'] = False
    row['self_rated_pid'] = np.nan
        
    for c in ['w1', 'w2', 'l1', 'l2']:
        p_id = row[c+'_id']
        pname = row[c]
 
        if p_id is None or pd.isna(p_id):
            continue

        if p_id not in player_db:
            player_db[p_id] = Player(pname, p_id)

        # idempotent. sets new year in case the year has changed and updates ratings (if needed)
        # for example, a player maybe self-rated in the new year, in that case we need to erase the old ratings
        # a player maybe 'D' or 'A' rated in the new year, in that case we need to erase the old ratings and appropriately update the new ratings
        # (see set_year in player.py) for more details
        player_db[p_id].set_year(row['Year'], row[c+'_rating'])

        row[c+'_l3_dynamic_before'] = player_db[p_id].get_all_dynamic()

        if player_db[p_id].is_self_rated(row['Year']):
            if player_db[p_id].count_ratings() >= 3:
                # We have a self-rated player that has played 3 or matches so we can treat as normal
                continue
            self_rated_col = c
            self_rated_count += 1

    if self_rated_count > 1:
        # ignore matches with more than one self-rated player
        #print(f"Ignoring match with more than one self-rated player: {row}")
        #if (row['w1'] == 'Anderson,Scott'):
        #    breakpoint()
        return row
    
    
    ## Self-rated match
    if self_rated_count == 1:
        row['self_rated'] = True
        row['self_rated_col'] = self_rated_col
        row = pull_ratings_data(row, player_db)
        row = process_self_rated_match(row, player_db, self_rated_col)
        # only update the self-rated player's ratings
        update_player_ratings(row, player_db, [self_rated_col+'_id'])

    else:
        # Normal match
        # Prepare data      
        row = pull_ratings_data(row, player_db)
        row = process_normal_match(row)
        update_player_ratings(row, player_db, ['w1_id', 'w2_id', 'l1_id', 'l2_id'])

    # Add the new 3 ratings to the row
    for c in ['w1', 'w2', 'l1', 'l2']:
        p_id = row[c+'_id']
        if p_id is not None and not pd.isna(p_id) and p_id in player_db:
            row[c+'_l3_dynamic_after'] = player_db[p_id].get_all_dynamic()

    # save row in each player object
    srow = row.copy()
    for c in ['w1', 'w2', 'l1', 'l2']:
        p_id = srow[c+'_id']
        if p_id is not None and not pd.isna(p_id) and p_id in player_db:
            player_db[p_id].save_match_row(srow)

    return row

def filter_out_violations(df):
    # Check for '(Rules Violation)' in each relevant column
    conditions = (
        df['Winner1'].str.contains("\(Rules Violation\)", na=False) |
        df['Winner2'].str.contains("\(Rules Violation\)", na=False) |
        df['Loser1'].str.contains("\(Rules Violation\)", na=False) |
        df['Loser2'].str.contains("\(Rules Violation\)", na=False) |

        df['Winner1'].str.contains("\(Three Strikes Disqualification\)", na=False) |
        df['Winner2'].str.contains("\(Three Strikes Disqualification\)", na=False) |
        df['Loser1'].str.contains("\(Three Strikes Disqualification\)", na=False) |
        df['Loser2'].str.contains("\(Three Strikes Disqualification\)", na=False)
    )
    
    # Filter the DataFrame based on the conditions
    violation_df = df[conditions]
    # remove violation_df from df
    df = df[~conditions]

    # remove any rows where Winner1 or Loser1 or Winner2 or Loser 2 = 'player, tennis'
    conditions = (
        df['Winner1'].str.contains("player,tennis", na=False) |
        df['Winner2'].str.contains("player,tennis", na=False) |
        df['Loser1'].str.contains("player,tennis", na=False) |
        df['Loser2'].str.contains("player,tennis", na=False)
    )
    df = df[~conditions]

    return df

def load_filter_and_consolidate_matches(start_year):
    """
    Consolidates tennis match data to address a specific quirk where each match is recorded twice in the dataset: 
    once with the end-year ratings for one team and again with the end-year ratings for the opposing team. 
    This function merges these two entries into a single comprehensive record, ensuring complete rating information for both teams in one entry.

    Detailed Steps:
    - **Data Loading**: Loads the match data from a CSV file for the specified start year.
    - **Preprocessing**: Applies sorting by match date and basic data integrity checks.
    - **Identifying Duplicates**: Identifies duplicate match entries based on consistent match identifiers like match date, location, and score, which due to a quirk in the collection process, 
    result in two entries per match.
    - **Merging Entries**: Merges pairs of entries for each match, combining rating information from both to ensure that each match is represented by a single, complete record.
    - **Final Sorting**: Sorts the merged entries by match date to ensure the data is in chronological order for any subsequent analysis.

    This method effectively eliminates the redundancies caused by the scraping quirk and is essential for ensuring the data's accuracy and reliability for performance analytics and rating assessments.

    Parameters:
        start_year (int): The year for which the match data needs to be consolidated and cleaned.

    Returns:
        DataFrame: A cleaned and consolidated DataFrame where each match is uniquely represented, incorporating comprehensive rating data for both teams involved.
    """
    # Implementation would include the steps outlined: loading data, preprocessing, identifying duplicate entries based on match identifiers, merging these entries, and re-sorting the data.

    console.print(f"\nLoading Matches from {start_year} onwards:", style="bold magenta")
    
    data_directory = "./data/"
    df_filename = os.path.join(data_directory, f"all_matches_df_{start_year}.csv")
    with console.status(f"Fetching {df_filename}...", spinner="dots"):
        combined_df = pd.read_csv(df_filename)
    console.print(f"Fetching {df_filename}...done",style="bold blue")



    with console.status(f"Fetching {df_filename}...", spinner="dots"):
        df = pd.read_csv(df_filename)
    

    # Convert 'Match Date' to datetime format, specify the exact format to increase conversion efficiency
    combined_df['Match Date'] = pd.to_datetime(combined_df['Match Date'], format='%m/%d/%Y', errors='coerce')
    combined_df['Year'] = combined_df['Match Date'].dt.year

    combined_df['Match Date'] = combined_df['Match Date'].dt.strftime('%Y-%m-%d')

    combined_df = combined_df.sort_values(by='Match Date', ascending=True)
    with console.status(f"Filtering out matches with violations", spinner="dots"):
        combined_df = filter_out_violations(combined_df)
    console.print(f"Filtering out matches with violations...done",style="bold blue")

    with console.status(f"Removing duplicates and consolidating", spinner="dots"):
        # Partition the DataFrame into singles and doubles
        singles_df = combined_df[combined_df['Line'].str.contains('Singles')]
        doubles_df = combined_df[combined_df['Line'].str.contains('Doubles')]

        # Process singles matches
        # Define key columns and aggregation dictionary for singles
        singles_keys = ['Match Date', 'Line', 'Winner1','Winner1_ID', 'Loser1','Loser1_ID', 'Score', 'Year', 'Home Team', 'Visiting Team']
        singles_agg_dict = {
            'Winner1_Rating': 'first',
            'Loser1_Rating': 'first',
            
        }
        singles_df = singles_df.groupby(singles_keys).agg(singles_agg_dict).reset_index()

        # Process doubles matches
        # Define key columns and aggregation dictionary for doubles
        doubles_keys = ['Match Date', 'Line', 'Winner1', 'Winner1_ID', 'Winner2','Winner2_ID', 'Loser1', 'Loser1_ID', 'Loser2', 'Loser2_ID', 'Score', 'Year', 'Home Team', 'Visiting Team']
        doubles_agg_dict = {
            'Winner1_Rating': 'first',
            'Winner2_Rating': 'first',
            'Loser1_Rating': 'first',
            'Loser2_Rating': 'first'
        }
        doubles_df = doubles_df.groupby(doubles_keys).agg(doubles_agg_dict).reset_index()

        # Merge singles and doubles dataframes
        combined_df = pd.concat([singles_df, doubles_df], ignore_index=True)
        combined_df = combined_df.sort_values(by='Match Date', ascending=True)

        combined_df = combined_df.sort_values(by='Match Date', ascending=True)
    console.print(f"Removing duplicates and consolidating...done",style="bold blue")  
    # Convert player id columns to strings, removing any trailing '.0'
    id_columns = ['Winner1_ID', 'Winner2_ID', 'Loser1_ID', 'Loser2_ID']
    for column in id_columns:
        combined_df[column] = combined_df[column].apply(lambda x: str(int(x)) if pd.notna(x) and isinstance(x, float) else str(x) if pd.notna(x) else np.nan)

    combined_df['Win Ratio'] = combined_df['Score'].apply(calculate_win_ratio)
    combined_df['raw_perf_gap'] = combined_df.apply(lambda row: compute_raw_performance_gap(row['Win Ratio']), axis=1)

    # Mapping old names to new names where 'Winner' is replaced by 'w_' and 'Loser' by 'l_'
    new_columns = {col: col.replace('Winner', 'w').replace('Loser', 'l') for col in combined_df.columns}
    combined_df = combined_df.rename(columns=new_columns)
    combined_df.columns = [col.replace('ID', 'id') for col in combined_df.columns]
    combined_df.columns = [col.replace('Rating', 'rating') for col in combined_df.columns]


    for c in ['w1', 'w2', 'l1', 'l2']:
        combined_df[c+'_dynamic'] = np.nan
        combined_df[c+'_new_match'] = np.nan
 

    return combined_df


def create_json_with_byte_index(player_db, start_year):
    data_directory = "./data"
    json_filename = f"{data_directory}/player_db_processed_{start_year}.json"
    index_filename = f"{data_directory}/player_db_processed_{start_year}_index.json"
    
    index = {}
    with console.status(f"Saving ratings data to {json_filename}...", spinner="dots"):
        with open(json_filename, 'w') as file:
            current_position = 0
            for player_id, player in player_db.items():
                player_json = json.dumps({player_id: player.to_dict_all()})
                file.write(player_json + "\n")  # Write each player's JSON followed by a newline
                index[player_id] = current_position
                current_position += len(player_json) + 1  # Update position after each player

    console.print(f"Saving ratings data to {json_filename}...done",style="bold blue")

    with console.status(f"Writing byte-level index to {index_filename}...", spinner="dots"):
        # Write index data
        with open(index_filename, 'w') as file:
            json.dump(index, file)
    console.print(f"Writing byte-level index to {index_filename}...done",style="bold blue")
    return

# write a description of the function
# This function creates an index of player names to player IDs
# The index is a dictionary where the key is the player name and the value is a list of tuples
# Each tuple contains the player ID and the original name
# The original name is the name as it appears in the dataset
# The player name is reformatted to "First Last"
# The index is used to quickly find player IDs given a player name

def create_name_index(player_db):
    name_index = {}
    
    for player_id, player in player_db.items():
        original_name = player.name  # Assuming this is "Last,First M"
        # Splitting the name into components
        parts = original_name.split(',')
        last_name = parts[0].strip()  # Ensure any extra spaces are removed
        try:
            first_parts = parts[1].strip().split(' ')
        except:
            print(f"Error in {player_id} {original_name}")
            breakpoint()
        first_name = first_parts[0].strip()  # Ensure any extra spaces are removed
        
        # Reformat to "First Last"
        new_name = f"{first_name} {last_name}"
        
        # Add to index
        if new_name not in name_index:
            name_index[new_name] = []
        name_index[new_name].append((player_id, original_name))
    
    return name_index



### Main function
@click.command()
@click.option('--start_year', prompt='Start year', default=2019, help='The year from which to start collecting match data.')
@click.option('--max_iterations', default=-1, help='How many iterations to run. Default is -1 which is for the entire dataset.')
def main(start_year, max_iterations):

    input_details = '''\n- DataFrame titled 'all_matches_df_{start_year}'. This DataFrame contains detailed records of all tennis matches from the specified year within the Norcal section. The data excludes specific match types such as combo, mixed doubles, and 55 and over leagues.\n
'''
    engine_details = ''' This utility reverse engineers USTA player ratings from match data. It sequentially processes matches starting from a designated year, updating each player's ratings based on their performance in each match. By the end of its run, every player's rating is updated to reflect their performance up to their last match.\n
'''
    output_details = '''\n- Upon processing, the engine outputs a JSON file named 'player_db_processed_{start_year}.json'. This file includes comprehensive updated ratings and match information for all players covered by the engine.
'''

    next_steps = '''\nOnce you've run this utility you can the estimated current ratings for any given player by running :
'''

    # Create a rich text object to hold all the content for a single panel
    info_text = Text.assemble(
    ("Welcome to Norcal USTA Ratings Calculator!\n", "bold magenta"),
    ("Version: ", "bold"), "0.95.0\n",
    ("Last Updated: ", "bold"), "July 2024\n\n",
    
    ("Purpose:", "bold underline"), engine_details,
    ("Usage:", "bold underline"), ("- python rate_engine.py --start_year YEAR --max_iterations ITERATIONS\n", "italic"),
    ("  --start_year: ", "bold"), "Start year for collecting match data. Default is 2019.\n",
    ("  --max_iterations: ", "bold"), "Limit the number of matches processed. Default is -1 which means process the entire dataset.\n\n",
    ("Examples:\n", "bold"),
    ("- To process all matches from 2020 with the default iteration limit:\n", "italic"),
    ("  python rate_engine.py --start_year 2020\n", "bold"),
    ("- To process 50 matches starting from 2019:\n", "italic"),
    ("  python rate_engine.py --start_year 2019 --max_iterations 50", "bold"),
    ("\n\nInput:", "bold underline"), input_details,
    ("Output:", "bold underline"), output_details,
    ("\nNext Steps:", "bold underline"), next_steps,
    ("streamlit run app.py\n", "bold italic"),
    ("\nThis app loads the processed JSON data and displays it through an interactive web interface, allowing users to explore player ratings and historical performance data effectively.\n"),
    justify="left"
    )
    # Create a panel with the constructed text
    panel = Panel(info_text, expand=True)
    console.print(panel)

    #df, player_db = load_all(info_text, start_year)

    
    df = load_filter_and_consolidate_matches(start_year)
    player_db = {}
    if max_iterations > 0:
        df = df.head(max_iterations)


    # get number of rows in df
    df_len = len(df)
    console.print(f"The rating engine is processing [bold green]{df_len:,} matches[/bold green]:")
    tqdm.pandas()
    df = df.progress_apply(process_one_match, axis=1, args = (player_db,))

    # dump df to disk
    df.to_csv("./data/processed_matches.csv", index=False)

    create_json_with_byte_index(player_db, start_year)

    name_index = create_name_index(player_db)
    # Now write the index to a JSON file
    console.print(f"\nSaving profrom {start_year} onwards:", style="bold magenta")
    with console.status(f"Saving Player dictionary to ./data/name_{start_year}_index.json...", spinner="dots"):
        with open(f"./data/name_{start_year}_index.json", 'w') as file:
            json.dump(name_index, file, indent=4)

    console.print(f"Saving Player dictionary to ./data/name_{start_year}_index.json...done",style="bold blue")


if __name__ == '__main__':
    main()


