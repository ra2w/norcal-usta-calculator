"""
Norcal USTA Ratings Calculator - Streamlit App

This Streamlit application provides a user-friendly interface to explore and visualize 
the player ratings data generated by the Norcal USTA Ratings Calculator engine. It allows 
users to look up individual players, view their rating history, and analyze their match 
performance.

Features:
- Player lookup by first and last name
- Display of player's last three dynamic ratings
- Detailed view of player's match history, including opponents and scores
- Calculation and display of performance metrics for each match

Input:
- JSON files generated by the ratings engine:
  - 'player_db_processed_{start_year}.json': Processed player database
  - 'player_db_processed_{start_year}_index.json': Index for efficient player lookup
  - 'name_{start_year}_index.json': Name index for player search

Usage:
streamlit run app.py

The app will open in your default web browser, allowing you to:
1. Select a start year (between 2014 and 2024)
2. Enter a player's first and last name to look up their data
3. View the player's last three dynamic ratings
4. See detailed match history and performance metrics

This application is part of the Norcal USTA Ratings Calculator project and serves
as the front-end interface for exploring the processed ratings data.

Author: Ramu Arunachalam
Email: rar204@gmail.com
Created: July 2024

License
This project is licensed under the MIT License. See the LICENSE file in the repository for the full license text.

"""

# Standard library imports
import os
import sys
import json
import warnings

# Third-party library imports
import ijson
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# Local module imports
from player import Player, NUM_DYNAMIC_RATINGS
from engine import compute_raw_performance_gap, compute_adj_raw_performance_gap, get_non_self_player_dynamic, calculate_self_rate_dynamic, process_normal_match

start_year = 2014

import gzip
import json
import streamlit as st

def load_from_json(key):

    current_file = Path(__file__).resolve()
    current_directory = current_file.parent
    data_directory = current_directory / "data"
    json_filename = f"{data_directory}/player_db_processed_2014.json.gz"
    index_filename = f"{data_directory}/player_db_processed_2014_index.json"

    try:
        with open(index_filename, 'r') as file:
            index = json.load(file)
    except FileNotFoundError:
        st.error(f"Error: Index file not found: {index_filename}")
        return None

    try:
        with gzip.open(json_filename, 'rt') as file:
            file.seek(index[key])
            player_json = file.readline()  # Read the JSON string at the indexed position
            player_data = json.loads(player_json)
            if key in player_data:
                return Player.from_dict_all(player_data[key])
            else:
                st.error(f"Error: Player ID not found in data: {key}")
                return None
    except FileNotFoundError:
        st.error(f"No data found for key: {key}")
        return None
# def load_from_json(key):
#         data_directory = "./data/"
#         json_filename = f"{data_directory}/player_db_processed_{start_year}.json"
#         index_filename = f"{data_directory}/player_db_processed_{start_year}_index.json"

#         try:
#             with open(index_filename, 'r') as file:
#                 index = json.load(file)
#         except FileNotFoundError:
#             st.error(f"Error: Index file not found: {index_filename}")
#             return None
#         try:
#             with open(json_filename, 'r') as file:
#                 file.seek(index[key])
#                 player_json = file.readline()  # Read the JSON string at the indexed position
#                 player_data = json.loads(player_json)
#                 if key in player_data:
#                     return Player.from_dict_all(player_data[key])
#                 else:
#                     st.error(f"Error: Player ID not found in data: {key}")
#                     return None
#         except FileNotFoundError:
#             st.error(f"No data found for key: {key}")
#             return None
        
class LazyLoadDict(dict):
    def __init__(self, *args, **kwargs):
        self.load_func = kwargs.pop('load_func', None)
        super().__init__(*args, **kwargs)

    def get(self, key, default=None):
        if key not in self:
            # Load data from disk
            data = self.load_func(key)
            if data:
                self[key] = data
                return data
            else:
                return default
        return super().get(key, default)


def check_json_structure():
    data_directory = "./data/"

    st.write(f"Checking JSON structure for {start_year}...")
    player_db_filename = os.path.join(data_directory, f"player_db_processed_{start_year}.json")

    with open(player_db_filename, 'rb') as file:
        try:
            # Attempt to iterate over every item to check for errors
            for prefix, event, value in ijson.parse(file):
                print(prefix, event, value)
                break  # You might want to remove or comment out this line after testing
        except Exception as e:
            st.write(f"Error: {e}")
    st.write("JSON structure check complete.")


# Streamlit app function
def player_lookup(first_name, last_name, name_index):
    pid = None

    if first_name and last_name:
        # Create the search key from user input
        search_name = f"{first_name} {last_name}"
        
        # Display matches using radio buttons
        if search_name in name_index:
            player_options = name_index[search_name]
            # List of formatted strings showing "player_id: original_name"
            options = [f"{pid}: {orig_name}" for pid, orig_name in player_options]
            choice = st.radio("Select a player:", options)
            pid = choice.split(":")[0].strip()
            st.text(f"You selected:{choice}")
        else:
            st.write("No players found with that name.")
    return pid


def print_details(details):
    block = ""

    for key, value in details.items():
        value_str = f"{value:.2f}" if isinstance(value, float) else value
        block += f"{key}: {value_str}\n"

    st.code(block)

from content import match_to_html
from jinja2 import Template

def html_display(team_details, score_details, match_details, is_self_rated=False):
    # Data to be inserted into the template
    # make a copy of details and store into data

    team_details = team_details.copy()
    score_details = score_details.copy()
    match_details = match_details.copy()

    #for data in [team_details, score_details, match_details]:
    #    # not iterate through data and convern all floats to 2 decimal places
    #    for key, value in data.items():
    #        if isinstance(value, float):
    #            data[key] = f"{value:.2f}"

    rendered_html = match_to_html(team_details, score_details, match_details, is_self_rated)
    # Display the rendered HTML using st.components.v1.html
    st.components.v1.html(rendered_html, height=600, scrolling=True)

def self_rated_match(row):
    player_db = st.session_state['player_db']
    self_rated_col = row['self_rated_col']

    self_rated_pid = row[f"{self_rated_col}_id"]
    self_rated_player = player_db.get(self_rated_pid, None)

    w1_dynamic, w1_dynamic_text = get_dynamic_ratings(row,'w1')
    w2_dynamic, w2_dynamic_text = get_dynamic_ratings(row,'w2')
    l1_dynamic, l1_dynamic_text = get_dynamic_ratings(row,'l1')
    l2_dynamic, l2_dynamic_text = get_dynamic_ratings(row,'l2')

    team_details = team_to_dict(row, w1_dynamic_text, w2_dynamic_text, l1_dynamic_text, l2_dynamic_text)
    # set avg_rating to Nan
    avg_rating_gap = np.nan
    raw_perf_gap = np.nan

    score_details = scorecard_to_dict(row, avg_rating_gap, raw_perf_gap)

    self_rated_col = row['self_rated_col']
    # Calculate the performance gap based on match result
    raw_perf_gap = row['raw_perf_gap'] if 'w' in self_rated_col else -row['raw_perf_gap']
    num_partners = 1 if 'Singles' in row['Line'] else 2
    
    sr_dynamic = row[f'{self_rated_col}_l3_dynamic_before']
    num_self_ratings = sum(not np.isnan(r) for r in sr_dynamic)

    with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sr_avg_dynamic= np.nanmean(sr_dynamic)

    # Retrieve dynamics of other involved players
    partner_dynamic, o1_dynamic, o2_dynamic = get_non_self_player_dynamic(row, self_rated_col)

    # Using list comprehension and join to build sr_dynamic_text
    sr_dynamic_text = ','.join(
    f"{value:.2f}" for value in sr_dynamic if not np.isnan(value)
    )   

    match_details = {'avg_self_rate_dynamic': f'avg({sr_dynamic_text}) = {sr_avg_dynamic:.2f}'
               }
    match_details['sr_avg_dynamic'] = sr_avg_dynamic
    match_details['Score'] = row['Score']
    # iterate through sr_dynamic
    for i, value in enumerate(sr_dynamic):
        match_details[f'sr_dynamic_{i}'] = value

    # Calculate new dynamic rating for self-rated player
    sr_dynamic = calculate_self_rate_dynamic(
        o1_dynamic, o2_dynamic, partner_dynamic, raw_perf_gap, 
        num_self_ratings, sr_avg_dynamic, num_partners, match_details)


    prefix = self_rated_col
    match_details[f'{prefix}'] = row[f'{prefix}']
    match_details[f'{prefix}_dynamic'] = row[f'{prefix}_dynamic']
    match_details[f'{prefix}_adj'] = match_details[f'adj']
    match_details[f'{prefix}_new_match'] = sr_dynamic

    # l3 --> last 3
    match_details[f'{prefix}_dynamic_calcs'] = (row[f'{prefix}_l3_dynamic_before'], 
                                                row[f'{prefix}_l3_dynamic_after'][-1])

    return (team_details, score_details, match_details)

def regular_match(row):

    w1_dynamic, w1_dynamic_text = get_dynamic_ratings(row,'w1')
    w2_dynamic, w2_dynamic_text = get_dynamic_ratings(row,'w2')
    l1_dynamic, l1_dynamic_text = get_dynamic_ratings(row,'l1')
    l2_dynamic, l2_dynamic_text = get_dynamic_ratings(row,'l2')
    
    team_details = team_to_dict(row, w1_dynamic_text, w2_dynamic_text, l1_dynamic_text, l2_dynamic_text)

    avg_rating_gap = np.nanmean([w1_dynamic, (w2_dynamic if not pd.isna(w2_dynamic) else 0)]) - np.nanmean([l1_dynamic, l2_dynamic if not pd.isna(l2_dynamic) else 0])
    raw_perf_gap = compute_adj_raw_performance_gap(compute_raw_performance_gap(row['Win Ratio']), avg_rating_gap)

    score_details = scorecard_to_dict(row, avg_rating_gap, raw_perf_gap)

    match_details = {}
    process_normal_match(row, match_details)

    match_details['w1'] = row['w1']
    match_details['w2'] = row['w2']
    match_details['l1'] = row['l1']
    match_details['l2'] = row['l2']


    num_partners = 1 if 'Singles' in row['Line'] else 2
    
    p_prefix = []

    if num_partners >= 1:
        p_prefix.append('w1')
        p_prefix.append('l1')
    if num_partners == 2:
        # insert w2 after w1 and before l1
        p_prefix.insert(1, 'w2')
        p_prefix.append('l2')

    for prefix in p_prefix:
        # l3 --> last 3
        match_details[f'{prefix}_dynamic_calcs'] = (row[f'{prefix}_l3_dynamic_before'], 
                                                    row[f'{prefix}_l3_dynamic_after'][-1])
        
    return (team_details, score_details, match_details)


def get_dynamic_ratings(row,position):
    dynamic_col = position + '_dynamic'
    dynamic = row[dynamic_col] if row[dynamic_col] is not None else np.nan
    dynamic_text = f"{row[dynamic_col]:.2f}" if not np.isnan(dynamic) else "N/A"
    return dynamic,dynamic_text


def team_to_dict(row, w1_dynamic_text, w2_dynamic_text, l1_dynamic_text, l2_dynamic_text):
    team_details = {}
    team_details['Match Date'] = row['Match Date']
    w1_text = f"{row['w1']}({row['w1_rating']}: {w1_dynamic_text})"
    team_details['w1'] = w1_text
    if row['w2'] is not None:
        w2_text = f"{row['w2']}({row['w2_rating']}: {w2_dynamic_text})"
        team_details['w2'] = w2_text

    l1_text = f"{row['l1']}({row['l1_rating']}: {l1_dynamic_text})"
    team_details['l1'] = l1_text

    if row['l2'] is not None:
        l2_text = f"{row['l2']}({row['l2_rating']}: {l2_dynamic_text})"
        team_details['l2'] = l2_text
    
    return team_details

def scorecard_to_dict(row, avg_rating_gap, raw_perf_gap):

    win_ratio = float(row['Win Ratio'])*100
    score_details = {'score': row['Score'],
                     'win_ratio': f"{win_ratio:.2f}%",
                    }

    return score_details

########################  Main Streamlit App  ########################
st.title("Ratings Evaluation")
with open(f'data/name_{start_year}_index.json', 'r') as file:
    name_index =  json.load(file)
    

#if st.session_state.get('player_db', None) is None:
st.session_state['player_db'] = LazyLoadDict(load_func=load_from_json)

# User inputs for first and last name
st.sidebar.subheader("Player Lookup")
first_name = st.sidebar.text_input("Enter First Name", "Ramu")
last_name = st.sidebar.text_input("Enter Last Name", "Arunachalam")
pid = player_lookup(first_name, last_name, name_index)


if pid is not None:
    player = st.session_state['player_db'].get(pid, None)
    with st.container(border=True):
        placeholder = st.empty()
        placeholder2 = st.empty()
        
        text = ""
        for label, d in zip(reversed(["Estimate From Two Matches Ago", "Previous Estimated Dynamic Rating"]), reversed(player.get_all_dynamic()[:-1])):
            text = text + f"{label}: {d:.2f}\n"
        st.code(text)

    most_recent = st.sidebar.checkbox("Show most recent matches first", True)
    #begin_year = st.sidebar.slider("Start Year", 2014, 2024, 2014)
    #st.write(begin_year)
    # sort keys by match date
    keys = sorted(player.matches_dict.keys(), key=lambda x: player.matches_dict[x]['row']['Match Date'], reverse=most_recent)


    most_recent_match_date = keys[0] if most_recent else keys[-1]
    with placeholder:
        st.markdown(f"**{first_name} {last_name} ({pid})**")
    with placeholder2:
        st.code(f"Most recent Estimated Dynamic Rating : {player.get_all_dynamic()[-1]:.2f} (as of {most_recent_match_date})")

    st.caption("Match History")
    prev_year = None
    for match_date in keys:
        # Extract the inner dictionary that actually contains the data you want
        row = player.matches_dict[match_date]['row']
        # extract year from player.matches_dict[match_date]['row']['Match Date']
        year = row['Match Date'].split('-')[0]
        if year != prev_year:
            st.code(year)
            prev_year = year

        with st.container(border=True):
            if row['self_rated']:
                team_details, score_details, match_details = self_rated_match(row)
            else:
                team_details, score_details, match_details = regular_match(row)
            for l in ['w1','w2','l1','l2']:
                if row[f"{l}_id"] == pid:
                    dynamic = row[f"{l}_l3_dynamic_after"][-1]
                    break
            st.text(f"{first_name} {last_name} : {dynamic:.2f}")
            with st.expander(f"{row['Match Date']}: Match Details"):
                html_display(team_details, score_details, match_details, row['self_rated'])

else:
    st.warning("No player found!")

st.sidebar.markdown('''<small>[Norcal USTA Calculator v0.95](https://github.com/daniellewisDL/streamlit-cheat-sheet)  
                    | July 2024 | 
                    [Ramu Arunachalam](https://daniellewisdl.github.io/)</small>''', unsafe_allow_html=True)




