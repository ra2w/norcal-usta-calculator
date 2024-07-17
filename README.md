# Norcal USTA Ratings Calculator

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

A comprehensive tool for analyzing and visualizing Norcal USTA player ratings based on match performance data.

## Table of Contents
- [Overview](#overview)
- [Quickstart](#quickstart)
- [Components](#components)
- [Rating Calculation Methodology](#rating-calculation-methodology)
- [Installation and Usage](#installation-and-usage)
- [Data Requirements](#data-requirements)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [Future Developments](#future-developments)
- [License](#license)

## Overview

The Norcal USTA Ratings Calculator is a comprehensive tool designed to analyze match data for Norcal USTA players. It consists of a ratings engine to process player data and a Streamlit app for visualizing player information. This project aims to provide insights into player ratings based on their match performance data.

## Quickstart

To quickly explore the Norcal USTA Ratings:

```bash
git clone https://github.com/yourusername/norcal-usta-calculator.git
cd norcal-usta-calculator
conda create -n usta-ratings python=3.9 && conda activate usta-ratings
pip install -r requirements.txt
streamlit run rating_engine/app.py
```

The app will open in your default web browser. You can then:
- Select a start year (between 2014 and 2024)
- Enter a player's first and last name to look up their data
- View the player's last three dynamic ratings
- See detailed match history, including opponents, scores, and performance metrics

This quickstart uses pre-processed data included in the repository, allowing you to explore player ratings without running the ratings engine.

## Components

### 1. Ratings Engine

The ratings engine processes player data to calculate and update player ratings.

Key Features:
- Sequential processing of matches from a specified start year
- Dynamic updating of player ratings based on match performance
- Handling of self-rated players and various match scenarios
- Generation of comprehensive player rating data

### 2. Streamlit App

The Streamlit app provides a user-friendly interface to view player data and ratings.

Key Features:
- Player lookup by first and last name
- Display of player's last three dynamic ratings
- Detailed view of player's match history
- Calculation and display of performance metrics for each match

## Rating Calculation Methodology

The rating system uses a dynamic approach to calculate player ratings based on match outcomes. Here's a brief overview of the process:

1. Calculate the win ratio for each match based on games won and total games played.
2. Compute the raw performance gap using the win ratio.
3. Adjust the raw performance gap based on the current rating difference between players.
4. Update player ratings based on the adjusted performance gap.
5. Handle special cases for self-rated players with fewer than three matches.

The system takes into account factors such as:
- Singles vs. doubles matches
- Self-rated players vs. computer-rated players
- Progressive adjustment of ratings as more matches are played

For a detailed understanding of the rating calculation, please refer to the `engine.py` file in the project.

## Installation and Usage

### Setting up the Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/norcal-usta-calculator.git
   cd norcal-usta-calculator
   ```

2. Create and activate the Conda environment:
   ```bash
   conda create -n usta-ratings python=3.9
   conda activate usta-ratings
   ```

3. Install dependencies:
   ```bash
   conda install --file requirements.txt
   ```
   
   Note: If some packages are not available via conda, you may need to use pip:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Ratings Engine

1. Activate the Conda environment:
   ```bash
   conda activate usta-ratings
   ```

2. Navigate to the project directory:
   ```bash
   cd path/to/norcal-usta-calculator
   ```

3. Run the ratings engine:
   ```bash
   python rating_engine/engine.py [OPTIONS]
   ```

   Options:
   - `--start_year INTEGER`: The year from which to start processing match data (default: 2014)
   - `--max_iterations INTEGER`: Limit the number of matches processed (-1 for entire dataset, default: -1)

### Running the Streamlit App

1. Ensure your Conda environment is activated:
   ```bash
   conda activate usta-ratings
   ```

2. Navigate to the project directory:
   ```bash
   cd path/to/norcal-usta-calculator
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run rating_engine/app.py
   ```

4. The app will open in your default web browser. You can then use it as described in the Quickstart section.

## Data Requirements

To run the ratings engine and app, you need the following data:

- Match data for Norcal USTA players, including player IDs, names, scores, and initial ratings.
- The data should be preprocessed and stored in the appropriate format in the `data/` directory.

**Important Note:** 
This repository comes with a CSV file containing match data from 2014 to June 2024. This data is sufficient to run the ratings engine and app for this time period. 

If you want to include matches from before 2013 or update the data with more recent matches (after June 2024), you'll need to:
1. Collect those additional matches.
2. Preprocess the new data to match the format of the existing CSV.
3. Add the new data to the existing CSV file in the `data/` directory.


## Output Files

The ratings engine generates the following output files:
- `data/player_db_processed_{start_year}.json`: JSON file containing comprehensive updated ratings and match information for all players
- `data/player_db_processed_{start_year}_index.json`: JSON file with byte-level index for efficient data access
- `data/name_{start_year}_index.json`: JSON file mapping player names to their IDs and original names

## Project Structure

```
norcal-usta-calculator/
│
├── data/
│   ├── all_matches_df_2014.csv
│   ├── name_2014_index.json
│   ├── player_db_processed_2014.json
│   └── player_db_processed_2014_index.json
│
├── rating_engine/
│   ├── app.py
│   ├── engine.py
│   └── player.py
│
├── LICENSE
├── README.md
└── requirements.txt
```

## Contributing

We welcome contributions to the Norcal USTA Ratings Calculator! Here's how you can contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Troubleshooting

If you encounter any issues:
1. Ensure all dependencies are correctly installed.
2. Check that the required data files are present in the `data/` directory.
3. Verify that you're using the correct Python version (3.9 recommended).
4. For app-specific issues, check the Streamlit documentation.

For further assistance, please open an issue on the GitHub repository.

## Future Developments

- Enhanced visualization options in the Streamlit app
- Using ML to fine-tune ratings calculator
- Support for custom rating algorithms

## License

MIT License

Copyright (c) 2024 Ramu Arunachalam

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
