from jinja2 import Template
import streamlit as st
import re
from pathlib import Path
from content_dict import CALCULATION_DICT

def format_calculation(calc):
    def format_part(part):
        if part.startswith('"') and part.endswith('"'):
            return f'<span class="score-red">{part}</span>'
        elif part.lower() == 'nan':
            return f'<span class="nan">{part}</span>'
        elif part.replace('.', '', 1).replace('-', '', 1).isdigit():  # Allow one decimal point and one minus sign
            if part.startswith('-'):
                return f'<span class="number negative">{part}</span>'
            else:
                return f'<span class="number">{part}</span>'
        else:
            return part

    # Handle function calls like avg(x,y)
    calc = re.sub(r'\b(avg|min|max)\b', r'<span class="function-blue">\1</span>', calc)
    
    # Split the string, but keep delimiters
    parts = re.split(r'(\s+|[(),])', calc)
    
    formatted_parts = [format_part(part) for part in parts if part]
    
    return ''.join(formatted_parts)


def match_to_html(team_details, score_details, match_details, match_with_self_rate):
    current_file = Path(__file__).resolve()
    current_directory = current_file.parent
    filename = current_directory / 'content.html'
    with open(filename, 'r') as file:
        template = Template(file.read())

    teams = [{"w": team_details['w1'], "l": team_details['l1']}]
    if match_details['num_partners'] == 2:
        teams.append({"w": team_details['w2'], "l": team_details['l2']})

    calculations = []
    match_updates = []


    # Create the context dictionary
    context = {
        "team_details": team_details,
        "score_details": score_details,
        "match_details": match_details,
        "match_with_self_rate": match_with_self_rate
    }


    if match_with_self_rate:
        calc_key = "self_rated_first_two_ratings" if match_details['first_two_ratings'] else "self_rated_after_two_ratings"
    else:
        calc_key = "regular"
    
    match_type = "doubles" if match_details['num_partners'] == 2 else "singles"

    # Add match-type specific calculations
    calculations.extend([
        {"description": calc["description"], "calculation": format_calculation(calc["calculation"](context))}
        for calc in CALCULATION_DICT[calc_key][match_type]
    ])

    # Add common calculations
    calculations.extend([
        {"description": calc["description"], "calculation": format_calculation(calc["calculation"](context))}
        for calc in CALCULATION_DICT[calc_key]["common"]
    ])


    # Add assertions for singles matches
    if match_type == "singles":
        if match_with_self_rate:
            assert match_details['avg_opponents'] == match_details['o1_dynamic'], "Avg opponents should match o1_dynamic in singles"
        else:
            assert match_details['avg_winning_team'] == match_details['w1_dynamic'], "Avg winning team should match w1_dynamic in singles"
            assert match_details['avg_losing_team'] == match_details['l1_dynamic'], "Avg losing team should match l1_dynamic in singles"

    match_updates = CALCULATION_DICT[calc_key]["match_updates"]["calculation"](context)

    if match_type == "singles":
        if match_with_self_rate:
            assert match_details['avg_opponents'] == match_details['o1_dynamic'], "Avg opponents should match o1_dynamic in singles"
        else:
            assert match_details['avg_winning_team'] == match_details['w1_dynamic'], "Avg winning team should match w1_dynamic in singles"
            assert match_details['avg_losing_team'] == match_details['l1_dynamic'], "Avg losing team should match l1_dynamic in singles"

    render_context = {
        'match_type': 'Self-Rated' if match_with_self_rate else 'Regular',
        'match_date': team_details['Match Date'],
        'score': score_details['score'],
        'win_ratio': score_details['win_ratio'],
        'teams': teams,
        'calculations': calculations,
        'match_updates': match_updates
    }

    return template.render(render_context)

