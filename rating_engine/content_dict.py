import numpy as np

CALCULATION_DICT = {
    "self_rated_first_two_ratings": {
        "doubles": [
            {
                "description": "Opponent rating avg.",
                "calculation": lambda context: f"=avg({context['match_details']['o1_dynamic']:.2f}, {context['match_details']['o2_dynamic']:.2f})\n={context['match_details']['avg_opponents']:.2f}"
            },
            {
                "description": "Partner's rating",
                "calculation": lambda context: f"={context['match_details']['partner_dynamic']:.2f}"
            }
        ],
        "singles": [
            {
                "description": "Opponent rating",
                "calculation": lambda context: f"={context['match_details']['o1_dynamic']:.2f}"
            }
        ],
        "common": [
            {
                "description": "Self-rated player avg.",
                "calculation": lambda context: "N/A (First two ratings are independently calculated)"
            },
            {
                "description": "Measured gap",
                "calculation": lambda context: f'="{context["score_details"]["score"]}" --> {context["match_details"]["perf_gap"]:.4f}'
            },
            {
                "description": "Rating for self-rated player",
                "calculation": lambda context: (
                    f'={context["match_details"]["avg_opponents"]:.2f}' +
                    (f'*{context["match_details"]["num_partners"]:.2f}' if context["match_details"]["num_partners"] > 1 else '') +
                    f' + ({context["match_details"]["perf_gap"]:.4f})' +
                    (f'*{context["match_details"]["num_partners"]:.2f}' if context["match_details"]["num_partners"] > 1 else '') +
                    (f' - {context["match_details"]["partner_dynamic"]:.2f}' if context["match_details"]["num_partners"] > 1 else '') +
                    f'\n={context["match_details"]["sr_dynamic"]:.2f}'
                )
            }
        ],
        "match_updates": {
            "calculation": lambda context: [
                {
                    "name": f"{context['match_details'][p]}",
                    "match_rating": f"={context['match_details'][f'{p}_new_match']:.2f}",
                    "dynamic_rating": f"[<span class='new-match'>{context['match_details'][f'{p}_new_match']:.2f}</span>,{','.join(f'{x:.2f}' for x in context['match_details'][f'{p}_dynamic_calcs'][0] if not np.isnan(x))}]</span>"
                }
                for p in ['w1', 'w2', 'l1', 'l2']
                if f"{p}_dynamic_calcs" in context['match_details']
            ]
        }     
    },
    "self_rated_after_two_ratings": {
        "doubles": [
            {
                "description": "Opponent rating avg.",
                "calculation": lambda context: f"=avg({context['match_details']['o1_dynamic']:.2f}, {context['match_details']['o2_dynamic']:.2f})\n={context['match_details']['avg_opponents']:.2f}"
            },
            {
                "description": "Partner's rating",
                "calculation": lambda context: f"={context['match_details']['partner_dynamic']:.2f}"
            }
        ],
        "singles": [
            {
                "description": "Opponent rating",
                "calculation": lambda context: f"={context['match_details']['o1_dynamic']:.2f}"
            }
        ],
        "common": [
            {
                "description": "Self-rated player avg.",
                "calculation": lambda context: f"=avg({context['match_details']['sr_dynamic_0']:.2f}, {context['match_details']['sr_dynamic_1']:.2f}, {context['match_details']['sr_dynamic_2']:.2f})\n= {context['match_details']['sr_avg_dynamic']:.2f}"
            },
            {
                "description": "Self-rate <u>team</u> avg.",
                "calculation": lambda context: f"=avg({context['match_details']['sr_avg_dynamic']:.2f}, {context['match_details']['partner_dynamic']:.2f})\n= {context['match_details']['avg_self_rated_team']:.2f}"
            },
            {
                "description": "Rating Gap",
                "calculation": lambda context: f"={context['match_details']['avg_self_rated_team']:.2f} - {context['match_details']['avg_opponents']:.2f}\n= {context['match_details']['rating_gap']:.2f}"
            },
            {
                "description": "Measured gap",
                "calculation": lambda context: f'="{context["score_details"]["score"]}" --> {context["match_details"]["perf_gap"]:.4f}'
            },
            {
                "description": "Measured Gap (adjusted)",
                "calculation": lambda context: f"={context['match_details']['perf_gap']:.4f}" + (f"+ {context['match_details']['gap_adjustment']:.4f}\n ={context['match_details']['adj_perf_gap']:.4f}")
            },
            {
                "description": "Rating for self-rated player",
                "calculation": lambda context: f'={context["match_details"]["sr_avg_dynamic"]:.2f} + ({context["match_details"]["perf_gap"]:.2f} - {context["match_details"]["rating_gap"]:.2f})\n={context["match_details"]["sr_dynamic"]:.2f}'
            }
        ],
        "match_updates": {
            "calculation": lambda context: [
                {
                    "name": f"{context['match_details'][p]}",
                    "match_rating": f"={context['match_details'][f'{p}_new_match']:.2f}",
                    "dynamic_rating": f"[<span class='new-match'>{context['match_details'][f'{p}_new_match']:.2f}</span>,{','.join(f'{x:.2f}' for x in context['match_details'][f'{p}_dynamic_calcs'][0] if not np.isnan(x))}]</span>"
                }
                for p in ['w1', 'w2', 'l1', 'l2']
                if f"{p}_dynamic_calcs" in context['match_details']
            ]
        }   
    },

    "regular": {
        "doubles": [
            {
                "description": "Winner rating avg.",
                "calculation": lambda context: f"=avg({context['match_details']['w1_dynamic']:.2f}, {context['match_details']['w2_dynamic']:.2f})\n={context['match_details']['avg_winning_team']:.2f}"
            },
            {
                "description": "Losers rating avg.",
                "calculation": lambda context: f"=avg({context['match_details']['l1_dynamic']:.2f}, {context['match_details']['l2_dynamic']:.2f})\n={context['match_details']['avg_losing_team']:.2f}"
            }
        ],
        "singles": [
            {
                "description": "Winner rating",
                "calculation": lambda context: f"={context['match_details']['w1_dynamic']:.2f}"
            },
            {
                "description": "Loser rating",
                "calculation": lambda context: f"={context['match_details']['l1_dynamic']:.2f}"
            }
        ],
        "common": [
            {
                "description": "Rating Gap",
                "calculation": lambda context: f"={context['match_details']['avg_winning_team']:.2f} - {context['match_details']['avg_losing_team']:.2f}\n={context['match_details']['rating_gap']:.4f}"
            },
            {
                "description": "Measured gap",
                "calculation": lambda context: f'="{context["score_details"]["score"]}" --> {context["match_details"]["perf_gap"]:.4f}'
            },
            {
                "description": "Measured Gap (adjusted)",
                "calculation": lambda context: f"={context['match_details']['perf_gap']:.4f}" + (f"+ {context['match_details']['gap_adjustment']:.4f}\n ={context['match_details']['adj_perf_gap']:.4f}")
            },
            {
                "description": "Adjustment",
                "calculation": lambda context: f"={context['match_details']['adj_perf_gap']:.4f}-({context['match_details']['rating_gap']:.4f})\n={context['match_details']['adj']:.4f}"
            }
        ],
        "match_updates": {
            "calculation": lambda context: [
                {
                    "name": f"{context['match_details'][p]}",
                    "match_rating": f"={context['match_details'][f'{p}_dynamic']:.2f} + ({context['match_details'][f'{p}_adj']:.4f})\n={context['match_details'][f'{p}_new_match']:.2f}",
                    "dynamic_rating": f"avg(<span class='new-match'>{context['match_details'][f'{p}_new_match']:.2f}</span>,{','.join(f'{x:.2f}' for x in reversed(context['match_details'][f'{p}_dynamic_calcs'][0]))})\n=<span class='new-dynamic'>{context['match_details'][f'{p}_dynamic_calcs'][1]:.2f}</span>"
                }
                for p in ['w1', 'w2', 'l1', 'l2']
                if f"{p}_dynamic_calcs" in context['match_details']
            ]
        }   
    }
}
