import json
import re
import os
from collections import defaultdict

# --- Configuration: Merged Categories and Keywords ---
KEYWORD_CONFIG = {
    "Weather & Light": {
        "Time": {
            "Daytime": [r"\bdaytime\b", r"\bday\b", r"\bsunny\b", r"\bbright\b"],
            "Nighttime": [r"\bnight\b", r"\bnighttime\b", r"\bdark\b", r"\bheadlights\b"],
            "Dusk/Dawn": [r"\bdusk\b", r"\bdawn\b", r"\btwilight\b", r"\bevening\b"]
        },
        "Sky": {
            "Clear": [r"\bclear\b", r"\bsunny\b", r"\bblue sky\b"],
            "Cloudy/Overcast": [r"\bcloudy\b", r"\bovercast\b", r"\bgray\b"],
            "Rain/Wet": [r"\brain\b", r"\brainy\b", r"\braining\b", r"\bwet\b", r"\bdamp\b"],
            "Snow/Ice": [r"\bsnow\b", r"\bicy\b", r"\bsnow-covered\b", r"\bfrost\b"]
        },
        "Road Surface Condition": {
            "Dry": [r"\bdry\b"],
            "Wet/Slippery": [r"\bwet\b", r"\bslippery\b", r"\bdamp\b", r"\bwater\b"],
            "Icy/Snowy": [r"\bice\b", r"\bicy\b", r"\bsnow\b"]
        },
        "Visibility": {
            "Good": [r"\bgood visibility\b", r"\bclear visibility\b"],
            "Poor/Reduced": [r"\bpoor visibility\b", r"\blow visibility\b", r"\breduced visibility\b", r"\bfog\b", r"\bhaze\b", r"\bmist\b"]
        }
    },
    "Traffic Environment": {
        "Area Type": {
            "Urban/City": [r"\burban\b", r"\bcity\b", r"\bdowntown\b", r"\bcommercial\b", r"\bshops\b", r"\bbuildings\b"],
            "Suburban": [r"\bsuburban\b", r"\bresidential\b"],
            "Rural/Open": [r"\brural\b", r"\bcountryside\b", r"\bopen road\b", r"\bfield\b"],
            "Highway/Expressway": [r"\bhighway\b", r"\bexpressway\b", r"\bfreeway\b", r"\binterstate\b"]
        },
        "Traffic Density": {
            "Heavy/Congested": [r"\bheavy\b", r"\bcongested\b", r"\bbusy\b", r"\bjam\b", r"\bqueuing\b"],
            "Moderate": [r"\bmoderate\b", r"\bflowing\b"],
            "Light/Sparse": [r"\blight\b", r"\bsparse\b", r"\bempty\b"]
        }
    },
    "Road Configuration": {
        "Lane Count": {
            "Multi-lane": [r"\bmulti-lane\b", r"\bmultiple lanes\b", r"\bthree-lane\b", r"\bfour-lane\b"],
            "Two-lane": [r"\btwo-lane\b", r"\bsingle lane\b"]
        },
        "Intersection Presence": {
            "Intersection": [r"\bintersection\b", r"\bcrossroad\b", r"\bjunction\b", r"\bsignalized\b", r"\btraffic light\b", r"\bstop sign\b", r"\byield sign\b", r"\broundabout\b", r"\bt-intersection\b"]
        },
        "Road Structure": {
            "Straight": [r"\bstraight\b"],
            "Curved/Winding": [r"\bcurve\b", r"\bwinding\b", r"\bbend\b"],
            "Divided": [r"\bdivided\b", r"\bmedian\b", r"\bbarrier\b"]
        }
    },
    "Accident Type": {
        "Collision Mode": {
            "Side Collision (T-Bone/Sideswipe)": [r"\bt-bone\b", r"\bside-impact\b", r"\bbroadside\b", r"\bflank\b", r"\bside\b", r"\bsideswipe\b", r"\bscrape\b"],
            "Rear-End": [r"\brear-end\b", r"\brear\b"],
            "Head-On": [r"\bhead-on\b", r"\bfrontal\b", r"\bfront\b"]
        },
        "Impact Location": {
            "Front": [r"\bfront\b"],
            "Side": [r"\bside\b", r"\bleft side\b", r"\bright side\b"],
            "Rear": [r"\brear\b"]
        }
    },
    "Accident Cause": {
        "Violation": {
            "Right-of-Way Violation": [r"\byield\b", r"\bright-of-way\b", r"\bfailed to yield\b"],
            "Unsafe Lane Change": [r"\blane change\b", r"\bcut off\b", r"\bmerging\b", r"\bblind spot\b"],
            "Improper Turn": [r"\bturn\b", r"\bu-turn\b"],
            "Signal Violation": [r"\bred light\b", r"\bstop sign\b", r"\bgreen light\b"]
        },
        "Driver Error": {
            "Speeding/Aggressive": [r"\bspeed\b", r"\bfast\b", r"\baggressive\b", r"\breckless\b"],
            "Distraction/Inattention": [r"\bdistract\b", r"\binattentive\b", r"\blook\b", r"\battention\b", r"\bnotice\b"]
        }
    },
    "Accident Prevention": {
        "Defensive Driving": {
            "Observation (Check Mirrors/Blind Spots)": [r"\bmirror\b", r"\bblind spot\b", r"\bcheck\b", r"\bscan\b", r"\blook\b", r"\bobserv\b"],
            "Patience (Wait/Yield)": [r"\bwait\b", r"\byield\b", r"\bstop\b", r"\bpatience\b"],
            "Speed Control (Slow Down/Brake)": [r"\bslow\b", r"\bbrake\b", r"\breduc\b", r"\bspeed\b"],
            "Maintain Safe Distance": [r"\bdistance\b", r"\bgap\b", r"\bfollow\b"],
            "Signaling": [r"\bsignal\b", r"\bindicat\b", r"\bhorn\b", r"\bhonk\b"]
        }
    }
}

def analyze_dataset():
    input_file = "/home/24068286g/CCD_VQA/VRU/src/description_generation/generated_vqa_344.json"
    output_file = "dimension_keyword_analysis.json"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize stats structure
    stats = {
        "summary": {
            "total_questions": 0,
            "dimensions_count": len(KEYWORD_CONFIG)
        },
        "dimensions": {}
    }

    for dim in KEYWORD_CONFIG:
        stats["dimensions"][dim] = {
            "total_questions": 0,
            "extracted_options_count": 0,
            "categories": {},
            "others": {
                "count": 0,
                "percentage": 0.0,
                "examples": []
            }
        }
        for cat_grp in KEYWORD_CONFIG[dim]:
            stats["dimensions"][dim]["categories"][cat_grp] = {
                "total_count": 0,
                "percentage": 0.0,
                "keywords": defaultdict(lambda: {"count": 0, "percentage": 0.0})
            }

    # Processing
    total_q = 0
    
    for entry in data:
        vqa_list = entry.get('generated_vqa', [])
        for item in vqa_list:
            dim = item.get('dimension')
            if dim not in KEYWORD_CONFIG:
                continue
                
            stats["dimensions"][dim]["total_questions"] += 1
            total_q += 1

            # Get Answer Text
            answer = item.get('answer')
            if isinstance(answer, int):
                options = item.get('options', [])
                if 0 <= answer < len(options):
                    answer_text = options[answer]
                else:
                    answer_text = ""
            else:
                answer_text = str(answer)
            
            answer_lower = answer_text.lower()
            
            # Analyze Keywords
            matched_any_Category = False
            
            dim_config = KEYWORD_CONFIG[dim]
            
            # Identify which labels match
            matched_labels_in_dim = set()
            
            for cat_grp, labels_dict in dim_config.items():
                for label, patterns in labels_dict.items():
                    if any(re.search(pat, answer_lower) for pat in patterns):
                        if label not in matched_labels_in_dim:
                             stats["dimensions"][dim]["categories"][cat_grp]["keywords"][label]["count"] += 1
                             stats["dimensions"][dim]["categories"][cat_grp]["total_count"] += 1
                             matched_labels_in_dim.add(label)
                             matched_any_Category = True
            
            if matched_any_Category:
                stats["dimensions"][dim]["extracted_options_count"] += 1
            else:
                stats["dimensions"][dim]["others"]["count"] += 1
                if len(stats["dimensions"][dim]["others"]["examples"]) < 10:
                     stats["dimensions"][dim]["others"]["examples"].append(answer_text)

    stats["summary"]["total_questions"] = total_q

    # Calculate Percentages
    for dim, dim_data in stats["dimensions"].items():
        dim_total = dim_data["total_questions"]
        if dim_total > 0:
            dim_data["others"]["percentage"] = round((dim_data["others"]["count"] / dim_total) * 100, 1)
            
            for cat_grp, grp_data in dim_data["categories"].items():
                grp_data["percentage"] = round((grp_data["total_count"] / dim_total) * 100, 1)
                for kw, kw_data in grp_data["keywords"].items():
                    kw_data["percentage"] = round((kw_data["count"] / dim_total) * 100, 1)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Analysis complete. Saved to {output_file}")

if __name__ == "__main__":
    print("DEBUG: Running MERGED LOGIC script")
    analyze_dataset()
