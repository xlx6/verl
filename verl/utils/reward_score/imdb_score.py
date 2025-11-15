import requests
import sys
def compute_score(solution_str_list: list):
    try:
        resp = requests.post("http://localhost:8000/score", json={
            "solution_str": solution_str_list,
        }, timeout=10)

        resp.raise_for_status() 
        response_data = resp.json() 

        score_list = response_data.get("score") 

        if score_list is None:
            print("Reward API Error: JSON response missing 'score' key.", file=sys.stderr)
            return [0.0] * len(solution_str_list)
            
        if len(score_list) != len(solution_str_list):
            print(f"Reward API Error: Input {len(solution_str_list)} texts, but got {len(score_list)} scores.", file=sys.stderr)
            return [0.0] * len(solution_str_list)

        return score_list

    except requests.exceptions.RequestException as e:
        print(f"Reward API call failed: {e}", file=sys.stderr)

        return [0.0] * len(solution_str_list)