# # debate_cli.py
# import json
# import sys
# import requests

# def main():
#     if len(sys.argv) != 2:
#         print("Usage: python debate_cli.py input.json")
#         sys.exit(1)

#     # Load JSON from file
#     with open(sys.argv[1], "r", encoding="utf-8") as f:
#         payload = json.load(f)

#     # Send request
#     resp = requests.post(
#         "http://localhost:8000/debate/multi_agent",
#         json=payload,
#         timeout=120,
#     )

#     # Print response JSON
#     try:
#         print(json.dumps(resp.json(), indent=2))
#     except ValueError:
#         print("Non-JSON response:", resp.text)

# if __name__ == "__main__":
#     main()

# debate_cli.py (UPDATED VERSION)
import json
import sys
import requests
from debate_formatter import clean_and_format_debate

def main():
    if len(sys.argv) != 2:
        print("Usage: python debate_cli.py input.json")
        sys.exit(1)

    # Load JSON from file
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Send request
    resp = requests.post(
        "http://localhost:8000/debate/multi_agent",
        json=payload,
        timeout=120,
    )

    # Process response
    try:
        response_data = resp.json()
        
        # Check if it's a successful multi-agent debate
        if response_data.get("status") == "completed" and "result" in response_data:
            result = response_data["result"]
            
            # Format the debate output cleanly
            if "multi_agent_debate" in result:
                formatted_debate = clean_and_format_debate(result)
                print(formatted_debate)
            else:
                # Fallback to regular JSON output
                print(json.dumps(response_data, indent=2))
        else:
            # Error case - show JSON
            print(json.dumps(response_data, indent=2))
            
    except ValueError:
        print("Non-JSON response:", resp.text)

if __name__ == "__main__":
    main()