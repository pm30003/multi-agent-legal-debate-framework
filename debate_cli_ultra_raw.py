# debate_cli_ultra_raw.py
"""
Ultra-Raw Legal Debate CLI - Saves COMPLETELY unprocessed output
Preserves every character of the original judicial decision
"""

import json
import sys
import requests
from datetime import datetime
import os

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python debate_cli_ultra_raw.py input.json [output_filename]")
        sys.exit(1)

    input_file = sys.argv[1]
    
    # Generate output filename if not provided
    if len(sys.argv) == 3:
        output_file = sys.argv[2]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"raw_debate_{timestamp}.txt"

    print(f"🔄 Starting legal debate...")
    print(f"📥 Input: {input_file}")
    print(f"📤 Output will be saved to: {output_file}")

    # Load JSON from file
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in '{input_file}': {e}")
        sys.exit(1)

    # Send request
    print("🚀 Sending request to legal framework...")
    try:
        resp = requests.post(
            "http://localhost:8000/debate/multi_agent",
            json=payload,
            timeout=300,
        )
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to legal framework. Is the server running?")
        print("💡 Start the server with: python main.py")
        sys.exit(1)

    # Process response - ZERO FORMATTING
    try:
        response_data = resp.json()
        
        if response_data.get("status") == "completed" and "result" in response_data:
            result = response_data["result"]
            
            if "multi_agent_debate" in result:
                debate = result["multi_agent_debate"]
                
                # Extract RAW data
                rounds = debate.get("debate_rounds", "N/A")
                consensus_summary = debate.get("consensus_summary", {})
                consensus_progression = debate.get("consensus_progression", [])
                all_plaintiff_args = debate.get("all_plaintiff_arguments", [])
                all_defendant_args = debate.get("all_defendant_arguments", [])
                judicial_decision_raw = debate.get("judicial_decision", "")
                
                # Build COMPLETELY unprocessed output
                raw_content = f"""LEGAL DEBATE OUTPUT
==============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Case: {payload.get('case_description', 'N/A')}
Question: {payload.get('legal_question', 'N/A')}
Total Rounds: {rounds}
Final Consensus: {consensus_summary.get('final_consensus_score', 'N/A')}
==============================

CONSENSUS PROGRESSION:
"""
                
                # Add consensus progression
                for i, metrics in enumerate(consensus_progression, 1):
                    raw_content += f"Round {i}: {metrics}\n"
                
                raw_content += "\n" + "="*80 + "\n"
                raw_content += "ALL ROUND ARGUMENTS\n"
                raw_content += "="*80 + "\n\n"
                
                # Add all plaintiff arguments (zero processing)
                for i, arg in enumerate(all_plaintiff_args, 1):
                    raw_content += f"PLAINTIFF ROUND {i} :\n"
                    raw_content += "-" * 40 + "\n"
                    raw_content += arg + "\n\n"
                
                # Add all defendant arguments (zero processing)  
                for i, arg in enumerate(all_defendant_args, 1):
                    raw_content += f"DEFENDANT ROUND {i} :\n"
                    raw_content += "-" * 40 + "\n"
                    raw_content += arg + "\n\n"
                
                raw_content += "="*80 + "\n"
                raw_content += "JUDICIAL DECISION \n"
                raw_content += "="*80 + "\n\n"
                
                # Add judicial decision with ZERO processing
                raw_content += judicial_decision_raw
                
                raw_content += "\n\n" + "="*80 + "\n"
                raw_content += "\n"
                raw_content += "="*80 + "\n"
                
                # Save to file
                try:
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(raw_content)
                        
                    print(f"✅ Debate results saved to: {output_file}")
                    print(f"📊 File size: {os.path.getsize(output_file)} bytes")
                    
                    # Show preview of judicial decision length
                    print(f"⚖️ Judicial decision length: {len(judicial_decision_raw)} characters")
                    print(f"🔍 Preview (first 200 chars): {judicial_decision_raw[:200]}...")
                    
                except IOError as e:
                    print(f"❌ Error saving file: {e}")
                    
            else:
                print("❌ No multi-agent debate found in response")
        else:
            print("❌ Debate failed or returned error")
            
    except ValueError as e:
        print(f"❌ Invalid JSON response: {e}")

if __name__ == "__main__":
    main()