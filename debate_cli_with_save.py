# debate_cli_with_save.py
import json
import sys
import requests
from datetime import datetime
import os
from debate_formatter import clean_and_format_debate

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python debate_cli_with_save.py input.json [output_filename]")
        print("Examples:")
        print("  python debate_cli_with_save.py input.json")
        print("  python debate_cli_with_save.py input.json my_debate_results.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    
    # Generate output filename if not provided
    if len(sys.argv) == 3:
        output_file = sys.argv[2]
    else:
        # Auto-generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"legal_debate_{timestamp}.txt"

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
            timeout=300,  # 5 minutes timeout for long debates
        )
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to legal framework. Is the server running?")
        print("💡 Start the server with: python main.py")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("❌ Error: Request timed out. The debate took too long.")
        sys.exit(1)

    # Process response
    try:
        response_data = resp.json()
        
        # Check if it's a successful multi-agent debate
        if response_data.get("status") == "completed" and "result" in response_data:
            result = response_data["result"]
            
            # Format the debate output
            if "multi_agent_debate" in result:
                formatted_debate = clean_and_format_debate(result)
                
                # Save to file
                try:
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write("LEGAL DEBATE RESULTS\n")
                        f.write("=" * 50 + "\n")
                        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Case: {payload.get('case_description', 'N/A')}\n")
                        f.write(f"Question: {payload.get('legal_question', 'N/A')}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(formatted_debate)
                        
                    print(f"✅ Debate results saved to: {output_file}")
                    print(f"📊 File size: {os.path.getsize(output_file)} bytes")
                    
                    # Also display summary in console
                    summary = extract_debate_summary(result)
                    print("\n" + "="*50)
                    print("📋 DEBATE SUMMARY")
                    print("="*50)
                    print(summary)
                    
                except IOError as e:
                    print(f"❌ Error saving file: {e}")
                    print("📺 Displaying output in console instead:")
                    print(formatted_debate)
                    
            else:
                # Fallback to regular JSON output
                output_content = json.dumps(response_data, indent=2)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write("LEGAL DEBATE ERROR RESULTS\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(output_content)
                print(f"⚠ Non-standard response saved to: {output_file}")
                
        else:
            # Error case - save JSON
            error_content = json.dumps(response_data, indent=2)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("LEGAL DEBATE ERROR\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                f.write(error_content)
            print(f"❌ Error response saved to: {output_file}")
            
    except ValueError:
        error_msg = f"Non-JSON response: {resp.text}"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("LEGAL DEBATE COMMUNICATION ERROR\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            f.write(error_msg)
        print(f"❌ Communication error saved to: {output_file}")

def extract_debate_summary(result):
    """Extract key summary information from debate result"""
    
    debate = result.get("multi_agent_debate", {})
    
    rounds = debate.get("debate_rounds", "N/A")
    consensus_summary = debate.get("consensus_summary", {})
    final_consensus = consensus_summary.get("final_consensus_score", "N/A")
    consensus_reached = consensus_summary.get("consensus_reached", False)
    
    summary = f"""Rounds Completed: {rounds}
Final Consensus Score: {final_consensus}
Consensus Target: 0.7
Status: {'✅ CONSENSUS REACHED' if consensus_reached else '❌ MAX ROUNDS REACHED'}

💼 Case Processing Complete
📁 Full details saved to output file
🔍 Review the file for complete round-by-round analysis"""

    return summary

if __name__ == "__main__":
    main()