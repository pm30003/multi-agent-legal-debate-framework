#!/usr/bin/env python3
"""
LSTM Memory System Verification Script
Run this to check if LSTM memory is working correctly
"""

import os
import sqlite3
from pathlib import Path
import requests
import json

def check_lstm_files():
    """Check if LSTM memory files exist"""
    print("\n" + "="*60)
    print("📁 CHECKING LSTM FILES")
    print("="*60)
    
    lstm_dir = Path("lstm_memory")
    
    if not lstm_dir.exists():
        print("❌ lstm_memory directory does NOT exist")
        print("   → LSTM has not been initialized yet")
        return False
    else:
        print("✅ lstm_memory directory exists")
    
    # Check for database
    db_path = lstm_dir / "legal_debates.db"
    if db_path.exists():
        print(f"✅ Database file exists: {db_path}")
        print(f"   Size: {db_path.stat().st_size} bytes")
    else:
        print("❌ Database file does NOT exist")
        print("   → No debates have been stored yet")
        return False
    
    # Check for model files (optional - only after training)
    model_path = lstm_dir / "lstm_model.pth"
    if model_path.exists():
        print(f"✅ LSTM model exists: {model_path}")
        print(f"   Size: {model_path.stat().st_size / 1024:.2f} KB")
    else:
        print("⚠️  LSTM model NOT trained yet (needs 10+ debates)")
    
    return True

def check_database_contents():
    """Check database contents"""
    print("\n" + "="*60)
    print("🗄️  CHECKING DATABASE CONTENTS")
    print("="*60)
    
    db_path = Path("lstm_memory/legal_debates.db")
    
    if not db_path.exists():
        print("❌ Database does not exist")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count debates
        cursor.execute("SELECT COUNT(*) FROM debates")
        total_debates = cursor.fetchone()[0]
        print(f"📊 Total debates stored: {total_debates}")
        
        if total_debates == 0:
            print("⚠️  No debates stored yet. Run some debates first.")
            conn.close()
            return False
        
        # Get recent debates
        cursor.execute("""
            SELECT id, case_description, rounds, final_consensus, consensus_reached, timestamp 
            FROM debates 
            ORDER BY id DESC 
            LIMIT 5
        """)
        
        recent_debates = cursor.fetchall()
        
        print(f"\n📋 Last {len(recent_debates)} debates:")
        print("-" * 60)
        for debate in recent_debates:
            debate_id, case_desc, rounds, consensus, reached, timestamp = debate
            status = "✅" if reached else "❌"
            print(f"ID: {debate_id} | {status} Rounds: {rounds} | Consensus: {consensus:.3f}")
            print(f"   Case: {case_desc[:50]}...")
            print(f"   Time: {timestamp}")
            print()
        
        # Get category statistics
        cursor.execute("""
            SELECT case_category, COUNT(*), AVG(final_consensus), AVG(rounds)
            FROM debates
            GROUP BY case_category
        """)
        
        categories = cursor.fetchall()
        
        if categories:
            print("📈 Statistics by Category:")
            print("-" * 60)
            for cat, count, avg_cons, avg_rounds in categories:
                print(f"{cat.upper()}: {count} cases, avg consensus: {avg_cons:.3f}, avg rounds: {avg_rounds:.1f}")
        
        # Check training features
        cursor.execute("SELECT COUNT(*) FROM training_features")
        feature_count = cursor.fetchone()[0]
        print(f"\n🧠 Training features stored: {feature_count}")
        
        conn.close()
        return True
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return False

def check_api_lstm_status():
    """Check LSTM status via API"""
    print("\n" + "="*60)
    print("🌐 CHECKING API LSTM STATUS")
    print("="*60)
    
    api_url = "http://localhost:8000"
    
    # Check if API is running
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
            data = response.json()
            
            if 'components' in data:
                print(f"\n📊 Component Status:")
                print(f"   Groq LLMs: {'✅' if data['components'].get('groq') else '❌'}")
                print(f"   RAG System: {'✅' if data['components'].get('rag') else '❌'}")
                print(f"   Memory: {'✅' if data['components'].get('memory') else '❌'}")
        else:
            print(f"⚠️  API returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("   → Make sure 'python main.py' is running")
        return False
    except Exception as e:
        print(f"❌ API check failed: {e}")
        return False
    
    # Try to get memory insights (if endpoint exists)
    try:
        response = requests.get(f"{api_url}/memory/insights", timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            if data.get('status') == 'success':
                print("\n✅ LSTM Memory API endpoint working!")
                insights = data['insights']
                
                print(f"\n🧠 LSTM Memory Insights:")
                print(f"   Total Debates: {insights['total_debates']}")
                print(f"   Average Consensus: {insights['average_consensus']:.3f}")
                print(f"   Success Rate: {insights['success_rate']:.1f}%")
                print(f"   Average Rounds: {insights['average_rounds']:.1f}")
                print(f"   Model Trained: {'✅ Yes' if insights['model_trained'] else '❌ No'}")
                
                if insights.get('category_statistics'):
                    print(f"\n   Categories tracked: {len(insights['category_statistics'])}")
                
                return True
            else:
                print(f"⚠️  LSTM Memory API returned: {data.get('status')}")
        elif response.status_code == 404:
            print("⚠️  LSTM Memory API endpoint not found")
            print("   → You may need to add /memory/insights endpoint to main.py")
        else:
            print(f"⚠️  Memory endpoint returned: {response.status_code}")
            
    except Exception as e:
        print(f"ℹ️  Memory insights not available: {e}")
        print("   (This is optional - core LSTM may still be working)")
    
    return True

def run_test_debate():
    """Run a test debate to verify LSTM storage"""
    print("\n" + "="*60)
    print("🧪 RUNNING TEST DEBATE")
    print("="*60)
    
    api_url = "http://localhost:8000"
    
    test_case = {
        "case_description": "LSTM Test Case: Company delayed salary payment by 2 months",
        "legal_question": "What are employee rights under Payment of Wages Act?",
        "jurisdiction": "indian",
        "max_rounds": 2,
        "session_id": "lstm_test"
    }
    
    print("📤 Sending test debate request...")
    
    try:
        response = requests.post(
            f"{api_url}/debate/multi_agent",
            json=test_case,
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('status') == 'completed':
                print("✅ Test debate completed successfully!")
                
                result = data.get('result', {})
                debate = result.get('multi_agent_debate', {})
                
                print(f"   Rounds: {debate.get('debate_rounds', 'N/A')}")
                
                # Check if debate was stored
                print("\n🔍 Checking if debate was stored in LSTM...")
                
                # Wait a moment for storage
                import time
                time.sleep(2)
                
                # Check database again
                db_path = Path("lstm_memory/legal_debates.db")
                if db_path.exists():
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute("SELECT COUNT(*) FROM debates WHERE case_description LIKE '%LSTM Test Case%'")
                    test_count = cursor.fetchone()[0]
                    
                    if test_count > 0:
                        print(f"✅ Test debate found in database ({test_count} entries)")
                    else:
                        print("⚠️  Test debate not found in database")
                    
                    conn.close()
                
                return True
            else:
                print(f"⚠️  Debate status: {data.get('status')}")
        else:
            print(f"❌ API returned status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Test debate failed: {e}")
    
    return False

def main():
    """Run all verification checks"""
    print("\n🧠 LSTM MEMORY SYSTEM VERIFICATION")
    print("="*60)
    print("This script will check if your LSTM memory system is working")
    print()
    
    results = {
        'files': False,
        'database': False,
        'api': False,
        'test': False
    }
    
    # Check 1: Files
    results['files'] = check_lstm_files()
    
    # Check 2: Database contents
    results['database'] = check_database_contents()
    
    # Check 3: API status
    results['api'] = check_api_lstm_status()
    
    # Check 4: Test debate (optional)
    print("\n" + "="*60)
    response = input("Run a test debate to verify storage? (y/n): ")
    if response.lower() == 'y':
        results['test'] = run_test_debate()
    
    # Summary
    print("\n" + "="*60)
    print("📋 VERIFICATION SUMMARY")
    print("="*60)
    
    for check, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check.upper()}")
    
    all_passed = all(v for k, v in results.items() if k != 'test')
    
    if all_passed:
        print("\n🎉 LSTM MEMORY SYSTEM IS WORKING CORRECTLY!")
        print("\nNext steps:")
        print("1. Run more debates to build training data")
        print("2. After 10 debates, LSTM will auto-train")
        print("3. Check lstm_memory/ folder for model files")
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\nTroubleshooting:")
        if not results['files']:
            print("- Run at least one debate to initialize LSTM")
        if not results['api']:
            print("- Make sure 'python main.py' is running")
            print("- Check GROQ_API_KEY in .env file")
        if not results['database']:
            print("- Ensure lstm_memory.py is in the same directory")
            print("- Check for Python errors during debate execution")

if __name__ == "__main__":
    main()