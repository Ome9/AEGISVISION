#!/usr/bin/env python3
"""
Windows Kaggle Setup Script (Python version)
Save as: windows_kaggle_setup.py
"""

import os
import json
from pathlib import Path
import platform

def windows_kaggle_setup():
    """Setup Kaggle credentials on Windows"""
    print("AegisVision - Windows Kaggle Setup")
    print("=" * 40)
    
    # Get Windows user profile
    userprofile = os.environ.get('USERPROFILE')
    if not userprofile:
        userprofile = os.path.expanduser('~')
    
    kaggle_dir = Path(userprofile) / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    print(f"Your Kaggle directory should be: {kaggle_dir}")
    print(f"Your credentials file should be: {kaggle_json}")
    
    # Create .kaggle directory
    print("\nCreating Kaggle directory...")
    if not kaggle_dir.exists():
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {kaggle_dir}")
    else:
        print(f"Directory already exists: {kaggle_dir}")
    
    # Check if kaggle.json exists
    if kaggle_json.exists():
        print(" Found existing kaggle.json")
        print(" Contents:")
        try:
            with open(kaggle_json, 'r') as f:
                creds = json.load(f)
                print(f"   Username: {creds.get('username', 'Unknown')}")
                print(f"   Key: {'*' * 20} (hidden for security)")
        except Exception as e:
            print(f" Error reading kaggle.json: {e}")
    else:
        print(" kaggle.json not found")
        print("\n TO COMPLETE SETUP:")
        print("1. Go to: https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Download kaggle.json file")
        print(f"4. Copy it to: {kaggle_dir}")
        print("\n Alternative - Open File Explorer and paste this path:")
        print(f"   {kaggle_dir}")
        
        # Offer to open the directory
        try:
            response = input("\n Want to open the .kaggle folder now? (y/n): ").lower().strip()
            if response == 'y':
                os.startfile(str(kaggle_dir))  # Windows-specific
                print(" Opened folder in File Explorer")
        except Exception as e:
            print(f"Could not open folder: {e}")
        
        # Offer manual credential entry
        print("\n" + "="*40)
        response = input(" Want to enter credentials manually? (y/n): ").lower().strip()
        if response == 'y':
            username = input("Enter Kaggle username: ").strip()
            key = input("Enter Kaggle key: ").strip()
            
            if username and key:
                credentials = {
                    "username": username,
                    "key": key
                }
                
                try:
                    with open(kaggle_json, 'w') as f:
                        json.dump(credentials, f, indent=2)
                    print(f"✅ Credentials saved to: {kaggle_json}")
                    return True
                except Exception as e:
                    print(f" Error saving credentials: {e}")
    
    print("\n After setup, run: python download_test_video.py")
    print("=" * 40)
    
    return kaggle_json.exists()

if __name__ == "__main__":
    success = windows_kaggle_setup()
    
    if success:
        # Test the setup
        try:
            import kagglehub
            print("\nTesting Kaggle connection...")
            print("Kaggle API is ready!")
        except ImportError:
            print("\nInstall kagglehub first: pip install kagglehub")
        except Exception as e:
            print(f"\nKaggle API test failed: {e}")
    
    input("\n⏸Press Enter to exit...")