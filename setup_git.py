#!/usr/bin/env python3
"""
Git Setup Script for Enterprise ASL Interpreter
Automates the process of setting up Git repository and pushing to GitHub
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"   Error: {e.stderr.strip()}")
        return False

def check_git_installed():
    """Check if Git is installed"""
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def setup_git_repository():
    """Set up the Git repository"""
    print("🚀 Setting up Git repository for Enterprise ASL Interpreter")
    print("=" * 60)
    
    # Check if Git is installed
    if not check_git_installed():
        print("❌ Git is not installed or not in PATH")
        print("   Please install Git from: https://git-scm.com/downloads")
        return False
    
    # Check if already a Git repository
    if Path(".git").exists():
        print("📁 Git repository already exists")
        response = input("   Continue with existing repository? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            return False
    else:
        # Initialize Git repository
        if not run_command("git init", "Initializing Git repository"):
            return False
    
    # Add all files
    if not run_command("git add .", "Adding all files to Git"):
        return False
    
    # Create initial commit
    commit_message = "feat: Enterprise ASL Interpreter v2.0 - Production-ready with MediaPipe integration"
    if not run_command(f'git commit -m "{commit_message}"', "Creating initial commit"):
        return False
    
    # Set main branch
    run_command("git branch -M main", "Setting main branch")
    
    print("\n✅ Local Git repository setup complete!")
    return True

def setup_github_remote():
    """Help user set up GitHub remote"""
    print("\n🌐 GitHub Setup Instructions")
    print("=" * 40)
    print("1. Go to https://github.com and create a new repository")
    print("2. Repository name: 'enterprise-asl-interpreter' (recommended)")
    print("3. Description: 'Enterprise-grade ASL interpreter with real-time analytics'")
    print("4. Make it PUBLIC to showcase in your portfolio")
    print("5. DO NOT initialize with README (we already have one)")
    print()
    
    github_url = input("📝 Enter your GitHub repository URL (e.g., https://github.com/username/repo.git): ").strip()
    
    if not github_url:
        print("❌ No URL provided. You can add the remote later with:")
        print(f"   git remote add origin YOUR_GITHUB_URL")
        print(f"   git push -u origin main")
        return False
    
    # Add remote origin
    if not run_command(f"git remote add origin {github_url}", "Adding GitHub remote"):
        # If remote already exists, try to set the URL
        run_command(f"git remote set-url origin {github_url}", "Updating GitHub remote URL")
    
    # Push to GitHub
    if not run_command("git push -u origin main", "Pushing to GitHub"):
        print("❌ Push failed. This might be due to:")
        print("   • Authentication issues (set up SSH keys or personal access token)")
        print("   • Repository doesn't exist on GitHub")
        print("   • Network connectivity issues")
        print("\n🔧 Manual push command:")
        print(f"   git push -u origin main")
        return False
    
    print(f"\n🎉 Successfully pushed to GitHub!")
    print(f"🔗 Your repository: {github_url.replace('.git', '')}")
    return True

def create_github_readme_badges():
    """Create additional GitHub-specific files"""
    print("\n📋 Creating GitHub-specific files...")
    
    # Create a GitHub Actions workflow (optional)
    github_dir = Path(".github/workflows")
    github_dir.mkdir(parents=True, exist_ok=True)
    
    workflow_content = """name: ASL Interpreter CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python 3.12
      uses: actions/setup-python@v3
      with:
        python-version: 3.12
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements_enterprise.txt
    
    - name: Run syntax check
      run: |
        python -m py_compile enterprise_asl_interpreter.py
        python -m py_compile demo_script.py
"""
    
    with open(github_dir / "ci.yml", "w") as f:
        f.write(workflow_content)
    
    print("✅ Created GitHub Actions workflow")

def main():
    """Main setup function"""
    print("🚀 ENTERPRISE ASL INTERPRETER - GIT SETUP")
    print("=" * 50)
    print("This script will help you set up Git and push to GitHub")
    print("=" * 50)
    
    # Verify we're in the right directory
    required_files = ["enterprise_asl_interpreter.py", "README.md", "PROJECT_DOCUMENTATION.md"]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("   Please run this script from the project directory")
        return
    
    # Setup local Git repository
    if not setup_git_repository():
        print("❌ Failed to set up local Git repository")
        return
    
    # Create GitHub-specific files
    create_github_readme_badges()
    
    # Add and commit the new files
    run_command("git add .", "Adding GitHub-specific files")
    run_command('git commit -m "docs: Add GitHub Actions workflow and setup files"', "Committing GitHub files")
    
    # Setup GitHub remote and push
    response = input("\n🌐 Ready to push to GitHub? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        setup_github_remote()
    else:
        print("\n📝 Manual GitHub setup:")
        print("1. Create repository on GitHub")
        print("2. git remote add origin YOUR_GITHUB_URL")
        print("3. git push -u origin main")
    
    print("\n" + "=" * 60)
    print("🎯 PORTFOLIO READY!")
    print("=" * 60)
    print("Your Enterprise ASL Interpreter is now ready for:")
    print("• Technical interviews at Lockheed Martin")
    print("• Computer Vision engineering roles")
    print("• ML/AI engineering positions")
    print("• Defense contractor opportunities")
    print("=" * 60)
    print("📁 Key files for presentation:")
    print("• README.md - Professional project overview")
    print("• PROJECT_DOCUMENTATION.md - Technical deep dive")
    print("• enterprise_asl_interpreter.py - Main application")
    print("• demo_script.py - Professional demo")
    print("=" * 60)

if __name__ == "__main__":
    main()