#!/usr/bin/env python3
"""
Script to update deprecated GitHub Actions to their latest versions.
"""

import os
import re
from pathlib import Path

def update_workflow_file(file_path):
    """Update a single workflow file with latest action versions."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Dictionary of action updates
    updates = {
        # Artifact actions
        r'actions/upload-artifact@v3': 'actions/upload-artifact@v4',
        r'actions/download-artifact@v3': 'actions/download-artifact@v4',
        
        # Checkout and setup actions
        r'actions/checkout@v3': 'actions/checkout@v4',
        r'actions/setup-python@v3': 'actions/setup-python@v5',
        r'actions/setup-python@v4': 'actions/setup-python@v5',
        r'actions/setup-node@v3': 'actions/setup-node@v4',
        
        # Cache action
        r'actions/cache@v3': 'actions/cache@v4',
        
        # Docker actions
        r'docker/setup-buildx-action@v2': 'docker/setup-buildx-action@v3',
        r'docker/login-action@v2': 'docker/login-action@v3',
        r'docker/build-push-action@v4': 'docker/build-push-action@v6',
        r'docker/metadata-action@v4': 'docker/metadata-action@v5',
        
        # GitHub actions
        r'github/codeql-action/upload-sarif@v2': 'github/codeql-action/upload-sarif@v3',
        
        # Security actions
        r'aquasecurity/trivy-action@master': 'aquasecurity/trivy-action@0.24.0',
        
        # Codecov
        r'codecov/codecov-action@v3': 'codecov/codecov-action@v4',
        
        # Other actions
        r'peaceiris/actions-gh-pages@v3': 'peaceiris/actions-gh-pages@v4',
    }
    
    # Apply updates
    original_content = content
    for old_version, new_version in updates.items():
        content = re.sub(old_version, new_version, content)
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated: {file_path}")
        return True
    else:
        print(f"No changes needed: {file_path}")
        return False

def main():
    """Main function to update all workflow files."""
    
    workflows_dir = Path('.github/workflows')
    
    if not workflows_dir.exists():
        print("Error: .github/workflows directory not found!")
        return
    
    updated_files = []
    
    # Update all YAML files in workflows directory
    for yaml_file in workflows_dir.glob('*.yml'):
        if update_workflow_file(yaml_file):
            updated_files.append(yaml_file)
    
    for yaml_file in workflows_dir.glob('*.yaml'):
        if update_workflow_file(yaml_file):
            updated_files.append(yaml_file)
    
    print(f"\nSummary: Updated {len(updated_files)} files")
    for file in updated_files:
        print(f"  - {file}")

if __name__ == "__main__":
    main()
