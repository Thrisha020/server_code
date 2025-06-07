"""
SEARCH FILE NAME FROM THE USER INPUT

"""



import re
import yaml
from github import Github
from openai import OpenAI
from adalflow.core.prompt_builder import Prompt
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from newFile_servicenow.servicenow_api import get_user_sys_id, get_user_incidents_by_sys_id, get_incident, update_incident, upload_attachment
from adalflow.core.prompt_builder import Prompt
import os
import yaml

# Load configuration

def load_config(yaml_file):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    yaml_path = os.path.join(script_dir, yaml_file)
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    else:
        raise FileNotFoundError(f"Configuration file '{yaml_file}' not found at path: {yaml_path}")

# Initialize OpenAI client
openai = OpenAI(
    api_key="bNodVKshzstyAhrHikPOyiOo8Cs0oSnS",
    base_url="https://api.deepinfra.com/v1/openai",
)

# Base Prompt
base_prompt = """
Instruction:
You are tasked to check the user query is related to which of the below functions, classify it and respond only with the function name. Do not provide any explanation or additional information. Do not answer the query.

Functions:
1. check Internet: Function to detect the speed of the internet.
2. install extenstion: Function for installing extensions.
3. clone: Function for cloning a repository.
4. list_branches: Function listing branches in a repository.
5. checkout_branch: Function For switching to a specific branch in a repository.
6. open_file: Function For opening a file in a repository.
7. commit_changes: Function For making some changes in the file and committing those changes in a repository.
8. lpar_list: Function For listing or retrieving information about LPAR configurations and also this function is related to Mainframe systems.
9. get_incident_ticket: Function For retrieving incident tickets from Service Now.
"""

# Optimized Prompt
optimization_prompt = """
Instruction:
You are tasked with analyzing the user query and determining which of the following functions it is most related to. Even if the query is short, vague, or ambiguous, you should carefully evaluate its context and provide the most accurate function name based on your understanding. Do not provide any explanation or additional information. Just classify and return the function name.

Consider the base prompt provided along with the user query and decide if it is highly related to one of the predefined functions. Provide the correct function name with no further explanation.

Now, classify the following user query:
"""

# Function to classify the user query
def classify_query(input_text: str) -> str:
    prompt = Prompt(
        template=optimization_prompt,
        prompt_kwargs={
            "base_prompt": base_prompt,
            "input_text": input_text,
        },
    )
    chat_completion = openai.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[
            {"role": "system", "content": str(prompt)},
            {"role": "user", "content": f'Question: {input_text}\nResponse: '},
        ],
    )
    return chat_completion.choices[0].message.content.strip()

# Extract file name from input text
def extract_file_name(input_text: str) -> str:
    """Extracts the file name with an extension from the input text."""
    file_pattern = r"\b\w+\.\w+\b"  # Regex to match file names with extensions (e.g., hello.cbl)
    match = re.search(file_pattern, input_text)
    return match.group(0) if match else None

# Load credentials from YAML file
def load_credentials(yaml_path: str):
    """Load credentials from the YAML file."""
    try:
        with open(yaml_path, 'r') as yaml_file:
            credentials = yaml.safe_load(yaml_file)
            return credentials['git_username'], credentials['git_password'], credentials['git_token']
    except Exception as e:
        print(f"Error reading YAML file: {e}")
        return None, None, None

# Search for a file in repositories
def search_file_in_repos(github_client, file_name):
    """Search for a file in all accessible repositories."""
    try:
        repos_with_file = []

        # List all repositories accessible by the user
        for repo in github_client.get_user().get_repos():
            try:
                contents = repo.get_contents("")

                while contents:
                    file_or_dir = contents.pop(0)
                    if file_or_dir.type == "dir":
                        contents.extend(repo.get_contents(file_or_dir.path))
                    elif file_or_dir.type == "file" and file_or_dir.name == file_name:
                        repos_with_file.append(repo.full_name)
                        break
            except Exception as repo_error:
                print(f"Error processing repository {repo.full_name}: {repo_error}")

        return repos_with_file
    except Exception as e:
        print(f"Error during file search: {e}")
        return []

# Main function that combines everything

def repository_manager(user_in: dict):
    action = classify_query(user_in['query'])

    print('Action:', action)

    # Action based on classified query
    if action == 'hi':
        return {"action": "open_vscode", 'name': '3sha'}

    elif action == 'get_incident_ticket':
        user_id = get_user_sys_id('new_user18')  # Dummy function
        incidents = get_user_incidents_by_sys_id(user_id)  # Dummy function
        return {'incident_tickets': incidents}

    elif action == 'update_incident':
        return {'action': 'update_incident'}

    elif action == 'upload_attachment':
        return {'action': 'log_file'}

    elif action == 'check_internet':
        return {"action": "check_internet"}

    elif action == 'install_extensions':
        config = load_config('/root/Desktop/Chatbot/vscode_extension.yaml')  # Assuming load_config() is defined elsewhere
        if not config:
            return
        required_extensions = config['required_extensions']
        return {"action": "install_extensions", 'required_extensions': required_extensions}

    elif action == 'clone':
        config = load_config('/root/Desktop/Chatbot/pull_clone.yaml')  # Assuming load_config() is defined elsewhere
        if not config:
            return
        base_url = config['repository']['base_url']
        return {"action": "clone", 'base_url': base_url}

    elif action == 'list_branches':
        config = load_config('/root/Desktop/Chatbot/pull_clone.yaml')
        if not config:
            return
        base_url = config['repository']['base_url']
        return {"action": "list_branches", 'base_url': base_url}

    elif action == 'checkout_branch':
        config = load_config('/root/Desktop/Chatbot/pull_clone.yaml')
        if not config:
            return
        base_url = config['repository']['base_url']
        return {"action": "checkout_branch", 'base_url': base_url}

    elif action == 'open_file':
        config = load_config('/root/Desktop/Chatbot/pull_clone.yaml')
        if not config:
            return
        # Extract file name from input query
        file_name = extract_file_name(user_in['query'])
        base_url = config['repository']['base_url']
        if not file_name:
            return {"error": "No file name detected in the input."}

        # Load GitHub credentials and authenticate
        yaml_path = "credentials.yaml"
        git_username, git_password, git_token = load_credentials(yaml_path)
        if not git_token:
            return {"error": "Git token is required for authentication."}

        github_client = Github(git_token)

        # Search for the file in repositories
        repos_with_file = search_file_in_repos(github_client, file_name)

        return {
            "action": "open_file",
            "file_name": file_name,
            "repositories": repos_with_file,
            'base_url': base_url
        }

    elif action == 'commit_changes':
        config = load_config('/root/Desktop/Chatbot/pull_clone.yaml')
        if not config:
            return
        base_url = config['repository']['base_url']
        return {"action": "commit_changes", 'base_url': base_url}

    elif action == 'lpar_list':
        config = load_config('/root/Desktop/Chatbot/pull_clone.yaml')
        ssh_config = load_config("/root/Desktop/Chatbot/timeout_dynamic.yaml")
        if not config:
            return
        base_url = config['repository']['base_url']
        return {"action": "lpar_list", 'base_url': base_url, 'ssh_iconfig': ssh_config}

    else:
        return {"action": "unknown"}



# Example Usage
if __name__ == "__main__":
    user_input = {"query": "I need to open the hello.cbl file"}
    response = repository_manager(user_input)
    print(response)
