"""
FINAL CUSTOM TOOL GETTING OBSERVATION IN THE RESPONSE BODY

28th April 2025
"""




from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import tool, Tool  # Add Tool import here
from langchain_openai import ChatOpenAI
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
import json
import yaml
import os
import re
from typing import Optional, List, Dict, Any, Union

from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from fastapi import FastAPI, Depends, Security
from fastapi.security import HTTPAuthorizationCredentials
import re
import os
import yaml
from fastapi import FastAPI
from pydantic import BaseModel
import re
import yaml
from fastapi import FastAPI, UploadFile, File, Form
from github import Github
from openai import OpenAI
from adalflow.core.prompt_builder import Prompt

from typing import List, Optional
from newFile_servicenow.servicenow_api import get_user_sys_id, get_user_incidents_by_sys_id, get_incident, update_incident, upload_attachment, get_sys_id_from_incident_number
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from starlette.middleware.sessions import SessionMiddleware
from jose import JWTError, jwt
import uuid, requests, pytz
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, String, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone
from typing import List, Optional
import json
from Dynamic_EA_bokeh import main 
from dynamicEAbokeh_v2 import ea_main
#from double_click import main
from pathlib import Path
from dynamic_table import *
from curl_3 import *
#from checj import *
from checkjv2 import jenkins_main
from fastapi.responses import FileResponse,JSONResponse
#while test use below 
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from jen_trig_wb import jenkins_main
import openai

# Keep all your imports as is

app = FastAPI()

# --- Pydantic model for API input ---
class AutomationInput(BaseModel):
    query: str

# --- Helper functions ---

optimization_prompt = """
Instruction:
You are tasked with analyzing the user query and determining which of the following functions it is most related to. Even if the query is short, vague, or ambiguous, you should carefully evaluate its context and provide the most accurate function name based on your understanding. Do not provide any explanation or additional information. Just classify and return the function name.

{base_prompt}

Now, classify the following user query:
{input_text}
"""


class AutomationInput(BaseModel):
    query: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def load_config(yaml_file):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    yaml_path = os.path.join(script_dir, yaml_file)
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    else:
        raise FileNotFoundError(f"Configuration file '{yaml_file}' not found at path: {yaml_path}")


def extract_file_name(input_text: str) -> Optional[str]:
    file_pattern = r"\b\w+\.\w+\b"  # Regex to match file names with extensions (e.g., hello.cbl)
    match = re.search(file_pattern, input_text)
    return match.group(0) if match else None


def load_credentials(yaml_path: str):
    try:
        with open(yaml_path, 'r') as yaml_file:
            credentials = yaml.safe_load(yaml_file)
            return credentials['git_username'], credentials['git_password'], credentials['git_token']
    except Exception as e:
        print(f"Error reading YAML file: {e}")
        return None, None, None



def search_file_in_repos(github_client, file_name):
    try:
        print(f'user repose:\n{github_client.get_user().get_repos()}')
        repos_with_file = []
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




# --- Define Tools as custom tool @tool functions ---
@tool
def clone() -> dict:
    """
    Clone a repository using the configured base URL.
    """
    config = load_config('/root/Desktop/Chatbot/pull_clone.yaml')
    base_url = config.get('repository', {}).get('base_url')
    return {"action": "clone", "base_url": base_url}


@tool
def open_file(user_query: str) -> dict:
    """
    Open a file in a repository based on the user query.
    """
    config = load_config('/root/Desktop/Chatbot/pull_clone.yaml')
    file_name = extract_file_name(user_query)
    base_url = config['repository']['base_url'] if config else None

    if not file_name:
        return {"error": "No file name detected in the input.", "action": "open_file"}

    yaml_path = "/root/Desktop/Chatbot/credentials.yaml"
    _, _, git_token = load_credentials(yaml_path)
    if not git_token:
        return {"error": "Git token is required for authentication.", "action": "open_file"}

    try:
        github_client = Github(git_token)
    except Exception as e:
        return {"error": f"GitHub authentication failed: {str(e)}", "action": "open_file"}

    repos_with_file = search_file_in_repos(github_client, file_name)

    if not repos_with_file:
        return {
            "action": "None",
            "file_name": file_name,
            "repositories": "null",
            "base_url": "null",
            "required_extensions": "null"
        }

    return {
        "action": "open_file",
        "file_name": file_name,
        "repositories": repos_with_file,
        "base_url": base_url,
        "required_extensions": "null"
    }




@tool
def commit_changes(user_query: str) -> dict:
    """Commit file changes in the repository"""
    config = load_config('/root/Desktop/Chatbot/pull_clone.yaml')
    file_name = extract_file_name(user_query)  # FIX: Directly use the string
    base_url = config['repository']['base_url'] if config else None

    return {
        "action": "commit_changes",
        "file_name": file_name,
        "base_url": base_url
    }

@tool
def lpar_list(base_url, ssh_iconfig):
    """List LPAR configurations and SSH-related data"""
    try:
        config = load_config('/root/Desktop/Chatbot/application_details.yaml')
        ssh_config = load_config("/root/Desktop/Chatbot/timeout_dynamic.yaml")

        base_url = config['repositories']['git_url'] if config else None

        return {
            "action": "lpar_list",
            "base_url": base_url,
            "ssh_iconfig": ssh_config
        }

    except Exception as e:
        return {
            "error": f"An error occurred while processing the LPAR list: {str(e)}",
            "action": "lpar_list"
        }


@tool
def dependency_graph(user_query: str) -> dict:
    """Generate and return dependency graph details for a file"""
    try:
        # Load required extension config
        config = load_config('/root/Desktop/Chatbot/vscode_extension.yaml')
        required_extensions = config['required_extensions'] if config else []

        # Load repository base URL
        config = load_config('/root/Desktop/Chatbot/pull_clone.yaml')
        base_url = config['repository']['base_url'] if config else None

        # Extract file name from user input
        file_name = extract_file_name(user_query)
        if not file_name:
            return {
                "error": "No file name detected in the input.",
                "action": "dependency_graph"
            }

        # Load Git credentials
        yaml_path = "/root/Desktop/Chatbot/credentials.yaml"
        _, _, git_token = load_credentials(yaml_path)
        if not git_token:
            return {
                "error": "Git token is required for authentication.",
                "action": "dependency_graph"
            }

        # Initialize GitHub client
        try:
            github_client = Github(git_token)
        except Exception as e:
            return {
                "error": f"GitHub authentication failed: {str(e)}",
                "action": "dependency_graph"
            }

        # Search for the file in repositories
        repos_with_file = search_file_in_repos(github_client, file_name)
        if not repos_with_file:
            return {
                "text": "No repositories contain the file you requested.",
                "action": "None",
                "file_name": file_name,
                "repositories": [],
                "base_url": base_url,
                "required_extensions": required_extensions
            }

        # Format output for dependency visualization
        text, graphs, htm_link_1, htm_link_2 = format_output(file_name)
        graphs = [
            f"https://grtapp.genairesonance.com/chatagent/html_template?html_filename={graph}"
            for graph in graphs
        ]

        return {
            "text": "The following repository(s) contain the file you requested.",
            "action": "open_file_2",
            "file_name": file_name,
            "repositories": repos_with_file,
            "base_url": base_url,
            "required_extensions": required_extensions,
            "dependency_text": text,
            "graph_path": graphs,
            "html_templates": {
                "MBANK70.BANK70A": "https://grtapp.genairesonance.com/chatagent/html_template?html_filename=MBANK70.BANK70A.htm",
                "MBANK70.HELP70A": "https://grtapp.genairesonance.com/chatagent/html_template?html_filename=MBANK70.HELP70A.htm"
            }
        }

    except Exception as e:
        return {
            "error": f"An error occurred while processing the dependency graph: {str(e)}",
            "action": "dependency_graph"
        }
    
    
@tool
def variable_table(query: str):
    """Display affected variable data table from COBOL-style variable input"""
    if 'BANK-SCR70-RATE' in query.split():
        variable = "BANK-SCR70-RATE"
    else:
        prompt = Prompt(
            template=optimization_prompt,
            prompt_kwargs={
                "base_prompt": "Extract all variable names from the following text. A variable name follows COBOL-style naming conventions, typically consisting of uppercase letters, numbers, and hyphens. Identify and list all such variables found in the text\n Example question: list me all the affected variable fields for BANK-SCR70-RATE\n output :BANK-SCR70-RATE ",
                "input_text": query,
            },
        )
        chat_completion = openai.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[
                {"role": "system", "content": str(prompt)},
                {"role": "user", "content": f'Question: {query}\noutput: '},
            ],
        )
        variable = chat_completion.choices[0].message.content.strip()

    variable_ids = get_variable_id(variable)
    affected_data = get_affected_data_items(variable_ids[0])
    image_path = display_affected_data_table(affected_data)

    return {
        "action": "id_variable",
        "message": "Processing completed",
        "image_table_path": "https://grtapp.genairesonance.com/chatagent/table_image/affected_data_table.png"
    }
# --- Initialize LLM ---
# Now passes the query to your function

def clone_wrapper(query):
    """Wrapper for clone function to handle the tool_input parameter"""
    return clone(query)  

def open_file_wrapper(query):
    """Wrapper for open_file function to handle the tool_input parameter"""
    return open_file(query)  

def commit_changes_wrapper(query):
    """Wrapper for commit_changes function to handle the tool_input parameter"""
    return commit_changes(query)  

def lpar_list_wrapper(query):
    """Wrapper for lpar_list function to handle the tool_input parameter"""
    return lpar_list(None, None)  

def dependency_graph_wrapper(query):
    """Wrapper for dependency_graph function to handle the tool_input parameter"""
    return dependency_graph(query)  

def variable_table_wrapper(query):
    """Wrapper for variable_table function to handle the tool_input parameter"""
    return variable_table(query)  # Now passes the query to your function

# Create tools with our wrapper functions
langchain_tools = [
    Tool(name="clone", func=clone_wrapper, description="Clone a repository."),
    Tool(name="open_file", func=open_file_wrapper, description="return the correct values in the open_file_wrapper function."),
    Tool(name="commit_changes", func=commit_changes_wrapper, description="return the correct values in the commit_changes_wrapper function."),
    Tool(name="lpar_list", func=lpar_list_wrapper, description="List LPAR configurations."),
    Tool(name="dependency_graph", func=dependency_graph_wrapper, description="Show dependency graph."),
    Tool(name="variable_table", func=variable_table_wrapper, description="Show affected variables.")
]

# Initialize LLM
llm = ChatOpenAI(
    model="Qwen/Qwen2.5-72B-Instruct",
    openai_api_key="bNodVKshzstyAhrHikPOyiOo8Cs0oSnS",
    openai_api_base="https://api.deepinfra.com/v1/openai",
    temperature=0
)

# Use a simpler agent that works well with the tool format
agent = initialize_agent(
    tools=langchain_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

def get_db():
    # Keep your original DB session implementation
    return None

# Need to add this function that was referenced but not defined
def classify_query(input_text: str) -> str:
    """Use the LLM to decide which tool to use based on the input text"""
    try:
        # Create a prompt that asks the LLM to classify the query
        classification_prompt = f"""
Functions:
        1. check_internet: Function to detect the speed of the internet.
        2. install_extensions: Function for installing extensions.
        3. clone: Function for cloning a repository.
        4. list_branches: Function listing branches in a repository.
        5. checkout_branch: Function for switching to a specific branch in a repository.
        6. open_file: Function for opening a file in a repository.
        7. commit_changes: Function for making some changes in the file and committing those changes in a repository.
        8. lpar_list: Function for listing or retrieving information about LPAR configurations and also this function is related to Mainframe systems.
        9. get_incident_ticket: Function for retrieving incident tickets from ServiceNow.
        10. dependency_graph: Function to show all the dependency files.
        11. variable_table: Function to show the affected variables.


        ### Examples:

        #### **English Examples**
        Question: What is the speed of the internet? / Is the internet available? / What is the speed of the internet?
        Response: check_internet

        Question: Install extension / install extension here / install these extensions / install the required extension
        Response: install_extensions

        Question: How can I clone a Git repository? / I want to clone the repo / Clone repository / Clone repo
        Response: clone

        Question: What branches are available in the repository? / List the feature branch / Show me the feature branch 
        Response: list_branches

        Question: How do I switch to a different branch? / I want to checkout this repo to the main branch
        Response: checkout_branch

        Question: How do I open a file in the repository? / Need to edit the file / Need to work on hello.cbl / Open the MBANK70P.bms file
        Response: open_file

        Question: How can I commit my changes? / I need to modify and commit the file
        Response: commit_changes

        Question: Can you list the available logical partitions? / Run the USS command / Connect to SSH / The build is successful / USS command is clean?
        Response: lpar_list

        Question: List my tickets / What are all the tickets assigned to me?
        Response: get_incident_ticket

        Question: I need to update the tickets / Update tickets / Ticket update
        Response: update_incident

        Question: Upload the log file / Upload this log file
        Response: upload_attachment

        Question: I want to see all the dependent files / Show me the dependency files of MBANK70.bms / Open dependency file
        Response: dependency_graph

        Question: List me all the affected variable fields for BANK-SCR70-RATE / Show me the affected variables / List all affected variables
        Response: variable_table


        Make sure to use only one tool per query. Do not return explanations. Just invoke the appropriate tool.

        
        User Query: {input_text}
        """
        
        print("Sending classification request to the model...")
        # Use the same LLM instance that's used for the agent
        response = llm.invoke(classification_prompt)
        
        # Extract just the tool name from the response
        tool_name = response.content.strip().lower()
        
        # Ensure we only return valid tool names
        valid_tools = ["clone", "open_file", "commit_changes", "lpar_list", "dependency_graph", "variable_table"]
        if tool_name in valid_tools:
            print(f"LLM classified query as: {tool_name}")
            return tool_name
        else:
            print(f"LLM returned invalid tool name: {tool_name}")
            return None 
            
    except Exception as e:
        print(f"Error during classification: {e}")
        return None  


@app.post("/chatagent/repository")
async def repository_manager(user_in: AutomationInput, db=Depends(get_db)):
    try:
        # Let the agent make decisions about which tool to use
        result = agent.invoke({
            "input": user_in.query
        })
        
        # Extract the tool observation from the intermediate steps
        if isinstance(result, dict) and "intermediate_steps" in result and result["intermediate_steps"]:
            # Get the last tool observation
            last_step = result["intermediate_steps"][-1]
            tool_output = last_step[1]  # This is the actual tool output/observation
            
            # Return the tool output directly
            return {"observation": tool_output}
        elif "output" in result:
            # Fallback to final answer if intermediate steps not available
            return {"observation": result["output"]}
        else:
            # Last resort fallback
            return {"observation": str(result)}
            
    except Exception as e:
        print(f"Error in repository_manager: {str(e)}")
        return {"error": f"Processing error: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("final_custom_agent_api:app", host="172.17.0.8", port=8000, workers=2)