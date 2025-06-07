"""

CUSTOM TOOL FULL API CODE & WEBSOCKET INTEGRATION
This code is a custom tool for Langchain that integrates with a Websockets API.
22nd May 2025
classify_query function used
"""



from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from fastapi import FastAPI, Depends, Security, WebSocket, WebSocketDisconnect
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

from typing import List, Optional, Dict, Any
from newFile_servicenow.servicenow_api import get_user_sys_id, get_user_incidents_by_sys_id, get_incident, update_incident, upload_attachment, get_sys_id_from_incident_number
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from starlette.middleware.sessions import SessionMiddleware
from jose import JWTError, jwt
import uuid, requests, pytz, json
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
#from curl_3 import *
from update_curl_3 import *
#from checj import *
#from checkjv2 import jenkins_main
from fastapi.responses import FileResponse, JSONResponse
from jen_trig_wb import jenkins_main
import openai
from langchain.tools import Tool
from openai import OpenAI
import traceback
from fastapi.staticfiles import StaticFiles
# FastAPI app
app = FastAPI()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

# Initialize the manager
manager = ConnectionManager()

# --- Prompt Template for LangChain Agent ---

optimization_prompt = """
Instruction:
You are tasked with analyzing the user query and determining which of the following functions it is most related to. Even if the query is short, vague, or ambiguous, you should carefully evaluate its context and provide the most accurate function name based on your understanding. Do not provide any explanation or additional information. Just classify and return the function name.

Now, classify the following user query:
"""

from openai import OpenAI

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
Base = declarative_base()


dep_dep = []
dep_plot = ''

# Define the ChatSessionHistory table
class ChatSessionHistory(Base):
    __tablename__ = "chat_session_history2"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, index=True)
    session_id = Column(String, index=True)
    conversation = Column(String)  # JSON string for the conversation
    timestamp = Column(DateTime, default=datetime.utcnow)
    last_message = Column(DateTime)

# Database configuration
DATABASE_URL = "mysql+mysqlconnector://resonance:VBhkk!op@172.17.0.6:3306/SSOauthentication"
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=3600,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

SECRET_KEY="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InlGMEcwck9VSmNScWxNOFdmNExXYSJ9.eyJpc3MiOiJodHRwczovL2Rldi1reDJqenFjNWJkbjhpMzc0LnVzLmF1dGgwLmNvbS8iLCJzdWIiOiJqSkIxY0FrZzRTaUFzTnhNWFNsV0paWXh0U1l3NkU1cEBjbGllbnRzIiwiYXVkIjoiaHR0cHM6Ly91c2VyY3JlZGVudGlhbHMvYXBpL2VuY3J5cHQiLCJpYXQiOjE3MjU0NzcxNzIsImV4cCI6MTcyNTU2MzU3Miwic2NvcGUiOiJhZG1pbiIsImd0eSI6ImNsaWVudC1jcmVkZW50aWFscyIsImF6cCI6ImpKQjFjQWtnNFNpQXNOeE1YU2xXSlpZeHRTWXc2RTVwIn0.XP1ndRZkrh6N49I-BqM9KUFYJTKVnYhDfO0-jPM9CdmxkMjvKkPQUL8E6Sj0AKPtapjoVwdZ5Tcnj4gO_XUQsbNdfN4GCDM2j6eVXCz4Q-KfvkLMoMyFEWmKryIhg5BzcZY3sEHv7MhVAGssjqpyxQcE41i9ePRoFCUk2NU2BI_nF-zuk8sE-lazlu4cCEXafGPtIxthidDVDSlHJze8Kf_8zYPrMjPPtYMsE7hSrC9u1YQyMLkImrDRziv7v3moaZs4zL6Wh_BW__pDJBoUYBu69esAljFSprGbkKt7ZMDLvEGN8v3-I6cy-Kkh25t4QqAiRuj9ckRC3vz8K7wyKw"
  # Replace with your secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_MINUTES = 1440
TIMEZONE = pytz.timezone("Asia/Kolkata")
from fastapi import FastAPI, HTTPException, Depends, Security, Request
security = HTTPBearer()


async def verify_token_v2(credentials: HTTPAuthorizationCredentials, db: Session):
    try:
        # Decode the JWT to get the payload
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        print(payload)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        login_type: str = payload.get("login_type")
        if login_type.lower() == 'github_cus':
            #validity_data = await validate_github_token(str(payload.get("github_token")))
        # Check if the token and user_id exist in the database
            token_query = text("SELECT token FROM refresh_tokens WHERE user_id = :user_id")
            result = db.execute(token_query, {'user_id': user_id}).fetchone()
            if result and credentials.credentials == result[0]: #and validity_data['status'] == 'valid':
                return {"user_id":user_id,"status":True,"git_token": str(payload.get("github_token"))}  # Token is valid and matches the DB entry
            else:
                return {"user_id":None,"status":False,"git_token":None}
                #raise HTTPException(status_code=401, detail="Invalid token or user ID")

        elif login_type.lower() == 'custom':
            token_query = text("SELECT token FROM refresh_tokens WHERE user_id = :user_id")
            result = db.execute(token_query, {'user_id': user_id}).fetchone()
            if result and credentials.credentials == result[0]:
                return {"user_id":user_id,"status":True}  # Token is valid and matches the DB entry
            else:
                return {"user_id":None,"status":False}


    except JWTError:
        print('{"user_id":user_id,"status":True}')
        raise HTTPException(status_code=401, detail="Invalid token")
        
    

# --- Define Tools as custom tool @tool functions ---


@tool
def get_incident_ticket() -> dict:
    """
    Retrieve a list of incident tickets from ServiceNow for the current user.
    """
    try:
        # Get the incidents for a specific user ID
        
        incidents = get_user_incidents_by_sys_id('800b174138d089c868d09de320f9833b')
        return {'incident_tickets': incidents}
    except Exception as e:
        return {"error": f"Failed to retrieve incident tickets: {str(e)}"}

@tool
def update_incident1() -> dict:
    """
    Tool for Updating an incident ticket in ServiceNow with new information.
    Returns a dict having the action and scroll_down_key.
    """
    try:
        return {
            'action': 'update_incident',
            'scroll_down_key': ["short_description"]
        }
    except Exception as e:
        return {"error": f"Failed to update incident: {str(e)}"}

@tool
def upload_attachment1() -> dict:
    """
    Upload an attachment to an incident ticket in ServiceNow.
    """
    try:
        # First get the incidents to select which one to attach to
        incidents = get_user_incidents_by_sys_id('800b174138d089c868d09de320f9833b')
        return {'action': 'log_file'}
    except Exception as e:
        return {"error": f"Failed to upload attachment: {str(e)}"}


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
    
app.mount("/chatagentws/table_image", StaticFiles(directory="table_variable"), name="table_image")

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

    app.mount("/chatagentws/table_image", StaticFiles(directory="table_variable"), name="table_image")


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

def get_incident_ticket_wrapper(query):
    """Wrapper for get_incident_ticket function to handle the tool_input parameter"""
    return get_incident_ticket(query)

def update_incident_wrapper(query):
    """Wrapper for update_incident function to handle the tool_input parameter"""
    return update_incident1(query)

def upload_attachment_wrapper(query):
    """Wrapper for upload_attachment function to handle the tool_input parameter"""
    return upload_attachment1(query)


# Create tools with our wrapper functions
langchain_tools = [
    Tool(name="clone", func=clone_wrapper, description="Clone a repository."),
    Tool(name="open_file", func=open_file_wrapper, description="return the correct values in the open_file_wrapper function."),
    Tool(name="commit_changes", func=commit_changes_wrapper, description="return the correct values in the commit_changes_wrapper function."),
    Tool(name="lpar_list", func=lpar_list_wrapper, description="List LPAR configurations."),
    Tool(name="dependency_graph", func=dependency_graph_wrapper, description="Show dependency graph."),
    Tool(name="variable_table", func=variable_table_wrapper, description="Show affected variables."),
    Tool(name="get_incident_ticket", func=get_incident_ticket_wrapper, description="Function for retrieving incident tickets from ServiceNow."),
    Tool(name="update_incident1", func=update_incident_wrapper, description="Function for updating an incident ticket in ServiceNow."),
    Tool(name="upload_attachment1", func=upload_attachment_wrapper, description="Function for uploading an attachment to an ServiceNow.")
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

#------------------------------------------

@app.websocket("/chatagentws/super")
async def websocket_endpoint_super(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.send_json({'test':'test'})
            await asyncio.sleep(60)  # Send test data every minute
    except WebSocketDisconnect:
        manager.disconnect(websocket)


class RepoInput(BaseModel):
    repo_name: str


class AutomationInput(BaseModel):
    query: str

class UpdateIncidentInput(BaseModel):
    incident_number: str
    update_fields: dict

class Textformatterinput(BaseModel):
    text : List


@app.websocket("/chatagentws/trigger-jenkins-build/{repo_name}")
async def websocket_jenkins_build(websocket: WebSocket, repo_name: str):
    await manager.connect(websocket)
    try:
        b = jenkins_main(repo_name,WebSocket)  # Get raw Jenkins output
        await websocket.send_json({'output': b})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    
@app.websocket("/ws/jenkins_progress/{repo_name}")
async def websocket_endpoint(websocket: WebSocket,repo_name:str):
    await websocket.accept()
    # file_path = "/root/amflw_chatbot/chabot_v4/upload_files/ml_test.pdf"
    # filename = "ml_test.pdf"
    #await pdf_extracted_images_pg2(file_path, filename, 48, websocket)
    await jenkins_main(repo_name,websocket)


@app.websocket("/chatagentws/dataformatting")
async def websocket_dataformatting(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            text = data.get('text', [])
            print(text)
            await websocket.send_json({'status': 'success'})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/chatagentws/get_ticket_details/{ticket_number}")
async def websocket_get_ticket_details(websocket: WebSocket, ticket_number: str):
    await manager.connect(websocket)
    try:
        ticket_details = get_incident(ticket_number)
        await websocket.send_json(ticket_details)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/chatagentws/update_incident")
async def websocket_update_incident(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        data = await websocket.receive_json()
        incident_number = data.get('incident_number')
        update_fields = data.get('update_fields')

        try:
            updated_incident = update_incident(incident_number, update_fields)
            ticket_details = get_incident(incident_number)
            
            await websocket.send_json({
                "message": f"Incident {incident_number} has been successfully updated.",
                "updated_incident": updated_incident,
                "ticket_details": ticket_details
            })
        except Exception as e:
            await websocket.send_json({"error": f"Failed to update incident: {str(e)}"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


class variable_id(BaseModel):
    variable_id: str


class SessionData(BaseModel):
    session_id: str
    session_data: dict


@app.websocket("/chatagentws/variable_id_table")
async def websocket_variable_id_table(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            variable_id_value = data.get('variable_id')
            
            affected_data = get_affected_data_items(variable_id_value)
            image_path = display_affected_data_table(affected_data)  # Call function to display & save table
            
            await websocket.send_json({
                "image_table_path": "https://grtapp.genairesonance.com/chatagent/table_image/affected_data_table.png"
            })
            await websocket.close()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/chatagent/table_image/{image_name}")
async def get_icoon_image(image_name: str):
    """
    Serve an image from the IMAGE_DIR by name.
    """
    image_path = f"/root/Desktop/Chatbot/table_variable/{image_name}"
    # if not image_path.exists() or not image_path.is_file():
    #     raise HTTPException(status_code=404, detail="Image not found")
    if image_name.endswith(".svg"):
        media_type = "image/svg+xml"
    elif image_name.endswith(".png"):
        media_type = "image/png"
    elif image_name.endswith(".jpg") or image_name.endswith(".jpeg"):
        media_type = "image/jpeg"
    else:
        return {"error": "Unsupported file format"}

    return FileResponse(image_path, media_type=media_type)


@app.post("/chatagent/upload_log_file")
async def upload_log_file(ticket_number: str , file: List[UploadFile]):
   
    # Specify the full path where files will be saved
    full_path = "/root/Desktop/Chatbot/uploaded_log/"
    
    # Ensure the directory exists
    os.makedirs(full_path, exist_ok=True)
    file = file[0]
    # Save the log file
    file_location = f"{full_path}{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    sys_id = get_sys_id_from_incident_number(ticket_number)
    upload_attachment(file_location,'incident',sys_id)

    return {
        "message": "Log file has been successfully uploaded and processed.",
    }



@app.get("/chatagent/upload_attachment/{ticket_number}")
async def upload_attachment_details(ticket_number: str):
   
    return upload_attachment(ticket_number)


# @app.websocket("/chatagentws/upload_log_file")
# async def websocket_upload_log_file(websocket: WebSocket):
#     await manager.connect(websocket)
#     try:
#         while True:
#             # Get the initial message with ticket number
#             data = await websocket.receive_json()
#             ticket_number = data.get('ticket_number')
            
#             # Wait for the file data
#             file_data = await websocket.receive_bytes()
#             file_name = data.get('file_name', 'uploaded_file.log')
            
#             # Specify the full path where files will be saved
#             full_path = "/root/Desktop/Chatbot/uploaded_log/"
            
#             # Ensure the directory exists
#             os.makedirs(full_path, exist_ok=True)
            
#             # Save the log file
#             file_location = f"{full_path}{file_name}"
#             with open(file_location, "wb") as f:
#                 f.write(file_data)

#             sys_id = get_sys_id_from_incident_number(ticket_number)
#             upload_attachment(file_location, 'incident', sys_id)

#             await websocket.send_json({
#                 "message": "Log file has been successfully uploaded and processed."
#             })
#     except WebSocketDisconnect:
#         manager.disconnect(websocket)


# @app.websocket("/chatagentws/upload_attachment/{ticket_number}")
# async def websocket_upload_attachment(websocket: WebSocket, ticket_number: str):
#     await manager.connect(websocket)
#     try:
#         result = upload_attachment(ticket_number)
#         await websocket.send_json(result)
#     except WebSocketDisconnect:
#         manager.disconnect(websocket)

###################################################

def get_recent_session(user_id: int, db: Session):
    """
    Fetch the most recent session for a given user_id.
    """
    try:
        # SQL Query to Get the Latest Session by Last Message
        query = text(
            """
            SELECT session_id, conversation
            FROM chat_session_history2
            WHERE user_id = :user_id
            ORDER BY last_message DESC
            LIMIT 1
            """
        )

        # Execute Query with the Provided User ID
        result = db.execute(query, {"user_id": user_id}).fetchone()
        
        if result:
            return {"result": "sessions available"}
        else:
            return {"result": "sessions not available"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching recent session: {e}")


@app.websocket("/chatagentws/recent_session")
async def get_recent_session_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            client_id = data.get("client_id")
            credentials = data.get("credentials")
            
            # Verify token and get user_id
            result = await verify_token_v2(credentials)
            
            if result['status']:
                user_id = result["user_id"]
                session_data = get_recent_session(user_id=user_id, db=next(get_db()))
                await websocket.send_json(session_data)
            else:
                await websocket.send_json({'message': 'invalid token'})
    except WebSocketDisconnect:
        pass

# WebSocket for EnterpriseAnalysis
@app.websocket("/chatagentws/EnterpriseAnalysis")
async def repository_manager_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            file_name = data.get("file_name")
            
            dep, html = main(file_name)
            response = {
                'dependencies_list': dep,
                'html_path': 'https://grtapp.genairesonance.com/chatagent/html_template?html_filename=dependency_graph.html',
                'EA_link': 'http://10.190.226.42:8080/EAWeb/EAWS_BankingDemoWS/'
            }
            await websocket.send_json(response)
    except WebSocketDisconnect:
        pass

# WebSocket for copybookname
@app.websocket("/chatagentws/copybookname")
async def copybook_name_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            file_name = data.get("file_name")
            
            dep, html = main(file_name)
            response = {
                'dependencies_list': dep,
                'html_path': 'https://grtapp.genairesonance.com/chatagent/html_template?html_filename=dependency_graph.html',
                'EA_link': 'http://10.190.226.42:8080/EAWeb/EAWS_BankingDemoWS/'
            }
            await websocket.send_json(response)
    except WebSocketDisconnect:
        pass

# WebSocket for EnterpriseAnalysisv2
@app.websocket("/chatagentws/EnterpriseAnalysisv2")
async def repository_manager_v2_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            repo_name = data.get("repo_name")
            file_name = data.get("file_name")
            
            dep, html, ea_url = ea_main(repo_name, file_name)
            response = {
                'dependencies_list': dep,
                'html_path': 'https://grtapp.genairesonance.com/chatagent/html_template?html_filename=dependency_graph1.html',
                'EA_link': ea_url
            }
            await websocket.send_json(response)
    except WebSocketDisconnect:
        pass

# WebSocket for copybooknamev2
@app.websocket("/chatagentws/copybooknamev2")
async def repository_manager_copybook_v2_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            repo_name = data.get("repo_name")
            file_name = data.get("file_name")
            
            
            dep, html, ea_url = ea_main(repo_name, file_name)
            response = {
                'dependencies_list': dep,
                'html_path': 'https://grtapp.genairesonance.com/chatagent/html_template?html_filename=dependency_graph1.html',
                'EA_link': ea_url
            }
            await websocket.send_json(response)
    except WebSocketDisconnect:
        pass

# WebSocket for chatstest
@app.websocket("/chatagentws/chatstest")
async def stream_html_response_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            filename = data.get("filename")
            
            print('\n double click node:  ', filename)
            await websocket.send_json({'status': 'success', 'name': filename})
    except WebSocketDisconnect:
        pass

# WebSocket for chatsingleclik
@app.websocket("/chatagentws/chatsingleclick")
async def click_resp_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Waiting for any message to trigger the response
            await websocket.receive_text()
            
            global dep_dep, dep_plot
            await data_ready.wait()
            async with lock:  # Lock to ensure proper synchronization
                dep = dep_dep.copy()
                html = dep_plot
                dep_dep.clear()
                dep_plot = ''
                data_ready.clear()

            await websocket.send_json({
                'dependencies_list': dep,
                'html_path': html
            })
    except WebSocketDisconnect:
        pass


def save_to_database(user_id, session_id: str, session_data: dict, db:Session):
    try:
        current_conversation_query = text(
            "SELECT conversation FROM chat_session_history2 WHERE user_id = :user_id AND session_id = :session_id"
        )
        result = db.execute(current_conversation_query, {'user_id': user_id, 'session_id': session_id}).fetchone()

        # Initialize conversation as an empty list if no previous conversation exists
        if result:
            #print(result)
            current_conversation = result[0]
            conversation = json.loads(current_conversation)
            #print('\nexisting')
        else:
            conversation = []
        #print(f'\nhistory:  {conversation}')
        conversation.append(session_data)
        # Convert DialogTurn objects to JSON for storage
        # conversation = json.dumps([
        #     {
        #         "user_message": turn.user_message if turn.user_message else None,
        #         "bot_response": turn.bot_response if turn.bot_response else None
        #     }
        #     for turn in session_data
        # ])
        #print(f'\nnew history:  {conversation}')
        conversation_json = json.dumps(conversation)
        # Store the timestamp of the last message
        last_message_timestamp = datetime.now(timezone.utc)
        if result:
            update_query = text(
                "UPDATE chat_session_history2 SET conversation = :conversation, timestamp = :time_stamp, last_message = :last_message "
                "WHERE user_id = :user_id AND session_id = :session_id"
            )
            db.execute(update_query, {'conversation': conversation_json, 'user_id': user_id, 'session_id': session_id,"time_stamp":last_message_timestamp,"last_message":last_message_timestamp})
        else:
            insert_query = text(
                "INSERT INTO chat_session_history2 (user_id, session_id, conversation,timestamp,last_message) VALUES (:user_id, :session_id, :conversation, :time_stamp,:last_message)"
            )
            db.execute(insert_query, {'user_id': user_id, 'session_id': session_id, 'conversation': conversation_json,"time_stamp":last_message_timestamp,"last_message":last_message_timestamp})

        db.commit()

        # # Insert a new record into the table
        # chat_session = ChatSessionHistory(
        #     user_id=user_id,
        #     session_id=session_id,
        #     conversation=conversation_json,
        #     timestamp=datetime.now(timezone.utc),
        #     last_message=last_message_timestamp,
        # )
        # db.add(chat_session)
        # db.commit()
        # db.refresh(chat_session)
        return {"message": "Session stored successfully!", "session_id": session_id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error saving to database: {e}")



import asyncio
lock = asyncio.Lock()

data_ready = asyncio.Event()

from fastapi.responses import HTMLResponse
@app.get("/chathtml_template", response_class=HTMLResponse)
async def html_template(html_filename : str):
    html_file = Path('/root/Desktop/Chatbot/html_paths') / html_filename
    #html_file = f"/root/amflw_chatbot/chabot_v4/upload_files/csv_html_reports/{html_filename}"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(), status_code=200)
    return HTMLResponse(content="<h1> Sorry, We are not able to generate a report at the moment..</h1>", status_code=404)

@app.get("/chatstest")
async def stream_html_response(fileanme: str):
    print('\n double click node:  ',fileanme)
    return {'status':'success','name':'fileanme'}
    # global dep_dep, dep_plot
    # dep,html = main(fileanme)
    # async with lock:  # Lock to safely update globals
    #     dep_dep = dep
    #     dep_plot = html
    #     data_ready.set()
   
    # return {'dependencies_list':dep,'html_path':'https://grtapp.genairesonance.com/chatagent/html_template?html_filename=dependency_graph.html'}

@app.get("/chatsingleclik")
async def click_resp():
    global dep_dep, dep_plot
    await data_ready.wait()
    async with lock:  # Lock to ensure proper synchronization
        dep = dep_dep.copy()
        html = dep_plot
        dep_dep.clear()
        dep_plot = ''
        data_ready.clear()

    return {'dependencies_list':dep,'html_path':html}



from fastapi.responses import HTMLResponse
@app.get("/chatagent/html_template", response_class=HTMLResponse)
async def html_template(html_filename : str):
    html_file = Path('/root/Desktop/Chatbot/html_paths') / html_filename
    #html_file = f"/root/amflw_chatbot/chabot_v4/upload_files/csv_html_reports/{html_filename}"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(), status_code=200)
    return HTMLResponse(content="<h1> Sorry, We are not able to generate a report at the moment..</h1>", status_code=404)




# Endpoint to store session history
@app.post("/chatagent/history_storage")
async def store_session_history(session: SessionData,credentials: HTTPAuthorizationCredentials = Security(security), db=Depends(get_db)):
    """
    Endpoint to receive session data from the frontend and store it in the database.
    """
    result = await verify_token_v2(credentials, db)
    #print(session)
    if result['status']:
        user_id = result["user_id"]
        res = save_to_database(
        user_id =user_id,
        session_id=session.session_id,
        session_data=session.session_data,
        db=db
        )
    
    else:
        res = {"message": "Invalid token", "session_id": None}
    # Save session data to the database
    return res

# Function to retrieve chat history from the database
def retrieve_chat_history(user_id: int, session_id: str, db):
    try:
        # Query the chat history for the given user and session
        chat_session = (
            db.query(ChatSessionHistory)
            .filter(
                ChatSessionHistory.user_id == user_id,
                ChatSessionHistory.session_id == session_id,
            )
            .first()
        )
        if chat_session:
            # Parse the conversation JSON into a list of dictionaries
            return json.loads(chat_session.conversation)
        else:
            raise HTTPException(status_code=404, detail="Session not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {e}")




#   uvicorn open_vscode:app --host 172.17.0.8 --port 5000


@app.get("/chatagent/all-automationsessions-history")
async def get_file_sessions(credentials: HTTPAuthorizationCredentials = Security(security), db: Session = Depends(get_db)):
    #user_id = verify_token(credentials, db)

    result = await verify_token_v2(credentials, db)
    if result['status']:
        user_id = result["user_id"]
    
        try:
            # Get all distinct session IDs for the user
            session_ids_query = text(
                "SELECT session_id FROM chat_session_history2 WHERE user_id = :user_id"
            )
            session_ids_result = db.execute(session_ids_query, {'user_id': user_id}).fetchall()

            sessions = []
            for session_id_row in session_ids_result:
                session_id = session_id_row[0]

                # Get the first message of the session
                first_message_query = text(
                    "SELECT conversation FROM chat_session_history2 WHERE user_id = :user_id AND session_id = :session_id"
                    " ORDER BY timestamp ASC LIMIT 1"
                )
                result = db.execute(first_message_query, {'user_id': user_id, 'session_id': session_id}).fetchone()

                if result:
                    conversation = json.loads(result[0])
                    if conversation:
                        first_message = conversation[0]
                        sessions.append({
                            "session_id": session_id,
                            "first_message": first_message
                        })
                    else:
                        sessions.append({
                            "session_id": session_id,
                            "first_message": None
                        })
                else:
                    sessions.append({
                        "session_id": session_id,
                        "first_message": None
                    })

            return {"sessions": sessions}
        
        except Exception as e:
            #log_error(f"\nError occurred in all-filesessions-history: {e}")
            raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    else:
        return {"detail":"Invalid Token"}



@app.get("/chatagent/single-automationsession-history/{session_id}")
async def get_filesession_messages(session_id: str, credentials: HTTPAuthorizationCredentials = Security(security), db: Session = Depends(get_db)):
    #user_id = verify_token(credentials, db)

    result = await verify_token_v2(credentials, db)
    if result['status']:
        user_id = result["user_id"]
    
        try:
            # Get all the messages for the given session_id and user_id
            messages_query = text(
                "SELECT conversation FROM chat_session_history2 WHERE user_id = :user_id AND session_id = :session_id"
            )
            result = db.execute(messages_query, {'user_id': user_id, 'session_id': session_id}).fetchone()

            if result:
                conversation = json.loads(result[0])
                #print('\n\nConversation without correction:',conversation)  # Parse the JSON conversation
                # for item in conversation:
                #     bot_response = item.get("bot_response", {})
                #     image_urls = bot_response.get("image_urls",{})
                #     #urls = re.findall(r'"(.*?)"', input_string)   text.replace("\\", "")
                #     if image_urls and isinstance(image_urls, str):
                #         # Convert the JSON-encoded string to a list
                #         for url in image_urls:
                #             url = url.replace("\\", "")
                #         #bot_response["image_urls"] = json.loads(image_urls)
                #         bot_response["image_urls"]  = image_urls
                #         item['bot_response'] = bot_response

                # Output the modified data
                # conversation = json.dumps(conversation, indent=4)
                return {"session_id": session_id, "messages": conversation}
            else:
                #log_error(f"\nNo chat history found for this session")
                return {'detail':'No chat history found for this session'}

        except Exception as e:
            #log_error(f"\nError occurred in single-filesessions-history: {e}")
            raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

    else:
        return {"detail":"Invalid Token"}

class SessionResponse(BaseModel):
    session_id: str
    message: str


@app.post("/chatagent/start-automationsession", response_model=SessionResponse)
async def start_session(credentials: HTTPAuthorizationCredentials = Security(security), db: Session = Depends(get_db)):
    # Verify the JWT and get the user ID
    #user_id = verify_token(credentials, db)

    result = await verify_token_v2(credentials, db)
    if result['status']:
        user_id = result["user_id"]
    
    
    # Generate a new session ID (UUID)
        session_id = str(uuid.uuid4())
        
        # Insert the session into the user_sessions table
        try:
            db.execute(
                text("INSERT INTO user_sessions_data (session_id, user_id, start_time) VALUES (:session_id, :user_id, :start_time)"),
                {'session_id': session_id, 'user_id': user_id, 'start_time': datetime.now()}
            )
            db.commit()
        except Exception as e:
            #log_error(f"\nError occurred in starting new session: {e}")
            raise HTTPException(status_code=500, detail=f"Error creating session: {e}")
        
        # Return the session ID to the frontend
        return {"session_id": session_id, "message": "Session started successfully"}

    else:
        return {"session_id": '', "message": "Session Creation Failed","detail": "Invalid Token"}



def classify_query(input_text: str) -> str:
    """Use the LLM to decide which tool to use based on the input text"""
    try:
        # Create a prompt that asks the LLM to classify the query
        classification_prompt = f"""
        You are a helpful assistant. Based on the user query, classify it into one of the following functions:
        Instruction:
        **Important Rules:**
        - The query may be in English, French, or German.
        - First detect the language, then translate to English before classifying.
        - Respond with ONLY the exact function name like `check_internet` with no other text.
        - Do NOT include translation, quotes, explanations, punctuation, or newlines.


        Additionally, detect the **language** of the user query. The query can be in **English, French, or German**. If the query is in French or German, first translate it into English before classifying it.

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
        Tool: check_internet

        Question: Install extension / install extension here / install these extensions / install the required extension
        Tool: install_extensions

        Question: How can I clone a Git repository? / I want to clone the repo / Clone repository / Clone repo
        Tool: clone

        Question: What branches are available in the repository? / List the feature branch / Show me the feature branch 
        Tool: list_branches

        Question: How do I switch to a different branch? / I want to checkout this repo to the main branch
        Tool: checkout_branch

        Question: How do I open a file in the repository? / Need to edit the file / Need to work on hello.cbl / Open the MBANK70P.bms file
        Tool: open_file

        Question: How can I commit my changes? / I need to modify and commit the file
        Tool: commit_changes

        Question: Can you list the available logical partitions? / Run the USS command / Connect to SSH / The build is successful / USS command is clean?
        Tool: lpar_list

        Question: List my tickets / What are all the tickets assigned to me?
        Tool: get_incident_ticket

        Question: I need to update the tickets / Update tickets / Ticket update
        Tool: update_incident1

        Question: Upload the log file / Upload this log file / attach the log file
        Tool: upload_attachment

        Question: I want to see all the dependent files / Show me the dependency files of MBANK70.bms / Open dependency file
        Tool: dependency_graph

        Question: List me all the affected variable fields for BANK-SCR70-RATE / Show me the affected variables / List all affected variables
        Tool: variable_table

        #### **French Examples (Auto-translate & Classify)**
        Question: Quelle est la vitesse de l'internet? / L'internet est-il disponible?  
        (Translation: What is the speed of the internet?)  
        Tool: check_internet

        Question: Veuillez installer l'extension. / Installez ces extensions.  
        (Translation: Please install the extension.)  
        Tool: install_extensions

        Question: Veuillez cloner le dépôt. / Je dois cloner le référentiel.  
        (Translation: Please clone the repository.)  
        Tool: clone

        Question: Quelles sont les branches disponibles dans le référentiel? / Lister les branches.  
        (Translation: What branches are available in the repository?)  
        Tool: list_branches

        Question: Ouvrir le fichier MBANK70P.bms. / Besoin de travailler sur MBANK70P.bms.  
        (Translation: Open the MBANK70P.bms file.)  
        Tool: open_file

        Question: Comment puis-je valider mes modifications ? / Je dois modifier et valider le fichier.  
        (Translation: How can I commit my changes? / I need to modify and commit the file.)  
        Tool: commit_changes

        Question: Pouvez-vous lister les partitions logiques disponibles ? / Exécuter la commande USS / Se connecter en SSH / La construction est réussie / La commande USS est propre ?  
        (Translation: Can you list the available logical partitions? / Run the USS command / Connect to SSH / The build is successful / USS command is clean?)  
        Tool: lpar_list

        Question: Listez mes tickets. / Quels sont les tickets qui me sont attribués ?  
        (Translation: List my tickets. / What are all the tickets assigned to me?)  
        Tool: get_incident_ticket

        Question: Je dois mettre à jour les tickets. / Mettre à jour les tickets. / Mise à jour du ticket.  
        (Translation: I need to update the tickets. / Update tickets. / Ticket update.)  
        Tool: update_incident

        Question: Téléchargez le fichier journal. / Téléchargez ce fichier journal.  
        (Translation: Upload the log file. / Upload this log file.)  
        Tool: upload_attachment

        Question: Je veux voir tous les fichiers dépendants. / Montrez-moi les fichiers dépendants de MBANK70.bms. / Ouvrir le fichier de dépendance.  
        (Translation: I want to see all the dependent files. / Show me the dependency files of MBANK70.bms. / Open dependency file.)  
        Tool: dependency_graph

        Question: Listez tous les champs de variables affectés pour BANK-SCR70-RATE. / Montrez-moi les variables affectées. / Lister toutes les variables affectées.  
        (Translation: List me all the affected variable fields for BANK-SCR70-RATE. / Show me the affected variables. / List all affected variables.)  
        Tool: variable_table

        #### **German Examples (Auto-translate & Classify)**  
        Question: Wie schnell ist das Internet? / Ist das Internet verfügbar?  
        (Translation: What is the speed of the internet?)  
        Tool: check_internet  

        Question: Bitte installieren Sie die Erweiterung. / Installieren Sie diese Erweiterungen.  
        (Translation: Please install the extension.)  
        Tool: install_extensions  

        Question: Bitte klonen Sie das Repository. / Ich muss das Repository klonen.  
        (Translation: Please clone the repository.)  
        Tool: clone  

        Question: Welche Zweige sind im Repository verfügbar? / Liste die Zweige auf.  
        (Translation: What branches are available in the repository?)  
        Tool: list_branches  

        Question: Öffnen Sie die Datei MBANK70P.bms. / Ich muss mit MBANK70P.bms arbeiten.  
        (Translation: Open the MBANK70P.bms file.)  
        Tool: open_file  

        Question: Wie kann ich meine Änderungen übernehmen? / Ich muss die Datei ändern und übernehmen.  
        (Translation: How can I commit my changes? / I need to modify and commit the file.)  
        Tool: commit_changes  

        Question: Können Sie die verfügbaren logischen Partitionen auflisten? / Führen Sie den USS-Befehl aus / Verbinden Sie sich mit SSH / Der Build war erfolgreich / Der USS-Befehl ist sauber?  
        (Translation: Can you list the available logical partitions? / Run the USS command / Connect to SSH / The build is successful / USS command is clean?)  
        Tool: lpar_list  

        Question: Listen Sie meine Tickets auf. / Welche Tickets sind mir zugewiesen?  
        (Translation: List my tickets. / What are all the tickets assigned to me?)  
        Tool: get_incident_ticket  

        Question: Ich muss die Tickets aktualisieren. / Tickets aktualisieren. / Ticketaktualisierung.  
        (Translation: I need to update the tickets. / Update tickets. / Ticket update.)  
        Tool: update_incident  

        Question: Laden Sie die Protokolldatei hoch. / Laden Sie diese Protokolldatei hoch.  
        (Translation: Upload the log file. / Upload this log file.)  
        Tool: upload_attachment  

        Question: Ich möchte alle abhängigen Dateien sehen. / Zeigen Sie mir die Abhängigkeitsdateien von MBANK70.bms. / Öffnen Sie die Abhängigkeitsdatei.  
        (Translation: I want to see all the dependent files. / Show me the dependency files of MBANK70.bms. / Open dependency file.)  
        Tool: dependency_graph  

        Question: Listen Sie alle betroffenen Variablenfelder für BANK-SCR70-RATE auf. / Zeigen Sie mir die betroffenen Variablen. / Listen Sie alle betroffenen Variablen auf.  
        (Translation: List me all the affected variable fields for BANK-SCR70-RATE. / Show me the affected variables. / List all affected variables.)  
        Tool: variable_table  


        **Unrelated Input:**
            Question: "What's the weather today?"
            Tool: unknown
        
        User Query: {input_text}
        """
        
        print("Sending classification request to the model...")
        # Use the same LLM instance that's used for the agent
        response = llm.invoke(classification_prompt)
        
        # Extract just the tool name from the response
        tool_name = response.content.strip().lower()
        
        # Ensure we only return valid tool names
        valid_tools = ["clone", "open_file", "commit_changes", "lpar_list", "dependency_graph", "variable_table","update_incident"]
        if tool_name in valid_tools:
            print(f"LLM classified query as: {tool_name}")
            return tool_name
        else:
            print(f"LLM returned invalid tool name: {tool_name}")
            return None 
            
    except Exception as e:
        print(f"Error during classification: {e}")
        return None  

# @app.websocket("/chatagentws/repository")
# async def repository_manager_ws(websocket: WebSocket, db=Depends(get_db)):
#     await websocket.accept()
#     try:
#         while True:
#             # Wait for data from the client
#             data = await websocket.receive_json()
            
#             # Extract the query from the received data
#             query = data.get("query")
#             if not query:
#                 await websocket.send_json({"error": "Query parameter is required"})
#                 continue
                
#             try:
#                 # Let the agent make decisions about which tool to use
#                 result = agent.invoke({
#                     "input": query
#                 })
                
#                 # Extract the tool observation from the intermediate steps
#                 if isinstance(result, dict) and "intermediate_steps" in result and result["intermediate_steps"]:
#                     # Get the last tool observation
#                     last_step = result["intermediate_steps"][-1]
#                     tool_output = last_step[1]  # This is the actual tool output/observation
                    
#                     # Send the tool output directly
#                     await websocket.send_json({"observation": tool_output})
#                 elif "output" in result:
#                     # Fallback to final answer if intermediate steps not available
#                     await websocket.send_json({"observation": result["output"]})
#                 else:
#                     # Last resort fallback
#                     await websocket.send_json({"observation": str(result)})
                    
#             except Exception as e:
#                 print(f"Error in repository_manager_ws: {str(e)}")
#                 # Print the full traceback for better debugging
#                 traceback.print_exc()
#                 await websocket.send_json({"error": f"Processing error: {str(e)}"})
                
#     except WebSocketDisconnect:
#         print("Client disconnected from repository WebSocket")
#     except Exception as e:
#         print(f"Unexpected error in repository WebSocket: {str(e)}")
#         traceback.print_exc()

#-----------------------#-------------------#---------------------------------------#-------------------
# only observation ---- working

# @app.websocket("/chatagentws/repository")
# async def repository_manager_ws(websocket: WebSocket, db=Depends(get_db)):
#     await websocket.accept()
#     try:
#         while True:
#             # Wait for data from the client
#             data = await websocket.receive_json()
#             query = data.get("query")

#             if not query:
#                 await websocket.send_json({"error": "Query parameter is required"})
#                 continue

#             try:
#                 # --- Call classify_query() ---
#                 tool = classify_query(query)
#                 if not tool:
#                     await websocket.send_json({"error": "Could not classify query"})
#                     continue

#                 print(f"Classified tool: {tool}")

#                 # Optional: Attach classified tool to context for agent, or route manually
#                 result = agent.invoke({
#                     "input": query,
#                     "tool": tool  # optional: pass this to help agent
#                 })

#                 # Process agent result
#                 if isinstance(result, dict) and "intermediate_steps" in result and result["intermediate_steps"]:
#                     last_step = result["intermediate_steps"][-1]
#                     tool_output = last_step[1]

#                     if isinstance(tool_output, dict) and "observation" in tool_output:
#                         tool_output = tool_output["observation"]

#                     await websocket.send_json(tool_output)

#                 elif "output" in result:
#                     output = result["output"]
#                     if isinstance(output, dict) and "observation" in output:
#                         output = output["observation"]
#                     await websocket.send_json(output)
#                 else:
#                     await websocket.send_json(result)

#             except Exception as e:
#                 print(f"Error in repository_manager_ws: {str(e)}")
#                 traceback.print_exc()
#                 await websocket.send_json({"error": f"Processing error: {str(e)}"})

#     except WebSocketDisconnect:
#         print("Client disconnected from repository WebSocket")
#     except Exception as e:
#         print(f"Unexpected error in repository WebSocket: {str(e)}")
#         traceback.print_exc()


#############__________________________________________________################
# streaming 

from fastapi import WebSocket, WebSocketDisconnect, Depends
import traceback
import asyncio

@app.websocket("/chatagentws/repository")
async def repository_manager_ws(websocket: WebSocket, db=Depends(get_db)):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("query")
            if not query:
                await websocket.send_json({"error": "Query parameter is required"})
                continue

            try:
                # ✅ Classify the query first
                tool_name = classify_query(query)
                print(f"Classified tool: {tool_name}")

                # Optional: Send classification to the client for debugging/UX
                await websocket.send_json({"tool_classified": tool_name})

                # Now proceed with streaming agent events
                async for step in agent.astream_events({"input": query}, version="v1"):
                    if step["event"] == "on_tool_end":
                        tool_output = step["data"]
                        if isinstance(tool_output, dict) and "observation" in tool_output:
                            await websocket.send_json({"observation": tool_output["observation"]})
                        else:
                            await websocket.send_json({"observation": tool_output})

                    elif step["event"] == "on_chain_end" and "output" in step["data"]:
                        output = step["data"]["output"]
                        if isinstance(output, dict) and "observation" in output:
                            output = output["observation"]
                        await websocket.send_json({"final_output": output})

            except Exception as e:
                print(f"Error in repository_manager_ws: {str(e)}")
                traceback.print_exc()
                await websocket.send_json({"error": f"Processing error: {str(e)}"})

    except WebSocketDisconnect:
        print("Client disconnected from repository WebSocket")
    except Exception as e:
        print(f"Unexpected error in repository WebSocket: {str(e)}")
        traceback.print_exc()



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ws_custom_i18n:app", host="172.17.0.8", port=8000,workers = 2)


# python3 /root/Desktop/Chatbot/ws_custom_i18n.py


# source automation/bin/activate

#---------------------------------
# old code --working
# only observation ---- working

# @app.websocket("/chatagentws/repository")
# async def repository_manager_ws(websocket: WebSocket, db=Depends(get_db)):
#     await websocket.accept()
#     try:
#         while True:
#             # Wait for data from the client
#             data = await websocket.receive_json()
            
#             # Extract the query from the received data
#             query = data.get("query")
#             if not query:
#                 await websocket.send_json({"error": "Query parameter is required"})
#                 continue
                
#             try:
#                 # Let the agent make decisions about which tool to use
#                 result = agent.invoke({
#                     "input": query
#                 })
                
#                 # Extract the tool observation from the intermediate steps
#                 # Extract the tool observation from the intermediate steps
#                 if isinstance(result, dict) and "intermediate_steps" in result and result["intermediate_steps"]:
#                     # Get the last tool observation
#                     last_step = result["intermediate_steps"][-1]
#                     tool_output = last_step[1]  # This is the actual tool output/observation

#                     # If tool_output is a dict and has "observation", unwrap it
#                     if isinstance(tool_output, dict) and "observation" in tool_output:
#                         tool_output = tool_output["observation"]

#                     await websocket.send_json(tool_output)

#                 elif "output" in result:
#                     output = result["output"]
#                     if isinstance(output, dict) and "observation" in output:
#                         output = output["observation"]
#                     await websocket.send_json(output)
#                 else:
#                     await websocket.send_json(result)

                    
#             except Exception as e:
#                 print(f"Error in repository_manager_ws: {str(e)}")
#                 # Print the full traceback for better debugging
#                 traceback.print_exc()
#                 await websocket.send_json({"error": f"Processing error: {str(e)}"})
                
#     except WebSocketDisconnect:
#         print("Client disconnected from repository WebSocket")
#     except Exception as e:
#         print(f"Unexpected error in repository WebSocket: {str(e)}")
#         traceback.print_exc()
