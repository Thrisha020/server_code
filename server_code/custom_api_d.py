
"""
ONLY THE CUSTOM TOOL

"""



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

from fastapi import WebSocket, WebSocketDisconnect
# FastAPI app
app = FastAPI()

# # Instantiate the LLM
# openai = OpenAI(
#     api_key="bNodVKshzstyAhrHikPOyiOo8Cs0oSnS",
#     base_url="https://api.deepinfra.com/v1/openai",
# )



# Maintain active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

# Initialize the manager
manager = ConnectionManager()

# --- Prompt Template for LangChain Agent ---
base_prompt = """
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
"""

optimization_prompt = """
Instruction:
You are tasked with analyzing the user query and determining which of the following functions it is most related to. Even if the query is short, vague, or ambiguous, you should carefully evaluate its context and provide the most accurate function name based on your understanding. Do not provide any explanation or additional information. Just classify and return the function name.

Now, classify the following user query:
"""
# Function to classify the user query
# def classify_query(input_text: str) -> str:
#     prompt = Prompt(
#         template=optimization_prompt,
#         prompt_kwargs={
#             "base_prompt": base_prompt,
#             "input_text": input_text,
#         },
#     )
#     chat_completion = openai.chat.completions.create(
#         model="Qwen/Qwen2.5-72B-Instruct",
#         messages=[
#             {"role": "system", "content": str(prompt)},
#             {"role": "user", "content": f'Question: {input_text}\nResponse: '},
#         ],
#     )
#     return chat_completion.choices[0].message.content.strip()

from openai import OpenAI





def classify_query(input_text: str) -> str:
    prompt = optimization_prompt.format(base_prompt=base_prompt, input_text=input_text)

    try:
        print("Sending classification request to the model...")
        chat_completion = llm.invoke(prompt)
        
        print("Classification response received:", chat_completion)
        return chat_completion.content.strip()
    except Exception as e:
        print(f"Error during classification: {e}")
        raise ValueError("Failed to classify the query.")

# def classify_query(input_text: str) -> str:
#     prompt = Prompt(
#     template=optimization_prompt,
#     prompt_kwargs={
#         "base_prompt": base_prompt,
#         "input_text": input_text,
#     },
# )

#     chat_completion = openai.chat.completions.create(
#         model="Qwen/Qwen2.5-72B-Instruct",
#         messages=[
#             {"role": "system", "content": str(prompt)},
#             {"role": "user", "content": f'Question: {input_text}\nResponse: '},
#         ],
#     )
#     return chat_completion.choices[0].message.content.strip()


# from langchain.prompts import PromptTemplate

# def classify_query(input_text: str, base_prompt: str, model_name: str) -> str:
#     prompt_template = PromptTemplate(
#         input_variables=["base_prompt", "input_text"],
#         template="{base_prompt}\n{input_text}"
#     )

#     formatted_prompt = prompt_template.format({
#         "base_prompt": base_prompt,
#         "input_text": input_text
#     })

#     chat_completion = openai.chat.completions.create(
#         model="Qwen/Qwen2.5-72B-Instruct",
#         messages=[
#             {"role": "system", "content": formatted_prompt},
#             {"role": "user", "content": f"Question: {input_text}\nResponse:"},
#         ],
#         temperature=0.2,
#         max_tokens=1000,
#     )

#     return chat_completion.choices[0].message.content.strip()




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
def commit_changes(user_in: str) -> dict:
    """Commit file changes in the repository"""
    config = load_config('/root/Desktop/Chatbot/pull_clone.yaml')
    file_name = extract_file_name(user_in.query)
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


# Bind tools 
#tools = [clone, open_file, commit_changes, lpar_list, dependency_graph, variable_table]

from langchain.agents import initialize_agent, Tool

# Define LangChain tools
langchain_tools = [
    Tool(name="clone", func=clone, description="Function for cloning a repository"),
    Tool(name="open_file", func=open_file, description="Function for opening a file in a repository."),
    Tool(name="commit_changes", func=commit_changes, description="Function for making changes in the file and committing them in a repository."),
    Tool(name="lpar_list", func=lpar_list, description="Function for listing or retrieving information about LPAR configurations."),
    Tool(name="dependency_graph", func=dependency_graph, description="Function to show all the dependency files."),
    Tool(name="variable_table", func=variable_table, description="Function to show the affected variables.")
]

# Initialize the ChatOpenAI wrapper with DeepInfra configuration
llm = ChatOpenAI(
    model="Qwen/Qwen2.5-72B-Instruct",
    openai_api_key="bNodVKshzstyAhrHikPOyiOo8Cs0oSnS",  # Replace with your DeepInfra token
    openai_api_base="https://api.deepinfra.com/v1/openai"
)

# Initialize the agent with the ChatOpenAI wrapper
agent_executor = initialize_agent(
    tools=langchain_tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# FastAPI endpoint


# WebSocket endpoint
@app.websocket("/chatagentws/repository")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received via WebSocket: {data}")

            # Classify and process user input
            classify_result = classify_query(data)
            print(f"Classified WebSocket query: {classify_result}")

            response = agent_executor.run(data)
            await manager.send_personal_message(response, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected from WebSocket.")
    except Exception as e:
        print(f"WebSocket processing error: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("custom_api:app", host="0.0.0.0", port=8000, workers=2)


#  python3 /root/Desktop/Chatbot/custom_api.py




# Import things that are needed generically

# --- FastAPI Endpoint using the LangChain Agent ---

# @app.post("/chatagent/repository")
# async def repository_manager(
#     user_in: AutomationInput,
#     #credentials: HTTPAuthorizationCredentials = Security(security),
#     db=Depends(get_db)
# ):
#     try:
#         # Verify token and get user details
#         # result = await verify_token_v2(credentials, db)
#         # print("result:", result)
#         # if not result['status']:
#         #     raise HTTPException(status_code=401, detail="Unauthorized access")

#         # Classify the user query
#         classify_query_result = classify_query(user_in.query)
#         print("classify_query_result:  ",  classify_query_result)
    
#         # Construct the final prompt for the agent
#         user_prompt = f"{base_prompt}\nUser Query: {user_in.query}"
#         print(f"User Prompt: {user_prompt}")
#         try:
#             # Invoke the agent with the classified query
#             response = agent_executor.run(input=user_prompt)
#             return {"response": response}
#         except Exception as e:
#             print(f"LangChain agent execution failed: {e}")
#             return {"error": "Internal processing error", "details": str(e)}

#     except JWTError as jwt_error:
#         print(f"JWT decode error: {jwt_error}")
#         raise HTTPException(status_code=401, detail="Invalid token")
#     except Exception as e:
#         print(f"Unexpected error: {e}")
#         raise HTTPException(status_code=500, detail="Unexpected server error")

