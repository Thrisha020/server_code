import jenkins
import time
import yaml
import os
import json
import requests
import asyncio
from jenkins import JenkinsException
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatDeepInfra
import re

# Load configuration from config.yaml
CONFIG_FILE = "/root/Desktop/Chatbot/jen_config.yaml"

def load_yaml_config(yaml_file):
    try:
        with open(yaml_file, 'r') as file:
            return yaml.safe_load(file)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading YAML file {yaml_file}: {e}")
        return None

def trigger_jenkins_job_build(server_url, username, password, job_name, PARAMETERS):
    try:
        server = jenkins.Jenkins(server_url, username=username, password=password)
        
        if not server.job_exists(job_name):
            return {"output": f"❌ Error: Job '{job_name}' does not exist."}
        
        next_build_number = server.get_job_info(job_name)['nextBuildNumber']
        server.build_job(job_name, PARAMETERS)
        print(f"Build triggered for job '{job_name}' - Build Number: {next_build_number}")
        
        time.sleep(10)
        while True:
            build_info = server.get_build_info(job_name, next_build_number)
            if not build_info['building']:
                break
            time.sleep(5)
        
        stages_api_url = f"{server_url}/job/{job_name.replace('/', '/job/')}/{next_build_number}/wfapi/describe"
        response = requests.get(stages_api_url, auth=(username, password))
        
        stages_data = response.json() if response.status_code == 200 else {}
        
        stage_results = {
            stage["name"]: "Success" if stage["status"] == "SUCCESS" else "Failed" 
            for stage in stages_data.get("stages", [])
        }
        
            
        status_icon = "✅" if build_info["result"] == "SUCCESS" else "❌"
        
        output_message = {
            "job_name": job_name,
            "build_number": next_build_number,
            "status": f"{status_icon} {'Success' if build_info['result'] == 'SUCCESS' else 'Failed'}",
            "stages": stage_results,
            "job_url": f"{server_url}/job/{job_name.replace('/', '/job/')}/{next_build_number}/",
            "console_output_url": f"{server_url}/job/{job_name.replace('/', '/job/')}/{next_build_number}/console"
        }
       
        return output_message
    except JenkinsException as e:
        return {"output": f"❌ Error: {e}"}

def workspace_detector(repo_name):
    os.environ["DEEPINFRA_API_TOKEN"] = "ag1JDMRLMu7DS1hoQM0sM8HEPyrLKBav"
    
    template = """
    You are tasked to determine the correct workspace name for the given repository from the below workspaces.
    
    Repository: {input}
    
    Available Workspaces:
    1. "BankingDemoWS"
    2. "MortgageAppWS"
    
    Output: Provide only the name of the matching workspace name. Do not give anything else.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatDeepInfra(model_name="Qwen/Qwen2.5-72B-Instruct", max_tokens=10000, precision="float16", streaming=True, temperature=0.9) 
    
    chain = prompt | llm
    result = chain.invoke({"input": repo_name})
    return result.content.strip()



from bs4 import BeautifulSoup

def format_output_with_llm(response_text):
    os.environ["DEEPINFRA_API_TOKEN"] = "kEBKzSgJWEd34DgoFQvno6Xf1aiDN76a"
    
    llm_formatter = ChatDeepInfra(
        model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        max_tokens=2000,
        precision="float16",
        streaming=True,
        temperature=0.9
    )
    
    prompt = f"""
    Convert the following text into a properly formatted HTML report with correct line spacing, formatting, and structure. Give only the HTML formatted output, and ensure links are clean without unnecessary messages:
    
    {response_text}
    """
    
    # Get the raw response from the LLM
    formatted_response = llm_formatter.invoke(prompt)
    raw_html = formatted_response.content  # Assuming the response has a `content` attribute
    
    # Step 1: Remove Markdown-style backticks
    cleaned_html = raw_html.strip()
    if cleaned_html.startswith("```html"):
        cleaned_html = cleaned_html[len("```html"):].strip()  # Remove the opening ```html
    if cleaned_html.endswith("```"):
        cleaned_html = cleaned_html[:-3].strip()  # Remove the closing ```
    
    # Step 2: Parse and clean the HTML using BeautifulSoup
    try:
        soup = BeautifulSoup(cleaned_html, "html.parser")
        cleaned_html = soup.prettify()  # Ensure proper formatting and structure
    except Exception as e:
        print(f"Error while parsing HTML with BeautifulSoup: {e}")
        cleaned_html = f"Invalid HTML content. Error: {e}"
    
    return cleaned_html
   

def jenkins_main(repo_name):
    config = load_yaml_config(CONFIG_FILE)
    if not config:
        return "❌ Error: Failed to load configuration."
    
    workspace = workspace_detector(repo_name)
    if workspace == 'BankingDemoWS':
        git_url = config['jenkins_parameters']['banking_git_url']
    elif workspace == 'MortgageAppWS':
        git_url = config['jenkins_parameters']['mortagage_git_url']
    else:
        return "❌ Error: Workspace not found"
    
    Appname = git_url.split("/")[-1].replace(".git", "")
    job_name = f"Development/{Appname}-BuildFromChatbot"
    PARAMETERS = {'gitBranch': 'Feature/Demo'}
    
    job_info = trigger_jenkins_job_build(
        config['jenkins_parameters']['jenkins_server_url'],
        config['jenkins_parameters']['jenkins_username'],
        config['jenkins_parameters']['jenkins_password'],
        job_name,
        PARAMETERS
    )
    
    if "output" in job_info:
        return job_info["output"]
    
    output = f"""🚀 Job Name: {job_info['job_name']}\n📌 Job ID: {job_info['build_number']}\nStatus: {job_info['status']}\n"""
    
    for stage, result in job_info["stages"].items():
        #print(stage,result)
        if result.lower()=='success':
          output += f"✅{stage}...{result}\n"
        else:
          output += f"❌{stage}...{result}\n"
        
    if job_info['status'] == '✅ Success':
        output += '✅ Jenkins Build Triggered Successfully!\n'
    else:
        output += '❌ Jenkins Build Failed!\n'
    output += f"\n🔗 Job URL: {job_info['job_url']}\n📜 Console Output: {job_info['console_output_url']}"
    print(output)
    formatted_output = format_output_with_llm(output)
    #print(formatted_output)
    return formatted_output

#if __name__ == "__main__":
    #print(jenkins_main('Banking_Demo'))
