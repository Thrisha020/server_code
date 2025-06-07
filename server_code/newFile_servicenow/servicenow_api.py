# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:03:07 2024

@author: samarth
"""

import requests
import yaml
import json

# Load configuration from config.yaml
with open("/root/Desktop/Chatbot/newFile_servicenow/config.yaml", "r") as file:
    config = yaml.safe_load(file)

INSTANCE_URL = config["servicenow"]["instance_url"]
AUTH = (config["servicenow"]["username"], config["servicenow"]["password"])

'''
INSTANCE_URL = config["servicenow"]["instance_url"]
CLIENT_ID = config["oauth"]["client_id"]
CLIENT_SECRET = config["oauth"]["client_secret"]
USERNAME = config["oauth"]["username"]
PASSWORD = config["oauth"]["password"]
TOKEN_URL = f"{INSTANCE_URL}/oauth_token.do"

# Function to get OAuth token
def get_oauth_token():
    try:
        payload = {
            'grant_type': 'password',
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'username': USERNAME,
            'password': PASSWORD
        }
        response = requests.post(TOKEN_URL, data=payload)
        response.raise_for_status()
        token = response.json().get("access_token")
        if token:
            return token
        else:
            raise Exception("Failed to retrieve OAuth token.")
    except Exception as e:
        print(f"Error getting OAuth token: {str(e)}")
        return None

# Get the OAuth token once
ACCESS_TOKEN = get_oauth_token()
HEADERS = {
    'Authorization': f'Bearer {ACCESS_TOKEN}',
    'Content-Type': 'application/json'
}
'''

# Helper function for error handling
def handle_response(response):
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Response: {response.text}")
    except json.JSONDecodeError:
        print("Failed to decode JSON response.")
    except Exception as err:
        print(f"An error occurred: {err}")
    return None

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}

def send_request(method, url, data=None, params=None):
    """Helper function to send HTTP requests with error handling."""
    try:
        response = requests.request(
            method=method,
            url=url,
            auth=AUTH,
            headers=HEADERS,
            json=data,
            params=params
        )
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
    return None

def get_table_records(table_name, query_params=None):
    url = INSTANCE_URL + config["endpoints"]["table"].format(table_name=table_name)
    response = requests.get(url, auth=AUTH, params=query_params)
    return response.json()

def create_incident(data):
    url = INSTANCE_URL + config["endpoints"]["incident"]
    response = requests.post(url, auth=AUTH, json=data)
    return response.json()

def get_incident1(sys_id):
    url = INSTANCE_URL + config["endpoints"]["incident"] + f"/{sys_id}"
    response = requests.get(url, auth=AUTH)
    return response.json()
'''
def update_incident(sys_id, data):
    url = INSTANCE_URL + config["endpoints"]["incident"] + f"/{sys_id}"
    response = requests.put(url, auth=AUTH, json=data)
    return response.json()

def delete_incident(sys_id):
    url = INSTANCE_URL + config["endpoints"]["incident"] + f"/{sys_id}"
    response = requests.delete(url, auth=AUTH)
    return response.status_code

'''

def create_user(data):
    url = INSTANCE_URL + config["endpoints"]["user"]
    response = requests.post(url, auth=AUTH, json=data, params="name")
    print(response.json())

def import_data(data, table_name):
    url = INSTANCE_URL + config["endpoints"]["import_set"].format(table_name=table_name)
    response = requests.post(url, auth=AUTH, json=data)
    print(response.json())

def send_event(data):
    url = INSTANCE_URL + config["endpoints"]["inbound_events"]
    response = requests.post(url, auth=AUTH, json=data)
    print(response.json())

def get_catalog_items():
    url = INSTANCE_URL + config["endpoints"]["catalog_items"]
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    response = requests.get(url, headers=headers, auth=AUTH)
    print(handle_response(response))

def get_incident(number):
    url = f"{INSTANCE_URL}/api/now/table/incident"
    params = {"sysparm_query": f"number={number}",
              "sysparm_fields": "number,short_description,caller_id,category,subcategory,cmdb_ci,opened_by,priority,state,assignment_group,business_service,sys_id",
              "sysparm_display_value": "true"}
    return (send_request("GET", url, params=params))
    
def update_incident(number, update_data):
    """Update an incident using its number."""
    url = f"{INSTANCE_URL}/api/now/table/incident"
    params = {"number": number}
    
    # Retrieve the incident matching the number
    response = send_request("GET", url, params=params)
    if not response or not response.get("result"):
        print(f"Incident with number {number} not found.")
        print(None)

    # Get the sys_id from the matching incident
    incident = response["result"][0]
    sys_id = incident["sys_id"]
    
    # Update the incident using its sys_id
    update_url = f"{url}/{sys_id}"
    updated_response = send_request("PUT", update_url, data=update_data)
    #print(updated_response)

def delete_incident(number):
    """Delete an incident using its number."""
    url = f"{INSTANCE_URL}/api/now/table/incident"
    params = {"number": number}
    
    # Retrieve the incident matching the number
    response = send_request("GET", url, params=params)
    if not response or not response.get("result"):
        print(f"Incident with number {number} not found.")
        print(None)

    # Get the sys_id from the matching incident
    incident = response["result"][0]
    sys_id = incident["sys_id"] 
    url = f"{INSTANCE_URL}/api/now/table/incident"
    params = {"number": number}
     
    # Retrieve the incident matching the number
    response = send_request("GET", url, params=params)
    if not response or not response.get("result"):
        print(f"Incident with number {number} not found.")
        print(None)

    # Get the sys_id from the matching incident
    incident = response["result"][0]
    sys_id = incident["sys_id"]

    url = f"{INSTANCE_URL}/api/now/table/incident/{sys_id}"
    response = send_request("DELETE", url)
    if response is None:
        print(f"Incident {number} deleted successfully.")
    print(response)

def upload_attachment(file_path, table_name, record_sys_id):
    url = INSTANCE_URL + config["endpoints"]["attachment"]
    headers = {"Content-Type": "multipart/form-data"}
    files = {"file": open(file_path, "rb")}
    params = {"table_name": table_name, "table_sys_id": record_sys_id, "file_name": files}
    response = requests.post(url, auth=AUTH, headers=headers, files=files, params=params)
    #print(response.json())
    return response.json()
    
def send_message_via_servicenow(email, subject, message):
    """
    Send an email message through ServiceNow's email API.
    """
    try:
        url = INSTANCE_URL + config["endpoints"]["email"]
        email_data = {
            "recipients": email,
            "subject": subject,
            "message": message
        }
        response = requests.post(url, auth=AUTH, json=email_data)
        response.raise_for_status()

        result = response.json()
        if result.get("status") == "success":
            print(f"Email sent successfully to {email}.")
        else:
            print(f"Failed to send email: {result}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending email: {e}")

def get_user_incidents(user_name):
    url = INSTANCE_URL + config["endpoints"]["table"].format(table_name="incident")
    # Query parameter to filter by assigned user
    query_params = {
        "sysparm_query": f"assigned_to.name={user_name}",
        "sysparm_fields": "number,short_description,assigned_to,priority,state,email",
        "sysparm_limit": "100"
    }
    response = requests.get(url, auth=AUTH, params=query_params)
    try:
        response.raise_for_status()
        incidents = response.json().get("result", [])
        if not incidents:
            print(f"No incidents found for user: {user_name}")
        return incidents
    except requests.exceptions.RequestException as e:
        print(f"Error fetching incidents for user {user_name}: {e}")
        return []
    
def get_user_sys_id(user_name):
    """
    Fetch the sys_id of a user by their name.
    """
    url = INSTANCE_URL + config["endpoints"]["table"].format(table_name="sys_user")
    query_params = {"sysparm_query": f"name={user_name}", "sysparm_fields": "sys_id,name,email"}
    response = requests.get(url, auth=AUTH, params=query_params)
    #print(response)
    try:
        response.raise_for_status()
        #print(response.json())
        users = response.json().get("result", [])
        if users:
            return users[0]["sys_id"]  # Return the first matched user's sys_id
        else:
            print(f"No user found with name: {user_name}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sys_id for user {user_name}: {e}")
        return None
    
def get_user_incidents_by_sys_id(user_sys_id):
    """
    Fetch incidents assigned to a user using their sys_id.
    """
    url = INSTANCE_URL + config["endpoints"]["table"].format(table_name="incident")
    query_params = {
        "sysparm_query": f"assigned_to={user_sys_id}",
        "sysparm_fields": "number,short_description,assigned_to,priority,state",
        "sysparm_limit": "100"
    }
    response = requests.get(url, auth=AUTH, params=query_params)
    print(response)
    try:
        response.raise_for_status()
        incidents = response.json().get("result", [])
        print(response.json())
        if not incidents:
            print(f"No incidents found for user with sys_id: {user_sys_id}")
        return incidents
    except requests.exceptions.RequestException as e:
        print(f"Error fetching incidents for user sys_id {user_sys_id}: {e}")
        return []
    
def get_sys_id_from_incident_number(incident_number):
    """Retrieve the sys_id of an incident using the incident number."""
    try:
        # Query the incident table
        query = f"number={incident_number}"
        response = get_table_records("incident", query)
        # Check if the response contains results
        if response and response.get("result"):
            # Extract sys_id from the first matching record
            sys_id = response["result"][0].get("sys_id")
            if sys_id:
                print(f"sys_id for incident {incident_number} is: {sys_id}")
                return sys_id
            else:
                print(f"No sys_id found for incident {incident_number}.")
        else:
            print(f"No matching incident found for number {incident_number}.")
    except Exception as e:
        print(f"An error occurred while fetching sys_id: {e}")
    return None


 #print(get_user_incidents_by_sys_id('800b174138d089c868d09de320f9833b')) 