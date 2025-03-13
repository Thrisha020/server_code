# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:04:56 2024

@author: samarth
"""

from servicenow_api import (
    get_table_records, create_incident, get_incident, update_incident,
    delete_incident, create_user, upload_attachment, import_data, send_event, 
    get_catalog_items, get_user_incidents, get_user_sys_id, 
    get_user_incidents_by_sys_id, send_message_via_servicenow,
    get_sys_id_from_incident_number
)

import json

def load_json(file_path):
    """Load data from a JSON file."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def send_email_on_update_or_attachment(incident_number, is_updated, is_attachment_uploaded):
    """Send email when incident is updated or an attachment is uploaded."""
    if is_updated or is_attachment_uploaded:
        email_data = load_json("email_data.json")
        if email_data:
            recipient_email = email_data["email"]
            subject = email_data["subject"]
            message = email_data["message"]
            print(f"\nSending email to {recipient_email}...")
            send_message_via_servicenow(recipient_email, subject, message)
        else:
            print("Email data not found.")

def main():
    try:
        # Step 1: Fetch and Display All Incidents
        user_name_input = load_json("/root/Desktop/Chatbot/newFile_servicenow/user_input.json")
        user_sys_id = get_user_sys_id(user_name_input["user_name"])
        
        if user_sys_id:
            incidents = get_user_incidents_by_sys_id(user_sys_id)
            if incidents:
                print(f"Incidents assigned to {user_name_input['user_name']}:")
                incident_list = incidents
                for i, incident in enumerate(incident_list, start=1):
                    print(f"{i}. Incident Number: {incident.get('number')}, Short Description: {incident.get('short_description')}, Priority: {incident.get('priority')}")
            else:
                print("No incidents found or failed to fetch incidents.")
                return
        else:
            print("User not found or failed to fetch user details.")
            return

        # Step 2: Select an Incident for Further Operations
        selected_index = int(input("\nEnter the number corresponding to the incident you want to work on: ")) - 1
        if 0 <= selected_index < len(incident_list):
            selected_incident = incident_list[selected_index]
            incident_number = selected_incident.get("number")
            print(selected_incident)
            print(f"\nSelected Incident: {incident_number}")
        else:
            print("Invalid selection. Exiting.")
            return

        # Step 3: Perform Operations on the Selected Incident
        print("\nWhat would you like to do?")
        print("1. View Incident Details")
        print("2. Update Incident")
        print("3. Upload Attachment")
        operation = input("Enter the number corresponding to the operation: ")

        is_updated = False
        is_attachment_uploaded = False

        if operation == "1":
            # View Incident Details
            fetched_incident = get_incident(incident_number)
            if fetched_incident == True:
                print("\nIncident Details:")
                print(json.dumps(fetched_incident, indent=2))
                #with open("output.json", "w") as outfile: 
                #    json.dump(fetched_incident, outfile) 

        elif operation == "2":
            # Update Incident
            update_data = load_json("update_incident.json")
            if update_data:
                update_fields = update_data["fields"]
                print(f"\nUpdating incident {incident_number}...")
                updated_incident = update_incident(incident_number, update_fields)
                if updated_incident:
                    print("Incident updated successfully:", json.dumps(updated_incident, indent=2))
                    is_updated = True

        elif operation == "3":
            # Upload Attachment
            attachment_data = load_json("attachment_data.json")
            if attachment_data:
                file_path = attachment_data["file_path"]
                print(f"\nUploading attachment for incident {incident_number}...")
                # Check if sys_id exists in selected incident
                #sys_id = selected_incident.get("sys_id")
                sys_id = get_sys_id_from_incident_number(incident_number)
                # Call upload_attachment
                attachment = upload_attachment(file_path, "incident", sys_id)
                if attachment:
                    print("\nAttachment uploaded successfully:", json.dumps(attachment, indent=2))
        else:
            print("Invalid operation selected.")

        # Send email if necessary
        #send_email_on_update_or_attachment(incident_number, is_updated, is_attachment_uploaded)

    except Exception as e:
        print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     #main()

sys_id = get_sys_id_from_incident_number('INC0010074')

print(upload_attachment("/root/Desktop/Chatbot/internet_connection_log.txt","incident",sys_id))

   
