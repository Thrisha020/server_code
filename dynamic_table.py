"""
FINAL 

TABLE FORMAT

"""

# import json
# import pandas as pd
# import requests
# import matplotlib.pyplot as plt

# def save_table_as_image(df, output_filename="affected_data_table.png"):
#     """Saves the affected data items table as an image using Matplotlib."""
#     fig, ax = plt.subplots(figsize=(10, len(df) * 0.3 + 1))  # Auto-scale based on rows
#     ax.axis("tight")
#     ax.axis("off")

#     # Create a table inside the figure
#     table = ax.table(cellText=df.values,
#                      colLabels=df.columns,
#                      cellLoc="center",
#                      loc="center")

#     table.auto_set_font_size(False)
#     table.set_fontsize(9)  # Adjust font size for readability
#     table.auto_set_column_width([0, 1, 2, 3])  # Adjust width

#     plt.savefig(f"/root/Desktop/Chatbot/table_variable/{output_filename}", bbox_inches="tight", dpi=300)  # Save as PNG with high resolution
#     print(f"\n✅ Table saved as image: {output_filename}")
#     return f"/root/Desktop/Chatbot/table_variable/{output_filename}"


# def display_affected_data_table(affected_data):
#     """Extracts and displays a formatted table of affected data items, and saves it as an image."""
#     table_data = []
    
#     for item in affected_data:
#         for field in item.get("affectedFields", []):
#             table_data.append([
#                 field.get("dataName", "N/A"),
#                 str(field.get("length", "N/A")),
#                 field.get("file", "N/A").split("\\")[-1],  # Extract filename only
#                 field.get("program", "N/A")
#             ])

#     if table_data:
#         df = pd.DataFrame(table_data, columns=["Data Name", "Length", "Defined In", "For Program"])
#         print("\n **Formatted Affected Data Items Table:**\n")

#         print(df.to_string(index=False))  # Pretty-print the table

#         return save_table_as_image(df)  # ✅ Save table as an image
#     else:
#         print("No valid affected data items to display.")
#         return "NONE"

# def get_variable_id(variable_name):
#     url = "http://172.17.64.4:1248/api/workspaces/BankingDemoWS/SourceObjects/Search"
#     params = {
#         "name": variable_name,
#         "objectTypes": "COBOL,NATURAL,PLI,RPG,JCL",
#         "sourceObjectTypes": "DECL,SECTION,PAR,VAR,DATASTORE",
#         "limit": 10000
#     }
#     headers = {"accept": "application/json"}
    
#     try:
#         response = requests.get(url, params=params, headers=headers)
#         response.raise_for_status()
#         data = response.json()
        
#         if data:
#             id_list = []
#             print("\nAvailable IDs:")
#             for item in data:
#                 print(f"Variable: {item['name']}, ID: {item['id']}")
#                 id_list.append(item['id'])
#             return id_list  # Return list of IDs found
#         else:
#             print("No data found for the given variable.")
#             return []
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching data: {e}")
#         return []

# def get_affected_data_items(variable_id):
#     url = "http://172.17.64.4:1248/api/workspaces/BankingDemoWS/SourceObjects/ChangeAnalyzer/AffectedDataItems"
#     encoded_id = variable_id.replace("| |", "%7C%7C")
#     params = {
#         "Ids": encoded_id,
#         "includeTrace": "false",
#         "crossProgramAnalysis": "false",
#         "defaultDepth": "10"
#     }
#     headers = {"accept": "application/json"}
    
#     try:
#         response = requests.get(url, params=params, headers=headers)
#         response.raise_for_status()
#         data = response.json()
        
#         if data:
#             formatted_output = []
#             for item in data:
#                 change_candidate = {
#                     "changeCandidate": item.get("changeCandidate", ""),
#                     "program": item.get("program", ""),
#                     "file": item.get("file", ""),
#                     "line": item.get("line", 0),
#                     "column": item.get("column", 0),
#                     "affectedFields": []
#                 }
                
#                 for field in item.get("affectedFields", []):
#                     affected_field = {
#                         "id": field.get("id", ""),
#                         "dataName": field.get("dataName", ""),
#                         "length": field.get("length", ""),
#                         "value": field.get("value", ""),
#                         "picture": field.get("picture", ""),
#                         "normalizedPicture": field.get("normalizedPicture", ""),
#                         "comment": field.get("comment", ""),
#                         "program": field.get("program", ""),
#                         "file": field.get("file", ""),
#                         "usage": field.get("usage", ""),
#                         "normalizedUsage": field.get("normalizedUsage", ""),
#                         "line": field.get("line", 0),
#                         "column": field.get("column", 0),
#                         "changeCandidate": field.get("changeCandidate", False),
#                         "traceItems": field.get("traceItems", [])
#                     }
#                     change_candidate["affectedFields"].append(affected_field)

#                 formatted_output.append(change_candidate)

#             print(json.dumps(formatted_output, indent=4))
#             return formatted_output
#         else:
#             print("No affected data items found.")
#             return []
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching affected data items: {e}")
#         return []
###########################
# Example usage
# variable_name = input("Enter variable name: ")
# variable_ids = get_variable_id(variable_name)

# if variable_ids:
#     while True:
#         selected_id = input("\nCopy and paste the ID you want to analyze from the list above: ")
#         if selected_id in variable_ids:
#             break
#         print("Invalid ID. Please select a valid ID from the list.")
    
#     affected_data = get_affected_data_items(selected_id)
#     display_affected_data_table(affected_data)  # Call function to display & save table



# BANK-SCR70-RATE


#  53||STU
############################

import json
import pandas as pd
import requests
import matplotlib.pyplot as plt

def save_table_as_image(df, output_filename="affected_data_table.png"):
    """Saves the affected data items table as an image using Matplotlib."""
    max_rows = 30  # Set a max limit for display
    display_df = df.head(max_rows)  # Prevent excessive height scaling

    row_count = len(display_df)
    fig_height = min(6, row_count * 0.4 + 1)  # Cap the max height at 15 inches

    fig, ax = plt.subplots(figsize=(6, fig_height))  # Adjust figure size dynamically
    ax.axis("tight")
    ax.axis("off")

    # Create a table inside the figure
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(5)  # Keep font size readable
    table.scale(0.87, 1)  # Adjust table scaling

    plt.savefig(f"/root/Desktop/Chatbot/table_variable/{output_filename}",
                bbox_inches="tight", dpi=200)  # Reduce DPI to control file size
    print(f"\n✅ Table saved as image: {output_filename}")
    return f"/root/Desktop/Chatbot/table_variable/{output_filename}"


def display_affected_data_table(affected_data):
    """Extracts and displays a formatted table of affected data items, and saves it as an image."""
    table_data = []
    
    for item in affected_data:
        for field in item.get("affectedFields", []):
            table_data.append([
                field.get("dataName", "N/A"),
                str(field.get("length", "N/A")),
                field.get("file", "N/A").split("\\")[-1],  # Extract filename only
                field.get("program", "N/A")
            ])

    if table_data:
        df = pd.DataFrame(table_data, columns=["Data Name", "Length", "Defined In", "For Program"])
        print("\n **Formatted Affected Data Items Table:**\n")

        print(df.to_string(index=False))  # Pretty-print the table

        return save_table_as_image(df)  # ✅ Save table as an image
    else:
        print("No valid affected data items to display.")
        return "NONE"

def get_variable_id(variable_name):
    url = "http://172.17.64.4:1248/api/workspaces/BankingDemoWS/SourceObjects/Search"
    params = {
        "name": variable_name,
        "objectId": "COBOL|45",  # Updated as per document
        "objectTypes": "COBOL,NATURAL,PLI,RPG,JCL",
        "sourceObjectTypes": "DECL",  # Updated as per document
        "limit": 10000
    }
    headers = {"accept": "application/json"}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if data:
            id_list = []
            print("\nAvailable IDs:")
            for item in data:
                print(f"Variable: {item['name']}, ID: {item['id']}")
                id_list.append(item['id'])
            return id_list  # Return list of IDs found
        else:
            print("No data found for the given variable.")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return []

def get_affected_data_items(variable_id):
    url = "http://172.17.64.4:1248/api/workspaces/BankingDemoWS/SourceObjects/ChangeAnalyzer/AffectedDataItems"
    encoded_id = variable_id.replace("| |", "%7C%7C")
    params = {
        "Ids": encoded_id,
        "includeTrace": "false",
        "crossProgramAnalysis": "true",  # Updated as per document
        "defaultDepth": "10"
    }
    headers = {"accept": "application/json"}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if data:
            formatted_output = []
            for item in data:
                change_candidate = {
                    "changeCandidate": item.get("changeCandidate", ""),
                    "program": item.get("program", ""),
                    "file": item.get("file", ""),
                    "line": item.get("line", 0),
                    "column": item.get("column", 0),
                    "affectedFields": []
                }
                
                for field in item.get("affectedFields", []):
                    affected_field = {
                        "id": field.get("id", ""),
                        "dataName": field.get("dataName", ""),
                        "length": field.get("length", ""),
                        "value": field.get("value", ""),
                        "picture": field.get("picture", ""),
                        "normalizedPicture": field.get("normalizedPicture", ""),
                        "comment": field.get("comment", ""),
                        "program": field.get("program", ""),
                        "file": field.get("file", ""),
                        "usage": field.get("usage", ""),
                        "normalizedUsage": field.get("normalizedUsage", ""),
                        "line": field.get("line", 0),
                        "column": field.get("column", 0),
                        "changeCandidate": field.get("changeCandidate", False),
                        "traceItems": field.get("traceItems", [])
                    }
                    change_candidate["affectedFields"].append(affected_field)

                formatted_output.append(change_candidate)

            print(json.dumps(formatted_output, indent=4))
            return formatted_output
        else:
            print("No affected data items found.")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching affected data items: {e}")
        return []

# Example usage
# variable_name = input("Enter variable name: ")
# variable_ids = get_variable_id(variable_name)

# if variable_ids:
#     while True:
#         selected_id = input("\nCopy and paste the ID you want to analyze from the list above: ")
#         if selected_id in variable_ids:
#             break
#         print("Invalid ID. Please select a valid ID from the list.")
    
#     affected_data = get_affected_data_items(selected_id)
#     display_affected_data_table(affected_data)  # Call function to display & save table


############################################



# import json
# import pandas as pd
# import requests
# import matplotlib.pyplot as plt
# import json
# import pandas as pd
# import requests
# import matplotlib.pyplot as plt


# def save_table_as_image(df, output_filename="affected_data_table.png"):
#     """Saves the affected data items table as an image using Matplotlib."""
#     max_rows = 30  # Set a max limit for display
#     display_df = df.head(max_rows)  # Prevent excessive height scaling

#     row_count = len(display_df)
#     fig_height = min(6, row_count * 0.4 + 1)  # Cap the max height at 15 inches

#     fig, ax = plt.subplots(figsize=(6, fig_height))  # Adjust figure size dynamically
#     ax.axis("tight")
#     ax.axis("off")

#     # Create a table inside the figure
#     table = ax.table(
#         cellText=display_df.values,
#         colLabels=display_df.columns,
#         cellLoc="center",
#         loc="center"
#     )

#     table.auto_set_font_size(False)
#     table.set_fontsize(5)  # Keep font size readable
#     table.scale(0.87, 1)  # Adjust table scaling

#     plt.savefig(f"/root/Desktop/Chatbot/table_variable/{output_filename}",
#                 bbox_inches="tight", dpi=200)  # Reduce DPI to control file size
#     print(f"\n✅ Table saved as image: {output_filename}")
#     return f"/root/Desktop/Chatbot/table_variable/{output_filename}"

# def display_affected_data_table(variable_name):
#     url = "http://172.17.64.4:1248/api/workspaces/BankingDemoWS/SourceObjects/Search"
#     params = {
#         "name": variable_name,
#         "objectId": "COBOL|45",  # Updated as per document
#         "objectTypes": "COBOL,NATURAL,PLI,RPG,JCL",
#         "sourceObjectTypes": "DECL",  # Updated as per document
#         "limit": 10000
#     }
#     headers = {"accept": "application/json"}
    
#     try:
#         response = requests.get(url, params=params, headers=headers)
#         response.raise_for_status()
#         data = response.json()
        
#         if data:
#             first_id = data[0]['id']  # Automatically selecting the first ID
#             print(f"Selected Variable ID: {first_id}")
#             return first_id
#         else:
#             print("No data found for the given variable.")
#             return None
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching data: {e}")
#         return None


# def get_variable_id(variable_id):
#     if not variable_id:
#         print("Invalid variable ID.")
#         return []
    
#     url = "http://172.17.64.4:1248/api/workspaces/BankingDemoWS/SourceObjects/ChangeAnalyzer/AffectedDataItems"
#     encoded_id = variable_id.replace("| |", "%7C%7C")
#     params = {
#         "Ids": encoded_id,
#         "includeTrace": "false",
#         "crossProgramAnalysis": "true",
#         "defaultDepth": "10"
#     }
#     headers = {"accept": "application/json"}
    
    
#     try:
#         response = requests.get(url, params=params, headers=headers)
#         response.raise_for_status()
#         data = response.json()
        
#         if data:
#             id_list = []
#             print("\nAvailable IDs:")
#             for item in data:
#                 print(f"Variable: {item['name']}, ID: {item['id']}")
#                 id_list.append(item['id'])
#             return id_list  # Return list of IDs found
#         else:
#             print("No data found for the given variable.")
#             return []
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching data: {e}")
#         return []

# def get_affected_data_items(variable_id):
#     url = "http://172.17.64.4:1248/api/workspaces/BankingDemoWS/SourceObjects/ChangeAnalyzer/AffectedDataItems"
#     encoded_id = variable_id.replace("| |", "%7C%7C")
#     params = {
#         "Ids": encoded_id,
#         "includeTrace": "false",
#         "crossProgramAnalysis": "true",  # Updated as per document
#         "defaultDepth": "10"
#     }
#     headers = {"accept": "application/json"}
    
#     try:
#         response = requests.get(url, params=params, headers=headers)
#         response.raise_for_status()
#         data = response.json()
        
#         if data:
#             formatted_output = []
#             for item in data:
#                 change_candidate = {
#                     "changeCandidate": item.get("changeCandidate", ""),
#                     "program": item.get("program", ""),
#                     "file": item.get("file", ""),
#                     "line": item.get("line", 0),
#                     "column": item.get("column", 0),
#                     "affectedFields": []
#                 }
                
#                 for field in item.get("affectedFields", []):
#                     affected_field = {
#                         "id": field.get("id", ""),
#                         "dataName": field.get("dataName", ""),
#                         "length": field.get("length", ""),
#                         "value": field.get("value", ""),
#                         "picture": field.get("picture", ""),
#                         "normalizedPicture": field.get("normalizedPicture", ""),
#                         "comment": field.get("comment", ""),
#                         "program": field.get("program", ""),
#                         "file": field.get("file", ""),
#                         "usage": field.get("usage", ""),
#                         "normalizedUsage": field.get("normalizedUsage", ""),
#                         "line": field.get("line", 0),
#                         "column": field.get("column", 0),
#                         "changeCandidate": field.get("changeCandidate", False),
#                         "traceItems": field.get("traceItems", [])
#                     }
#                     change_candidate["affectedFields"].append(affected_field)

#                 formatted_output.append(change_candidate)

#             print(json.dumps(formatted_output, indent=4))
#             return formatted_output
#         else:
#             print("No affected data items found.")
#             return []
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching affected data items: {e}")
#         return []
