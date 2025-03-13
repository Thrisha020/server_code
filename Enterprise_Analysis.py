import requests

def get_object_id(file_path):
    """
    Step 1: Get the object ID by replacing the file path in curl command 2.
    """
    url = "http://172.17.64.4:1248/api/workspaces/BankingDemoWS/ObjectByPath"
    params = {
        "path": file_path
    }
    headers = {"accept": "application/json"}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        result = response.json()
        object_id = result.get("id")  # Assuming the response has an 'id' field
        return object_id
    else:
        print(f"Error getting object ID: {response.status_code}")
        print(response.text)
        return None


def get_dependencies(object_id):
    """
    Step 2: Get dependencies by replacing the object ID in curl command 1.
    """
    url = f"http://172.17.64.4:1248/api/workspaces/BankingDemoWS/ObjectDirectRelationship"
    params = {"id": object_id}
    headers = {"accept": "application/json"}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        dependencies = response.json()
        return dependencies
    else:
        print(f"Error getting dependencies: {response.status_code}")
        print(response.text)
        return None


def display_dependencies(dependencies):
    """
    Step 3: Display dependencies in a chat-like format.
    """
    print("Chat: Please find the list of dependency code:")
    for dep in dependencies:
        print(f"{dep['name']} (Type: {dep['type']}, Relation: {dep['relation']})")


def main():
    # File path for Hello.cbl (replace with your desired file)
    file_path = r"C:\Users\Administrator.GRT-EA-WDC2\Downloads\Rocket EA\Banking_Demo_Sources\cobol\SBANK00P.cbl"
    
    print("Fetching object ID for the given file...")
    object_id = get_object_id(file_path)
    
    if object_id:
        print(f"Object ID retrieved: {object_id}")
        print("Fetching dependencies...")
        
        
        dependencies = get_dependencies(object_id)
        if dependencies:
            display_dependencies(dependencies)
        else:
            print("No dependencies found.")
    else:
        print("Failed to retrieve object ID.")


if __name__ == "__main__":
    main()
