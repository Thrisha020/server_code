"""

TAKE BANK AND THE MORTGAGE EA LINK DYNAMICALLY

"""
import os
import requests
import subprocess
import json
from bokeh.io import show, save
from bokeh.plotting import figure, from_networkx
from bokeh.models import HoverTool, MultiLine
import networkx as nx
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer

import json
from bokeh.io import show, save
from bokeh.plotting import figure, from_networkx
from bokeh.models import HoverTool, MultiLine
from bokeh.io import show, save
from bokeh.plotting import figure, from_networkx
from bokeh.models import HoverTool, MultiLine
from bokeh.models import ColumnDataSource, LabelSet
import networkx as nx
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
import os, yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatDeepInfra

def workspace_detector(repo_name):
    os.environ["DEEPINFRA_API_TOKEN"] = "ag1JDMRLMu7DS1hoQM0sM8HEPyrLKBav"

    template = """
    You are taskked to determine the correct workspace name for the given repository from the below given workspaces.

    Repository: {input}

    Available Workspaces:
    1. "BankingDemoWS"
    2. "MortgageAppWS"

    Output: Provide only the name of the matching workspace name. Do not give anything else.
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatDeepInfra(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        temperature=0.9,
    )

    chain = prompt | llm


    result = chain.invoke({
            "input": repo_name
        })
    return result.content
    

def get_object_id(file_path,ws_object_id_url):
    """
    Step 1: Get the object ID by replacing the file path in the API request.
    """
    #url = "http://10.190.226.42:1248/api/workspaces/MortgageAppWS/ObjectByPath"
    url = f'{ws_object_id_url}/ObjectByPath'
    params = {"path": file_path}
    headers = {"accept": "application/json"}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        result = response.json()
        object_id = result.get("id")  # Assuming the response has an 'id' field
        object_name = result.get("name")  # Assuming the response has a 'name' field
        return object_id, object_name
    else:
        print(f"Error getting object ID: {response.status_code}")
        print(response.text)
        return None, None

def get_dependencies(object_id,ws_object_id_url):
    """
    Step 2: Get dependencies by replacing the object ID in the API request.
    """
    #url = "http://10.190.226.42:1248/api/workspaces/MortgageAppWS/ObjectDirectRelationship"
    url = f"{ws_object_id_url}/ObjectDirectRelationship"
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

def open_copybook_in_vscode(file_name):
    """
    Open a copybook file in Visual Studio Code by searching in the current working directory and its subdirectories.
    """
    current_dir = os.getcwd()  
    # Use os.walk to search through all directories and subdirectories
    for root, dirs, files in os.walk(current_dir):
        if file_name in files:
            file_path = os.path.join(root, file_name)  # Full path to the file
            print(f"Found {file_name} at {file_path}")  # Print where the file was found
            subprocess.run(["code", file_path])  # Open the file in VS Code
            return

    print(f"File {file_name} not found in the current directory or any subdirectory.")

def display_dependencies1(dependencies, base_path):
    """
    Step 3: Display dependencies and optionally open copybooks in Visual Studio Code.
    """

    print("Chat: Please find the list of dependency code:")
    for dep in dependencies:

        print(f"{dep['name']} (Type: {dep['type']}, Relation: {dep['relation']})")
       
        if dep['type'] == "COPYBOOK":
            open_copybook_in_vscode(dep['name'])

        elif dep['type'] == "COBOL":
            open_copybook_in_vscode(dep['name'])
            
def display_dependencies(dependencies, base_path):
    """
    Step 3: Display dependencies and optionally open copybooks in Visual Studio Code.
    """
    print("Chat: Please find the list of dependency code:")
    dependencies_list = []
    for dep in dependencies:
        print(f"{dep['name']} (Type: {dep['type']}, Relation: {dep['relation']})")
        #open_copybook_in_vscode(dep['name'])
        dependencies_list.append(dep['name'])
    
    return dependencies_list
        


# def visualize_dependencies(object_name, dependencies):
#     """
#     Visualize dependencies as a graph using Bokeh and capture the URL.
#     """
#     # Create a directed graph
#     graph = nx.DiGraph()
#     graph.add_node(object_name)  # Add the main object as a node

#     # Add edges from the main object to its dependencies
#     for dep in dependencies:
#         graph.add_edge(object_name, dep['name'])

#     # Create a Bokeh plot
#     plot = figure(title="Dependency Graph", x_range=(-2, 2), y_range=(-2, 2),
#                   tools="pan,wheel_zoom,reset", active_scroll="wheel_zoom",
#                   background_fill_color="#f7f9fc", border_fill_color="#ffffff")
#     plot.title.text_font_size = "18pt"
#     plot.title.align = "center"
#     plot.grid.grid_line_color = "lightgray"

#     # Convert the NetworkX graph to a Bokeh graph
#     bokeh_graph = from_networkx(graph, nx.spring_layout, scale=1.5, center=(0, 0))
#     bokeh_graph.node_renderer.data_source.data["node_label"] = list(graph.nodes)

#     # Customize node colors and sizes
#     node_colors = ["#87ceeb" if node == object_name else "#ffcccb" for node in graph.nodes]
#     node_sizes = [50 if node == object_name else 40 for node in graph.nodes]

#     bokeh_graph.node_renderer.data_source.data["fill_color"] = node_colors
#     bokeh_graph.node_renderer.data_source.data["size"] = node_sizes

#     bokeh_graph.node_renderer.glyph.fill_color = "fill_color"
#     bokeh_graph.node_renderer.glyph.line_color = "black"
#     bokeh_graph.node_renderer.glyph.line_width = 1.5
#     bokeh_graph.node_renderer.glyph.size = "size"

#     # Customize edge appearance
#     bokeh_graph.edge_renderer.glyph = MultiLine(line_color="#a3a3a3", line_alpha=0.8, line_width=2)

#     # Add hover tool
#     hover_tool = HoverTool(tooltips=[("Node", "@node_label")])
#     plot.add_tools(hover_tool)

#     # Add the graph to the plot
#     plot.renderers.append(bokeh_graph)

#     # Hide axes and outline
#     plot.xaxis.visible = False
#     plot.yaxis.visible = False
#     plot.outline_line_color = None

#     # Save the plot to an HTML file and get the file path
#     output_file = "dependency_graph.html"
#     save(plot, filename=output_file)

#     # Print the URL in JSON format
#     url = f"file://{os.path.abspath(output_file)}"
#     print(json.dumps({"plot_url": url}, indent=4))

#     # Open the plot in the default web browser
#     show(plot)
def visualize_dependencies1(object_name, dependencies):
    """
    Visualize dependencies as a graph using Bokeh and capture the URL.
    """
    # Create a directed graph
    graph = nx.DiGraph()
    graph.add_node(object_name)  # Add the main object as a node

    # Add edges from the main object to its dependencies
    for dep in dependencies:
        graph.add_edge(object_name, dep['name'])

    # Create a Bokeh plot
    plot = figure(title="Dependency Graph", x_range=(-2, 2), y_range=(-2, 2),
                  tools="pan,wheel_zoom,reset", active_scroll="wheel_zoom",
                  background_fill_color="#f7f9fc", border_fill_color="#ffffff")
    plot.title.text_font_size = "12pt"
    plot.title.align = "center"
    #plot.grid.grid_line_color = "lightgray"
    plot.background_fill_color = "LightBlue"

    # Use NetworkX to compute the layout positions for the graph
    pos = nx.spring_layout(graph, scale=1.5, center=(0, 0))

    # Convert the NetworkX graph to a Bokeh graph using the computed positions
    bokeh_graph = from_networkx(graph, pos)
    bokeh_graph.node_renderer.data_source.data["node_label"] = list(graph.nodes)

    # Customize node colors and sizes
    node_colors = ["#87ceeb" if node == object_name else "#ffcccb" for node in graph.nodes]
    node_sizes = [80 if node == object_name else 70 for node in graph.nodes]

    bokeh_graph.node_renderer.data_source.data["fill_color"] = node_colors
    bokeh_graph.node_renderer.data_source.data["size"] = node_sizes

    bokeh_graph.node_renderer.glyph.fill_color = "fill_color"
    bokeh_graph.node_renderer.glyph.line_color = "black"
    bokeh_graph.node_renderer.glyph.line_width = 2
    bokeh_graph.node_renderer.glyph.size = "size"

    # Customize edge appearance
    bokeh_graph.edge_renderer.glyph = MultiLine(line_color="#a3a3a3", line_alpha=0.9, line_width=3)

    # Add hover tool
    hover_tool = HoverTool(tooltips=[("Node", "@node_label")])
    plot.add_tools(hover_tool)

    # Add the graph to the plot
    plot.renderers.append(bokeh_graph)

    # Add text labels to the nodes
    from bokeh.models import ColumnDataSource, LabelSet

    # Extract node positions from the layout
    x_coords = [pos[node][0] for node in graph.nodes]  # X-coordinates of nodes
    y_coords = [pos[node][1] for node in graph.nodes]  # Y-coordinates of nodes
    labels = list(graph.nodes)  # Node labels

    # Create a ColumnDataSource for the labels
    labels_source = ColumnDataSource(data=dict(
        x=x_coords,
        y=y_coords,
        labels=labels
    ))

    # Add labels to the plot
    labels = LabelSet(x='x', y='y', text='labels', source=labels_source,
                      text_font_size="12pt", text_color="#333333",
                      text_align="center", text_baseline="middle")
    plot.add_layout(labels)

    # Hide axes and outline
    plot.xaxis.visible = False
    plot.yaxis.visible = False
    plot.outline_line_color = None

    # Save the plot to an HTML file and get the file path
    # output_file = "dependency_graph.html"
    # save(plot, filename=output_file)

    # Print the URL in JSON format
    # url = f"file://{os.path.abspath(output_file)}"
    # print(json.dumps({"plot_url": url}, indent=4))

    # Open the plot in the default web browser
    show(plot)
    output_file = "/root/Desktop/Chatbot/html_paths/dependency_graph1.html"
    save(plot, filename=output_file)

    return output_file


from bokeh.models import HoverTool, MultiLine, TapTool, CustomJS, ColumnDataSource
def visualize_dependencies(object_name, dependencies):
    """
    Visualize dependencies as a graph using Bokeh and capture the URL.
    """
    # Create a directed graph
    graph = nx.DiGraph()
    graph.add_node(object_name)  # Add the main object as a node
    # Add edges from the main object to its dependencies
    for dep in dependencies:
        graph.add_edge(object_name, dep['name'])

    # Create a Bokeh plot
    plot = figure(title="Dependency Graph", x_range=(-2, 2), y_range=(-2, 2),
                  tools="pan,wheel_zoom,reset", active_scroll="wheel_zoom",
                  background_fill_color="#f7f9fc", border_fill_color="#ffffff")
    plot.title.text_font_size = "12pt"
    plot.title.align = "center"

    # Use NetworkX to compute the layout positions for the graph
    pos = nx.spring_layout(graph, scale=1.5, center=(0, 0))

    # Convert the NetworkX graph to a Bokeh graph using the computed positions
    bokeh_graph = from_networkx(graph, pos)
    bokeh_graph.node_renderer.data_source.data["node_label"] = list(graph.nodes)

    # Customize node colors and sizes
    node_colors = ["#87ceeb" if node == object_name else "#ffcccb" for node in graph.nodes]
    node_sizes = [80 if node == object_name else 70 for node in graph.nodes]
    bokeh_graph.node_renderer.data_source.data["fill_color"] = node_colors
    bokeh_graph.node_renderer.data_source.data["size"] = node_sizes
    bokeh_graph.node_renderer.glyph.fill_color = "fill_color"
    bokeh_graph.node_renderer.glyph.line_color = "black"
    bokeh_graph.node_renderer.glyph.line_width = 2
    bokeh_graph.node_renderer.glyph.size = "size"

    # Customize edge appearance
    bokeh_graph.edge_renderer.glyph = MultiLine(line_color="#a3a3a3", line_alpha=0.9, line_width=3)

    # Add hover tool
    hover_tool = HoverTool(tooltips=[("Node", "@node_label")])
    plot.add_tools(hover_tool)

    # Add the graph to the plot
    plot.renderers.append(bokeh_graph)

    # Add text labels to the nodes
    
    x_coords = [pos[node][0] for node in graph.nodes]
    y_coords = [pos[node][1] for node in graph.nodes]
    labels = list(graph.nodes)
    labels_source = ColumnDataSource(data=dict(x=x_coords, y=y_coords, labels=labels))
    labels = LabelSet(x='x', y='y', text='labels', source=labels_source,
                      text_font_size="12pt", text_color="#333333",
                      text_align="center", text_baseline="middle")
    plot.add_layout(labels)

    # Hide axes and outline
    plot.xaxis.visible = False
    plot.yaxis.visible = False
    plot.outline_line_color = None

    # Save the plot to an HTML file and get the file path
    output_file = "dependency_graph.html"
    save(plot, filename=output_file)

    # Print the URL in JSON format
    url = f"file://{os.path.abspath(output_file)}"
    print(json.dumps({"plot_url": url}, indent=4))

    # Add a TapTool for detecting single and double clicks
    tap_tool = TapTool()
    plot.add_tools(tap_tool)

    #Custom JavaScript for handling single and double clicks
    callback = CustomJS(args=dict(source=bokeh_graph.node_renderer.data_source), code="""
    const indices = cb_data.source.selected.indices;
    if (indices.length > 0) {
        const selected_node = source.data['node_label'][indices[0]];

        // Send the selected node value as a query parameter in the GET request
        const apiUrl = `http://10.190.226.6:8000/chatstest?fileanme=${encodeURIComponent(selected_node)}`;
                        
        fetch(apiUrl)
        .then(response => response.json())
        .then(data => {
            console.log('API Response:', data);
                        
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
""")

#     callback = CustomJS(args=dict(source=bokeh_graph.node_renderer.data_source), code="""
#     const indices = cb_data.source.selected.indices;
#     if (indices.length > 0) {
#         const selected_node = source.data['node_label'][indices[0]];

#         // Send the selected node value as a query parameter in the GET request
#         const apiUrl = 'https://grtapp.genairesonance.com/stest?input_text=${encodeURIComponent(selected_node)}';
        
#         fetch(apiUrl)
#         .then(response => response.json())
#         .then(data => {
#             console.log('API Response:', data);
#         })
#         .catch(error => {
#             console.error('Error:', error);
#         });
#     }
# """)    

    #tap_tool.callback = callback

    # Open the plot in the default web browser
    # show(plot)
    output_file = "/root/Desktop/Chatbot/html_paths/dependency_graph1.html"
    save(plot, filename=output_file)

    return output_file


def get_base_path(repo_name,file_name):
    """
    Determine the base path based on the file extension.
    
    """
    workspace_name = workspace_detector(repo_name)
    workspace_name = workspace_name.replace("'","")
    with open('/root/Desktop/Chatbot/ea_ws.yaml', 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    # Define the base directories
    if workspace_name == 'MortgageAppWS':
        #cobol_base_path = r"C:\Users\Administrator.GRT-EA-WDC2\Downloads\Rocket EA\MortgageApplication-main\cobol"
        #copybook_base_path = r"C:\Users\Administrator.GRT-EA-WDC2\Downloads\Rocket EA\MortgageApplication-main\copybook"
        
        cobol_base_path = data['MortgageAppWS']['cobol_base_path']
        copybook_base_path = data['MortgageAppWS']['copybook_base_path']
        
    if workspace_name == 'BankingDemoWS':
        #cobol_base_path : "C:/Users/Administrator.GRT-EA-WDC2/Downloads/Rocket EA/Banking_Demo_Sources/cobol"
        #copybook_base_path : "C:/Users/Administrator.GRT-EA-WDC2/Downloads/Rocket EA/Banking_Demo_Sources/copybook"

        cobol_base_path = data['BankingDemoWS']['cobol_base_path']
        copybook_base_path = data['BankingDemoWS']['copybook_base_path']

    # Check the file extension
    if file_name.lower().endswith((".cbl", ".cobol")):
        return cobol_base_path, workspace_name
    elif file_name.lower().endswith(".cpy"):
        return copybook_base_path, workspace_name
    else:
        raise ValueError("Unsupported file type. Please provide a COBOL (.cbl, .cobol) or copybook (.cpy) file.")




###########
#### finding workspace....

#workspace_detector('banking demo repository')


def ea_main(repo_name,file_name):
    # Ask the user for the COBOL file name
    #file_name = input("Please enter the COBOL file name: ")
    response = requests.get("http://172.17.64.4:1248/api/workspaces")
    if response.status_code == 200:
                #print("✅ Success! Response Data:")
        workspaces = response.json()

    # Determine the base path based on the file extension
    try:
        base_path, workspace_name = get_base_path(repo_name,file_name)
    except ValueError as e:
        print(e)
        return

    # Construct the full file path
    file_path = os.path.join(base_path, file_name)
    file_path = os.path.normpath(file_path)
    file_path = file_path.replace("/", "\\")

    print(f"Constructed file path: {file_path}")
    
    for ws in workspaces:
        if ws['name'] == workspace_name:
            ws_object_id_url = ws['url']
            ws_object_id_url = ws_object_id_url.replace('10.190.226.42','172.17.64.4')
        # if ws['name'] == workspace_name:
        #     ws_object_id_url = ws['url']

    # Fetch the object ID and name
    print(f"Fetching object ID for the given file '{file_name}'...")
    object_id, object_name = get_object_id(file_path,ws_object_id_url)

    if object_id:
        print(f"Object ID retrieved: {object_id}")
        print("Fetching dependencies...")

        # Fetch the dependencies
        dependencies = get_dependencies(object_id,ws_object_id_url)
        print(f'\n dep b4 removal:   {dependencies}')
        dependencies = [dep for dep in dependencies if dep['name']!=file_name.split('.')[0]]
        print(f'\n dep after removal:   {dependencies}')
        if dependencies:
            dependencies_list = display_dependencies(dependencies, base_path)
            html_path = visualize_dependencies(object_name, dependencies)
            
            response = requests.get(f"{ws_object_id_url}/EAWeb")
            if response.status_code == 200:
                        #print("✅ Success! Response Data:")
                workspace_ea_details = response.json()
                workspace_ea_url = workspace_ea_details['url']
                workspace_ea_url = workspace_ea_url.replace('172.17.64.4','10.190.226.42')
            else:
                workspace_ea_url = None
                
            return dependencies_list,html_path,workspace_ea_url
            
            
        else:
            print("No dependencies found.")
            return [],None,None
    else:
        print("Failed to retrieve object ID.")



if __name__ == "__main__":
    print(main('BankingDemo','sbank10p.cbl'))
    
    
    #BankingDemo
    #sbank10p.cbl