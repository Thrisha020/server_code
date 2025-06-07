
"""
FINAL

Enterprise Analysis
updated: trisha
date:29th jan

"""

import os
import requests
import subprocess
import json
import bokeh
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
import networkx as nx
import os
import json
import math
from bokeh.plotting import figure, save, show
from bokeh.models import MultiLine, HoverTool, LabelSet, ColumnDataSource
from bokeh.io import output_file
from bokeh.plotting.graph import from_networkx

def get_object_id(file_path):
    """
    Step 1: Get the object ID by replacing the file path in the API request.
    """
    url = "http://172.17.64.4:1248/api/workspaces/BankingDemoWS/ObjectByPath"
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

def get_dependencies(object_id):
    """
    Step 2: Get dependencies by replacing the object ID in the API request.
    """
    url = "http://172.17.64.4:1248/api/workspaces/BankingDemoWS/ObjectDirectRelationship"
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
   
    for root, dirs, files in os.walk(current_dir):
        if file_name in files:
            file_path = os.path.join(root, file_name) 
            print(f"Found {file_name} at {file_path}")  # Print where the file was found
            subprocess.run(["code", file_path])  # Open the file in VS Code
            return

    print(f"File {file_name} not found in the current directory or any subdirectory.")

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

from bokeh.io import show, save
from bokeh.plotting import figure, from_networkx
from bokeh.models import HoverTool, MultiLine, TapTool, CustomJS, ColumnDataSource
import networkx as nx
from bokeh.models import LabelSet

def visualize_dependencies_old(object_name, dependencies):
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
    plot.title.align = "left"
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

    output_file = "/root/Desktop/Chatbot/html_paths/dependency_graph.html"
    save(plot, filename=output_file)

    # Print the URL in JSON format
    # url = f"file://{os.path.abspath(output_file)}"
    # print(json.dumps({"plot_url": url}, indent=4))

    # # Open the plot in the default web browser
    # show(plot)



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

    tap_tool.callback = callback

    # Open the plot in the default web browser
    # show(plot)
    output_file = "/root/Desktop/Chatbot/html_paths/dependency_graph.html"
    save(plot, filename=output_file)

    return output_file


def visualize_dependencies1(object_name, dependencies):
    """
    Visualize dependencies as a graph using Bokeh and serve the HTML file over HTTP.
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
    plot.title.text_font_size = "18pt"
    plot.title.align = "center"
    plot.grid.grid_line_color = "lightgray"

    # Convert the NetworkX graph to a Bokeh graph
    bokeh_graph = from_networkx(graph, nx.spring_layout, scale=1.5, center=(0, 0))
    bokeh_graph.node_renderer.data_source.data["node_label"] = list(graph.nodes)

    # Customize node colors and sizes
    node_colors = ["#87ceeb" if node == object_name else "#ffcccb" for node in graph.nodes]
    node_sizes = [50 if node == object_name else 40 for node in graph.nodes]

    bokeh_graph.node_renderer.data_source.data["fill_color"] = node_colors
    bokeh_graph.node_renderer.data_source.data["size"] = node_sizes

    bokeh_graph.node_renderer.glyph.fill_color = "fill_color"
    bokeh_graph.node_renderer.glyph.line_color = "black"
    bokeh_graph.node_renderer.glyph.line_width = 2
    bokeh_graph.node_renderer.glyph.size = "size"

    pos = nx.spring_layout(graph, scale=1.5, center=(0, 0))
    
    # Customize edge appearance
    bokeh_graph.edge_renderer.glyph = MultiLine(line_color="#a3a3a3", line_alpha=0.9, line_width=3)

    # Add hover tool
    hover_tool = HoverTool(tooltips=[("Node", "@node_label")])
    plot.add_tools(hover_tool)

    # Add the graph to the plot
    plot.renderers.append(bokeh_graph)

    # Add text labels to the nodes
    

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


    # Save the plot to an HTML file
    output_file = "/root/Desktop/Chatbot/html_paths/dependency_graph.html"
    save(plot, filename=output_file)

    # Serve the HTML file over HTTP
    #port = 8000
    # os.chdir(os.path.dirname(os.path.abspath(output_file)))  # Change to the directory of the HTML file
    # httpd = TCPServer(("", port), SimpleHTTPRequestHandler)
    # print(f"Serving the plot at http://10.190.226.6:{port}/{output_file}")
    # print("Press Ctrl+C to stop the server.")

    # # Print the URL in JSON format
    # print(json.dumps({"plot_url": f"http://10.190.226.6:{port}/{output_file}"}, indent=4))

    return output_file

    # Open the plot in the default web browser (only works locally)
    # try:
    #     show(plot)
    # except Exception as e:
    #     print(f"Could not open the plot in the browser: {e}")

    # # Start the HTTP server
    # httpd.serve_forever()

def get_base_path(file_name):
    """
    Determine the base path based on the file extension.
    """
    # Define the base directories
    cobol_base_path = r"C:\Users\Administrator.GRT-EA-WDC2\Downloads\Rocket EA\Banking_Demo_Sources\cobol"
    copybook_base_path = r"C:\Users\Administrator.GRT-EA-WDC2\Downloads\Rocket EA\Banking_Demo_Sources\copybook"

    # Check the file extension
    if file_name.lower().endswith((".cbl", ".cobol")):
        return cobol_base_path
    elif file_name.lower().endswith(".cpy"):
        return copybook_base_path
    else:
        raise ValueError("Unsupported file type. Please provide a COBOL (.cbl, .cobol) or copybook (.cpy) file.")





def visualize_dependencies12(object_name, dependencies):
    """
    Visualize dependencies as a graph using Bokeh and serve the HTML file over HTTP.
    Uses Kamada-Kawai algorithm for better node positioning with minimal overlap.
    """
    # Create a directed graph
    graph = nx.DiGraph()
    graph.add_node(object_name)  # Add the main object as a node

    # Add edges from the main object to its dependencies
    for dep in dependencies:
        graph.add_edge(object_name, dep['name'])

    # Create a Bokeh plot with more space
    plot = figure(title="Dependency Graph", 
                  width=700, height=600,  # Larger plot size
                  x_range=(-6, 6), y_range=(-6, 6),  # Wider ranges
                  tools="pan,wheel_zoom,reset,save", 
                  active_scroll="wheel_zoom")
    plot.title.text_font_size = "16pt"
    plot.title.align = "center"
    plot.background_fill_color = "lightgrey"

    # Get the number of dependencies
    num_dependencies = len(dependencies)
    
    # Create a custom circular layout
    pos = {}
    
    # Place the main object in the center
    pos[object_name] = (0, 0)
    
    # Place dependencies in a circle around the main object
    radius = 4  # Adjust radius based on number of nodes
    
    # Calculate positions for the dependencies in a circular layout
    for i, dep in enumerate(dependencies):
        angle = 2 * math.pi * i / num_dependencies
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        pos[dep['name']] = (x, y)
    
    # Convert the NetworkX graph to a Bokeh graph
    bokeh_graph = from_networkx(graph, pos)
    bokeh_graph.node_renderer.data_source.data["node_label"] = list(graph.nodes)

    # Customize node colors and sizes
    node_colors = ["#87ceeb" if node == object_name else "#ffcccb" for node in graph.nodes]
    node_sizes = [100 if node == object_name else 80 for node in graph.nodes]

    bokeh_graph.node_renderer.data_source.data["fill_color"] = node_colors
    bokeh_graph.node_renderer.data_source.data["size"] = node_sizes

    bokeh_graph.node_renderer.glyph.fill_color = "fill_color"
    bokeh_graph.node_renderer.glyph.line_color = "black"
    bokeh_graph.node_renderer.glyph.line_width = 2
    bokeh_graph.node_renderer.glyph.size = "size"

    # Customize edge appearance
    bokeh_graph.edge_renderer.glyph = MultiLine(line_color="#a3a3a3", line_alpha=0.8, line_width=2)

    # Add hover tool
    hover_tool = HoverTool(tooltips=[("Node", "@node_label")])
    plot.add_tools(hover_tool)

    # Add the graph to the plot
    plot.renderers.append(bokeh_graph)

    # Add text labels to the nodes with slight offset from nodes for better readability
    x_coords = []
    y_coords = []
    labels = []
    
    # Calculate label positions with a slight offset
    for node in graph.nodes:
        x, y = pos[node]
        # For dependency nodes, slightly increase the offset in the direction from center
        if node != object_name:
            # Normalize direction vector
            dist = math.sqrt(x*x + y*y)
            if dist > 0:
                # Add a slight offset in the same direction
                offset_factor = 1.15  # 15% further out than the node
                x = x * offset_factor
                y = y * offset_factor
        
        x_coords.append(x)
        y_coords.append(y)
        labels.append(node)

    labels_source = ColumnDataSource(data=dict(
        x=x_coords,
        y=y_coords,
        labels=labels
    ))

    label_set = LabelSet(x='x', y='y', text='labels', source=labels_source,
                          text_font_size="10pt", text_color="#333333",
                          text_align="center", text_baseline="middle")
    plot.add_layout(label_set)

    # Hide axes and outline
    plot.xaxis.visible = False
    plot.yaxis.visible = False
    plot.outline_line_color = None
    plot.grid.visible = False

    # Save the plot to an HTML file
    output_file = "/root/Desktop/Chatbot/html_paths/dependency_graph.html"
    save(plot, filename=output_file)

    return output_file




def visualize_dependencies_advanced(object_name, dependencies):
    """
    Create an advanced dependency graph that matches the sample topology image.
    """
    if not dependencies:
        print("No dependencies to visualize.")
        return

    # Create a directed graph
    graph = nx.DiGraph()
    graph.add_node(object_name)

    # Add edges and collect node types
    node_types = {object_name: "MAIN"}  # Main object type
    
    for dep in dependencies:
        graph.add_edge(object_name, dep['name'])
        node_types[dep['name']] = dep.get('type', 'UNKNOWN')

    # Create a larger Bokeh plot
    plot = figure(
        title="Topology example", 
        width=1200, 
        height=800,
        x_range=(-12, 12),
        y_range=(-10, 10),
        tools="pan,wheel_zoom,reset,save", 
        active_scroll="wheel_zoom",
        background_fill_color="white"
    )
    plot.title.text_font_size = "18pt"
    plot.title.align = "left"
    plot.title.text_color = "black"

    # Create layered layout with alternating up/down positioning
    pos = {}
    pos[object_name] = (0, 0)  # Center position for main object
    
    num_dependencies = len(dependencies)
    
    # Layer 1: First 6 dependencies with alternating up/down
    layer1_radius = 6.0
    layer1_nodes = min(6, num_dependencies)
    
    # Layer 2: Additional dependencies (farther from center)  
    layer2_radius = 9.5
    layer2_nodes = max(0, num_dependencies - 6)
    
    for i, dep in enumerate(dependencies):
        if i < 6:  # First layer with alternating pattern
            base_angle = 2 * math.pi * i / layer1_nodes
            
            # Alternating up/down positioning
            if i % 2 == 0:  # Even indices go up
                y_offset = 1.0
            else:  # Odd indices go down
                y_offset = -1.0
            
            # Calculate position with alternating y-offset
            x = layer1_radius * math.cos(base_angle)
            y = layer1_radius * math.sin(base_angle) + y_offset
            
            pos[dep['name']] = (x, y)
        else:  # Second layer
            layer2_index = i - 6
            angle = 2 * math.pi * layer2_index / layer2_nodes
            # Offset angle to prevent overlap with first layer
            angle_offset = math.pi / layer2_nodes
            
            # Alternating pattern for second layer too
            if layer2_index % 2 == 0:
                y_offset = 0.8
            else:
                y_offset = -0.8
            
            x = layer2_radius * math.cos(angle + angle_offset)
            y = layer2_radius * math.sin(angle + angle_offset) + y_offset
            pos[dep['name']] = (x, y)

    # Calculate proper node widths based on text length for capsule shape
    def get_node_width(text):
        base_width = 2.8
        char_width = 0.12
        return max(base_width, len(text) * char_width + 0.8)

    # Add main node as capsule (ellipse) with exact purple colors
    main_width = get_node_width(object_name)
    main_source = ColumnDataSource(data=dict(
        x=[pos[object_name][0]],
        y=[pos[object_name][1]],
        width=[main_width],
        height=[0.8],
        labels=[object_name]
    ))
    
    main_renderer = plot.ellipse(x='x', y='y', width='width', height='height', 
                                source=main_source, 
                                fill_color="#A56EFF",  # Light purple
                                line_color="#774ECB",  # Dark purple border
                                line_width=2,
                                fill_alpha=1.0)

    # Store renderers for hover coordination
    node_renderers = []
    line_renderers = []
    dot_renderers = []

    # Add dependency nodes and their connections with improved spacing
    for i, dep in enumerate(dependencies):
        dep_pos = pos[dep['name']]
        dep_width = get_node_width(dep['name'])
        
        # Create curved connection line with better spacing
        start_pos = pos[object_name]
        end_pos = pos[dep['name']]
        
        # Create curved path with unique curve for each line to prevent overlap
        num_points = 30
        path_x = []
        path_y = []
        
        # Calculate control point with better spacing
        mid_x = (start_pos[0] + end_pos[0]) / 2
        mid_y = (start_pos[1] + end_pos[1]) / 2
        
        # Create unique curve offset for each line based on angle
        node_angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        curve_base = 2.5
        
        # Vary curve offset based on node index to spread lines
        curve_variation = (i % 4 - 1.5) * 0.8
        
        # Calculate perpendicular offset to curve direction
        perp_angle = node_angle + math.pi/2
        curve_offset_x = curve_base * math.cos(perp_angle) + curve_variation * math.cos(perp_angle)
        curve_offset_y = curve_base * math.sin(perp_angle) + curve_variation * math.sin(perp_angle)
        
        control_x = mid_x + curve_offset_x
        control_y = mid_y + curve_offset_y
        
        # Generate curved path using quadratic Bezier
        for j in range(num_points + 1):
            t = j / num_points
            x = (1-t)**2 * start_pos[0] + 2*(1-t)*t * control_x + t**2 * end_pos[0]
            y = (1-t)**2 * start_pos[1] + 2*(1-t)*t * control_y + t**2 * end_pos[1]
            path_x.append(x)
            path_y.append(y)
        
        # Create line data source (without dynamic color column)
        line_source = ColumnDataSource(data=dict(
            x=path_x, 
            y=path_y,
            node_id=[i] * len(path_x)
        ))
        
        # Draw the curved dashed line with fixed color (will be changed via JavaScript)
        line_renderer = plot.line(x='x', y='y', source=line_source,
                                 line_color="#BBBBBB",  # Fixed gray color
                                 line_width=2, 
                                 line_dash="dashed",
                                 line_alpha=0.8)
        line_renderers.append(line_renderer)
        
        # Add dots along the curved path
        dot_indices = [5, 10, 15, 20, 25]
        dot_x_line = [path_x[j] for j in dot_indices if j < len(path_x)]
        dot_y_line = [path_y[j] for j in dot_indices if j < len(path_y)]
        
        if dot_x_line and dot_y_line:
            dot_source = ColumnDataSource(data=dict(
                x=dot_x_line, 
                y=dot_y_line,
                node_id=[i] * len(dot_x_line)
            ))
            
            dot_renderer = plot.circle(x='x', y='y', size=5,
                                      source=dot_source,
                                      fill_color="#BBBBBB",  # Fixed gray color
                                      line_color="#BBBBBB",
                                      alpha=0.8)
            dot_renderers.append(dot_renderer)

        # Create individual data source for each dependency node
        dep_source = ColumnDataSource(data=dict(
            x=[dep_pos[0]], 
            y=[dep_pos[1]],
            width=[dep_width],
            height=[0.8],
            labels=[dep['name']],
            node_id=[i]
        ))

        # Add dependency node as capsule (ellipse)
        dep_renderer = plot.ellipse(x='x', y='y', width='width', height='height',
                                   source=dep_source,
                                   fill_color="#3DDBD9",  # Keep teal fill
                                   line_color="#2CBCBA",  # Dark teal border
                                   line_width=1,
                                   fill_alpha=1.0,
                                   hover_fill_color="#3DDBD9",  # Keep same teal fill on hover
                                   hover_line_color="#A56EFF",  # Only border turns purple
                                   hover_line_width=3)
        
        node_renderers.append(dep_renderer)

    # JavaScript callback for coordinated hover effects
    from bokeh.models import CustomJS
    
    # Create callback to change line colors on node hover
    hover_callback = CustomJS(
        args=dict(line_renderers=line_renderers, dot_renderers=dot_renderers),
        code="""
        // Reset all lines and dots to default color
        for (let i = 0; i < line_renderers.length; i++) {
            line_renderers[i].glyph.line_color = '#BBBBBB';
            if (i < dot_renderers.length) {
                dot_renderers[i].glyph.fill_color = '#BBBBBB';
                dot_renderers[i].glyph.line_color = '#BBBBBB';
            }
        }
        
        // If hovering over a node, change its corresponding line
        if (cb_data.index.indices.length > 0) {
            const node_idx = source.data.node_id[cb_data.index.indices[0]];
            
            if (node_idx < line_renderers.length) {
                // Change line to purple
                line_renderers[node_idx].glyph.line_color = '#A56EFF';
                
                // Change dots to purple
                if (node_idx < dot_renderers.length) {
                    dot_renderers[node_idx].glyph.fill_color = '#A56EFF';
                    dot_renderers[node_idx].glyph.line_color = '#A56EFF';
                }
            }
        }
        """
    )

    # Reset callback for mouse leave
    reset_callback = CustomJS(
        args=dict(line_renderers=line_renderers, dot_renderers=dot_renderers),
        code="""
        // Reset all lines and dots to default color
        for (let i = 0; i < line_renderers.length; i++) {
            line_renderers[i].glyph.line_color = '#BBBBBB';
            if (i < dot_renderers.length) {
                dot_renderers[i].glyph.fill_color = '#BBBBBB';
                dot_renderers[i].glyph.line_color = '#BBBBBB';
            }
        }
        """
    )

    # Add hover tool with callbacks
    hover_tool = HoverTool(
        renderers=node_renderers,
        tooltips=[("File", "@labels")],
        callback=hover_callback
    )
    plot.add_tools(hover_tool)

    # Add mouse leave callback to reset colors
    for renderer in node_renderers:
        renderer.data_source.js_on_change('selected', reset_callback)

    # Add colored dots inside nodes (darker shade)
    # Main node dot
    main_dot_source = ColumnDataSource(data=dict(
        x=[pos[object_name][0] - (main_width/2 - 0.3)],
        y=[pos[object_name][1]],
        color=["#774ECB"]  # Dark purple dot
    ))
    plot.circle(x='x', y='y', size=8,
                source=main_dot_source,
                fill_color='color',
                line_color='color',
                alpha=1.0)

    # Dependency node dots (darker teal, will change to purple on hover)
    for dep in dependencies:
        dep_pos = pos[dep['name']]
        dep_width = get_node_width(dep['name'])
        
        dot_source = ColumnDataSource(data=dict(
            x=[dep_pos[0] - (dep_width/2 - 0.3)],
            y=[dep_pos[1]],
            color=["#2CBCBA"]  # Dark teal (darker shade)
        ))
        plot.circle(x='x', y='y', size=6,
                    source=dot_source,
                    fill_color='color',
                    line_color='color',
                    alpha=1.0,
                    hover_fill_color="#774ECB",  # Dark purple on hover
                    hover_line_color="#774ECB")

    # Add text labels for main node
    main_label_source = ColumnDataSource(data=dict(
        x=[pos[object_name][0] - (main_width/2 - 0.6)],
        y=[pos[object_name][1]],
        labels=[object_name]
    ))
    
    main_labels = LabelSet(
        x='x', y='y', text='labels',
        source=main_label_source,
        text_font_size="10pt",
        text_color="white",
        text_font_style="bold",
        text_align="left",
        text_baseline="middle"
    )
    plot.add_layout(main_labels)

    # Add text labels for dependency nodes
    for dep in dependencies:
        dep_pos = pos[dep['name']]
        dep_width = get_node_width(dep['name'])
        
        dep_label_source = ColumnDataSource(data=dict(
            x=[dep_pos[0] - (dep_width/2 - 0.6)],
            y=[dep_pos[1]],
            labels=[dep['name']]
        ))
        
        dep_labels = LabelSet(
            x='x', y='y', text='labels',
            source=dep_label_source,
            text_font_size="8pt",
            text_color="#2F4F4F",
            text_align="left",
            text_baseline="middle"
        )
        plot.add_layout(dep_labels)

    # Hide axes and grid
    plot.xaxis.visible = False
    plot.yaxis.visible = False
    plot.xgrid.visible = False
    plot.ygrid.visible = False
    plot.outline_line_color = None

    # Save and display
    output_file_name = "/root/Desktop/Chatbot/html_paths/dependency_graph.html"
    output_file(output_file_name)
    save(plot)

    url = f"file://{os.path.abspath(output_file_name)}"
    print(json.dumps({"plot_url": url}, indent=4))
    
    show(plot)
    return output_file_name



def main(file_name):
    # Ask the user for the COBOL file name
    #file_name = input("Please enter the COBOL file name: ")

    # Determine the base path based on the file extension
    try:
        base_path = get_base_path(file_name)
    except ValueError as e:
        print(e)
        return

    # Construct the full file path
    file_path = os.path.join(base_path, file_name)
    file_path = os.path.normpath(file_path)
    file_path = file_path.replace("/", "\\")

    # print(f"Constructed file path: {file_path}")

    # Fetch the object ID and name
    # print(f"Fetching object ID for the given file '{file_name}'...")
    object_id, object_name = get_object_id(file_path)

    if object_id:
        # print(f"Object ID retrieved: {object_id}")
        # print("Fetching dependencies...")

        # Fetch the dependencies
        dependencies = get_dependencies(object_id)
        dependencies = [dep for dep in dependencies if dep['name']!=file_name.split('.')[0]]
        #print("dependencies............",dependencies)
        if dependencies:
            dependencies_list = display_dependencies(dependencies, base_path)
            html_path = visualize_dependencies_advanced(object_name, dependencies)
            return dependencies_list,html_path
       
            #print('\ndepen:     ',dependencies_list)
            
        else:
            print("No dependencies found.")
            return [],None
    else:
        print("Failed to retrieve object ID.")
# if __name__ == "__main__":
#      print(main('SBANK00P.cbl'))



# SBANK00P.cbl

# 172.17.64.4:1248

# MBANKZZ.cpy

# def read_html_file(file_path):
#     try:
#         with open(file_path, 'r') as file:
#             html_content = file.read()
#         return html_content
#     except FileNotFoundError:
#         return f"Error: The file at {file_path} was not found."
#     except Exception as e:
#         return f"Error: {str(e)}"

# # Example usage
# file_path = "/root/Desktop/Chatbot/html_paths/dependency_graph.html"
# html_code = read_html_file(file_path)
# print(html_code)
