import requests
import networkx as nx
import webbrowser
import os
import time
import math
from bokeh.plotting import figure, output_file, save
from bokeh.models import (
    GraphRenderer, StaticLayoutProvider, Circle, LabelSet, ColumnDataSource,
    NodesAndLinkedEdges, EdgesAndLinkedNodes, MultiLine, TapTool, HoverTool, 
    BoxSelectTool, Range1d, Label
)
from bokeh.palettes import Spectral8

BASE_URL = "http://172.17.64.4:1248/api/workspaces/BankingDemoWS"

def get_bms_details(file_name):
    url = f"{BASE_URL}/SearchObjects?name={file_name}&types=%2A"
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200 and response.json():
        return response.json()[0]
    return None

def get_bms_dependencies(bms_id):
    url = f"{BASE_URL}/ObjectRelationships?id={bms_id}"
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None

def get_programs_using_map(map_id):
    url = f"{BASE_URL}/ObjectRelationships?id={map_id}"
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None

def bezier(p0, p1, control, steps):
    """Create a bezier curve between two points with a control point"""
    return [(1 - t) ** 2 * p0 + 2 * (1 - t) * t * control + t ** 2 * p1 for t in steps]

def optimize_layout(pos, scale_factor=2.5, min_distance=0.5):
    """
    Optimize node positions to ensure minimum distance between nodes
    and provide better spacing
    """
    # Scale positions for better spacing
    optimized_pos = {k: (v[0] * scale_factor, v[1] * scale_factor) for k, v in pos.items()}
    
    # Iteratively adjust positions to maintain minimum distance
    converged = False
    iterations = 0
    max_iterations = 50
    
    while not converged and iterations < max_iterations:
        converged = True
        iterations += 1
        
        for node1 in optimized_pos:
            for node2 in optimized_pos:
                if node1 != node2:
                    x1, y1 = optimized_pos[node1]
                    x2, y2 = optimized_pos[node2]
                    
                    # Calculate distance between nodes
                    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    
                    if dist < min_distance:
                        # Calculate repulsion force
                        force = min_distance - dist
                        angle = math.atan2(y2 - y1, x2 - x1)
                        
                        # Apply repulsion force
                        force_x = force * math.cos(angle) * 0.5
                        force_y = force * math.sin(angle) * 0.5
                        
                        optimized_pos[node1] = (x1 - force_x, y1 - force_y)
                        optimized_pos[node2] = (x2 + force_x, y2 + force_y)
                        converged = False
    
    return optimized_pos

def create_node_color_mapping(nodes, root_node):
    """Create a color mapping based on node type/level"""
    node_colors = {}
    node_sizes = {}
    
    # Set different colors based on node level or type
    for node in nodes:
        if node == root_node:
            node_colors[node] = "orange"  # Root node
            node_sizes[node] = 0.15
        elif "." in node:  # Actual files
            node_colors[node] = "skyblue"
            node_sizes[node] = 0.1
        else:  # Group relations - changed from lightgreen to blue
            node_colors[node] = "lightblue"
            node_sizes[node] = 0.12
    
    return node_colors, node_sizes

def calculate_label_offsets(positions, nodes):
    """Calculate optimal label offsets to avoid overlap"""
    x_offsets = {}
    y_offsets = {}
    
    for node in nodes:
        # Calculate offset direction based on position in graph
        x, y = positions[node]
        
        # Determine quadrant and set offset accordingly
        if x >= 0 and y >= 0:  # Q1
            x_offsets[node] = 15
            y_offsets[node] = 15
        elif x < 0 and y >= 0:  # Q2
            x_offsets[node] = -15
            y_offsets[node] = 15
        elif x < 0 and y < 0:  # Q3
            x_offsets[node] = -15
            y_offsets[node] = -15
        else:  # Q4
            x_offsets[node] = 15
            y_offsets[node] = -15
    
    return x_offsets, y_offsets



def format_output(bms_file_name):
    #bms_file_name = input("Enter the file name: ").strip()
    bms_details = get_bms_details(bms_file_name)
    text = ""
    if not bms_details:
        print(" BMS file not found.")
        exit()

    parent_id = bms_details["id"]
    parent_name = bms_details["name"]
    print("\n ID:", parent_id)
    text = text + f"ID :{parent_id}"
    print(" Parent Name:", parent_name, "\n")
    text = text + f"\nName :{parent_name}"

    dependencies_data = get_bms_dependencies(parent_id)
    if not dependencies_data:
        print(" No dependencies found.")
        text = text + f"\nNo dependencies found."
        exit()

    map_data = {}
    for relation in dependencies_data.get("relations", []):
        if relation["groupRelation"] == "Direct Relationships":
            for item in relation["data"]:
                if item["type"] == "MAP":
                    map_data[item["id"]] = item["name"]

    generated_files = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for map_id, map_name in map_data.items():
        program_data = get_programs_using_map(map_id)
        if not program_data:
            continue
        text = text + f"\n*Responsible Node: {map_name}**\n"
        print(f"\n **Responsible Node: {map_name}**\n")
        graph_data = {"nodes": [map_name], "edges": []}
        node_counts = {}

        for relation in program_data.get("relations", []):
            group_name = relation["groupRelation"]
            if not relation["data"]:
                continue

            print(f"📌 Group Relation: {group_name}")
            text = text + f"\nGroup Relation: {group_name}"
            graph_data["nodes"].append(group_name)
            graph_data["edges"].append((map_name, group_name))

            for item in relation["data"]:
                file_name = item["name"]

                if file_name in node_counts:
                    node_counts[file_name] += 1
                    file_name_unique = f"{file_name} ({node_counts[file_name]})"
                else:
                    node_counts[file_name] = 1
                    file_name_unique = file_name

                graph_data["nodes"].append(file_name_unique)
                graph_data["edges"].append((group_name, file_name_unique))
                text = text + f"\n* Name: {file_name_unique}"
                print(f"  -  Child Name: {file_name_unique}")
            print()

        # Create graph
        G = nx.DiGraph()
        for node in graph_data["nodes"]:
            G.add_node(node)
        for edge in graph_data["edges"]:
            G.add_edge(edge[0], edge[1])

        # Use a layout algorithm that provides better spacing
        if len(G.nodes()) < 50:
            pos = nx.kamada_kawai_layout(G)  # Better for smaller graphs
        else:
            pos = nx.spring_layout(G, k=1.5)  # Better for larger graphs
        
        # Further optimize layout for better spacing
        pos = optimize_layout(pos)
        
        # Create node color mapping
        node_colors_map, node_sizes_map = create_node_color_mapping(G.nodes(), map_name)
        
        # Calculate optimal label offsets
        x_offsets, y_offsets = calculate_label_offsets(pos, G.nodes())

        # Calculate the graph boundaries with padding
        all_xs = [pos[node][0] for node in G.nodes()]
        all_ys = [pos[node][1] for node in G.nodes()]
        x_range = max(all_xs) - min(all_xs)
        y_range = max(all_ys) - min(all_ys)
        padding = 0.5  # Add padding to ensure nodes aren't cut off
        
        # Configure plot with better dimensions and ranges
        plot = figure(
            title=f"Dependency Graph for {map_name}", 
            tools="tap,box_select,hover,pan,wheel_zoom,reset", 
            toolbar_location="above",
            background_fill_color="#f5f5f5",
            outline_line_color=None, 
            border_fill_color="#f5f5f5",
            x_axis_type=None, 
            y_axis_type=None,
            x_range=Range1d(min(all_xs) - padding, max(all_xs) + padding),
            y_range=Range1d(min(all_ys) - padding, max(all_ys) + padding),
            width=700, 
            height=500
        )
        
        # Add subtle grid for better visual alignment
        plot.grid.grid_line_color = "#eeeeee"
        plot.grid.grid_line_alpha = 0.5

        # Create graph renderer
        graph_renderer = GraphRenderer()
        
        # Set node data
        node_indices = list(G.nodes())
        node_colors = [node_colors_map[node] for node in node_indices]
        node_sizes = [node_sizes_map[node] for node in node_indices]
        
        graph_renderer.node_renderer.data_source.data = {
            'index': node_indices,
            'colors': node_colors,
            'sizes': node_sizes
        }

        # Configure edges with smoother curves for better visibility
        edge_start = [edge[0] for edge in G.edges()]
        edge_end = [edge[1] for edge in G.edges()]
        steps = [i / 50.0 for i in range(51)]
        xs, ys = [], []
        
        for s, e in zip(edge_start, edge_end):
            sx, sy = pos[s]
            ex, ey = pos[e]
            
            # Calculate edge curvature based on distance
            distance = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
            curvature = min(0.3, distance * 0.2)  # Adaptive curvature
            
            # Create curved edges with varying control points to reduce overlap
            control_x = (sx + ex) / 2
            control_y = (sy + ey) / 2 + curvature
            
            xs.append(bezier(sx, ex, control_x, steps))
            ys.append(bezier(sy, ey, control_y, steps))

        graph_renderer.edge_renderer.data_source.data = dict(
            start=edge_start, 
            end=edge_end, 
            xs=xs, 
            ys=ys
        )

        # Configure node appearance with variable size
        graph_renderer.node_renderer.glyph = Circle(
            radius='sizes', 
            fill_color='colors', 
            line_color='black',
            line_width=1.5
        )
        
        # Configure edge appearance
        graph_renderer.edge_renderer.glyph = MultiLine(
            line_color="#555555", 
            line_alpha=0.7, 
            line_width=1.5
        )

        # Configure selection and hover effects
        graph_renderer.node_renderer.selection_glyph = Circle(
            fill_color="red", 
            line_width=2
        )
        graph_renderer.node_renderer.hover_glyph = Circle(
            fill_color="#FFCC00", 
            line_width=2
        )
        graph_renderer.edge_renderer.selection_glyph = MultiLine(
            line_color="red", 
            line_width=2.5
        )
        graph_renderer.edge_renderer.hover_glyph = MultiLine(
            line_color="#FFCC00", 
            line_width=2.5
        )

        # Configure hover tool
        node_hover_tool = HoverTool(tooltips=[("Name", "@index")])
        plot.add_tools(node_hover_tool)

        # Set interactive policies
        graph_renderer.selection_policy = NodesAndLinkedEdges()
        graph_renderer.inspection_policy = EdgesAndLinkedNodes()

        # Set layout
        graph_renderer.layout_provider = StaticLayoutProvider(graph_layout=pos)

        # Create node labels with optimized placement
        label_data = ColumnDataSource(data=dict(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            names=[node for node in G.nodes()],
            x_offset=[x_offsets[node] for node in G.nodes()],

            y_offset=[y_offsets[node] for node in G.nodes()]
        ))

        # Add labels with dynamic offsets and better formatting
        labels = LabelSet(
            x='x', 
            y='y', 
            text='names', 
            source=label_data,
            text_font_size="10px", 
            text_font_style="bold",
            text_color="black",
            text_align='left',
            background_fill_color="white",
            background_fill_alpha=0.7,
            x_offset='x_offset', 
            y_offset='y_offset',
            border_line_color="lightgrey",
            border_line_alpha=0.5
        )

        # Add renderers
        plot.renderers.append(graph_renderer)
        plot.add_layout(labels)

        # Add legend note
        legend_note = Label(
            x=10, y=10, 
            x_units='screen', y_units='screen',
            #text=f"● Root node  ● Group relation  ● File",
            text_font_size="10px",
            background_fill_color="white",
            background_fill_alpha=0.7
        )
        plot.add_layout(legend_note)

        # Save file
        file_counter = 1
        file_name = f"dependency_graph_{map_name}_{timestamp}.html"

        while os.path.exists(file_name):
            file_name = f"dependency_graph_{map_name}_{timestamp}_{file_counter}.html"
            file_counter += 1

       # output_file(file_name)
        output_file(f"/root/Desktop/Chatbot/html_paths/{file_name}")
        save(plot)

        generated_files.append(file_name)

    if generated_files:
        print("\n Generated Graph Files:")
        for file in generated_files:
            abs_path = os.path.abspath(file)
            file_url = f"'http://10.190.226.6:8000/chathtml_template?html_filename={file_name}"
            print(f"{file} → Open in browser: {file_url}")
            webbrowser.open(file_url)
    else:
        print(" No graphs were generated.")

    html_link_1 = "file:///Users/thrisham/Desktop/MBANK70.BANK70A.htm"
    html_link_2 = "file:///Users/thrisham/Desktop/MBANK70.HELP70A.htm"

    print("MBANK70.BANK70A.htm :", html_link_1, "\n", "MBANK70.HELP70A.htm :", html_link_2)

    return text , generated_files , html_link_1,html_link_2

#print(bokeh_graph("MBANK70.bms"))
