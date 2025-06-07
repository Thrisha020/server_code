"""

FINAL

Fixed i18n network graph - updated: may 26th 2025
All messages come from .po files only
"""

import requests
import networkx as nx
import webbrowser
import os
import time
import math
from bokeh.plotting import figure, output_file, save
import gettext
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # Consistent language detection

try:
    BASE_DIR = os.path.dirname(__file__)
except NameError:
    BASE_DIR = os.getcwd()  # fallback if __file__ not defined

LOCALE_PATH = os.path.join(BASE_DIR, "locale")

# Global translation function - initialize with default
_ = lambda x: x  # Default fallback

def detect_language(user_input):
    try:
        lang = detect(user_input)
        return lang if lang in ["fr", "de"] else "en"
    except:
        return "en"

def setup_translation(lang):
    """Setup translation and return the translation function"""
    global _
    
    if lang == "en":
        # For English, just return the identity function
        _ = lambda x: x
        return _
    
    # For other languages, set up proper translation
    # Try both "fr" and "en-fr" formats
    language_codes = [lang, f"en-{lang}"]
    
    try:
        # Try different language code formats
        translation = None
        for lang_code in language_codes:
            try:
                translation = gettext.translation(
                    "messages",              
                    localedir=LOCALE_PATH,   
                    languages=[lang_code], 
                    fallback=False  # Don't fallback initially to catch errors
                )
                break
            except FileNotFoundError:
                continue
        
        if translation is None:
            # If no translation found, try with fallback
            translation = gettext.translation(
                "messages",              
                localedir=LOCALE_PATH,   
                languages=language_codes, 
                fallback=True
            )
        
        # Install globally
        translation.install()
        _ = translation.gettext
        
        return _
    except Exception as e:
        _ = lambda x: x
        return _

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
        else:  # Group relations
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

# def bokeh_graph(bms_file_name,user_input):
    

#     """Main function to create the bokeh graph - now uses the global _ function"""

#     #print(user_input)

#     detected_lang = detect_language(user_input)

#     #print(detected_lang)

#     setup_translation(detected_lang)

#     bms_details = get_bms_details(bms_file_name)
#     text = ""
#     if not bms_details:
#         print(_("BMS file not found."))
#         return text, [], "", ""

#     parent_id = bms_details["id"]
#     parent_name = bms_details["name"]
    
#     # Properly formatted translated strings
#     print(_("\n ID: {parent_id}").format(parent_id=parent_id))
#     print(_("Parent Name: {parent_name}\n").format(parent_name=parent_name))
    
#     text = text + f"ID: {parent_id}"
#     text = text + f"\nName: {parent_name}"

#     dependencies_data = get_bms_dependencies(parent_id)
#     if not dependencies_data:
#         print(_("No dependencies found."))
#         text = text + _("\nNo dependencies found.")
#         return text, [], "", ""

#     map_data = {}
#     for relation in dependencies_data.get("relations", []):
#         if relation["groupRelation"] == "Direct Relationships":
#             for item in relation["data"]:
#                 if item["type"] == "MAP":
#                     map_data[item["id"]] = item["name"]

#     generated_files = []
#     timestamp = time.strftime("%Y%m%d_%H%M%S")

#     for map_id, map_name in map_data.items():
#         program_data = get_programs_using_map(map_id)
#         if not program_data:
#             continue
            
#         text = text + _("\n**Responsible Node: {map_name}**\n".format(map_name=map_name))
#        # print(_("\n**Responsible Node: {map_name}**\n").format(map_name=map_name))
        
#         graph_data = {"nodes": [map_name], "edges": []}
#         node_counts = {}

#         for relation in program_data.get("relations", []):
#             group_name = relation["groupRelation"]
#             if not relation["data"]:
#                 continue

#             print(_("📌 Group Relation: {group_name}").format(group_name=group_name))
#             text = text + _("\nGroup Relation: {group_name}".format(group_name=group_name))
            
#             graph_data["nodes"].append(group_name)
#             graph_data["edges"].append((map_name, group_name))

#             for item in relation["data"]:
#                 file_name = item["name"]

#                 if file_name in node_counts:
#                     node_counts[file_name] += 1
#                     file_name_unique = f"{file_name} ({node_counts[file_name]})"
#                 else:
#                     node_counts[file_name] = 1
#                     file_name_unique = file_name

#                 graph_data["nodes"].append(file_name_unique)
#                 graph_data["edges"].append((group_name, file_name_unique))
#                 text = text + f"{_("\n* Child Name: {file_name_unique}".format(file_name_unique=file_name_unique))}"
#                 print(_("  -  Child Name: {file_name_unique}").format(file_name_unique=file_name_unique))
                
#         print()

def bokeh_graph(bms_file_name, user_input):
    """Main function to create the bokeh graph - now uses the global _ function"""
    
    # Ensure translation is set up (defensive programming)
    detected_lang = detect_language(user_input)
    setup_translation(detected_lang)
    
    # DEBUG: Test if translation is working
    test_translation = _("Test")
    print(f"DEBUG: Translation test - 'Test' becomes: '{test_translation}'")

    bms_details = get_bms_details(bms_file_name)
    text = ""
    if not bms_details:
        print(_("BMS file not found."))
        return text, [], "", ""

    parent_id = bms_details["id"]
    parent_name = bms_details["name"]
    
    # Properly formatted translated strings for print
    print(_("ID: {parent_id}").format(parent_id=parent_id))
    print(_("Parent Name: {parent_name}").format(parent_name=parent_name))
    
    # Apply translation to text variable too
    text += _("ID: {parent_id}").format(parent_id=parent_id)
    text += _("\nName: {parent_name}").format(parent_name=parent_name)

    dependencies_data = get_bms_dependencies(parent_id)
    if not dependencies_data:
        print(_("No dependencies found."))
        text = text + _("\nNo dependencies found.")
        return text, [], "", ""

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
            
        # Apply translation to BOTH print and text consistently
        responsible_node_text = _("**Responsible Node: {map_name}**").format(map_name=map_name)
        text += f"\n{responsible_node_text}\n"
        print(f"\n{responsible_node_text}\n")
        
        graph_data = {"nodes": [map_name], "edges": []}
        node_counts = {}

        for relation in program_data.get("relations", []):
            group_name = relation["groupRelation"]
            if not relation["data"]:
                continue

            # For print - use one format
            group_relation_print = _("📌 Group Relation: {group_name}").format(group_name=group_name)
            print(group_relation_print)
            
            # For text - use different format but SAME translation
            group_relation_text = _("Group Relation: {group_name}").format(group_name=group_name)
            text += f"\n{group_relation_text}"
            
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
                
                # For print - use one format
                print_child_text = _("  -  Child Name: {file_name_unique}").format(file_name_unique=file_name_unique)
                print(print_child_text)
                
                # For text - use different format but SAME translation
                child_name_text = _("* Child Name: {file_name_unique}").format(file_name_unique=file_name_unique)
                text += f"\n{child_name_text}"
                
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

        

    # if generated_files:
    #     generated_files_header = _("Generated Graph Files:")
    #     print(generated_files_header)
    #     text += f"\n{generated_files_header}"
        
    #     for file in generated_files:
    #         abs_path = os.path.abspath(file)
    #         file_url = f"'http://10.190.226.6:8000/chathtml_template?html_filename={file_name}"
            
    #         # FIXED: Correct way to translate formatted strings
    #         print_message = _("{file} → Open in browser: {file_url}").format(file=file, file_url=file_url)
    #         print(print_message)
            
    #         text_message = _("{file} → Open in browser: {file_url}").format(file=file, file_url=file_url)
    #         text += f"\n{text_message}"

    #         webbrowser.open(file_url)
    # else:
    #     no_graphs_message = _("No graphs were generated.")
    #     print(no_graphs_message)
    #     text += f"\n{no_graphs_message}"

    html_link_1 = "file:///Users/thrisham/Desktop/MBANK70.BANK70A.htm"
    html_link_2 = "file:///Users/thrisham/Desktop/MBANK70.HELP70A.htm"

    print("MBANK70.BANK70A.htm :", html_link_1, "\n", "MBANK70.HELP70A.htm :", html_link_2)

    print("\n\n", text)

    return text, generated_files, html_link_1, html_link_2

# def main():
#     """Main function to run the application"""
#     # Get user input and detect language
#     # user_input = "Je veux voir toutes les dépendances du fichier MBANK70.bms"  # French input
#     # detected_lang = detect_language(user_input)

#     # print(detected_lang)
    
#     # Setup translation BEFORE calling bokeh_graph
#     #setup_translation(detected_lang)
    
#     # Now call the main function
#     result = bokeh_graph("MBANK70.bms","Je veux voir toutes les dépendances du fichier MBANK70.bms")
#     return result

# # Run the main function
# if __name__ == "__main__":
#     main()


# Guten Morgen

# Bonjour

# python3 /root/Desktop/Chatbot/update_curl_3_i18n.py