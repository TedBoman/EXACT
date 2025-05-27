import json
import sys
import os
import urllib.parse
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback, no_update, dash_table
import plotly.graph_objects as go
import pandas as pd
import traceback
from datetime import datetime, timedelta, timezone
from io import StringIO 
from pages.job_page import get_display_job_name
from get_handler import get_handler 
from visualisations.feature_importance_plot import plot_aggregated_feature_importance_comparison

XAI_DIR = "/app/data" # The path INSIDE the container

# --- Helper Function to Generate Asset URL ---
def get_asset_url(job_name, method_name, filename):
    """Constructs the URL for accessing assets via the Flask static route."""
    # Quote each part to handle special characters in names
    quoted_job = urllib.parse.quote(job_name)
    quoted_method = urllib.parse.quote(method_name)
    quoted_file = urllib.parse.quote(filename)
    return f"/xai-assets/{quoted_job}/{quoted_method}/{quoted_file}"
# -------------------------------------------

# --- Helper function to create XAI Evaluation Table ---
def create_xai_evaluation_table(xai_eval_data: dict, theme_colors: dict):
    """
    Creates a Dash DataTable for XAI evaluation metrics (like NDCG).
    Input: xai_eval_data = {"ndcg_scores": {"ShapExplainer": {"NDCG@3": 0.8, "NDCG@5": 0.75}, ...}}
    """
    if not xai_eval_data or "ndcg_scores" not in xai_eval_data:
        return None

    ndcg_data = xai_eval_data.get("ndcg_scores", {})
    if not ndcg_data: # If ndcg_scores is empty
        return html.P("No NDCG scores available.", style={'color': theme_colors.get('text_medium', '#aaa'), 'marginTop': '10px'})

    # Prepare data for the table
    # Rows: Metrics (e.g., NDCG@3, NDCG@5)
    # Columns: XAI Methods (e.g., SHAP, LIME, DiCE)

    all_metrics = set()
    all_methods = sorted(list(ndcg_data.keys())) # Ensures consistent column order

    for method_scores in ndcg_data.values():
        all_metrics.update(method_scores.keys())
    
    def get_ndcg_k_value(metric_name_str):
        """Extracts the k value from NDCG@k string, returns large number if not found for sorting."""
        if metric_name_str.startswith("NDCG@"):
            try:
                return int(metric_name_str.split('@')[1])
            except (IndexError, ValueError):
                return float('inf') # Put non-standard NDCG names at the end
        return float('inf') # Put other metric types (if any) at the end

    # Sort metrics: first by whether they are NDCG@k, then by the k value
    # Non-NDCG@k metrics will be sorted alphabetically after all NDCG@k metrics.
    sorted_metrics = sorted(
        list(all_metrics),
        key=lambda m: (not m.startswith("NDCG@"), get_ndcg_k_value(m), m)
        # The tuple in key does:
        # 1. (not m.startswith("NDCG@")): False (0) for NDCG@k, True (1) for others. So NDCG@k comes first.
        # 2. get_ndcg_k_value(m): Sorts NDCG@k by their k value numerically.
        # 3. m: Alphabetical sort as a tie-breaker or for non-NDCG@k metrics among themselves.
    )

    table_data = []
    for metric_name in sorted_metrics:
        row = {'Metric': metric_name}
        for method_name in all_methods:
            # Get the score, default to 'N/A' or 0.0 if not present
            score = ndcg_data.get(method_name, {}).get(metric_name, "N/A")
            if isinstance(score, float):
                row[method_name] = f"{score:.4f}"
            else:
                row[method_name] = score # Keep as 'N/A' or other string
        table_data.append(row)

    if not table_data:
        return html.P("Could not parse NDCG scores for table display.", style={'color': theme_colors.get('text_medium', '#aaa')})

    table_columns = [{"name": "Metric", "id": "Metric"}] + \
                    [{"name": method, "id": method} for method in all_methods]

    data_table = dash_table.DataTable(
        columns=table_columns,
        data=table_data,
        style_table={'overflowX': 'auto', 'marginTop': '10px', 'marginBottom': '20px'},
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_cell={
            'backgroundColor': 'rgb(50, 50, 50)',
            'color': 'white',
            'border': f"1px solid {theme_colors.get('border_light', '#555')}",
            'textAlign': 'center', # Center align all cells
            'padding': '8px',
            'minWidth': '100px', 'width': '120px', 'maxWidth': '150px',
        },
        style_cell_conditional=[ # Left align the 'Metric' column
            {
                'if': {'column_id': 'Metric'},
                'textAlign': 'left',
                'fontWeight': 'bold'
            }
        ]
    )

    return html.Div([
        html.H4("XAI Evaluation Metrics (e.g., NDCG)", style={
            'borderBottom': f"1px solid {theme_colors.get('border_light', '#555')}",
            'paddingBottom': '5px',
            'marginTop': '25px', # More space before this table
            'marginBottom': '10px',
            'color': theme_colors.get('text_light', '#eee')
        }),
        data_table
    ], style={'gridColumn': '1 / -1'}) # Make this section span all columns

# --- Helper function to format nested dicts/lists prettily ---
def create_pretty_dict_list_display(data, indent=0):
    """Recursively creates html.Divs/Lists to display nested data."""
    items = []
    indent_space = "  " * indent # Using non-breaking space for indent
    if isinstance(data, dict):
        for key, value in data.items():
            # Display key
            item_content = [html.Span(f"{indent_space}{key}: ", style={'fontWeight': 'bold'})]
            # Display value (recurse if nested)
            if isinstance(value, (dict, list)):
                 # Add newline before nested structure for clarity
                item_content.append(html.Br())
                item_content.append(create_pretty_dict_list_display(value, indent + 1))
            else:
                item_content.append(html.Span(f"{value}"))
            items.append(html.Div(item_content))
    elif isinstance(data, list):
         # Special handling for list of dicts (like xai_params)
        is_list_of_dicts = all(isinstance(i, dict) for i in data)
        for index, value in enumerate(data):
            if is_list_of_dicts:
                 # Add a separator/header for each item in the list
                 items.append(html.Div(f"{indent_space}Item {index+1}:", style={'marginTop': '5px', 'fontStyle':'italic'}))
                 items.append(create_pretty_dict_list_display(value, indent + 1))
            elif isinstance(value, (dict, list)):
                items.append(create_pretty_dict_list_display(value, indent + 1))
            else:
                items.append(html.Div(f"{indent_space}- {value}"))
    return html.Div(items)

# --- Helper function to create a simple key-value table section ---
def create_info_section(title, data_dict, theme_colors, format_floats=True):
    """Creates a styled Div with H4 title and key-value pairs."""
    rows = []
    for key, value in data_dict.items():
        # Format float values nicely
        if format_floats and isinstance(value, float):
            display_value = f"{value:.4f}" # Adjust precision as needed
        else:
            display_value = str(value)

        # Improve readability of keys
        display_key = key.replace('_', ' ').title()

        rows.append(html.Div([
            html.Span(f"{display_key}:", style={'fontWeight': 'bold', 'minWidth': '200px', 'display': 'inline-block'}),
            html.Span(display_value)
        ], style={'marginBottom': '5px'}))

    return html.Div([
        html.H4(title, style={'borderBottom': f"1px solid {theme_colors.get('border_light', '#555')}", 'paddingBottom': '5px', 'marginTop': '15px', 'marginBottom': '10px', 'color': 'white'}),
        *rows
    ], style={'padding': '15px', 'border': f"1px solid {theme_colors.get('border_light', '#444')}", 'borderRadius':'5px', 'backgroundColor': 'rgba(40,40,40,0.3)', 'marginBottom': '15px'}) # Slightly different background

# --- Helper function to create the detailed performance metrics explanation section ---
def create_performance_metrics_explanation(metrics_data: dict, theme_colors: dict):
    """
    Creates a collapsible HTML Div that explains common performance metrics,
    their values from the provided data, descriptions, and formulas,
    with all text styled white and larger equation fonts.
    """
    if not metrics_data:
        return None

    # This list will hold all the individual metric explanation Divs
    metric_detail_elements = []

    # --- Define the base components: TP, TN, FP, FN ---
    base_components_details = [
        {
            "key": "correct_anomalies", "name": "True Positives (TP)",
            "description": "Cases where the model correctly predicted an anomaly (actual anomaly, predicted anomaly)."
        },
        {
            "key": "false_positives", "name": "False Positives (FP)",
            "description": "Cases where the model incorrectly predicted an anomaly when it was actually normal (actual normal, predicted anomaly). Also known as a Type I error."
        },
        {
            "key": "false_negatives", "name": "False Negatives (FN)",
            "description": "Cases where the model incorrectly predicted normal when it was actually an anomaly (actual anomaly, predicted normal). Also known as a Type II error."
        },
        {
            "key": "correct_non_anomalies", "name": "True Negatives (TN)",
            "description": "Cases where the model correctly predicted a normal instance (actual normal, predicted normal)."
        },
        {
            "key": "total_predictions", "name": "Total Predictions",
            "description": "The total number of instances evaluated by the model."
        }
    ]

    for detail in base_components_details:
        key = detail["key"]
        if key in metrics_data:
            value = metrics_data[key]
            metric_detail_elements.append(html.Div([
                html.Strong(f"{detail['name']}: ", style={'color': 'white'}), 
                html.Span(str(value), style={'color': 'white', 'fontWeight': 'bold'}), 
                html.P(detail["description"], style={
                    'fontSize': '1.2em',
                    'color': 'white', 
                    'marginTop': '3px',
                    'marginBottom': '10px',
                    'marginLeft': '10px'
                })
            ], style={'marginBottom': '10px'}))

    # --- Define the calculated metrics: Accuracy, Precision, Recall, F1, Specificity ---
    tp_val = metrics_data.get("correct_anomalies", "TP")
    tn_val = metrics_data.get("correct_non_anomalies", "TN")
    fp_val = metrics_data.get("false_positives", "FP")
    fn_val = metrics_data.get("false_negatives", "FN")

    calculated_metrics_details = [
        {
            "key": "accuracy", "name": "Accuracy",
            "description": "The proportion of all predictions that were correct. It's a general measure of how often the model is right.",
            "formula": rf"$\text{{Accuracy}} = \frac{{{tp_val} + {tn_val}}}{{{tp_val} + {tn_val} + {fp_val} + {fn_val}}} = \frac{{\text{{TP + TN}}}}{{\text{{Total Predictions}}}}$"
        },
        {
            "key": "precision", "name": "Precision (Positive Predictive Value)",
            "description": "Of all instances the model predicted as anomalies, what proportion were actual anomalies? High precision indicates few false positives.",
            "formula": rf"$\text{{Precision}} = \frac{{{tp_val}}}{{{tp_val} + {fp_val}}} = \frac{{\text{{TP}}}}{{\text{{TP + FP}}}}$"
        },
        {
            "key": "recall_tpr", "name": "Recall (Sensitivity, True Positive Rate - TPR)",
            "description": "Of all actual anomalies, what proportion did the model correctly identify? High recall indicates few false negatives.",
            "formula": rf"$\text{{Recall (TPR)}} = \frac{{{tp_val}}}{{{tp_val} + {fn_val}}} = \frac{{\text{{TP}}}}{{\text{{TP + FN}}}}$"
        },
        {
            "key": "f1_score", "name": "F1-Score",
            "description": "The harmonic mean of Precision and Recall. It provides a balance between the two, especially useful when class distribution is uneven.",
            "formula": r"$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$"
        },
        {
            "key": "specificity_tnr", "name": "Specificity (True Negative Rate - TNR)",
            "description": "The proportion of normal instances that were correctly identified as normal. High specificity indicates the model is good at identifying normal data.",
            "formula": rf"$\text{{Specificity (TNR)}} = \frac{{{tn_val}}}{{{tn_val} + {fp_val}}} = \frac{{\text{{TN}}}}{{\text{{TN + FP}}}}$"
        }
    ]

    for detail in calculated_metrics_details:
        key = detail["key"]
        if key in metrics_data:
            value = metrics_data[key]
            value_display = f"{value:.4f}" if isinstance(value, float) else str(value)
            metric_detail_elements.append(html.Div([
                html.Strong(f"{detail['name']}: ", style={'color': 'white'}), 
                html.Span(value_display, style={'color': 'white', 'fontWeight': 'bold'}), 
                html.P(detail["description"], style={
                    'fontSize': '1.1em',
                    'color': 'white', 
                    'marginTop': '3px',
                    'marginLeft': '10px'
                }),
                dcc.Markdown(f"**Formula:** {detail['formula']}",
                             mathjax=True, # Good to keep for explicitness
                             style={
                                 'fontSize': '1.2em',  # Increased font size for equations
                                 'color': 'white',    
                                 'marginBottom': '10px',
                                 'marginLeft': '10px'
                             })
            ], style={'marginBottom': '15px'}))

    if not metric_detail_elements: # If no metric details were actually generated
        return None

    # Create the collapsible section using html.Details and html.Summary
    collapsible_section = html.Details([
        html.Summary(
            "Performance Metrics Definitions",  # This is the clickable title
            style={
                'color': 'white',  
                'fontWeight': 'bold',
                'fontSize': '1.1em', # Header-like font size
                'cursor': 'pointer',
                'paddingBottom': '10px',
                'borderBottom': f"1px solid {theme_colors.get('border_light', '#444')}",
                'marginBottom': '15px' # Space between summary and content when open
            }
        ),
        html.Div(metric_detail_elements, style={'paddingTop': '10px'}) # Wrapper for the content elements
    ], style={  # Styles for the overall <details> block
        'padding': '20px',
        'border': f"1px solid {theme_colors.get('border_light', '#444')}",
        'borderRadius': '8px',
        'backgroundColor': 'rgba(40,40,40,0.3)',
        'marginBottom': '20px',
        'gridColumn': '1 / -1' # Make this section span all columns in the grid
    })

    return collapsible_section

# --- Helper Function for CSV Display ---
def create_datatable(file_path):
    """Reads a CSV and returns a Dash DataTable or an error message."""
    try:
        # Use StringIO to handle potential encoding issues if needed, though direct path usually works
        df = pd.read_csv(file_path)
        # Limit rows for display performance if necessary
        max_rows = 50
        if len(df) > max_rows:
             df_display = df.head(max_rows)
             disclaimer = html.P(f"(Displaying first {max_rows} rows)", style={'fontSize':'small', 'color':'#ccc'})
        else:
             df_display = df
             disclaimer = None

        table = dash_table.DataTable(
             columns=[{"name": i, "id": i} for i in df_display.columns],
             data=df_display.to_dict('records'),
             style_table={'overflowX': 'auto', 'marginTop': '10px'},
             style_header={
                 'backgroundColor': 'rgb(30, 30, 30)',
                 'color': 'white',
                 'fontWeight': 'bold'
             },
             style_cell={
                 'backgroundColor': 'rgb(50, 50, 50)',
                 'color': 'white',
                 'border': '1px solid #555',
                 'textAlign': 'left',
                 'padding': '5px',
                 'minWidth': '80px', 'width': '150px', 'maxWidth': '300px', # Adjust width constraints
                 'overflow': 'hidden',
                 'textOverflow': 'ellipsis',
             },
             tooltip_data=[ # Add tooltips for potentially truncated cells
                {
                    column: {'value': str(value), 'type': 'markdown'}
                    for column, value in row.items()
                } for row in df_display.to_dict('records')
            ],
            tooltip_duration=None # Keep tooltip visible indefinitely on hover
        )
        if disclaimer:
             return html.Div([disclaimer, table])
        else:
             return table
    except Exception as e:
        print(f"Error reading/parsing CSV {file_path}: {e}")
        return html.P(f"Error displaying CSV: {os.path.basename(file_path)} - {e}", style={'color':'red'})
# ------------------------------------

# --- Helper Function for counterfactual CSV Display ---
def create_cfe_delta_table(file_path):
    """
    Reads a CFE CSV and returns a single styled DataTable containing both
    the original row and counterfactual rows, highlighting differences.
    """
    try:
        df_cfe = pd.read_csv(file_path)

        # --- Find Original and Counterfactual Data ---
        original_rows = df_cfe[df_cfe['type'].str.lower() == 'original']
        if original_rows.empty:
            return html.P(f"Error: 'original' row not found in {os.path.basename(file_path)}.", style={'color': 'red'})
        original_row = original_rows.iloc[0] # Use the first original row found

        cf_rows = df_cfe[df_cfe['type'].str.lower() == 'counterfactual']
        # No error if cf_rows is empty, we'll just show the original

        # Identify feature columns (exclude 'type', but keep others like 'label')
        all_display_cols = list(df_cfe.columns)
        feature_cols = [col for col in all_display_cols if col.lower() != 'type']

        combined_data = []
        style_conditions = []

        # --- Process Original Row (Row Index 0) ---
        original_display = {'Row Type': 'Original', 'CF #': 'N/A', 'Changes': 'N/A'}
        for col in feature_cols:
             original_display[col] = original_row.get(col, 'N/A') # Use .get for safety
        combined_data.append(original_display)

        # Add style to distinguish the original row
        style_conditions.append({
            'if': {'row_index': 0},
            'fontWeight': 'bold',
            'backgroundColor': 'rgba(100, 100, 100, 0.15)' # Slightly different background
        })

        # --- Process Counterfactual Rows (Starting from Row Index 1) ---
        if not cf_rows.empty:
            for cf_idx, (row_label, cf_row) in enumerate(cf_rows.iterrows(), start=1):
                # combined_row_index corresponds to cf_idx since original is row 0
                cf_display = {'Row Type': f'CF {cf_idx}', 'CF #': cf_idx}
                changed_features_list = []

                for col in feature_cols:
                    original_val = original_row.get(col)
                    cf_val = cf_row.get(col)

                    # Check if the value changed
                    changed = False
                    try:
                        if pd.isna(original_val) and pd.isna(cf_val):
                            changed = False
                        elif pd.isna(original_val) or pd.isna(cf_val):
                            changed = True
                        elif isinstance(original_val, (int, float)) and isinstance(cf_val, (int, float)):
                            if not np.isclose(original_val, cf_val, rtol=1e-05, atol=1e-08, equal_nan=True):
                                changed = True
                        elif original_val != cf_val:
                            changed = True
                    except TypeError:
                        if str(original_val) != str(cf_val):
                            changed = True

                    if changed:
                        cf_display[col] = cf_val
                        changed_features_list.append(col)
                        # Add style condition to highlight this changed cell
                        style_conditions.append({
                            'if': {'row_index': cf_idx, 'column_id': col},
                            'backgroundColor': '#3D9970', 
                            'color': 'white',
                            'fontWeight': 'bold'
                        })
                    else:
                        # Indicate no change with em dash
                        cf_display[col] = "—"

                cf_display['Changes'] = ", ".join(changed_features_list) if changed_features_list else "None"
                combined_data.append(cf_display)

        # --- Define Columns for the Combined DataTable ---
        # Start with the special columns, then add the feature columns
        table_columns = [
            {"name": "Row Type", "id": "Row Type"},
            {"name": "CF #", "id": "CF #"},
            {"name": "Changed", "id": "Changes"}
        ] + [{"name": i, "id": i} for i in feature_cols]


        # --- Create the Combined DataTable ---
        combined_table = dash_table.DataTable(
            columns=table_columns,
            data=combined_data,
            style_table={'overflowX': 'auto', 'marginTop': '10px'},
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_cell={ # Default cell style for all cells
                'backgroundColor': 'rgb(50, 50, 50)',
                'color': 'white',
                'border': '1px solid #555',
                'textAlign': 'left',
                'padding': '5px',
                'minWidth': '60px', 'width': '100px', 'maxWidth': '150px', # Adjust width
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            # Apply conditional styles (original row + changed cells)
            style_data_conditional=style_conditions,
            tooltip_data=[ # Show full value on hover
                {
                    column: {'value': str(value), 'type': 'markdown'}
                    for column, value in row.items()
                } for row in combined_data
            ],
            tooltip_duration=None # Keep tooltip visible
        )

        # Return a Div containing the combined table
        return html.Div([
            html.H5("Original vs. Counterfactuals:", style={'marginTop':'20px', 'fontWeight':'bold'}),
            combined_table
        ])

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        P_component = getattr(html, 'P', lambda *args, **kwargs: f"Error: File not found - {os.path.basename(file_path)}")
        return P_component(f"Error: File not found - {os.path.basename(file_path)}", style={'color':'red'})
    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_path} is empty.")
        P_component = getattr(html, 'P', lambda *args, **kwargs: f"Error: The file is empty - {os.path.basename(file_path)}")
        return P_component(f"Error: The file is empty - {os.path.basename(file_path)}", style={'color':'red'})
    except KeyError as e:
        print(f"Error processing CFE file {file_path}: Missing expected column {e}")
        traceback.print_exc()
        P_component = getattr(html, 'P', lambda *args, **kwargs: f"Error: Missing expected column {e} in {os.path.basename(file_path)}.")
        return P_component(f"Error: Missing expected column {e} in {os.path.basename(file_path)}.", style={'color': 'red'})
    except Exception as e:
        print(f"Error creating combined CFE table for {file_path}: {e}")
        traceback.print_exc()
        P_component = getattr(html, 'P', lambda *args, **kwargs: f"Error processing counterfactuals file: {os.path.basename(file_path)} - {e}")
        return P_component(f"Error processing counterfactuals file: {os.path.basename(file_path)} - {e}", style={'color':'red'})


def register_job_page_callbacks(app):
    print("Registering job page callbacks...")
    
    theme_colors = {
        'background': '#0D3D66', 'header_background': '#1E3A5F',
        'content_background': '#104E78', 'status_background': '#145E88',
        'text_light': '#E0E0E0', 'text_medium': '#C0C0C0',
        'text_dark': '#FFFFFF', 'border_light': '#444'
    }

    # --- Callback to parse job name from URL ---
    @app.callback(
        Output('job-page-job-name-store', 'data'),
        Input('url', 'pathname') # Triggered when URL pathname changes (including initial load)
    )
    def update_job_store_from_url(pathname):
        """
        Parses the job name from the URL pathname (/job/job_name)
        and updates the job name store.
        """
        print(f"(URL Parser CB) Pathname received: {pathname}")
        if pathname and pathname.startswith('/job/'):
            try:
                job_name = pathname.split('/')[-1]
                if job_name:
                    print(f"(URL Parser CB) Extracted job name: {job_name}. Updating store.")
                    return job_name
                else:
                    print("(URL Parser CB) Job name empty after split.")
                    return None
            except Exception as e:
                print(f"(URL Parser CB) Error parsing pathname '{pathname}': {e}")
                return None
        else:
            print("(URL Parser CB) Pathname doesn't match expected '/job/...' format.")
            return None # Clear job name if path doesn't match

    @app.callback(
        Output('job-metadata-display', 'children'),
        Input('job-page-job-name-store', 'data')
    )
    def update_metadata_display(job_name):
        if not job_name:
            # Return a styled message consistent with theme
            return html.Div("Select a job to view metadata.", style={'color': theme_colors['text_medium'], 'padding': '10px'})

        print(f"(Metadata CB) Attempting to load metadata for job: {job_name}")

        # --- Construct the path to the metadata logfile ---
        metadata_filename = f"logfile"
        metadata_filepath = os.path.join(XAI_DIR, job_name, metadata_filename)
        print(f"(Metadata CB) Expecting metadata file at: {metadata_filepath}")
        # ----------------------------------------------------

        # --- Read metadata from the file ---
        metadata_json = None
        try:
            with open(metadata_filepath, 'r', encoding='utf-8') as f:
                metadata_json = f.read()
            print(f"(Metadata CB) Successfully read metadata file: {metadata_filepath}")

        except FileNotFoundError:
            print(f"(Metadata CB) Metadata file not found: {metadata_filepath}")
            # Return a clear message if the file is missing
            return html.Div([
                html.Strong("Metadata file not found."),
                html.P(f"Expected location: {metadata_filepath}", style={'fontSize':'small', 'color':theme_colors['text_medium']})
            ], style={'color': 'orange', 'padding': '10px', 'border': f"1px solid {theme_colors['border_light']}", 'borderRadius':'5px', 'backgroundColor':'rgba(255, 165, 0, 0.1)'}) # Orange theme for warning
        except Exception as e:
            print(f"(Metadata CB) Error reading metadata file {metadata_filepath}: {e}")
            traceback.print_exc()
            return html.Div(f"Error reading metadata file: {e}", style={'color': 'red', 'padding': '10px'}) # Red theme for error
        # -----------------------------------
        
        # --- If file read successfully, proceed with parsing and display ---
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
                print(f"(Metadata CB) Successfully parsed JSON metadata for {job_name}")

                # --- Build Display Components (using helpers defined earlier) ---
                display_elements = []

                # 1. General Job Info
                job_info = {
                    "Run Timestamp (UTC)": metadata.get("run_timestamp_utc"),
                    "Status": metadata.get("status"),
                    "Dataset Path": metadata.get("dataset_path"),
                    "Model Name": metadata.get("model_name"),
                    "Label Column": metadata.get("label_column_used"),
                    "Sequence Length": metadata.get("sequence_length")
                }
                display_elements.append(create_info_section("Job Summary", {k: v for k, v in job_info.items() if v is not None}, theme_colors, format_floats=False))

                # 2. Data Summary
                data_summary = {
                    "Total Rows": metadata.get("data_total_rows"),
                    "Training Rows": metadata.get("data_training_rows"),
                    "Testing Rows": metadata.get("data_testing_rows"),
                    "Features": metadata.get("data_num_features"),
                    "Anomalies (Ground Truth)": metadata.get("data_num_anomalies_ground_truth"),
                    "Anomalies (Predicted)": metadata.get("data_num_anomalies_predicted")
                }
                data_summary_filtered = {k: v for k, v in data_summary.items() if v is not None}
                if data_summary_filtered:
                     display_elements.append(create_info_section("Data Summary", data_summary_filtered, theme_colors, format_floats=False))

                cv_metrics_data = metadata.get("cross_validation_metrics", {})
                if cv_metrics_data: # Only display if the section exists and is not empty
                    display_elements.append(create_info_section("Cross-Validation Metrics", cv_metrics_data, theme_colors))

                # 3. Performance Metrics
                metrics = metadata.get("evaluation_metrics", {})
                if metrics:
                    display_elements.append(create_info_section("Performance Metrics (Testing data only)", metrics, theme_colors))
                    
                # 4. Execution Times
                exec_times = {
                    "Total (s)": metadata.get("execution_time_total_seconds"),
                    "Simulation (s)": metadata.get("execution_time_simulation_seconds"),
                    "Training (s)": metadata.get("execution_time_training_seconds"),
                    "Detection (s)": metadata.get("execution_time_detection_seconds"),
                    "XAI (s)": metadata.get("execution_time_xai_seconds")
                }
                exec_times_filtered = {k: v for k, v in exec_times.items() if v is not None}
                if exec_times_filtered:
                    display_elements.append(create_info_section("Execution Times", exec_times_filtered, theme_colors))

                # 5. Model Parameters (collapsible)
                model_params = metadata.get("model_params")
                if model_params:
                    display_elements.append(html.Div([
                         html.Details([
                             html.Summary("Model Parameters", style={'fontWeight':'bold', 'cursor': 'pointer', 'color': theme_colors.get('text_light', '#eee'), 'marginBottom':'5px'}),
                             create_pretty_dict_list_display(model_params)
                         ], style={'padding': '15px', 'border': f"1px solid {theme_colors.get('border_light', '#444')}", 'borderRadius':'5px', 'backgroundColor': 'rgba(40,40,40,0.3)', 'marginBottom': '15px'})
                    ]))

                # 6. XAI Settings (collapsible)
                xai_settings = metadata.get("xai_settings")
                if xai_settings:
                     display_elements.append(html.Div([
                         html.Details([
                             html.Summary("XAI Settings", style={'fontWeight':'bold', 'cursor': 'pointer', 'color': theme_colors.get('text_light', '#eee'), 'marginBottom':'5px'}),
                            create_pretty_dict_list_display(xai_settings)
                         ], style={'padding': '15px', 'border': f"1px solid {theme_colors.get('border_light', '#444')}", 'borderRadius':'5px', 'backgroundColor': 'rgba(40,40,40,0.3)', 'marginBottom': '15px'})
                    ]))

                # 7. Anomaly Injection Params (collapsible, if any)
                injection_params = metadata.get("anomaly_injection_params")
                if injection_params: # Check if list is not empty or None
                    display_elements.append(html.Div([
                        html.Details([
                            html.Summary("Anomaly Injection Parameters", style={'fontWeight':'bold', 'cursor': 'pointer', 'color': theme_colors.get('text_light', '#eee'), 'marginBottom':'5px'}),
                            create_pretty_dict_list_display(injection_params)
                        ], style={'padding': '15px', 'border': f"1px solid {theme_colors.get('border_light', '#444')}", 'borderRadius':'5px', 'backgroundColor': 'rgba(40,40,40,0.3)', 'marginBottom': '15px'})
                    ]))
                    
                # 8. Performance metrics explanations
                metrics_explanation_section = create_performance_metrics_explanation(metrics, theme_colors)
                if metrics_explanation_section:
                    display_elements.append(metrics_explanation_section)
                    
                # 9. XAI Evaluation Metrics Table (NDCG)
                xai_eval_metrics_data = metadata.get("xai_evaluation_metrics", {})
                if xai_eval_metrics_data:
                    xai_eval_table = create_xai_evaluation_table(xai_eval_metrics_data, theme_colors)
                    if xai_eval_table:
                        display_elements.append(xai_eval_table)
                # --- END ---
                
                # 10. All data evaluation results
                # 3. Performance Metrics
                metrics = metadata.get("all_data_evaluation_metrics", {})
                if metrics:
                    display_elements.append(create_info_section("All Data Performance Metrics (Can be misleading)", metrics, theme_colors))
                    

                # --- Create the Grid Container ---
                grid_container = html.Div(
                    children=display_elements, # Place all section divs inside the container
                    style={
                        'display': 'grid',
                        'gridTemplateColumns': 'repeat(auto-fit, minmax(350px, 1fr))',
                        'gap': '20px', # Space between grid items (rows and columns)
                        'width': '100%' # Ensure container takes full width
                    }
                )
                
                print(f"(Metadata CB) Successfully generated display components for {job_name} from file.")
                return grid_container 

            except json.JSONDecodeError as e:
                print(f"(Metadata CB) Error decoding metadata JSON from file {metadata_filepath}: {e}")
                return html.Div(f"Error loading metadata: Invalid JSON format in {metadata_filename} - {e}", style={'color': 'red', 'padding': '10px'})
            except Exception as e:
                print(f"(Metadata CB) Error generating metadata display for {job_name} from file: {e}")
                traceback.print_exc()
                return html.Div(f"An error occurred while displaying metadata: {e}", style={'color': 'red', 'padding': '10px'})
        else:
            # This case should theoretically not be reached if file reading failed earlier,
            # but included for completeness.
             return html.Div("Failed to load metadata content.", style={'color': 'red', 'padding': '10px'})

    # --- Callback 1: Fetch and Store Data ---
    @app.callback(
        [
            Output('job-page-data-store', 'data'),
            Output('job-status-display', 'children'),
            Output('loading-output-jobpage', 'children') # Controls the loading indicator text/presence
        ],
        [
            Input('job-page-job-name-store', 'data'),      # Trigger on job change
            Input('job-page-interval-component', 'n_intervals') # Trigger on interval
        ],
    )
    def update_data_store(job_name, n_intervals):
        """
        Fetches data based on triggers:
        - Always fetches when job_name changes (initial load).
        - Fetches periodically ONLY if job_name starts with 'job_stream_'.
        - Stores fetched data in dcc.Store. Updates status.
        """
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'initial load'
        current_time_display = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        display_name = get_display_job_name(job_name) # Get display name

        if not job_name:
            return None, f"No job selected. Last checked: {current_time_display}", None # Clear data, update status, clear loading

        is_streaming_job = job_name.startswith("job_stream_")
        is_batch_job = job_name.startswith("job_batch_")
        triggered_by_job_change = trigger_id == 'job-page-job-name-store' or trigger_id == 'initial load'
        triggered_by_interval = trigger_id == 'job-page-interval-component'

        should_fetch = False
        fetch_reason = ""
        start_time_iso = None

        if triggered_by_job_change:
            if is_streaming_job or is_batch_job:
                should_fetch = True
                fetch_reason = f"Job selected/changed to '{display_name}'"
                if is_batch_job:
                    start_time_iso = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc).isoformat()
                    print(f"(Data Fetch CB) Using epoch start for batch job '{display_name}'.") 
                else: # is_streaming_job
                    lookback_minutes = 60
                    start_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
                    start_time_iso = start_time.isoformat()
                    print(f"(Data Fetch CB) Using recent start ({lookback_minutes} min ago) for streaming job '{display_name}' initial load.") 
            else:
                print(f"(Data Fetch CB) Job changed to unrecognized type: {job_name}. No fetch.")

        elif triggered_by_interval:
            if is_streaming_job:
                should_fetch = True
                fetch_reason = f"Interval trigger for streaming job '{display_name}'" 
                lookback_minutes = 10 # Fetch recent data
                start_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
                start_time_iso = start_time.isoformat()
                print(f"(Data Fetch CB) Using recent start ({lookback_minutes} min ago) for streaming interval fetch.")
            elif is_batch_job:
                print(f"(Data Fetch CB) Interval trigger ignored for batch job '{display_name}'.") 
                status_msg = f"Data loaded for batch job: '{display_name}'. Last Update: {current_time_display} UTC" 
                return dash.no_update, status_msg, dash.no_update # Only update status
            else:
                print(f"(Data Fetch CB) Interval trigger for unrecognized job type: {job_name}. No fetch.")

        if should_fetch and start_time_iso:
            print(f"(Data Fetch CB) Condition met: {fetch_reason}. Fetching data from {start_time_iso}...")
            status_msg = f"Fetching data for job '{display_name}' ({'Streaming' if is_streaming_job else 'Batch'})..." 
            data_json = None
            try:
                handler = get_handler()
                df = handler.handle_get_data(timestamp=start_time_iso, job_name=job_name) 

                if df is not None and not df.empty:
                    print(f"(Data Fetch CB) Successfully fetched data. Shape: {df.shape}")
                    data_json = df.to_json(date_format='iso', orient='split')
                    status_msg = f"Data updated for job '{display_name}'. Reason: {fetch_reason}. Records: {len(df)}. Timestamp: {current_time_display}" 
                else:
                    print(f"(Data Fetch CB) Received empty DataFrame or None for job '{display_name}'.") 
                    status_msg = f"No new data found for job '{display_name}'. Reason: {fetch_reason}. Timestamp: {current_time_display}" 
                    data_json = None

            except Exception as e:
                print(f"(Data Fetch CB) Error fetching data for job '{display_name}':") 
                traceback.print_exc()
                status_msg = f"Error fetching data for job '{display_name}': {e}. Timestamp: {current_time_display}" 

            return data_json, status_msg, None

        else:
            print("(Data Fetch CB) No fetch condition met or start_time_iso not set. Returning no_update.")
            return dash.no_update, dash.no_update, dash.no_update

    # --- Callback to populate feature selector dropdown ---
    @app.callback(
        [Output('feature-selector-dropdown', 'options'),
         Output('feature-selector-dropdown', 'value')],
        [Input('job-page-data-store', 'data')], # This is the critical Input
        [State('job-page-job-name-store', 'data')],
        prevent_initial_call=True 
    )
    def update_feature_selector_options(stored_data_json, job_name):
        """
        Populates the feature selector dropdown based on available numeric columns
        in the stored data. Selects the first feature by default if available.
        """
        print(f"(Feature Selector CB) ENTERED. Job: {get_display_job_name(job_name)}. Data store updated.")
        sys.stdout.flush() 

        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No trigger (initial or direct)'
        print(f"(Feature Selector CB) Triggered by: {trigger_id}. Job: {get_display_job_name(job_name)}")

        if not stored_data_json: 
            print("(Feature Selector CB) No data in store (stored_data_json is None or empty). Clearing dropdown.")
            return [], []

        try:
            df = pd.read_json(StringIO(stored_data_json), orient='split')
            print(f"(Feature Selector CB) DataFrame successfully loaded. Shape: {df.shape}. Columns: {df.columns.tolist()}")
            if df.empty:
                print("(Feature Selector CB) DataFrame is empty after loading. Clearing dropdown.")
                return [], []

            cols_to_exclude = {'timestamp', 'id'} 
            
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            print(f"(Feature Selector CB) Initial numeric columns: {numeric_cols}")

            available_y_cols = [col for col in numeric_cols if col.lower() not in cols_to_exclude] 
            print(f"(Feature Selector CB) Filtered plottable features: {available_y_cols}")

            options = [{'label': col, 'value': col} for col in available_y_cols]
            default_value = [available_y_cols[0]] if available_y_cols else []
            
            print(f"(Feature Selector CB) Generated options: {options}")
            print(f"(Feature Selector CB) Setting default value: {default_value}")
            sys.stdout.flush() 
            return options, default_value

        except ValueError as ve: 
            print(f"(Feature Selector CB) ValueError (likely JSON format issue for job '{get_display_job_name(job_name)}'): {ve}")
            traceback.print_exc(); sys.stdout.flush()
            return [], []
        except Exception as e:
            print(f"(Feature Selector CB) Generic error for job '{get_display_job_name(job_name)}': {e}")
            traceback.print_exc(); sys.stdout.flush()
            return [], []

    # --- Callback 2: Update Graph from Stored Data ---
    @app.callback(
        Output('timeseries-anomaly-graph', 'figure'),
        [Input('job-page-data-store', 'data'),
         Input('feature-selector-dropdown', 'value')],
        [State('job-page-job-name-store', 'data')]
    )
    def update_graph_from_data(stored_data_json, selected_features, job_name):
        #print(f"(Graph Update CB) ENTERED for job '{job_name}'. stored_data_json is None: {stored_data_json is None}") # DIAGNOSTIC
        sys.stdout.flush()
        job_title_name = get_display_job_name(job_name) if job_name else "No Job Selected"
        fig = go.Figure(layout=go.Layout(
            template='plotly_dark',
            title=f'Time Series Data: {job_title_name}',
            xaxis_title="Timestamp",
            yaxis_title="Value",
            legend_title_text='Features',
            uirevision=job_name,  # Persists zoom/pan state across updates for the same job_name
            hovermode='closest'   # Explicitly set hovermode
        ))
        
        current_annotations_init = fig.layout.annotations
        if current_annotations_init is None:
            fig.layout.annotations = []
        elif not isinstance(current_annotations_init, list):
            try:
                fig.layout.annotations = list(current_annotations_init) # Handles tuples
            except TypeError: # Fallback if not iterable for some reason
                fig.layout.annotations = []

        # Shapes
        current_shapes_init = fig.layout.shapes
        if current_shapes_init is None:
            fig.layout.shapes = []
        elif not isinstance(current_shapes_init, list):
            try:
                fig.layout.shapes = list(current_shapes_init) # Handles tuples
            except TypeError:
                fig.layout.shapes = []

        if stored_data_json is None:
            print("(Graph Update CB) No data in store for graph.")
            fig.update_layout(title=f'No Data Available for {job_title_name}. Waiting for data...',
                            xaxis={'visible': False}, yaxis={'visible': False})
            return fig

        print(f"(Graph Update CB) Data found. Selected features for plotting: {selected_features}")
        try:
            df = pd.read_json(StringIO(stored_data_json), orient='split')
            if df.empty:
                # print("(Graph Update CB) DataFrame from store is empty for graph.")
                fig.update_layout(title=f'No data points to plot for {job_title_name}.',
                                xaxis={'visible': False}, yaxis={'visible': False})
                return fig

            print(f"(Graph Update CB) DataFrame loaded for graph. Shape: {df.shape}. Columns: {df.columns.tolist()}")

            x_axis_source_name = 'index'
            x_axis_data = df.index
            is_timestamp_axis = False

            if 'timestamp' in df.columns:
                timestamp_numeric = pd.to_numeric(df['timestamp'], errors='coerce')
                if 'timestamp' not in df.columns: df['timestamp'] = pd.NaT
                valid_timestamps = timestamp_numeric.dropna()
                if not valid_timestamps.empty:
                    median_val = valid_timestamps.median()
                    SECONDS_THRESHOLD, MILLISECONDS_THRESHOLD = 4e10, 4e13
                    if median_val > MILLISECONDS_THRESHOLD: assumed_unit = 'ns'
                    elif median_val > SECONDS_THRESHOLD: assumed_unit = 'ms'
                    else: assumed_unit = 's'
                    df['timestamp'] = pd.to_datetime(timestamp_numeric, unit=assumed_unit, errors='coerce', utc=True)
                    if df['timestamp'].notna().any():
                        x_axis_data = df['timestamp']
                        is_timestamp_axis = True
                        x_axis_source_name = 'timestamp'
            elif 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                dt_series = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
                if dt_series.notna().any():
                    x_axis_data = dt_series
                    df['timestamp'] = dt_series # Ensure df uses the standardized datetime
                    is_timestamp_axis = True
                    x_axis_source_name = 'timestamp'
            
            if isinstance(x_axis_data, pd.Index): # Ensure x_axis_data is a Series for consistent operations
                x_axis_data = x_axis_data.to_series(name=x_axis_data.name or x_axis_source_name)

            print(f"(Graph Update CB) X-axis type: {x_axis_data.dtype}, is timestamp: {is_timestamp_axis}, source: {x_axis_source_name}")

            fig.layout.annotations = [] if fig.layout.annotations is None else list(fig.layout.annotations)
            fig.layout.shapes = []
            
            #print(f"(Graph Update CB) Final x_axis_source_name: '{x_axis_source_name}', is_timestamp_axis: {is_timestamp_axis}") # DIAGNOSTIC
            #if not x_axis_data.empty:
                #print(f"(Graph Update CB) x_axis_data head: \n{x_axis_data.head()}") # DIAGNOSTIC
                #print(f"(Graph Update CB) x_axis_data tail: \n{x_axis_data.tail()}") # DIAGNOSTIC
                #print(f"(Graph Update CB) x_axis_data NaNs: {x_axis_data.isna().sum()} out of {len(x_axis_data)}") # DIAGNOSTIC
            #else:
                #print("(Graph Update CB) x_axis_data is empty!") # DIAGNOSTIC
            #sys.stdout.flush()

            # --- Plotting Traces ---

            plotted_trace_count = 0
            if not selected_features:
                fig.layout.annotations.append(dict(
                    text="Use the dropdown to select features.", xref="paper", yref="paper",
                    y=0.5, showarrow=False, font=dict(size=16)
                ))
            else:
                for col_name in selected_features:
                    if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]):
                        fig.add_trace(go.Scattergl(
                            x=x_axis_data, y=df[col_name], mode='lines', name=col_name,
                            hoverinfo='x+y' # Simple hover info
                        ))
                        plotted_trace_count += 1
                if plotted_trace_count == 0 and selected_features:
                    fig.layout.annotations.append(dict(
                        text="Selected features are not plottable.", xref="paper", yref="paper",
                        y=0.5, showarrow=False, font=dict(size=16)
                    ))

            anomaly_col = 'is_anomaly'
            MAX_VRECT_ANOMALIES = 500 # Tune this based on desired performance vs. visibility

            condition_for_anomaly_plotting = (
                anomaly_col in df.columns and
                df[anomaly_col].isin([0, 1]).any() and # Check if 'is_anomaly' contains 0s or 1s
                (not df.empty and x_axis_data.notna().any()) # Check if x_axis has any valid (non-NaT) data
            )

            if condition_for_anomaly_plotting:
                anomalies_df = df[df[anomaly_col] == 1].copy()
                num_anomalies = len(anomalies_df)

                anomalies_df = df[df[anomaly_col] == 1].copy()
                num_anomalies = len(anomalies_df)
                print(f"(Graph Update CB) Found {num_anomalies} anomalies for '{anomaly_col}'.")
                sys.stdout.flush()

                if num_anomalies > 0:
                    legend_name = f'Anomaly as Xs ({num_anomalies})'

                    if x_axis_source_name == 'timestamp':
                        sample_x_coords = anomalies_df['timestamp'] if 'timestamp' in anomalies_df else anomalies_df.index.to_series()
                    else:
                        sample_x_coords = anomalies_df.index.to_series()

                    # Y value set to Y axis
                    y_coords = pd.Series([0.0] * len(anomalies_df), index=sample_x_coords.index if isinstance(sample_x_coords, pd.Series) else None) # Default y=0
                    
                    fig.add_trace(go.Scattergl(
                        x=sample_x_coords,
                        y=y_coords,
                        mode='markers',
                        marker=dict(color='rgba(255, 0, 0, 0.9)', symbol='x', size=8),
                        name=legend_name
                    ))
            
            # --- Apply Initial X-axis Zoom ---
            if not df.empty and x_axis_data.notna().any() and len(x_axis_data) > 1:
                INITIAL_POINTS_TO_SHOW = 1000 # Number of most recent data points to display initially

                if len(x_axis_data) > INITIAL_POINTS_TO_SHOW:
                    # x_axis_data is sorted because df was sorted based on the time column
                    x_start_zoom = x_axis_data.iloc[0]
                    x_end_zoom = x_axis_data.iloc[INITIAL_POINTS_TO_SHOW - 1]
                    
                    #print(f"(Graph Update CB) Initial calculated zoom range: [{x_start_zoom}, {x_end_zoom}]") # DIAGNOSTIC
                    sys.stdout.flush()
                    
                    if pd.notna(x_start_zoom) and pd.notna(x_end_zoom):
                        # Handle pd.Period type for xaxis_range if necessary
                        if isinstance(x_start_zoom, pd.Period) or isinstance(x_end_zoom, pd.Period):
                            try:
                                x_start_zoom_ts = x_start_zoom.to_timestamp() if isinstance(x_start_zoom, pd.Period) else x_start_zoom
                                x_end_zoom_ts = x_end_zoom.to_timestamp() if isinstance(x_end_zoom, pd.Period) else x_end_zoom
                                fig.update_layout(xaxis_range=[x_start_zoom_ts, x_end_zoom_ts])
                            except Exception: # Fallback if conversion fails
                                fig.update_layout(xaxis_range=[x_start_zoom, x_end_zoom])
                        else:
                            fig.update_layout(xaxis_range=[x_start_zoom, x_end_zoom])
                            print(f"(Graph Update CB) Applied initial zoom to show last {INITIAL_POINTS_TO_SHOW} points.")
                else:
                    print(f"(Graph Update CB) Not applying initial zoom, not enough data points ({len(x_axis_data)} available).")
                sys.stdout.flush()
        except Exception as e:
            print(f"(Graph Update CB) Error during graph generation: {e}")
            sys.stdout.flush()
            traceback.print_exc()
            # Robustly ensure fig.layout.annotations is a list before appending the error message
            current_fig_annotations_in_except = fig.layout.annotations
            if not isinstance(current_fig_annotations_in_except, list):
                if current_fig_annotations_in_except is None: current_fig_annotations_in_except = []
                else: 
                    try: current_fig_annotations_in_except = list(current_fig_annotations_in_except)
                    except TypeError: current_fig_annotations_in_except = [] 
                fig.layout.annotations = current_fig_annotations_in_except
            fig.layout.annotations.append({'text':f'Error: {str(e)}','xref':'paper','yref':'paper','y':0.5,'showarrow':False,'font':{'size':16,'color':'red'}})
            fig.update_layout(title=f'Error Displaying Data', xaxis={'visible':False}, yaxis={'visible':False})
            sys.stdout.flush()

        #print(f"(Graph Update CB) Returning figure. Traces: {len(fig.data)}, Annotations: {len(fig.layout.annotations)}, Shapes: {len(fig.layout.shapes)}") # DIAGNOSTIC
        sys.stdout.flush()
        return fig
    
    @app.callback(
        [Output('xai-results-content', 'children'),
         Output('xai-results-section', 'style')], # To hide/show the whole section
        [Input('job-page-job-name-store', 'data')] # Trigger when job name changes
    )
    def update_xai_display(job_name):
        """
        Scans the XAI directory for the current job, identifies subdirectories
        named after XAI methods, and displays their contents (images, html, csv).
        """
        sys.stdout.flush()

        if not job_name:
            return "No job selected.", {'display': 'none'} # Hide section if no job

        print(f"(XAI Display CB) Checking results for job: {job_name}")
        # Use the corrected XAI_DIR variable
        job_xai_base_path = os.path.join(XAI_DIR, job_name)
        xai_content_blocks = [] # List to hold Divs for each method
        found_any_results = False

        # Supported file extensions
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg']
        html_extension = '.html'
        csv_extension = '.csv'

        # Check if the base job directory exists
        if not os.path.isdir(job_xai_base_path):
            print(f"(XAI Display CB) Base XAI directory not found: {job_xai_base_path}")
            # Return message, show the section to display the message
            return f"No XAI results found for job '{job_name}'. Base directory missing: {job_xai_base_path}", {'display': 'block'}

        try:
            # List potential XAI method subdirectories
            # Define known/expected method names if possible for better filtering
            # known_methods = {'ShapExplainer', 'LimeExplainer', 'DiceExplainer'}
            method_subdirs = []
            for item in os.listdir(job_xai_base_path):
                item_path = os.path.join(job_xai_base_path, item)
                if os.path.isdir(item_path):
                     # Optional: Check if 'item' name matches known_methods if you have a list
                     method_subdirs.append(item) # Assume all subdirs are methods for now

            if not method_subdirs:
                print(f"(XAI Display CB) No subdirectories found within: {job_xai_base_path}")
                return f"No XAI method subdirectories found within '{job_xai_base_path}'.", {'display': 'block'}

            # Process each found method subdirectory
            for method_name in sorted(method_subdirs): # Sort for consistent order
                method_path = os.path.join(job_xai_base_path, method_name)
                method_file_components = [] # List to hold components for files within this method
                print(f"(XAI Display CB) Scanning method directory: {method_path}")

                try:
                    # List and process files within the method directory
                    files_in_method_dir = [f for f in os.listdir(method_path) if os.path.isfile(os.path.join(method_path, f))]

                    if not files_in_method_dir:
                         print(f"  - No files found in {method_path}")
                         method_file_components.append(html.P("(No displayable files found in this directory)", style={'color':'#aaa'}))
                    else:
                        for filename in sorted(files_in_method_dir): # Sort files
                            file_path = os.path.join(method_path, filename)
                            _, extension = os.path.splitext(filename.lower())

                            # Generate the correct URL using the helper
                            asset_url = get_asset_url(job_name, method_name, filename)

                            component_to_add = None
                            if extension in image_extensions:
                                print(f"  - Found Image: {filename}, URL: {asset_url}")
                                component_to_add = html.Img(src=asset_url,
                                                             alt=f"{method_name} - {filename}",
                                                             style={'maxWidth': '95%', 'height': 'auto', 'marginTop': '10px', 'border':'1px solid #444', 'display':'block', 'marginLeft':'auto', 'marginRight':'auto'}) # Center images
                                print(component_to_add)
                                component_with_header = html.Div([
                                    html.H5(filename, style={'marginTop':'15px', 'marginBottom':'5px', 'color':'#ddd', 'fontWeight':'normal', 'fontSize':'1em'}),
                                    component_to_add
                                ], style={'marginBottom':'20px'})
                                method_file_components.append(component_with_header)
                                found_any_results = True # Mark that we found at least one displayable file
                            
                            elif extension == html_extension:
                                print(f"  - Found HTML: {filename}, URL: {asset_url}")
                                component_to_add = html.Iframe(src=asset_url,
                                                               style={'width': '100%', 'height': '300px', 'marginTop': '10px', 'border': '1px solid #444', 'backgroundColor': 'white'})
                                component_with_header = html.Div([
                                    html.H5(filename, style={'marginTop':'15px', 'marginBottom':'5px', 'color':'#ddd', 'fontWeight':'normal', 'fontSize':'1em'}),
                                    component_to_add
                                ], style={'marginBottom':'20px'})
                                method_file_components.append(component_with_header)
                                found_any_results = True # Mark that we found at least one displayable file
                            
                            elif extension == csv_extension:
                                print(f"  - Found CSV: {filename}")
                                # Use the helper function to create a DataTable
                                component_to_add = create_cfe_delta_table(file_path)
                                component_with_header = html.Div([
                                    html.H5(filename, style={'marginTop':'15px', 'marginBottom':'5px', 'color':'#ddd', 'fontWeight':'normal', 'fontSize':'1em'}),
                                    component_to_add
                                ], style={'marginBottom':'20px'})
                                method_file_components.append(component_with_header)
                                found_any_results = True # Mark that we found at least one displayable file

                            sys.stdout.flush()

                except Exception as e:
                    print(f"(XAI Display CB) Error scanning/processing files in directory {method_path}: {e}")
                    method_file_components.append(html.P(f"Error processing results for {method_name}: {e}", style={'color':'red'}))

                #print(f"method_file_components (after last append): {method_file_components}")
                if method_file_components:
                    xai_content_blocks.append(html.Div([
                        # Method Title
                        html.H4(f"{method_name} Results", style={'borderBottom': '1px solid #555', 'paddingBottom': '5px', 'marginTop': '25px', 'marginBottom': '15px', 'color':'#eee'}),
                        # File Components
                        *method_file_components # Unpack the list of components
                    ], className="xai-method-block", style={'marginBottom': '30px', 'padding': '15px', 'border': '1px solid #555', 'borderRadius':'5px', 'backgroundColor': 'rgba(40,40,40,0.5)'})) # Style the block

            # --- Aggregated Feature Importance Comparison Plot ---
            aggregated_fi_path = os.path.join(job_xai_base_path, "aggregated_feature_importances.json")
            print(f"(XAI Display CB) Checking for aggregated FI summary: {aggregated_fi_path}")

            if os.path.exists(aggregated_fi_path):
                try:
                    with open(aggregated_fi_path, 'r') as f:
                        aggregated_scores_data = json.load(f)
                    
                    if aggregated_scores_data:
                        print("(XAI Display CB) Found aggregated_scores_data. Attempting to plot comparison.")
                        comparison_plot_filename = f"{job_name}_feature_importance_comparison.html"
                        
                        # Call the plotting function (it saves the file)
                        plot_aggregated_feature_importance_comparison(
                            aggregated_scores_data,
                            output_dir=job_xai_base_path, 
                            job_name=job_name
                        )
                        
                        comparison_plot_asset_url = f"/xai-assets/{urllib.parse.quote(job_name)}/{urllib.parse.quote(comparison_plot_filename)}"
                        expected_comparison_plot_path = os.path.join(job_xai_base_path, comparison_plot_filename)

                        if os.path.exists(expected_comparison_plot_path):
                            print(f"(XAI Display CB) Comparison plot HTML exists at: {expected_comparison_plot_path}")
                            comparison_plot_component = html.Div([
                                html.H4("Aggregated Feature Importance Comparison", style={'borderBottom': '1px solid #555', 'paddingBottom': '5px', 'marginTop': '30px', 'marginBottom': '15px', 'color':'#eee'}),
                                html.Iframe(
                                    src=comparison_plot_asset_url,
                                    style={'width': '100%', 'height': '700px', 'border': '1px solid #444', 'backgroundColor': 'rgba(40,40,40,0.5)'}
                                )
                            ], style={'marginTop': '20px', 'marginBottom': '20px', 'padding': '15px', 'border': '1px solid #555', 'borderRadius':'5px', 'backgroundColor': 'rgba(40,40,40,0.5)'})
                            xai_content_blocks.append(comparison_plot_component)
                            # found_any_results = True # No longer solely rely on found_any_results
                        else:
                            print(f"(XAI Display CB) Comparison plot HTML NOT found at {expected_comparison_plot_path} after calling plot function.")
                            # xai_content_blocks.append(html.P("Failed to generate comparison plot.", style={'color':'orange'}))
                    else:
                        print("(XAI Display CB) aggregated_scores_data loaded but was empty.")
                except Exception as e:
                    print(f"(XAI Display CB) Error loading or plotting aggregated feature importances: {e}")
                    traceback.print_exc()
                    xai_content_blocks.append(html.P(f"Error displaying aggregated feature importance comparison: {e}", style={'color':'red'}))
            else:
                print(f"(XAI Display CB) Aggregated feature importance file not found: {aggregated_fi_path}")

            # --- Final Return based on xai_content_blocks ---
            if xai_content_blocks: # If any content (method files, error messages, or aggregated plot) was added
                print("(XAI Display CB) xai_content_blocks is populated. Returning content.")
                return xai_content_blocks, {'display': 'block'}
            else:
                # This means base dir existed, but no method subdirs with files, AND no aggregated_fi.json (or it failed to process into content)
                print("(XAI Display CB) xai_content_blocks is empty. No displayable results found.")
                message = f"No displayable XAI results or summary plot found for job '{job_name}'."
                return message, {'display': 'block'}

        except Exception as e:
            print(f"(XAI Display CB) General error processing XAI for job {job_name}:")
            traceback.print_exc()
            return f"An error occurred while trying to load XAI results: {e}", {'display': 'block'}

    # --- Callback For The "Back To Home" Button ---
    @app.callback(
        Output('url', 'pathname'),            # Target the 'pathname' of dcc.Location(id='url')
        Input('back-to-home-button', 'n_clicks'), # Listen to button clicks
        prevent_initial_call=True             # IMPORTANT: Don't run when the page loads
    )
    def go_back_to_home(n_clicks):
        if n_clicks and n_clicks > 0:
            print("Back button clicked, navigating to home ('/')")
            # NOTE: Ensure 'dcc.Location(id="url")' exists in your main app layout (app.py)
            return "/"  # Return the path for the home page
        return dash.no_update # If no clicks (or initial call), do nothing
    # --- End Callback For "Back To Home" Button ---

    # --- Add other callbacks if needed ---

    print("Job page callbacks registered.")
    sys.stdout.flush() # Ensure registration message is flushed