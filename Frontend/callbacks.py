import base64
import re
import sys
import traceback
from dash import Dash, dcc, html, Input, Output, State, ALL, MATCH, callback, callback_context, no_update, ctx
import json
from get_handler import get_handler
import os
from ml_model_hyperparameters import HYPERPARAMETER_DESCRIPTIONS
from ml_model_hyperparameters import XAI_METHOD_DESCRIPTIONS
from pages.job_page import get_display_job_name

UPLOAD_DIRECTORY = "/app/Datasets"

# --- Create_active_jobs FUNCTION ---
def create_active_jobs(active_jobs):
    """
    Generates the HTML structure for the active jobs list, returning a single Div.
    """
    if not active_jobs:
        # Return a single Div containing the message
        return html.Div("No active jobs found.")

    job_divs = []
    for job in active_jobs:
        job_name = job.get("name", "Unknown Job") # Use .get for safety
        # Construct the internal Dash link
        dash_link = f'/job/{job_name}'

        # Create the Div for each job entry
        job_entry = html.Div([
            # Confirmation dialog for stopping the job
            dcc.ConfirmDialog(
                id={"type": "confirm-box", "index": job_name},
                message=f'Are you sure you want to cancel the job: {get_display_job_name(job_name)}?',
                displayed=False,
            ),
            # Link to the job's results page within the Dash app
            html.A(
                children=[get_display_job_name(job_name)],
                href=dash_link,
                # target="_blank", # Optional: uncomment to open in new tab
                style={
                    "marginRight": "15px",
                    "color": "#4CAF50", # Green link color
                    "textDecoration": "none",
                    "fontWeight": "bold",
                    "fontSize": "16px"
                }
            ),
            # Button to stop/cancel the job
            html.Button(
                "Stop Job",
                id={"type": "remove-dataset-btn", "index": job_name},
                n_clicks=0,
                style={
                    "fontSize": "12px",
                    "backgroundColor": "#e74c3c", # Red button color
                    "color": "#ffffff", 
                    "border": "none",
                    "borderRadius": "5px",
                    "padding": "5px 10px", # Slightly more padding
                    "cursor": "pointer" # Indicate it's clickable
                }
            )
        ], style={'paddingBottom': '8px', 'borderBottom': '1px solid #444'}) # Add padding and separator line

        job_divs.append(job_entry)

    # Return a single Div wrapping the title and the list container
    return html.Div([
        html.H4("Active Job List", style={'color': '#C0C0C0', 'marginBottom': '10px'}),
        html.Div(job_divs) # The list of job divs is the second child
    ])
    
# Helper function to list datasets
def get_available_datasets(upload_dir):
    """Scans the upload directory and returns a list of dataset filenames."""
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir) # Create if it doesn't exist
        return []
    try:
        return [f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))]
    except Exception as e:
        print(f"Error listing datasets in {upload_dir}: {e}")
        return []

# Define a function to handle saving and processing
def save_file(name, content):
    """Decode and save a file uploaded with dcc.Upload."""
    data = content.encode("utf8").split(b";base64,")[1]
    filepath = os.path.join(UPLOAD_DIRECTORY, name)
    try:
        with open(filepath, "wb") as fp:
            fp.write(base64.decodebytes(data))
        print(f"Saved file: {filepath}")
        return filepath # Return the path where the file was saved
    except Exception as e:
        print(f"Error saving file {name}: {e}")
        return None
    
    # --- Helper function to build the explanation content ---
def build_xai_explanation_content(method_name, current_index, total_methods):
    """Builds the HTML content for a single XAI method's explanation."""
    if not method_name or method_name == 'none':
        return [html.P("Invalid method selected.", style={'color':'#ffcc00'})]

    descriptions = XAI_METHOD_DESCRIPTIONS.get(method_name, {})
    if not descriptions:
        return [html.P(f"No description available for method: {method_name}", style={'color':'#ffcc00'})]

    # --- Navigation Elements ---
    nav_elements = []
    if total_methods > 1:
        nav_elements = [
            html.Div([
                html.Button('⬅️ Prev', id='xai-prev-btn', n_clicks=0,
                            disabled=(current_index == 0), # Disable if first item
                            style={'marginRight': '10px', 'padding': '5px 10px'}),
                html.Span(f"Method {current_index + 1} of {total_methods}",
                        style={'color': '#ffffff', 'fontWeight': 'bold'}),
                html.Button('Next ➡️', id='xai-next-btn', n_clicks=0,
                            disabled=(current_index >= total_methods - 1), # Disable if last item
                            style={'marginLeft': '10px', 'padding': '5px 10px'}),
            ], style={'textAlign': 'center', 'marginBottom': '15px'})
        ]

    # --- Description Content ---
    explanation_children = [
        html.H4(f"{method_name} Description:",),
        html.P(descriptions.get("description", "No description provided."),),
        html.H5("Capabilities:",),
        html.P(descriptions.get("capabilities", "Not specified."),),
        html.H5("Limitations:",),
        html.P(descriptions.get("limitations", "Not specified."),),
        html.H5("Parameters:",),
    ]

    params_dict = descriptions.get("parameters", {})
    if not params_dict:
        explanation_children.append(html.P("No specific parameters listed.", style={'fontStyle': 'italic', 'color':'#aaaaaa', 'marginLeft':'15px'}))
    else:
        for param, desc in params_dict.items():
            explanation_children.append(
                html.Div([
                    html.Strong(f"{param}:", style={'color':'#ffffff'}),
                    html.P(desc, style={'color':'#d0d0d0', 'marginTop':'2px', 'marginBottom':'10px', 'fontSize':'14px'})
                ], style={'marginLeft':'15px'})
            )

    # Combine navigation and content
    return nav_elements + explanation_children

# --- Callbacks for index page ---
def get_index_callbacks(app):

    # Upload dataset callback
    @callback(
        Output('dataset-dropdown', 'options', allow_duplicate=True),
        Output('dataset-dropdown', 'value', allow_duplicate=True),
        Output('output-upload-state', 'children'),
        Input('upload-dataset', 'contents'),
        State('upload-dataset', 'filename'),
        State('dataset-dropdown', 'options'), # Get current options to append
        prevent_initial_call=True # Don't run on page load
    )
    def update_output(uploaded_contents, uploaded_filename, existing_options):
        if uploaded_contents is not None:
            # Save the uploaded file
            filepath = save_file(uploaded_filename, uploaded_contents)

            if filepath:
                try:
                    print(f"File {uploaded_filename} saved. Handler should be notified.")

                    # --- Refresh dataset list
                    # Check if the dataset is already in the options to avoid duplicates
                    is_new = True
                    for option in existing_options:
                        if option['value'] == uploaded_filename:
                            is_new = False
                            break
                    if is_new:
                        new_options = existing_options + [{"label": uploaded_filename, "value": uploaded_filename}]
                    else:
                        new_options = existing_options

                    # Return updated options, select the new file, and show success message
                    return new_options, uploaded_filename, f"Uploaded: {uploaded_filename}"

                except Exception as e:
                    print(f"Error processing upload with handler: {e}")
                    return existing_options, no_update, f"Error processing: {uploaded_filename}"
            else:
                # File saving failed
                return existing_options, no_update, f"Error uploading: {uploaded_filename}"
        else:
            # Callback triggered without content (e.g., initial load, clearing upload)
            return existing_options, no_update, "" # No change, clear message


    # Callback to display the confirmation dialog
    @app.callback(
        Output('confirm-delete-dialog', 'displayed'),
        Input('delete-dataset-button', 'n_clicks'),
        State('dataset-dropdown', 'value'), # Check if a dataset is selected
        prevent_initial_call=True
    )
    def display_confirm_delete_dialog(n_clicks, selected_dataset):
        if n_clicks > 0 and selected_dataset: # Only show if button clicked AND a dataset is selected
            return True
        return False

    # Callback to handle the actual deletion after confirmation
    @app.callback(
        Output('dataset-dropdown', 'options', allow_duplicate=True),
        Output('dataset-dropdown', 'value', allow_duplicate=True),
        Output('output-delete-state', 'children'),
        Input('confirm-delete-dialog', 'submit_n_clicks'),
        State('dataset-dropdown', 'value'),
        prevent_initial_call=True
    )
    def handle_delete_dataset(submit_n_clicks, selected_dataset_to_delete):
        if submit_n_clicks and submit_n_clicks > 0:
            if selected_dataset_to_delete:
                file_to_delete_path = os.path.join(UPLOAD_DIRECTORY, selected_dataset_to_delete)
                try:
                    if os.path.exists(file_to_delete_path):
                        os.remove(file_to_delete_path)
                        print(f"Successfully deleted dataset: {file_to_delete_path}")

                        # Refresh dataset list from the directory
                        updated_datasets_list = get_available_datasets(UPLOAD_DIRECTORY)
                        new_options = [{"label": ds, "value": ds} for ds in updated_datasets_list]

                        # Determine the new value for the dropdown
                        new_selected_value = new_options[0]['value'] if new_options else None

                        return new_options, new_selected_value, f"Successfully deleted: {selected_dataset_to_delete}"
                    else:
                        # This case should ideally not happen if dropdown is synced with file system
                        print(f"File not found for deletion: {file_to_delete_path}")
                        # Refresh options anyway, as the file system is the source of truth
                        updated_datasets_list = get_available_datasets(UPLOAD_DIRECTORY)
                        new_options = [{"label": ds, "value": ds} for ds in updated_datasets_list]
                        new_selected_value = new_options[0]['value'] if new_options else None
                        return new_options, new_selected_value, f"Error: File '{selected_dataset_to_delete}' not found on server."

                except Exception as e:
                    print(f"Error deleting file {file_to_delete_path}: {e}")
                    # Even on error, refresh options to reflect actual state
                    updated_datasets_list = get_available_datasets(UPLOAD_DIRECTORY)
                    new_options = [{"label": ds, "value": ds} for ds in updated_datasets_list]
                    # Keep current selection if possible, or pick first if current was deleted
                    current_selection_still_valid = any(opt['value'] == selected_dataset_to_delete for opt in new_options)
                    new_selected_value = selected_dataset_to_delete if current_selection_still_valid else (new_options[0]['value'] if new_options else None)

                    return new_options, new_selected_value, f"Error deleting '{selected_dataset_to_delete}': {e}"
            else:
                # No dataset was selected in the dropdown when delete was confirmed (should be rare)
                return no_update, no_update, "No dataset selected to delete."
        return no_update, no_update, "" # No action if dialog not submitted

    # --- Callback to update Parameter Explanation Box ---
    @app.callback(
        Output("parameter-explanation-box", "children"),
        Input("detection-model-dropdown", "value"),
        prevent_initial_call=True
    )
    def update_parameter_explanations(selected_model):
        if not selected_model or selected_model == 'none':
            return [html.P("Select a model to see parameter explanations.", style={'color':'#b0b0b0'})]

        descriptions = HYPERPARAMETER_DESCRIPTIONS.get(selected_model, {})
        if not descriptions:
            return [html.P(f"No descriptions available for model: {selected_model}", style={'color':'#ffcc00'})]

        explanation_children = [
            html.H5(f"{selected_model} Parameters:", style={'color':'#ffffff', 'marginBottom':'15px', 'borderBottom': '1px solid #555', 'paddingBottom':'5px'})
        ]

        if not descriptions:
             explanation_children.append(html.P("No descriptions found for this model.", style={'fontStyle': 'italic', 'color':'#aaaaaa'}))
        else:
            for param, desc in descriptions.items():
                explanation_children.append(
                    html.Div([
                        html.Strong(f"{param}:", style={'color':'#ffffff'}),
                        html.P(desc, style={'color':'#d0d0d0', 'marginTop':'2px', 'marginBottom':'10px', 'fontSize':'14px'})
                    ])
                )

        return explanation_children
    
    # --- Callback 1: Update Store when Dropdown changes ---
    @app.callback(
        Output('xai-method-state-store', 'data'),
        Input("xai-method-dropdown", "value"),
        prevent_initial_call=True
    )
    def update_xai_method_store(selected_methods):
        if selected_methods is None:
            active_methods = []
        else:
            # Filter out 'none' and any potential falsy values
            active_methods = [m for m in selected_methods if m and m != 'none']

        if not active_methods:
            # Store empty data if no valid methods are selected
            return {'methods': [], 'index': 0}
        else:
            # Store the list of active methods and reset index to 0
            return {'methods': active_methods, 'index': 0}


    # --- Callback 2: Update Store based on Arrow Clicks ---
    @app.callback(
        Output('xai-method-state-store', 'data', allow_duplicate=True), # Allow duplicate needed
        Input('xai-prev-btn', 'n_clicks'),
        Input('xai-next-btn', 'n_clicks'),
        State('xai-method-state-store', 'data'),
        prevent_initial_call=True
    )
    def handle_xai_navigation(prev_clicks, next_clicks, current_state):
        if not current_state or not current_state.get('methods'):
            # Should not happen if buttons are only active when state is valid, but safety check
            return no_update

        methods = current_state.get('methods', [])
        index = current_state.get('index', 0)
        total_methods = len(methods)

        # Determine which button was clicked
        triggered_id = ctx.triggered_id

        if triggered_id == 'xai-prev-btn' and index > 0:
            index -= 1
        elif triggered_id == 'xai-next-btn' and index < total_methods - 1:
            index += 1
        else:
            # No valid navigation click occurred
            return no_update

        return {'methods': methods, 'index': index}


    # --- Callback 3: Update Explanation Box based on Store data ---
    @app.callback(
        Output("xai-explanation-box", "children"),
        Input('xai-method-state-store', 'data'),
        # No prevent_initial_call here, needs to run when store updates
    )
    def update_xai_explanations_from_store(state_data):
        # Handle initial load or empty state
        if not state_data or not state_data.get('methods'):
            return [html.P("Select an XAI method to see its description.", style={'color':'#b0b0b0'})]

        methods = state_data.get('methods', [])
        index = state_data.get('index', 0)
        total_methods = len(methods)

        # Validate index just in case
        if index >= total_methods or index < 0:
            index = 0 # Reset to 0 if index is somehow invalid

        # Check again if methods list became empty after potential validation
        if not methods:
            return [html.P("Select an XAI method to see its description.", style={'color':'#b0b0b0'})]

        current_method_name = methods[index]

        # Use the helper function to build the content
        return build_xai_explanation_content(current_method_name, index, total_methods)

    # --- Callback to toggle visibility of Injection panel ---
    @app.callback(
        Output("injection-panel", "style"), # Target the inner panel
        Input("injection-check", "value")
    )
    def toggle_injection_panel(selected_injection):
        if "use_injection" in selected_injection:
            # Return style to make it visible, keep other styles if needed
            return {"display": "block", "marginTop": "15px", "padding": "10px", "border": "1px solid #444", "borderRadius": "5px", "backgroundColor": "#145E88"}
        return {"display": "none"}

    # --- Callback to toggle visibility of Labeled panel ---
    @app.callback(
        Output("label-column-selection-div", "style"),
        Input("labeled-check", "value")
    )
    def toggle_labeled_panel(selected_labeled):
        if "is_labeled" in selected_labeled:
            return {"display": "block", "marginTop": "10px", "textAlign": "center"}
        return {"display": "none"}
    
    # --- Callback to generate dynamic ML model settings panel ---
    @app.callback(
        Output("model-settings-panel", "children"),
        Output("model-settings-panel", "style"), # Add style output to show/hide
        Input("detection-model-dropdown", "value")
    )
    def update_model_settings_panel(selected_model):
        if not selected_model:
            return [], {"display": "none"} # Hide if no model selected

        settings_children = []
        settings_children = [html.H5(f"Settings for {selected_model}:", style={'color':'#ffffff', 'marginBottom': '15px'})]
        panel_style = {"marginTop": "15px", "padding": "15px", "border": "1px solid #444", "borderRadius": "5px", "backgroundColor": "#145E88", "display": "block"} # Style to show panel

        # Use Pattern-Matching IDs for settings
        setting_id_base = {'type': 'ml-setting', 'model': selected_model}

        # Define settings based on selected_model
        if selected_model == "XGBoost":
            settings_children.extend([
                html.Div([
                    html.Label("Auto Tune:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight": "10px", "display": "inline-block", "width": "180px"}),
                    dcc.Checklist(
                        id={**setting_id_base, 'param': 'auto_tune'},
                        options=[{'label': '', 'value': 'auto'}],
                        value=[],
                        style={'display': 'inline-block', 'verticalAlign': 'middle'}
                    ),
                ], style={'marginBottom': '8px', 'textAlign': 'left'}),
                # --- Core Booster Params ---
                html.Div([
                    html.Label("Num Estimators (n_estimators):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'n_estimators'},
                                type="number", value=100, min=10, step=10, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("Learning Rate (learning_rate):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                        dcc.Input(id={**setting_id_base, 'param': 'learning_rate'},
                                type="number",
                                value=0.1,
                                min=0.0,  
                                step=0.01,
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("Max Depth (max_depth):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'max_depth'},
                                type="number", value=6, min=1, step=1, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("Min Child Weight (min_child_weight):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'min_child_weight'},
                                type="number", value=1, min=0, step=1, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("Gamma (min_split_loss):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'gamma'},
                                type="number", value=0, min=0, step=0.1, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Sampling Params ---
                    html.Div([
                    html.Label("Subsample Ratio (subsample):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'subsample'},
                                type="number", value=1.0, min=0.1, max=1.0, step=0.05, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("Col Sample by Tree (colsample_bytree):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'colsample_bytree'},
                                type="number", value=1.0, min=0.1, max=1.0, step=0.05,  
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("Col Sample by Level (colsample_bylevel):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'colsample_bylevel'},
                                type="number", value=1.0, min=0.1, max=1.0, step=0.05, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("Col Sample by Node (colsample_bynode):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'colsample_bynode'},
                                type="number", value=1.0, min=0.1, max=1.0, step=0.05, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Regularization ---
                    html.Div([
                    html.Label("L1 Reg Alpha (reg_alpha):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'reg_alpha'},
                                type="number", value=0, min=0, step=0.1, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("L2 Reg Lambda (reg_lambda):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'reg_lambda'},
                                type="number", value=1, min=0, step=0.1, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Other Params ---
                html.Div([
                    html.Label("Booster Type:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'booster'},
                                    options=[{'label': b, 'value': b} for b in ['gbtree', 'gblinear', 'dart']],
                                    value='gbtree', clearable=False, # Default gbtree
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("Random State (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'random_state'},
                                type="number", placeholder="None", step=1, # Allow None (empty) or int
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("Probability Calibration Method:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'calibration_method'},
                                    options=[{'label': m, 'value': m} for m in ['isotonic', 'sigmoid']],
                                    value='isotonic', clearable=False, # Default isotonic from backend
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("Number of jobs (-1 for all available):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "210px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'n_jobs'},
                                type="number", value=-1, min=-1, step=1, # Default -1 (all threads)
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                # --- Common wrapper parameters ---
                html.Hr(style={'borderColor': '#555', 'margin': '15px 0'}),
                html.P("Preprocessing & Cross-Validation (Wrapper):", style={"fontSize": "16px", "color": "#e0e0e0", "fontWeight": "bold"}),

                html.Div([
                    html.Label("Imputer Strategy:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'imputer_strategy'},
                                    options=[{'label': s, 'value': s} for s in ['mean', 'median', 'most_frequent', 'constant']],
                                    value='mean', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                html.Div([
                    html.Label("CV n_splits (>=2):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'n_splits'}, type="number", value=5, min=2, step=1,
                                style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                html.Div([
                    html.Label("CV Shuffle KFold:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight": "10px", "display": "inline-block", "width": "180px"}),
                    dcc.Checklist(id={**setting_id_base, 'param': 'shuffle_kfold'}, options=[{'label': '', 'value': 'true'}], value=['true'],
                                    style={'display': 'inline-block', 'verticalAlign': 'middle'}),
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                html.Div([
                    html.Label("CV Validation Metrics:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px", 'verticalAlign':'top'}),
                    dcc.Checklist(
                        id={**setting_id_base, 'param': 'validation_metrics'},
                        options=[
                            {'label': 'Accuracy', 'value': 'accuracy'},
                            {'label': 'F1-score (macro)', 'value': 'f1'},
                            {'label': 'ROC AUC', 'value': 'roc_auc'}
                        ],
                        value=['accuracy', 'f1', 'roc_auc'],
                        style={'display': 'inline-block', 'color': '#e0e0e0'},
                        labelStyle={'display': 'block', 'marginBottom': '5px'}
                    )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # Parameters for RandomizedSearchCV (conditional display can be handled in callback based on auto_tune)
                html.Div([
                    html.Label("Search n_iter (AutoTune):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'search_n_iter'}, type="number", value=10, min=1, step=1,
                                style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'},
                id={**setting_id_base, 'element': 'search_n_iterxgboost'}
                ),

                html.Div([
                    html.Label("Search Scoring (AutoTune):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'search_scoring'},
                                    options=[
                                        {'label': 'F1 (macro)', 'value': 'f1'},
                                        {'label': 'ROC AUC', 'value': 'roc_auc'},
                                        {'label': 'Accuracy', 'value': 'accuracy'}
                                    ],
                                    value='f1', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'},
                id={**setting_id_base,}
                ),
            ])
        elif selected_model == "lstm":
            settings_children.extend([
                # --- Architecture ---
                html.Div([
                    html.Label("LSTM Units per Layer (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'units'},
                                type="number", value=64, min=8, step=8, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("LSTM Activation:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'activation'},
                                    options=[{'label': act, 'value': act} for act in ['relu', 'tanh', 'sigmoid', 'elu']],
                                    value='relu', clearable=False, 
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                        html.Label("Dropout Rate (float 0-1):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                        dcc.Input(id={**setting_id_base, 'param': 'dropout'},
                                type="number", value=0.0, min=0.0, max=1.0, step=0.05, 
                                )
                    ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                        html.Label("Recurrent Dropout (float 0-1):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                        dcc.Input(id={**setting_id_base, 'param': 'recurrent_dropout'},
                                type="number", value=0.0, min=0.0, max=1.0, step=0.05, 
                                )
                    ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Compilation / Training ---
                    html.Div([
                    html.Label("Time Steps (Sequence Len):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'time_steps'},
                                type="number", value=10, min=1, step=1, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("Optimizer:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'optimizer'}, 
                                    options=[{'label': opt, 'value': opt} for opt in ['adam', 'rmsprop', 'sgd']],
                                    value='adam', clearable=False, 
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("Learning Rate (float):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'learning_rate'},
                                type="number", value=0.001, step=0.0001, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("Loss Function:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'loss'},
                                    options=[{'label': loss, 'value': loss} for loss in ['mse', 'mae']],
                                    value='mse', clearable=False, 
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("Training Epochs (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'epochs'},
                                type="number", value=10, min=1, step=1, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("Batch Size (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'batch_size'},
                                type="number", value=256, min=1, step=1, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
            ])
        elif selected_model == "svm":
            settings_children.extend([
                # --- Autoencoder Settings ---
                html.H6("Autoencoder Parameters:", style={'color':'#cccccc', 'marginTop':'10px', 'marginBottom': '8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("Encoding Dimension (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'encoding_dim'},
                                type="number", value=10, min=2, step=1, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("AE Hidden Activation:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'ae_activation'},
                                    options=[{'label': act, 'value': act} for act in ['relu', 'tanh', 'sigmoid', 'elu']],
                                    value='relu', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("AE Output Activation:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'ae_output_activation'},
                                    options=[{'label': act, 'value': act} for act in ['linear', 'sigmoid']], # Linear for StandardScaler, Sigmoid for MinMaxScaler
                                    value='linear', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("AE Optimizer:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'optimizer'},
                                    options=[{'label': opt, 'value': opt} for opt in ['adam', 'rmsprop', 'sgd']],
                                    value='adam', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("AE Learning Rate (float):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'learning_rate'},
                                type="number", value=0.001, step=0.0001,
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                html.Div([
                    html.Label("AE Loss Function:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'loss'},
                                    options=[{'label': loss, 'value': loss} for loss in ['mse', 'mae']],
                                    value='mse', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("AE Training Epochs (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'epochs'},
                                type="number", value=10, min=1, step=1, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    html.Div([
                    html.Label("AE Batch Size (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'batch_size'},
                                type="number", value=32, min=1, step=1, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- OneClassSVM Settings ---
                html.H6("OneClassSVM Parameters:", style={'color':'#cccccc', 'marginTop':'20px', 'marginBottom': '8px', 'textAlign':'left'}),
                # Kernel 
                html.Div([
                    html.Label("SVM Kernel:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'svm_kernel'}, 
                                    options=[{'label': k, 'value': k} for k in ['rbf', 'linear', 'poly', 'sigmoid']],
                                    value='rbf', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                # Nu 
                html.Div([
                    html.Label("SVM Nu (float 0-1):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'svm_nu'}, 
                                type="number", value=0.1, min=0.0, max=1.0, step=0.01, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                # Gamma 
                html.Div([
                    html.Label("SVM Gamma ('scale', 'auto', float):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'svm_gamma'}, 
                                type="text", value='scale', placeholder="'scale', 'auto' or float",
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                # Degree 
                html.Div([
                    html.Label("SVM Degree (int>=1, for Poly):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'svm_degree'}, 
                                type="number", value=3, min=1, step=1,
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                # Coef0 
                    html.Div([
                    html.Label("SVM Coef0 (float, Poly/Sigmoid):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'coef0'}, # Standard param name
                                type="number", value=0.0, step=0.1, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    # Shrinking 
                html.Div([
                    html.Label("SVM Shrinking Heuristic:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'shrinking'}, # Standard param name
                                    options=[{'label': 'True', 'value': True}, {'label': 'False', 'value': False}],
                                    value=True, 
                                    clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    # Tol 
                    html.Div([
                    html.Label("SVM Tolerance (float):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'tol'}, # Standard param name
                                type="number", value=1e-3, step=1e-4,
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
                    # Max Iter 
                    html.Div([
                    html.Label("SVM Max Iterations (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "190px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'max_iter'}, # Standard param name
                                type="number", placeholder="-1 (no limit)", step=1, min=-1, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

            ])
        elif selected_model == "isolation_forest":
            settings_children.extend([
                # --- Number of Estimators ---
                html.Div([
                    html.Label("Num Estimators (n_estimators):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'n_estimators'},
                                type="number", value=100, min=10, step=10, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Contamination ---
                html.Div([
                    html.Label("Contamination ('auto' or float):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'contamination'},
                                type="text", value='auto', placeholder="'auto' or float (0-0.5)", # Text allows 'auto' or float
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Max Samples ---
                html.Div([
                    html.Label("Max Samples ('auto', int, float):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'max_samples'},
                                type="text", value='auto', placeholder="'auto', int or float", # Text allows 'auto', int, or float
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Max Features ---
                html.Div([
                    html.Label("Max Features (float 0-1):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'max_features'},
                                type="number", value=1.0, min=0.0, max=1.0, step=0.05, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Bootstrap ---
                html.Div([
                    html.Label("Bootstrap Samples:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'bootstrap'},
                                    options=[{'label': 'False', 'value': False}, {'label': 'True', 'value': True}],
                                    value=False, 
                                    clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Random State ---
                html.Div([
                    html.Label("Random State (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'random_state'},
                                type="number", placeholder="None", step=1, # Allow None (empty) or int
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),
            ])
        elif selected_model == "decision_tree":
            settings_children.extend([
                html.Div([
                    html.Label("Auto Tune:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight": "10px", "display": "inline-block", "width": "180px"}),
                    dcc.Checklist(
                        id={**setting_id_base, 'param': 'auto_tune'},
                        options=[{'label': '', 'value': 'auto'}],
                        value=[],
                        style={'display': 'inline-block', 'verticalAlign': 'middle'}
                    ),
                ], style={'marginBottom': '8px', 'textAlign': 'left'}),
                
                # --- Criterion ---
                html.Div([
                    html.Label("Criterion:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'criterion'},
                                    options=[{'label': c, 'value': c} for c in ['gini', 'entropy', 'log_loss']], 
                                    value='gini', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Splitter ---
                html.Div([
                    html.Label("Splitter Strategy:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'splitter'},
                                    options=[{'label': s, 'value': s} for s in ['best', 'random']],
                                    value='best', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Max Depth ---
                html.Div([
                    html.Label("Max Depth (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'max_depth'},
                                type="number", placeholder="None (unlimited)", step=1, min=1, # Ensure positive integer if set
                                ) 
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Min Samples Split ---
                html.Div([
                    html.Label("Min Samples Split (int>=2):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'min_samples_split'},
                                type="number", value=2, min=2, step=1, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                    # --- Min Samples Leaf ---
                html.Div([
                    html.Label("Min Samples Leaf (int>=1):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'min_samples_leaf'},
                                type="number", value=1, min=1, step=1, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                    # --- Min Weight Fraction Leaf ---
                html.Div([
                    html.Label("Min Weight Fraction Leaf (float):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'min_weight_fraction_leaf'},
                                type="number", value=0.0, min=0.0, max=0.5, step=0.01, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Max Features ---
                html.Div([
                    html.Label("Max Features:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'max_features'},
                                    options=[{'label': mf, 'value': mf} for mf in ['sqrt', 'log2', 'None']],
                                    value='sqrt', # Cannot be none
                                    clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'}),
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                    # --- Random State ---
                html.Div([
                    html.Label("Random State (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'random_state'},
                                type="number", placeholder="None", step=1,
                                ) # Allow None (empty) or int
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Max Leaf Nodes ---
                html.Div([
                    html.Label("Max Leaf Nodes (int>=2):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'max_leaf_nodes'},
                                type="number", placeholder="None (unlimited)", step=1, min=2,
                                ) # Allow None (empty) or int >= 2
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Min Impurity Decrease ---
                html.Div([
                    html.Label("Min Impurity Decrease (float):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'min_impurity_decrease'},
                                type="number", value=0.0, min=0.0, step=0.01, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- CCP Alpha (Pruning) ---
                html.Div([
                    html.Label("CCP Alpha (Pruning, float):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'ccp_alpha'},
                                type="number", value=0.0, min=0.0, step=0.01, 
                                )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Common wrapper parameters ---
                html.Hr(style={'borderColor': '#555', 'margin': '15px 0'}),
                html.P("Preprocessing & Cross-Validation (Wrapper):", style={"fontSize": "16px", "color": "#e0e0e0", "fontWeight": "bold"}),

                html.Div([
                    html.Label("Imputer Strategy:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'imputer_strategy'},
                                    options=[{'label': s, 'value': s} for s in ['mean', 'median', 'most_frequent', 'constant']],
                                    value='mean', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                html.Div([
                    html.Label("CV n_splits (>=2):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'n_splits'}, type="number", value=5, min=2, step=1,
                                style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                html.Div([
                    html.Label("CV Shuffle KFold:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight": "10px", "display": "inline-block", "width": "180px"}),
                    dcc.Checklist(id={**setting_id_base, 'param': 'shuffle_kfold'}, options=[{'label': '', 'value': 'true'}], value=['true'],
                                    style={'display': 'inline-block', 'verticalAlign': 'middle'}),
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                html.Div([
                    html.Label("CV Validation Metrics:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px", 'verticalAlign':'top'}),
                    dcc.Checklist(
                        id={**setting_id_base, 'param': 'validation_metrics'},
                        options=[
                            {'label': 'Accuracy', 'value': 'accuracy'},
                            {'label': 'F1-score (macro)', 'value': 'f1'},
                            {'label': 'ROC AUC', 'value': 'roc_auc'}
                        ],
                        value=['accuracy', 'f1', 'roc_auc'],
                        style={'display': 'inline-block', 'color': '#e0e0e0'},
                        labelStyle={'display': 'block', 'marginBottom': '5px'}
                    )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # Parameters for RandomizedSearchCV
                html.Div([
                    html.Label("Search n_iter (AutoTune):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'search_n_iter'}, type="number", value=10, min=1, step=1,
                                style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'},
                id={**setting_id_base, 'element': 'search_n_iter_decitree'}
                ),

                html.Div([
                    html.Label("Search Scoring (AutoTune):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'search_scoring'},
                                    options=[
                                        {'label': 'F1 (macro)', 'value': 'f1'},
                                        {'label': 'ROC AUC', 'value': 'roc_auc'},
                                        {'label': 'Accuracy', 'value': 'accuracy'}
                                    ],
                                    value='f1', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'},
                id={**setting_id_base,}
                ),
            ])
        elif selected_model == "SGDClassifier":
            settings_children.extend([
                html.Div([
                    html.Label("Auto Tune:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight": "10px", "display": "inline-block", "width": "180px"}),
                    dcc.Checklist(
                        id={**setting_id_base, 'param': 'auto_tune'},
                        options=[{'label': '', 'value': 'auto'}],
                        value=[],
                        style={'display': 'inline-block', 'verticalAlign': 'middle'}
                    ),
                ], style={'marginBottom': '8px', 'textAlign': 'left'}),

                # --- Loss Function ---
                html.Div([
                    html.Label("Loss Function:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'loss'},
                                    options=[
                                        {'label': 'Hinge (Linear SVM)', 'value': 'hinge'},
                                        {'label': 'Log Loss (Logistic Regression)', 'value': 'log_loss'},
                                        {'label': 'Modified Huber', 'value': 'modified_huber'},
                                        {'label': 'Squared Hinge', 'value': 'squared_hinge'},
                                        {'label': 'Perceptron', 'value': 'perceptron'}
                                    ],
                                    value='hinge', clearable=False,
                                    style={'width': '250px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Penalty (Regularization) ---
                html.Div([
                    html.Label("Penalty:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'penalty'},
                                    options=[
                                        {'label': 'L2', 'value': 'l2'},
                                        {'label': 'L1', 'value': 'l1'},
                                        {'label': 'ElasticNet', 'value': 'elasticnet'}
                                    ],
                                    value='l2', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Alpha (Regularization Strength) ---
                html.Div([
                    html.Label("Alpha (Reg. Strength):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'alpha'},
                                type="number", value=0.0001, min=0.0, step='any', # Positive float
                                style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Max Iterations ---
                html.Div([
                    html.Label("Max Iterations (Epochs):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'max_iter'},
                                type="number", value=1000, min=1, step=1, # Positive integer
                                style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Tolerance (tol) ---
                html.Div([
                    html.Label("Tolerance (tol):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'tol'},
                                type="number", value=0.001, min=0.0, step='any', # Positive float
                                style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Learning Rate Schedule ---
                html.Div([
                    html.Label("Learning Rate Schedule:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'learning_rate'},
                                    options=[
                                        {'label': 'Optimal', 'value': 'optimal'},
                                        {'label': 'Constant', 'value': 'constant'},
                                        {'label': 'Inverse Scaling', 'value': 'invscaling'},
                                        {'label': 'Adaptive', 'value': 'adaptive'}
                                    ],
                                    value='optimal', clearable=False,
                                    style={'width': '180px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Eta0 (Initial Learning Rate) ---
                html.Div([
                    html.Label("Eta0 (Initial LR):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'eta0'},
                                type="number", value=0.0, min=0.0, step='any',
                                placeholder="For const, invscaling, adaptive",
                                style={'width': '180px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Early Stopping ---
                html.Div([
                    html.Label("Early Stopping:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight": "10px", "display": "inline-block", "width": "180px"}),
                    dcc.Checklist(
                        id={**setting_id_base, 'param': 'early_stopping'},
                        options=[{'label': '', 'value': 'true'}],
                        value=[], 
                        style={'display': 'inline-block', 'verticalAlign': 'middle'}
                    ),
                ], style={'marginBottom': '8px', 'textAlign': 'left'}),

                # --- Validation Fraction (for early stopping) ---
                html.Div([
                    html.Label("Validation Fraction:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'validation_fraction'},
                                type="number", value=0.1, min=0.01, max=0.99, step=0.01,
                                style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- N Iter No Change (for early stopping) ---
                html.Div([
                    html.Label("N Iter No Change:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'n_iter_no_change'},
                                type="number", value=5, min=1, step=1,
                                style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),


                # --- Class Weight ---
                html.Div([
                    html.Label("Class Weight:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'class_weight'},
                                    options=[
                                        {'label': 'balanced', 'value': 'balanced'},
                                        {'label': 'None', 'value': 'None'}
                                    ],
                                    value='balanced', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- Calibrate Probabilities ---
                html.Div([
                    html.Label("Calibrate Probabilities:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight": "10px", "display": "inline-block", "width": "180px"}),
                    dcc.Checklist(
                        id={**setting_id_base, 'param': 'calibrate_probabilities'},
                        options=[{'label': '(for hinge/sq_hinge loss)', 'value': 'true'}],
                        value=['true'], 
                        style={'display': 'inline-block', 'verticalAlign': 'middle'}
                    ),
                ], style={'marginBottom': '8px', 'textAlign': 'left'}),

                # --- Random State ---
                html.Div([
                    html.Label("Random State (int):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'random_state'},
                                type="number", placeholder="e.g., 42", step=1,
                                style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # --- n_jobs (for OvA in multiclass) ---
                html.Div([
                    html.Label("N Jobs (OvA multiclass):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'n_jobs'},
                                type="number", value=-1, step=1,
                                style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),


                # --- Common wrapper parameters ---
                html.Hr(style={'borderColor': '#555', 'margin': '15px 0'}),
                html.P("Preprocessing & Cross-Validation (Wrapper):", style={"fontSize": "16px", "color": "#e0e0e0", "fontWeight": "bold"}),

                html.Div([
                    html.Label("Scaler Type:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'scaler_type'},
                                    options=[
                                        {'label': 'StandardScaler', 'value': 'standard'},
                                        {'label': 'MinMaxScaler', 'value': 'minmax'}
                                    ],
                                    value='standard', clearable=False,
                                    style={'width': '180px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                html.Div([
                    html.Label("Imputer Strategy:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'imputer_strategy'},
                                    options=[{'label': s, 'value': s} for s in ['mean', 'median', 'most_frequent', 'constant']],
                                    value='mean', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                html.Div([
                    html.Label("CV n_splits (>=2):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'n_splits'}, type="number", value=5, min=2, step=1,
                                style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                html.Div([
                    html.Label("CV Shuffle KFold:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight": "10px", "display": "inline-block", "width": "180px"}),
                    dcc.Checklist(id={**setting_id_base, 'param': 'shuffle_kfold'}, options=[{'label': '', 'value': 'true'}], value=['true'],
                                    style={'display': 'inline-block', 'verticalAlign': 'middle'}),
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                html.Div([
                    html.Label("CV Validation Metrics:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px", 'verticalAlign':'top'}),
                    dcc.Checklist(
                        id={**setting_id_base, 'param': 'validation_metrics'},
                        options=[
                            {'label': 'Accuracy', 'value': 'accuracy'},
                            {'label': 'F1-score (macro)', 'value': 'f1'},
                            {'label': 'ROC AUC', 'value': 'roc_auc'}
                        ],
                        value=['accuracy', 'f1', 'roc_auc'],
                        style={'display': 'inline-block', 'color': '#e0e0e0'},
                        labelStyle={'display': 'block', 'marginBottom': '5px'}
                    )
                ], style={'marginBottom':'8px', 'textAlign':'left'}),

                # Parameters for RandomizedSearchCV 
                html.Div([
                    html.Label("Search n_iter (AutoTune):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Input(id={**setting_id_base, 'param': 'search_n_iter'}, type="number", value=10, min=1, step=1,
                                style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'},
                id={**setting_id_base, 'element': 'search_n_iter_sgdclassifier'}
                ),

                html.Div([
                    html.Label("Search Scoring (AutoTune):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px", "display": "inline-block", "width": "180px"}),
                    dcc.Dropdown(id={**setting_id_base, 'param': 'search_scoring'},
                                    options=[
                                        {'label': 'F1 (macro)', 'value': 'f1'},
                                        {'label': 'ROC AUC', 'value': 'roc_auc'},
                                        {'label': 'Accuracy', 'value': 'accuracy'}
                                    ],
                                    value='f1', clearable=False,
                                    style={'width': '150px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'})
                ], style={'marginBottom':'8px', 'textAlign':'left'},
                id={**setting_id_base,}
                ),
            ])
        # Add elif blocks for other models 

        # If the selected model doesn't have specific settings defined here, hide the panel
        if len(settings_children) == 1: 
            return [], {"display": "none"}

        return settings_children, panel_style

    # --- Callback to toggle visibility of XAI options panel ---
    @app.callback(
        Output("xai-options-div", "style"),
        Input("xai-check", "value")
    )
    def toggle_xai_panel(selected_xai):
        if "use_xai" in selected_xai:
             return {"display": "block", "marginTop": "10px", "textAlign": "center"}
        return {"display": "none"}

    # --- Callback to populate columns for Injection Dropdown AND Label Dropdown ---
    @app.callback(
        Output("injection-column-dropdown", "options"),
        Output("time-column-dropdown", "options"),
        Output("label-column-dropdown", "options"),
        Output("label-column-dropdown", "value"), # Reset label value when dataset changes
        Output("time-column-dropdown", "value"), # reset this aswell
        Input("dataset-dropdown", "value"),
        prevent_initial_call=True
    )
    def update_column_dropdown(selected_dataset):
        print(f"--- update_column_dropdown callback triggered ---")
        print(f"Selected Dataset: {selected_dataset!r}")

        if not selected_dataset:
            print("No dataset selected. Returning empty options.")
            return [], [], None

        handler = get_handler()
        print(f"Handler object: {handler}")
        columns = []
        options = []
        try:
            print(f"Calling handler.handle_get_dataset_columns for '{selected_dataset}'...")
            columns = handler.handle_get_dataset_columns(selected_dataset)
            print(f"Handler returned columns: {columns} (Type: {type(columns)})")

            if not isinstance(columns, list):
                print("Warning: Handler did not return a list for columns. Returning empty options.")
                return [], [], None, None

            options = [{"label": col, "value": col} for col in columns]
            print(f"Generated options for dropdowns: {options}")

            if not options: print("Warning: No column options remaining.")

            return options, options, options, None, None # Return options for all three, reset label value

        except Exception as e:
            print(f"!!! ERROR inside update_column_dropdown try block: {e}")
            traceback.print_exc()
            return [], [], None

    # --- Callback to generate dynamic XAI settings panel ---
    @app.callback(
        Output("xai-settings-panel", "children"),
        Input("xai-method-dropdown", "value"),
        State("dataset-dropdown", "value"), # Need dataset to potentially populate features
    )
    def update_xai_settings_panel(selected_xai_methods, selected_dataset):
        if not selected_xai_methods:
            return []

        active_methods = [m for m in selected_xai_methods if m != 'none']
        if not active_methods:
             return []

        all_settings_children = []
        handler = get_handler() # Get handler once if needed for multiple methods

        # Fetch columns once if needed by multiple XAI methods 
        column_options = []
        if selected_dataset:
            try:
                columns = handler.handle_get_dataset_columns(selected_dataset)
                if isinstance(columns, list):
                    column_options = [{"label": col, "value": col} for col in columns]
                else:
                    print(f"Warning: Handler did not return list for columns: {columns}")
            except Exception as e:
                print(f"!!! ERROR fetching columns for XAI settings panel: {e}")
                traceback.print_exc()

        # --- Loop through each selected method ---
        for i, selected_xai_method in enumerate(active_methods):
            if i > 0: all_settings_children.append(html.Hr(style={'borderColor': '#555', 'margin': '20px 0'}))

            method_settings = [html.H5(f"Settings for {selected_xai_method.upper()}:", style={'color':'#ffffff', 'marginTop':'15px', 'marginBottom': '10px'})]

            # --- Pattern-Matching IDs ---
            if selected_xai_method == "ShapExplainer":
                method_settings.extend([
                    html.Div([
                        html.Label("Indices to explain (n_explain_max):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'ShapExplainer', 'param': 'n_explain_max'}, type="number", value=1000, min=1, step=1, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Num Samples (nsamples):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'ShapExplainer', 'param': 'nsamples'}, type="number", value=100, min=1, step=1, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("K for Background Summary (k_summary):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'ShapExplainer', 'param': 'k_summary'}, type="number", value=50, min=1, step=1, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("K for L1 Reg Features (l1_reg_k):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'ShapExplainer', 'param': 'l1_reg_k'}, type="number", value=20, min=1, step=1, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Explainer method:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Dropdown(id={'type': 'xai-setting', 'method': 'ShapExplainer', 'param': 'shap_method'}, options=[{'label': 'KernelShap (default)', 'value': 'kernel'},{'label': 'TreeShap', 'value': 'tree'},], value='kernel', clearable=False, style={'width': '150px', 'display': 'inline-block', 'color': '#333'})
                    ], style={'marginBottom':'8px'})
                ])
            elif selected_xai_method == "LimeExplainer":
                method_settings.extend([
                    html.Div([
                        html.Label("Indices to explain (n_explain_max):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'LimeExplainer', 'param': 'n_explain_max'}, type="number", value=10, min=1, step=1, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Num Features to Explain:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'LimeExplainer', 'param': 'num_features'}, type="number", value=15, min=1, step=1, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Num Samples (Perturbations):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'LimeExplainer', 'param': 'num_samples'}, type="number", value=1000, min=100, step=100, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Kernel Width (kernel_width):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'LimeExplainer', 'param': 'kernel_width'}, type="number", placeholder="LIME default", min=0.01, step=0.1, style={'width':'110px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Feature Selection:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Dropdown(id={'type': 'xai-setting', 'method': 'LimeExplainer', 'param': 'feature_selection'}, options=[{'label': 'Auto', 'value': 'auto'},{'label': 'Highest Weights', 'value': 'highest_weights'},{'label': 'Forward Selection', 'value': 'forward_selection'},{'label': 'Lasso Path', 'value': 'lasso_path'},{'label': 'None', 'value': 'none'}], value='auto', clearable=False, style={'width': '180px', 'display': 'inline-block', 'color': '#333'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Discretize Continuous:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Dropdown(id={'type': 'xai-setting', 'method': 'LimeExplainer', 'param': 'discretize_continuous'}, options=[{'label': 'True', 'value': True}, {'label': 'False', 'value': False}], value=True, clearable=False, style={'width': '100px', 'display': 'inline-block', 'color': '#333'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Sample Around Instance:", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Dropdown(id={'type': 'xai-setting', 'method': 'LimeExplainer', 'param': 'sample_around_instance'}, options=[{'label': 'True', 'value': True}, {'label': 'False', 'value': False}], value=True, clearable=False, style={'width': '100px', 'display': 'inline-block', 'color': '#333'})
                    ], style={'marginBottom':'8px'})
                ])
            elif selected_xai_method == "DiceExplainer":
                dice_specific_settings = [
                    html.Div([
                        html.Label("Indices to explain (n_explain_max):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'DiceExplainer', 'param': 'n_explain_max'}, type="number", value=10, min=1, step=1, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Num Counterfactuals (total_CFs):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'DiceExplainer', 'param': 'total_CFs'}, type="number", value=5, min=1, step=1, style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Desired Class (desired_class):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Input(id={'type': 'xai-setting', 'method': 'DiceExplainer', 'param': 'desired_class'}, type="text", value="opposite", style={'width':'80px'})
                    ], style={'marginBottom':'8px'}),
                    # Features to vary dropdown
                    html.Div([
                        html.Label("Features to vary (features_to_vary):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Dropdown(
                            id={'type': 'xai-setting', 'method': 'DiceExplainer', 'param': 'features_to_vary'},
                            options=column_options, # Use pre-fetched options
                            value=[], # Default to empty list 
                            multi=True,
                            placeholder="Select features (leave empty to vary all mutable)",
                            style={'width': '90%', 'maxWidth':'500px', 'display': 'inline-block', 'color': '#333', 'verticalAlign':'middle'}
                        )
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("Backend (ML model framework):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Dropdown(id={'type': 'xai-setting', 'method': 'DiceExplainer', 'param': 'backend'}, options=[{'label': 'SciKit-Learn', 'value': 'sklearn'},{'label': 'Tensorflow 1', 'value': 'TF1'},{'label': 'Tensorflow 2', 'value': 'TF2'},{'label': 'PyTorch', 'value': 'pytorch'}], value='sklearn', clearable=False, style={'width': '150px', 'display': 'inline-block', 'color': '#333'})
                    ], style={'marginBottom':'8px'}),
                    html.Div([
                        html.Label("DiCE Method (dice_method):", style={"fontSize": "16px", "color": "#e0e0e0", "marginRight":"5px"}),
                        dcc.Dropdown(id={'type': 'xai-setting', 'method': 'DiceExplainer', 'param': 'dice_method'}, options=[{'label': 'Random', 'value': 'random'},{'label': 'Genetic', 'value': 'genetic'},{'label': 'KD-Tree', 'value': 'kdtree'}], value='genetic', clearable=False, style={'width': '150px', 'display': 'inline-block', 'color': '#333'})
                    ], style={'marginBottom':'8px'})
                ]
                method_settings.extend(dice_specific_settings)

            all_settings_children.extend(method_settings)

        return all_settings_children

    # --- Callback to toggle Speedup input based on mode ---
    # @app.callback(
    #     Output("speedup-input-div", "style"),
    #     Input("mode-selection", "value")
    # )
    # def toggle_speedup_input(selected_mode):
    #     if selected_mode == "stream":
    #         return {"display": "block", "marginTop": "10px", "textAlign": "center"}
    #     return {"display": "none"}

    # --- Callback to toggle visibility of Active Jobs section ---
    @app.callback(
        Output("active-jobs-section", "style"),
        Input("active-jobs-list", "children") # Trigger based on content changes
    )
    def toggle_active_jobs_section(children):
        # Show the active jobs section unless the content indicates no jobs
        no_jobs_message = "No active jobs found."
        display_style = {"display": "block", "marginTop": "30px"}
        hide_style = {"display": "none"}

        if isinstance(children, list) and len(children) > 0:
            first_child = children[0]
            if isinstance(first_child, html.Div) and getattr(first_child, 'children', None) == no_jobs_message:
                return hide_style
            else:
                return display_style # Assume list contains job divs
        elif isinstance(children, str) and children == no_jobs_message:
            return hide_style 
        elif children: # If children exist and are not the 'no jobs' message
            return display_style
        else: # If children are None or empty list
            return hide_style


    # --- Callback to display confirmation box for stopping a job ---
    @app.callback(
        Output({"type": "confirm-box", "index": MATCH}, "displayed"),
        Input({"type": "remove-dataset-btn", "index": MATCH}, "n_clicks"),
        prevent_initial_call=True # Don't display on page load
    )
    def display_confirm(n_clicks):
        return True if n_clicks and n_clicks > 0 else False

    # --- Callback to manage active jobs list ---
    @callback(
        Output("active-jobs-list", "children"),
        Input("job-interval", "n_intervals"),
        Input({"type": "confirm-box", "index": ALL}, "submit_n_clicks"), # Listen to confirmation clicks
        State("active-jobs-json", "data") # Store previous state as JSON
    )
    def manage_and_remove_active_jobs(n_intervals, submit_n_clicks_list, active_jobs_json_state):
        """
        Periodically fetches the list of active jobs and updates the display.
        Also handles job cancellation confirmation. Includes error handling.
        Returns a list containing one item for the single Output.
        """
        ctx = callback_context
        triggered_id = ctx.triggered_id
        #print(f"manage_and_remove_active_jobs triggered by: {triggered_id}") # Log trigger

        handler = get_handler()
        error_message = None # Initialize error message

        # --- Handle Job Cancellation ---
        # Check if the trigger was one of the confirmation buttons AND it was clicked
        if isinstance(triggered_id, dict) and triggered_id.get("type") == "confirm-box":
            # Find which button was clicked
            button_index = -1
            for i, n_clicks in enumerate(submit_n_clicks_list):
                # Check if this specific confirmation box was clicked (n_clicks > 0)
                if n_clicks and n_clicks > 0:
                    # Extract the job name from the ID of the confirmation box that triggered
                    all_confirm_ids = ctx.inputs_list[1] # Get list of Input dicts for confirm-box
                    if i < len(all_confirm_ids):
                        button_index = i
                        job_to_cancel = all_confirm_ids[i]['id']['index']
                        print(f"Confirmation received for job: {job_to_cancel}")
                        try:
                            response = handler.handle_cancel_job(job_to_cancel)
                            if response != "success":
                                print(f"Backend error cancelling job '{job_to_cancel}': {response}")
                                error_message = f"Error cancelling {job_to_cancel}: {response}"
                            # The callback will proceed to refresh the list anyway
                        except Exception as cancel_err:
                            print(f"!!! EXCEPTION during handle_cancel_job for '{job_to_cancel}': {cancel_err}")
                            traceback.print_exc()
                            error_message = f"Frontend error cancelling job {job_to_cancel}."
                        break # Assume only one confirmation can be submitted at a time

        # --- Fetch and Update Active Jobs List ---
        try:
            #print("Fetching active jobs from backend...")
            raw_response = handler.handle_get_running()

            if not raw_response: raise ValueError("Received empty response from handle_get_running.")

            active_jobs_data = json.loads(raw_response)
            #print(f"Parsed active_jobs_data: {active_jobs_data}") # Log parsed data

            if not isinstance(active_jobs_data, dict) or 'running' not in active_jobs_data: raise TypeError("Invalid data structure received. Expected {'running': [...]}")
            if not isinstance(active_jobs_data['running'], list): raise TypeError("Invalid data structure: 'running' key is not a list.")
            active_jobs_list = active_jobs_data["running"]
            #print(f"Extracted active_jobs_list: {active_jobs_list}") # Log the final list

            # --- Compare with previous state ---
            current_jobs_json = json.dumps(active_jobs_list, sort_keys=True) # Sort keys for consistent comparison
            prev_jobs_json_state = active_jobs_json_state if active_jobs_json_state else json.dumps([]) # Handle initial None state
            if current_jobs_json == prev_jobs_json_state and triggered_id == "job-interval":
                print("Job list hasn't changed. Returning no_update.")
                return no_update

            # --- Generate new layout component ---
            #print("Job list changed or cancellation may have occurred. Updating display.")
            new_children_component = create_active_jobs(active_jobs_list) # Gets the single Div

            # --- Wrap the single component in a list for the Output ---
            return [new_children_component]

        except Exception as e:
            print(f"!!! EXCEPTION in manage_and_remove_active_jobs callback: {e}")
            traceback.print_exc()
            error_output = html.Div([
                html.P("Error updating active jobs list:", style={'color': 'red', 'fontWeight': 'bold'}),
                html.Pre(f"{traceback.format_exc()}", style={'color': 'red', 'fontSize': 'small', 'whiteSpace': 'pre-wrap'})
            ])
            # --- Wrap the error component in a list for the Output ---
            return [error_output]


    # --- start_job_handler ---
    @app.callback(
        [Output("popup", "style"), Output("popup-interval", "disabled"), Output("popup", "children")],
        [Input("start-job-btn", "n_clicks"), Input("popup-interval", "n_intervals")],
        [
            State("dataset-dropdown", "value"), State("detection-model-dropdown", "value"),
            State("mode-selection", "value"), State("name-input", "value"),
            State("injection-method-dropdown", "value"), State("timestamp-input", "value"),
            State("magnitude-input", "value"), State("percentage-input", "value"),
            State("duration-input", "value"), State("injection-column-dropdown", "value"),
            State("injection-check", "value"), #State("speedup-input", "value"),
            State("xai-sampling-strategy-dropdown", "value"), State("xai-sample-seed", "value"),
            State("popup", "style"),
            State("labeled-check", "value"), State("label-column-dropdown", "value"),
            State("time-column-dropdown", "value"),
            State("xai-check", "value"), State("xai-method-dropdown", "value"),
            State({'type': 'xai-setting', 'method': ALL, 'param': ALL}, 'value'),
            State({'type': 'xai-setting', 'method': ALL, 'param': ALL}, 'id'),
            State({'type': 'ml-setting', 'model': ALL, 'param': ALL}, 'value'),
            State({'type': 'ml-setting', 'model': ALL, 'param': ALL}, 'id'),
        ]
    )
    def start_job_handler(
            n_clicks, n_intervals,
            selected_dataset, selected_detection_model, selected_mode, job_name,
            selected_injection_method, timestamp, magnitude, percentage, duration,
            injection_columns, inj_check, #speedup, 
            xai_sampling_strategy, xai_sample_seed,
            style,
            labeled_check_val, selected_label_col,
            selected_time_col,
            # --- ARGS for pattern-matching states ---
            xai_check_val,
            selected_xai_methods,
            xai_settings_values,
            xai_settings_ids,
            # --- ML Settings Args ---
            ml_settings_values,
            ml_settings_ids
            ):
        handler = get_handler()
        children = "Job submission processed."
        style_copy = style.copy() if style else {} # Ensure style_copy is a dict

        ctx = callback_context
        # Check if callback was triggered by button click or interval timeout
        triggered_prop_id = ctx.triggered[0]['prop_id'] if ctx.triggered else 'No trigger'

        # Handle popup closing
        if triggered_prop_id == 'popup-interval.n_intervals':
            style_copy.update({"display": "none"})
            # Return style, disable interval, keep children text
            return style_copy, True, children

        # Handle button click
        if triggered_prop_id != 'start-job-btn.n_clicks' or not n_clicks or n_clicks == 0:
            # If not triggered by button or button hasn't been clicked, do nothing
            return no_update, no_update, no_update

        # --- Proceed with Job Submission Logic (triggered by button) ---
        print(f"Start job button clicked (n_clicks={n_clicks})")

        # --- Basic Validation ---
        error_msg = None
        if not selected_dataset:
            error_msg = "Please select a dataset."
        elif not selected_detection_model:
            error_msg = "Please select a detection model."
        elif not job_name: # Check if job_name is empty
            error_msg = "Job name cannot be empty."
        # --- New validation for job_name format ---
        elif not re.match(r"^[a-z_][a-z0-9_]*$", job_name): # Regex for validation
            if job_name[0].isdigit():
                error_msg = "Job name cannot start with a number."
            elif any(c.isupper() for c in job_name):
                error_msg = "Job name cannot contain uppercase letters."
            else:
                error_msg = "Job name must start with a lowercase letter or underscore, and contain only lowercase letters, numbers, or underscores."
        # --- End of new validation ---
        else:
            response = handler.check_name(job_name) 
            if response != "success":
                error_msg = "Job name already exists!"

        if error_msg:
            style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
            return style_copy, False, error_msg # Show error popup
        
        # Process Labeled Data Info
        is_labeled = "is_labeled" in labeled_check_val
        label_col_to_pass = None
        if is_labeled:
            if not selected_label_col:
                style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                return style_copy, False, "Please select the label column."
            label_col_to_pass = selected_label_col

        # Process Injection Info
        inj_params_list = None
        if "use_injection" in inj_check:
            if not selected_injection_method or selected_injection_method == "None": error_msg = "Please select an injection method."
            elif not timestamp: error_msg = "Please enter an injection timestamp."

            if error_msg:
                style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                return style_copy, False, error_msg

            inj_params = {
                "anomaly_type": selected_injection_method, "timestamp": str(timestamp),
                "magnitude": str(magnitude if magnitude is not None else 1),
                "percentage": str(percentage if percentage is not None else 0),
                "duration": str(duration if duration else '0s'),
                "columns": injection_columns if injection_columns else []
            }
            inj_params_list = [inj_params]

        # Process XAI Info
        use_xai = "use_xai" in xai_check_val
        xai_params_list = None
        xai_settings = None
        if use_xai:
            active_methods = [m for m in selected_xai_methods if m != 'none'] if selected_xai_methods else []
            if not active_methods:
                style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                return style_copy, False, "Please select at least one XAI method."

            all_parsed_settings = {}
            print(f"DEBUG: Received XAI settings IDs: {xai_settings_ids}")
            print(f"DEBUG: Received XAI settings Values: {xai_settings_values}")

            for id_dict, value in zip(xai_settings_ids, xai_settings_values):
                method_name = id_dict.get('method')
                param_name = id_dict.get('param')
                if not method_name or not param_name or method_name not in active_methods: continue

                if method_name not in all_parsed_settings: all_parsed_settings[method_name] = {}
                all_parsed_settings[method_name][param_name] = value

            print(f"DEBUG: Parsed all XAI settings: {all_parsed_settings}")
            
            xai_settings = {}
            xai_settings.update({"xai_sampling_strategy":xai_sampling_strategy, "xai_sample_seed":xai_sample_seed})
            xai_params_list = []
            for method_name in active_methods:
                if method_name in all_parsed_settings:
                    current_settings = all_parsed_settings[method_name]
                    # Perform key renaming if needed (e.g., l1_reg_k for Shap)
                    if method_name == "ShapExplainer" and "l1_reg_k" in current_settings:
                        current_settings["l1_reg_k_features"] = current_settings.pop("l1_reg_k")

                    xai_params_list.append({"method": method_name, "settings": current_settings})
                else:
                    print(f"Warning: No settings found/parsed for selected method: {method_name}")
                    xai_params_list.append({"method": method_name, "settings": {}}) # Add with empty settings

            if not xai_params_list: # Should not happen if active_methods is not empty
                 style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                 return style_copy, False, "Error constructing XAI parameters."
             
            xai_settings.update({"xai_params":xai_params_list})

            print(f"DEBUG: Constructed xai_params_list for backend: {xai_params_list}")
            
        # --- Process ML Model Settings ---
        ml_params_dict = {}
        if selected_detection_model: # Only parse if a model is selected
            print(f"DEBUG: Received ML settings IDs: {ml_settings_ids}")
            print(f"DEBUG: Received ML settings Values: {ml_settings_values}")

            for id_dict, value in zip(ml_settings_ids, ml_settings_values):
                # Ensure we only parse settings for the *currently selected* model
                if id_dict['model'] == selected_detection_model:
                    actual_value = value
                    param_name = id_dict['param']
                    
                    if selected_detection_model == 'decision_tree':
                        if param_name == 'max_depth' and value is None:
                            actual_value = None
                        elif param_name == 'shuffle_kfold' and isinstance(value, list): # For the wrapper
                            actual_value = 'true' in value
                    elif selected_detection_model == 'isolation_forest' and param_name in ['contamination', 'max_samples'] and isinstance(value, str) and value.lower() != 'auto':
                        try:
                            actual_value = float(value)
                        except ValueError:
                            print(f"Warning: Invalid float value '{value}' for {param_name}. Using default.")
                            actual_value = 'auto' 
                    elif selected_detection_model == 'SGDClassifier':
                        if param_name == 'shuffle_kfold' and isinstance(value, list): # For the wrapper
                            actual_value = 'true' in value
                        elif param_name == 'early_stopping' and isinstance(value, list):
                            actual_value = 'true' in value
                        elif param_name == 'calibrate_probabilities' and isinstance(value, list):
                            actual_value = 'true' in value
                        elif param_name == 'random_state' and value == '': # For the model itself
                            actual_value = None
                        elif param_name == 'class_weight' and value == 'None': # For the model itself
                             actual_value = None
                    elif selected_detection_model == 'XGBoost':
                        if param_name == 'shuffle_kfold' and isinstance(value, list): # For the wrapper
                            actual_value = 'true' in value
                    else:
                        actual_value = value

                    ml_params_dict[param_name] = actual_value
                    print(f"DEBUG: Parsed ML Setting: {param_name} = {actual_value}")

            print(f"DEBUG: Constructed ml_params_dict for backend: {ml_params_dict}")

        # --- Call Backend Handler ---

        try:
            #print(f"Sending job '{job_name}' with mode '{selected_mode}'...")
            #print(f"  Dataset: {selected_dataset}, Model: {selected_detection_model}")
            #print(f"  Label Column: {label_col_to_pass}")
            #print(f"  Time Column: {selected_time_col if selected_time_col != None else 'None'}")
            #print(f"  XAI Params: {xai_settings}")
            #print(f"  Injection Params: {inj_params_list}")
            #print(f"  ML Model Params: {ml_params_dict}") # Print new params
            #sys.stdout.flush()

            # --- Modify backend call signatures to include ml_params_dict ---
            model_params_to_pass = ml_params_dict if ml_params_dict else None # Pass None if empty

            if selected_mode == "batch":
                response = handler.handle_run_batch(
                    selected_dataset, selected_detection_model, job_name,
                    label_column=label_col_to_pass, time_column=selected_time_col, xai_params=xai_settings, inj_params=inj_params_list,
                    model_params=model_params_to_pass
                )
            # else: # stream
            #     Validate speedup for stream mode
            #     speedup_val = 1.0 # Default
            #     try:
            #          speedup_val = float(speedup) if speedup is not None else 1.0
            #          if speedup_val <= 0: raise ValueError("Speedup must be positive.")
            #     except ValueError:
            #          style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
            #          return style_copy, False, "Invalid speedup value for stream mode."

            #     response = handler.handle_run_stream(
            #         selected_dataset, selected_detection_model, job_name,
            #         label_column=label_col_to_pass, xai_params=xai_settings, inj_params=inj_params_list,
            #         model_params=model_params_to_pass
            #     )

            if response == "success":
                style_copy.update({"backgroundColor": "#4CAF50", "display": "block"})
                children = f"Job '{job_name}' started successfully!"
            else:
                style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
                children = f"Backend error starting job: {response}"

        except Exception as e:
            print(f"Error calling backend handler: {e}")
            traceback.print_exc()
            style_copy.update({"backgroundColor": "#e74c3c", "display": "block"})
            children = "Error communicating with backend."

        # Return style to show popup, disable interval timer, set popup text
        return style_copy, False, children