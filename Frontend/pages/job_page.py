from dash import dcc, html
import plotly.graph_objects as go

# --- Helper function ---
def get_display_job_name(full_job_name):
    """Removes known prefixes ('job_batch_', 'job_stream_') using slicing."""
    if not full_job_name:
        return "No Job Selected"

    prefix_batch = "job_batch_"
    prefix_stream = "job_stream_"

    if full_job_name.startswith(prefix_batch):
        return full_job_name[10:]
    elif full_job_name.startswith(prefix_stream):
        return full_job_name[11:]
    else:
        return full_job_name
# --- End Helper Function ---

# This is the main layout function called by app.py
def layout(handler, job_name):
    """
    Defines the layout for the individual job results page.
    Header: Button (left), Title+Loading (center), Spacer (right).
    """
    display_name = get_display_job_name(job_name)
    print(f"Generating layout for job: {display_name}")

    # --- Theme Colors ---
    theme_colors = {
        'background': '#0D3D66',
        'header_background': '#1E3A5F',
        'content_background': '#104E78',
        'status_background': '#145E88',
        'text_light': '#E0E0E0',
        'text_black': '000000',
        'text_medium': '#C0C0C0',
        'text_dark': '#FFFFFF', # White for high contrast on dark buttons
        'border_light': '#444'
    }
    # --- End Theme Colors ---


    return html.Div([
        # --- Store Components ---
        dcc.Store(id='job-page-job-name-store', data=job_name),
        dcc.Store(id='job-page-data-store'),
        dcc.Store(id='job-page-xai-store'),
        dcc.Store(id='job-page-status-store'),

        # ... other stores ...
        dcc.Interval(id='job-page-interval-component', interval=10*1000, n_intervals=0),

        # --- Header ---
        html.Div([
            # --- Left Column (Button) ---
            html.Div([
                html.Button(
                    "<< Back",
                    id="back-to-home-button",
                    n_clicks=0,
                    style={
                        'backgroundColor': theme_colors['status_background'],
                        'color': theme_colors['text_dark'], # Use white/light text
                        'border': f"1px solid {theme_colors['border_light']}", # Subtle border
                        'borderRadius': '5px',         # Match other elements
                        'padding': '6px 12px',         # Padding
                        'fontSize': '14px',            # Font size
                        'cursor': 'pointer',           # Cursor
                        'fontWeight': 'bold'           # Make text bold
                    }
                )
            # Style for the left column div
            # Let flexbox determine width based on button size + padding
            ], style={'flex': '0 1 auto', 'textAlign': 'left'}), # Don't grow, allow shrink, auto basis

            # --- Center Column (Title + Loading) ---
            html.Div([
                html.H1(
                    f"Analysis Results: {display_name}",
                    style={
                        'color': theme_colors['text_light'],
                        'margin': '0',
                        'display': 'inline-block',
                        'verticalAlign': 'middle'
                        }
                ),
                dcc.Loading(
                    id="loading-job-page",
                    type="circle",
                    fullscreen=False,
                    children=[html.Div(id="loading-output-jobpage")],
                    style={
                        'display': 'inline-block',
                        'marginLeft': '15px',
                        'verticalAlign': 'middle',
                        'position': 'relative', # Helps vertical alignment sometimes
                        'top': '2px'            # Fine-tune vertical alignment if needed
                        }
                )
            # Style for the center column div
            # Allow grow/shrink, center content
            ], style={'flex': '1 1 auto', 'textAlign': 'center'}),

            # --- Right Column (Spacer) ---
            html.Div([
                # Empty div, its purpose is to take up space equal to the left column
                # We make it invisible but it still occupies layout space
                html.Button("<< Back", style={'visibility': 'hidden', 'fontSize': '14px', 'padding': '6px 12px'}) # Mimic button size invisibly
            ], style={'flex': '0 1 auto', 'visibility': 'hidden'}), # Match left flex, hide content

        ], style={ # Main Header Div Style
            'display': 'flex',
            'alignItems': 'center',
            # Use space-between ONLY if spacer exactly matches button width,
            # otherwise let flex grow handle centering. Let's stick with flex-grow.
            # 'justifyContent': 'space-between', # Use if spacer width is reliable
            'marginBottom': '20px',
            'padding': '15px',
            'backgroundColor': theme_colors['header_background'],
            'borderRadius': '5px'
        }), # --- End Header Div ---

        # --- Main Content Area ---
        html.Div([
            # Status Display
            html.Div(id='job-status-display', style={
                'marginBottom': '15px', 'padding': '10px',
                'border': f"1px solid {theme_colors['border_light']}",
                'borderRadius': '5px',
                'backgroundColor': theme_colors['status_background'],
                'color': theme_colors['text_dark']
                }),
            
            # Meta data            
            html.Div(id='job-metadata-display', style={'marginBottom': '20px', 'color': 'white'}),
            
            # Graph Area with Feature Selector
            html.Div([
                html.H3("Time Series Data & Detected Anomalies", style={'color': theme_colors['text_medium']}),
                
                # Feature Selector Dropdown
                html.Div([
                    html.Label("Select Features to Plot:", style={'color': theme_colors['text_light'], 'marginRight': '10px', 'display':'block', 'marginBottom':'5px'}),
                    dcc.Dropdown(
                        id='feature-selector-dropdown',
                        options=[], # Populated by callback
                        value=[],   # Initially empty or set by callback to first item
                        multi=True, # Allow selecting multiple features
                        placeholder="Select features...",
                        style={ # Styles for the dropdown input box itself
                            'width': '100%',
                            'color': theme_colors['text_black'], # Text color inside input box (e.g. "Select...")
                        },
                    ),
                ], style={'marginBottom': '15px', 'padding': '10px', 'backgroundColor': 'rgba(40,40,40,0.3)', 'borderRadius': '5px'}),
                
                dcc.Graph(id='timeseries-anomaly-graph', figure={}), # Empty figure initially
            ], style={'marginBottom': '20px'}),
            
            # XAI Section
            html.Div([
                html.H3("Explainability (XAI) Results", style={'color': '#C0C0C0'}),
                # Container where the callback will inject results
                html.Div(id='xai-results-content', children="Checking for XAI results...")
            ], id='xai-results-section', style={'display': 'block', 'marginBottom': '20px'}), 

            html.Div(id='other-plots-section')

        ], style={'padding': '20px', 'backgroundColor': theme_colors['content_background'], 'borderRadius': '10px'}),

    ], style={'padding': '30px', 'backgroundColor': theme_colors['background'], 'minHeight': '100vh'})