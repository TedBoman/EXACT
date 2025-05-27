
import dash
from dash import dcc, html, Input, Output, State
import os
import flask
import urllib.parse
from dotenv import load_dotenv
from pages.index import layout as index_layout
from pages.job_page import layout as job_layout
from callbacks import get_index_callbacks 
from job_page_callbacks import register_job_page_callbacks
from get_handler import get_handler

load_dotenv()
HOST = 'Backend' 
PORT = int(os.getenv('BACKEND_PORT'))
FRONTEND_PORT = int(os.getenv('FRONTEND_PORT'))
XAI_DIR = "/app/data"

# Dash application
app = dash.Dash(__name__, suppress_callback_exceptions = True)

# --- Add static route for XAI assets ---
@app.server.route(f'/xai-assets/<path:resource>')
def serve_xai_asset(resource):
    """Serves files from the XAI_DIR."""
    # Basic security: prevent navigating up directories
    resource = urllib.parse.unquote(resource) # Decode URL chars like %20
    safe_path = os.path.abspath(os.path.join(XAI_DIR, resource))
    if not safe_path.startswith(os.path.abspath(XAI_DIR)):
        print(f"Forbidden access attempt: {resource}")
        return flask.abort(403) 

    # Check if file exists before trying to send
    if not os.path.isfile(safe_path):
         print(f"XAI asset not found: {safe_path}")
         return flask.abort(404) # Not Found

    print(f"Serving XAI asset: {safe_path}")
    # Use send_from_directory for proper handling of MIME types etc.
    # The 'directory' argument should be the base directory (XAI_DIR)
    # The 'path' argument is the path relative to the directory
    try:
        directory, filename = os.path.split(safe_path)
        return flask.send_from_directory(directory, filename)
    except Exception as e:
        print(f"Error serving {safe_path}: {e}")
        return flask.abort(500) # Internal server error
# --------------------------------------

# Main layout
app.layout = html.Div([
    dcc.Store(id="store-data"), 
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])

handler = get_handler()

# Callback: Display the correct page content based on the URL
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname):
    #print(f"Routing: Pathname received: {pathname}") # Debugging line
    if pathname == "/":
        print("Routing to index page")
        return index_layout(handler)
    elif pathname and pathname.startswith("/job/"):
        # Extract job name from path like "/job/my_job_name"
        job_name = pathname.split("/job/", 1)[1]
        if not job_name: # Handle case like "/job/"
             print("Routing: Invalid job path.")
             return html.Div("Invalid Job Path", style={"textAlign": "center", "color": "orange"})
        print(f"Routing to job page for: {job_name}")
        # --- Call the layout function from job_page.py ---
        try:
            # Pass the handler instance and job_name to the job page layout
            return job_layout(handler, job_name)
        except Exception as e:
            print(f"Error generating job layout for {job_name}: {e}")
            return html.Div(f"Error loading layout for job '{job_name}'.", style={"textAlign": "center", "color": "red"})
    else:
        print(f"Routing: Path '{pathname}' not found.")
        # Return a more user-friendly 404 page
        return html.Div([
                html.H1("404 - Page Not Found", style={'color': '#E0E0E0'}),
                html.P(f"The requested path '{pathname}' was not recognized.", style={'color': '#C0C0C0'}),
                dcc.Link("Go back to Home Page", href="/", style={'color': '#7FDBFF'})
            ], style={"textAlign": "center", "padding": "50px", 'backgroundColor': '#104E78'})

get_index_callbacks(app)

register_job_page_callbacks(app)

if __name__ == "__main__":
    print(f"Starting the Dash server on http://0.0.0.0:{FRONTEND_PORT}")
    # Set debug=False for production if needed, True is useful for development
    app.run(debug=False, host="0.0.0.0", port=FRONTEND_PORT)