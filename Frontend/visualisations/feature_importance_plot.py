import os
from typing import Dict, Optional
import plotly.graph_objects as go 
import pandas as pd

def plot_aggregated_feature_importance_comparison(
    aggregated_scores_data: Dict[str, Dict[str, float]], # {"Method1": {"FeatA": score, "FeatB": score}, ...}
    output_dir: str, # Base output directory (e.g., /data/job_name)
    job_name: str,
    top_n_features: Optional[int] = 15 # Number of top features to show
    ):
    """
    Creates a grouped bar chart comparing aggregated feature importances across XAI methods.
    Saves the plot as an HTML file.
    """
    print("--- Plotting Aggregated Feature Importance Comparison ---")
    if not aggregated_scores_data:
        print("No aggregated scores data to plot.")
        return

    # Convert to DataFrame: Index=Feature, Columns=Method, Values=Score
    try:
        df_plot = pd.DataFrame(aggregated_scores_data).fillna(0) # Fill NaNs with 0
    except Exception as e:
        print(f"Error creating DataFrame from aggregated_scores_data: {e}")
        print(f"Data was: {aggregated_scores_data}")
        return

    if df_plot.empty:
        print("DataFrame for plotting aggregated scores is empty.")
        return

    # --- Determine top N features based on max importance across methods ---
    # For each feature, find its maximum importance score across all methods
    df_plot['max_importance'] = df_plot.max(axis=1)
    # Sort features by this max importance and take top N
    df_plot_sorted = df_plot.sort_values(by='max_importance', ascending=False)
    
    if top_n_features is not None and top_n_features > 0 and top_n_features < len(df_plot_sorted):
        df_to_display = df_plot_sorted.head(top_n_features)
        plot_title = f"Top {top_n_features} Aggregated Feature Importances by Method"
    else:
        df_to_display = df_plot_sorted
        plot_title = "Aggregated Feature Importances by Method"
    
    df_to_display = df_to_display.drop(columns=['max_importance']) # Drop the helper column

    if df_to_display.empty:
        print("No features to display after filtering for top_n.")
        return

    # --- Create Plotly Grouped Bar Chart ---
    fig = go.Figure()
    methods = df_to_display.columns.tolist() # XAI methods

    for method in methods:
        fig.add_trace(go.Bar(
            name=method,
            x=df_to_display.index, # Feature names
            y=df_to_display[method],
            text=df_to_display[method].apply(lambda x: f"{x:.3f}"), # Show value on bar
            textposition='auto'
        ))

    fig.update_layout(
        title=plot_title,
        xaxis_title="Features",
        yaxis_title="Aggregated Importance Score",
        barmode='group', 
        legend_title_text="XAI Method",
        template='seaborn',
        xaxis_tickangle=-45, # Angle feature names if many
        height=max(600, len(df_to_display) * 30) # Dynamic height
    )

    # --- Save Plot ---
    # Save it in the main job XAI directory, not under a specific method
    plot_output_dir = os.path.join(output_dir) 
    os.makedirs(plot_output_dir, exist_ok=True)
    plot_filepath = os.path.join(plot_output_dir, f"{job_name}_feature_importance_comparison.html")

    try:
        fig.write_html(plot_filepath)
        print(f"Saved feature importance comparison plot to: {plot_filepath}")
    except Exception as e:
        print(f"Error saving feature importance comparison plot: {e}")