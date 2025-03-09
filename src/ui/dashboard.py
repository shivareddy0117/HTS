import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys

# Add the parent directory to path for importing modules
# This approach is more robust across different platforms
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import from the package
from src.data.data_processor import DataProcessor
from src.models.nfq_controller import NFQController


class HTSMonitoringDashboard:
    """
    Real-time dashboard for monitoring HTS tape manufacturing process.
    
    This dashboard provides visualizations and controls for monitoring the
    high-temperature superconductor (HTS) tape manufacturing process in real-time,
    including critical current uniformity, process parameters, and optimization suggestions.
    """
    
    def __init__(self, data_processor=None, controller=None):
        """
        Initialize the HTS Monitoring Dashboard.
        
        Args:
            data_processor (DataProcessor, optional): Data processor instance
            controller (NFQController, optional): NFQ controller instance
        """
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.title = "HTS Tape Manufacturing Monitor"
        
        self.data_processor = data_processor
        self.controller = controller
        
        # Sample data for initial plots
        self.generate_sample_data()
        
        # Create layout
        self.create_layout()
        
        # Register callbacks
        self.register_callbacks()
    
    def generate_sample_data(self):
        """Generate sample data for initial dashboard state."""
        # Position data (cm)
        self.positions = np.arange(0, 1000, 2)
        
        # Critical current data (A) with some dropouts
        self.critical_current = 300 + 30 * np.sin(self.positions / 100) + np.random.normal(0, 10, len(self.positions))
        dropout_indices = np.random.choice(range(len(self.positions)), size=5, replace=False)
        for idx in dropout_indices:
            # Create a dropout region
            start = max(0, idx - 5)
            end = min(len(self.positions), idx + 5)
            self.critical_current[start:end] *= 0.6
        
        # Calculate CV
        window_size = 8
        step_size = 2
        cv_positions = []
        cv_values = []
        
        for i in range(0, len(self.positions) - int(window_size / step_size)):
            start_idx = i
            end_idx = i + int(window_size / step_size)
            
            window_slice = self.critical_current[start_idx:end_idx]
            std_dev = np.std(window_slice)
            mean_val = np.mean(window_slice)
            
            if mean_val > 0:
                cv = std_dev / mean_val
                cv_values.append(cv)
                cv_positions.append(self.positions[start_idx])
        
        self.cv_positions = np.array(cv_positions)
        self.cv_values = np.array(cv_values)
        
        # Generate optimized CV (simulated improvement)
        self.optimized_cv = self.cv_values * 0.95  # 5% improvement
        
        # Generate process parameters data
        self.process_params_df = pd.DataFrame({
            'Position': self.positions,
            'Substrate_Temp_1': 800 + 15 * np.sin(self.positions / 120) + np.random.normal(0, 5, len(self.positions)),
            'Substrate_Temp_2': 820 + 10 * np.sin(self.positions / 150 + 1) + np.random.normal(0, 3, len(self.positions)),
            'Reaction_Zone_Pressure': 15 + 0.5 * np.sin(self.positions / 80) + np.random.normal(0, 0.1, len(self.positions)),
            'Deposition_Voltage': 10 + 0.2 * np.sin(self.positions / 200) + np.random.normal(0, 0.05, len(self.positions)),
            'Oxygen_Flow': 5 + 0.3 * np.cos(self.positions / 100) + np.random.normal(0, 0.1, len(self.positions))
        })
        
        # Generate PCA components (simulated)
        self.pca_df = pd.DataFrame({
            'Position': self.positions,
            'PCA1': 2 * np.sin(self.positions / 100) + np.random.normal(0, 0.2, len(self.positions)),
            'PCA2': 1.5 * np.cos(self.positions / 120) + np.random.normal(0, 0.1, len(self.positions)),
            'PCA3': 1 * np.sin(self.positions / 80) + np.random.normal(0, 0.15, len(self.positions))
        })
        
        # Optimized PCA components
        self.optimized_pca_df = self.pca_df.copy()
        self.optimized_pca_df['PCA1'] = self.pca_df['PCA1'] * 1.1
        self.optimized_pca_df['PCA2'] = self.pca_df['PCA2'] * 0.9
        self.optimized_pca_df['PCA3'] = self.pca_df['PCA3'] * 1.05
    
    def create_layout(self):
        """Create the dashboard layout."""
        # Header
        header = dbc.Row([
            dbc.Col([
                html.H1("HTS Tape Manufacturing Monitoring Dashboard"),
                html.P("Real-time monitoring and optimization of High-Temperature Superconductor tape manufacturing"),
            ], width=8),
            dbc.Col([
                html.Div([
                    html.H5("Live Status:"),
                    html.Span("ONLINE", className="badge bg-success p-2")
                ], className="d-flex align-items-center justify-content-end h-100")
            ], width=4)
        ], className="mb-4 mt-2")
        
        # Critical Current & CV Monitoring Section
        critical_current_section = dbc.Card([
            dbc.CardHeader([
                html.H4("Critical Current Monitoring"),
                html.Small("Live measurements of critical current and uniformity metrics")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Critical Current (Ic)"),
                        dcc.Graph(
                            id='critical-current-plot',
                            figure=self.create_critical_current_plot()
                        )
                    ], width=6),
                    dbc.Col([
                        html.H6("Coefficient of Variation (CV)"),
                        dcc.Graph(
                            id='cv-plot',
                            figure=self.create_cv_plot()
                        )
                    ], width=6)
                ])
            ])
        ], className="mb-4")
        
        # Process Parameters Monitoring Section
        process_params_section = dbc.Card([
            dbc.CardHeader([
                html.H4("Process Parameters"),
                html.Small("Real-time process parameter monitoring")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Key Process Parameters"),
                        dcc.Dropdown(
                            id='process-param-dropdown',
                            options=[
                                {'label': 'Substrate Temperature 1', 'value': 'Substrate_Temp_1'},
                                {'label': 'Substrate Temperature 2', 'value': 'Substrate_Temp_2'},
                                {'label': 'Reaction Zone Pressure', 'value': 'Reaction_Zone_Pressure'},
                                {'label': 'Deposition Voltage', 'value': 'Deposition_Voltage'},
                                {'label': 'Oxygen Flow', 'value': 'Oxygen_Flow'}
                            ],
                            value=['Substrate_Temp_1', 'Substrate_Temp_2', 'Reaction_Zone_Pressure'],
                            multi=True
                        ),
                        dcc.Graph(
                            id='process-params-plot',
                            figure=self.create_process_params_plot()
                        )
                    ], width=12)
                ])
            ])
        ], className="mb-4")
        
        # PCA and Optimization Section
        pca_optimization_section = dbc.Card([
            dbc.CardHeader([
                html.H4("Parameter Optimization"),
                html.Small("PCA analysis and optimization suggestions")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("PCA Components"),
                        dcc.Dropdown(
                            id='pca-dropdown',
                            options=[
                                {'label': 'PCA Component 1', 'value': 'PCA1'},
                                {'label': 'PCA Component 2', 'value': 'PCA2'},
                                {'label': 'PCA Component 3', 'value': 'PCA3'}
                            ],
                            value='PCA1',
                        ),
                        dcc.Graph(
                            id='pca-plot',
                            figure=self.create_pca_plot()
                        )
                    ], width=6),
                    dbc.Col([
                        html.H6("Process Optimization"),
                        html.Div([
                            html.P("Current Optimization Status:", className="mb-2"),
                            html.Div([
                                html.Span("Optimized", className="badge bg-success me-2"),
                                html.Span(f"CV Improvement: 5.6%", className="text-success")
                            ], className="mb-3"),
                            
                            html.P("Key Parameter Adjustments:", className="mb-2"),
                            dbc.Row([
                                dbc.Col([
                                    html.Div("Substrate Temperature:", className="fw-bold"),
                                    html.Div("Reaction Zone Pressure:", className="fw-bold"),
                                    html.Div("Deposition Voltage:", className="fw-bold")
                                ], width=6),
                                dbc.Col([
                                    html.Div([
                                        html.I(className="bi bi-arrow-up me-1 text-danger"),
                                        "Increase by 3.2%"
                                    ]),
                                    html.Div([
                                        html.I(className="bi bi-arrow-down me-1 text-primary"),
                                        "Decrease by 1.5%"
                                    ]),
                                    html.Div([
                                        html.I(className="bi bi-arrow-up me-1 text-danger"),
                                        "Increase by 2.1%"
                                    ])
                                ], width=6)
                            ], className="mb-3"),
                            
                            html.Div([
                                dbc.Button("Apply Optimization", color="success", className="me-2"),
                                dbc.Button("Reset", color="secondary")
                            ])
                        ])
                    ], width=6)
                ])
            ])
        ], className="mb-4")
        
        # Alerts and Metrics Section
        alerts_metrics_section = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("System Alerts")),
                    dbc.CardBody([
                        html.Div([
                            html.I(className="bi bi-check-circle-fill text-success me-2"),
                            "All systems operating normally"
                        ], className="mb-2"),
                        
                        html.Div([
                            html.I(className="bi bi-info-circle-fill text-info me-2"),
                            "Substrate temperature showing minor fluctuations"
                        ], className="mb-2"),
                        
                        html.Div([
                            html.I(className="bi bi-exclamation-triangle-fill text-warning me-2"),
                            "Potential dropout detected at position 432 cm"
                        ], className="mb-2")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Performance Metrics")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div("Average Critical Current:", className="fw-bold"),
                                html.Div("Average CV:", className="fw-bold"),
                                html.Div("Dropout Count:", className="fw-bold"),
                                html.Div("Manufacturing Efficiency:", className="fw-bold")
                            ], width=6),
                            dbc.Col([
                                html.Div(f"{np.mean(self.critical_current):.2f} A"),
                                html.Div(f"{np.mean(self.cv_values):.4f}"),
                                html.Div("3"),
                                html.Div("94%")
                            ], width=6)
                        ])
                    ])
                ])
            ], width=6)
        ], className="mb-4")
        
        # Timeline and Controls Section
        timeline_controls_section = dbc.Card([
            dbc.CardHeader(html.H5("Manufacturing Timeline and Controls")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Span("Manufacturing Start: ", className="fw-bold"),
                            html.Span(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
                        ], className="mb-2"),
                        html.Div([
                            html.Span("Current Position: ", className="fw-bold"),
                            html.Span(f"{self.positions[-1]:.1f} cm")
                        ], className="mb-2"),
                        html.Div([
                            html.Span("Estimated Completion: ", className="fw-bold"),
                            html.Span(f"{(datetime.now().hour + 2) % 24:02d}:{datetime.now().minute:02d}")
                        ], className="mb-2")
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            dbc.Button("Pause Manufacturing", color="warning", className="me-2"),
                            dbc.Button("Emergency Stop", color="danger")
                        ], className="mb-2"),
                        html.Div([
                            html.Label("Manufacturing Speed:", className="me-2"),
                            dcc.Slider(
                                id='speed-slider',
                                min=0.5,
                                max=2.0,
                                step=0.1,
                                value=1.0,
                                marks={0.5: '0.5x', 1.0: '1.0x', 1.5: '1.5x', 2.0: '2.0x'},
                            )
                        ])
                    ], width=6)
                ])
            ])
        ], className="mb-4")
        
        # Data update interval
        update_interval = dcc.Interval(
            id='update-interval',
            interval=5000,  # 5 seconds in milliseconds
            n_intervals=0
        )
        
        # Main layout
        self.app.layout = dbc.Container([
            header,
            critical_current_section,
            process_params_section,
            pca_optimization_section,
            alerts_metrics_section,
            timeline_controls_section,
            update_interval
        ], fluid=True)
    
    def create_critical_current_plot(self):
        """Create a plot of critical current vs position."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.positions,
            y=self.critical_current,
            mode='lines',
            name='Critical Current (Ic)',
            line=dict(color='blue', width=2)
        ))
        
        # Add a horizontal line for average Ic
        avg_ic = np.mean(self.critical_current)
        fig.add_trace(go.Scatter(
            x=[self.positions[0], self.positions[-1]],
            y=[avg_ic, avg_ic],
            mode='lines',
            name=f'Average Ic: {avg_ic:.2f} A',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title='Position (cm)',
            yaxis_title='Critical Current (A)',
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=300
        )
        
        return fig
    
    def create_cv_plot(self):
        """Create a plot of CV vs position."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.cv_positions,
            y=self.cv_values,
            mode='lines',
            name='Actual CV',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.cv_positions,
            y=self.optimized_cv,
            mode='lines',
            name='Optimized CV',
            line=dict(color='green', width=2)
        ))
        
        # Add horizontal lines for average CV
        avg_cv = np.mean(self.cv_values)
        avg_opt_cv = np.mean(self.optimized_cv)
        
        fig.add_trace(go.Scatter(
            x=[self.cv_positions[0], self.cv_positions[-1]],
            y=[avg_cv, avg_cv],
            mode='lines',
            name=f'Avg CV: {avg_cv:.4f}',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=[self.cv_positions[0], self.cv_positions[-1]],
            y=[avg_opt_cv, avg_opt_cv],
            mode='lines',
            name=f'Avg Opt CV: {avg_opt_cv:.4f}',
            line=dict(color='darkgreen', width=1, dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title='Position (cm)',
            yaxis_title='Coefficient of Variation (CV)',
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=300
        )
        
        return fig
    
    def create_process_params_plot(self):
        """Create a plot of process parameters vs position."""
        fig = go.Figure()
        
        params = ['Substrate_Temp_1', 'Substrate_Temp_2', 'Reaction_Zone_Pressure']
        colors = ['red', 'orange', 'blue']
        
        for param, color in zip(params, colors):
            fig.add_trace(go.Scatter(
                x=self.process_params_df['Position'],
                y=self.process_params_df[param],
                mode='lines',
                name=param.replace('_', ' '),
                line=dict(color=color, width=2)
            ))
        
        fig.update_layout(
            xaxis_title='Position (cm)',
            yaxis_title='Parameter Value',
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=300
        )
        
        return fig
    
    def create_pca_plot(self):
        """Create a plot of PCA components vs position."""
        fig = go.Figure()
        
        component = 'PCA1'
        
        fig.add_trace(go.Scatter(
            x=self.pca_df['Position'],
            y=self.pca_df[component],
            mode='lines',
            name=f'Current {component}',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.optimized_pca_df['Position'],
            y=self.optimized_pca_df[component],
            mode='lines',
            name=f'Optimized {component}',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            xaxis_title='Position (cm)',
            yaxis_title=f'{component} Value',
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=300
        )
        
        return fig
    
    def register_callbacks(self):
        """Register callbacks for interactive elements."""
        
        @self.app.callback(
            Output('process-params-plot', 'figure'),
            [Input('process-param-dropdown', 'value'),
            Input('update-interval', 'n_intervals')]
        )
        def update_process_params_plot(selected_params, n_intervals):
            """Update process parameters plot based on dropdown selection."""
            if not selected_params:
                selected_params = ['Substrate_Temp_1']
                
            fig = go.Figure()
            
            colors = ['red', 'orange', 'blue', 'green', 'purple']
            
            for i, param in enumerate(selected_params):
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=self.process_params_df['Position'],
                    y=self.process_params_df[param],
                    mode='lines',
                    name=param.replace('_', ' '),
                    line=dict(color=color, width=2)
                ))
            
            fig.update_layout(
                xaxis_title='Position (cm)',
                yaxis_title='Parameter Value',
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=300
            )
            
            return fig
        
        @self.app.callback(
            Output('pca-plot', 'figure'),
            [Input('pca-dropdown', 'value'),
            Input('update-interval', 'n_intervals')]
        )
        def update_pca_plot(selected_component, n_intervals):
            """Update PCA plot based on dropdown selection."""
            if not selected_component:
                selected_component = 'PCA1'
                
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=self.pca_df['Position'],
                y=self.pca_df[selected_component],
                mode='lines',
                name=f'Current {selected_component}',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=self.optimized_pca_df['Position'],
                y=self.optimized_pca_df[selected_component],
                mode='lines',
                name=f'Optimized {selected_component}',
                line=dict(color='green', width=2)
            ))
            
            fig.update_layout(
                xaxis_title='Position (cm)',
                yaxis_title=f'{selected_component} Value',
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=300
            )
            
            return fig
    
    def update_data(self, new_data):
        """
        Update dashboard with new data.
        
        Args:
            new_data (dict): Dictionary containing new data for the dashboard
        """
        if 'critical_current' in new_data:
            self.critical_current = new_data['critical_current']
        
        if 'cv_values' in new_data:
            self.cv_values = new_data['cv_values']
            
        if 'process_params' in new_data:
            self.process_params_df = new_data['process_params']
            
        if 'pca_components' in new_data:
            self.pca_df = new_data['pca_components']
    
    def run_server(self, debug=True, port=8050):
        """
        Run the dashboard server.
        
        Args:
            debug (bool): Whether to run in debug mode
            port (int): Port to run the server on
        """
        self.app.run_server(debug=debug, port=port)


if __name__ == "__main__":
    # Create and run dashboard
    dashboard = HTSMonitoringDashboard()
    dashboard.run_server(debug=True) 