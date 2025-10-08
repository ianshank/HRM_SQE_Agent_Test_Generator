import plotly.graph_objects as go

# Create a flowchart using Plotly with improved layout
fig = go.Figure()

# Define node positions with better spacing
nodes = {
    'START': (2, 5, '#A5D6A7'),  # Light green for terminal
    'analyze_req': (2, 4, '#FFCDD2'),  # Light red/orange for decision
    'gen_test_plan': (2, 3, '#FFCDD2'),  # Light red/orange for decision  
    'finalize_out': (2, 2, '#B3E5EC'),  # Light cyan for process
    'END': (2, 1, '#A5D6A7')  # Light green for terminal
}

# Add nodes as scatter points with uniform sizing
for node, (x, y, color) in nodes.items():
    fig.add_trace(go.Scatter(
        x=[x], y=[y],
        mode='markers+text',
        marker=dict(size=100, color=color, line=dict(width=3, color='black')),
        text=[node],
        textposition='middle center',
        textfont=dict(size=12, color='black', family='Arial Black'),
        showlegend=False,
        hoverinfo='skip'
    ))

# Add arrows and labels for the flow
# START → analyze_requirements
fig.add_trace(go.Scatter(
    x=[2, 2], y=[4.8, 4.2],
    mode='lines',
    line=dict(color='black', width=3),
    showlegend=False,
    hoverinfo='skip'
))
fig.add_annotation(x=2, y=4.2, ax=2, ay=4.8, arrowhead=2, arrowsize=2, arrowwidth=3, arrowcolor='black', showarrow=True, text='')

# analyze_requirements → generate_test_plan (True path)
fig.add_trace(go.Scatter(
    x=[1.8, 1.8], y=[3.8, 3.2],
    mode='lines',
    line=dict(color='black', width=3),
    showlegend=False,
    hoverinfo='skip'
))
fig.add_annotation(x=1.8, y=3.2, ax=1.8, ay=3.8, arrowhead=2, arrowsize=2, arrowwidth=3, arrowcolor='black', showarrow=True, text='')
fig.add_annotation(x=1.4, y=3.5, text='analysis_complete?<br>True', showarrow=False, font=dict(size=10), bgcolor='white', bordercolor='black', borderwidth=1)

# analyze_requirements → END (False path)
fig.add_trace(go.Scatter(
    x=[2.2, 2.2], y=[3.8, 1.2],
    mode='lines',
    line=dict(color='black', width=3),
    showlegend=False,
    hoverinfo='skip'
))
fig.add_annotation(x=2.2, y=1.2, ax=2.2, ay=3.8, arrowhead=2, arrowsize=2, arrowwidth=3, arrowcolor='black', showarrow=True, text='')
fig.add_annotation(x=2.6, y=2.5, text='analysis_complete?<br>False', showarrow=False, font=dict(size=10), bgcolor='white', bordercolor='black', borderwidth=1)

# generate_test_plan → finalize_output (True path)
fig.add_trace(go.Scatter(
    x=[1.8, 1.8], y=[2.8, 2.2],
    mode='lines',
    line=dict(color='black', width=3),
    showlegend=False,
    hoverinfo='skip'
))
fig.add_annotation(x=1.8, y=2.2, ax=1.8, ay=2.8, arrowhead=2, arrowsize=2, arrowwidth=3, arrowcolor='black', showarrow=True, text='')
fig.add_annotation(x=1.3, y=2.5, text='test_generation_<br>complete? True', showarrow=False, font=dict(size=10), bgcolor='white', bordercolor='black', borderwidth=1)

# generate_test_plan → END (False path)
fig.add_trace(go.Scatter(
    x=[2.2, 2.2], y=[2.8, 1.2],
    mode='lines',
    line=dict(color='black', width=3),
    showlegend=False,
    hoverinfo='skip'
))
fig.add_annotation(x=2.2, y=1.2, ax=2.2, ay=2.8, arrowhead=2, arrowsize=2, arrowwidth=3, arrowcolor='black', showarrow=True, text='')
fig.add_annotation(x=2.7, y=2, text='test_generation_<br>complete? False', showarrow=False, font=dict(size=10), bgcolor='white', bordercolor='black', borderwidth=1)

# finalize_output → END
fig.add_trace(go.Scatter(
    x=[2, 2], y=[1.8, 1.2],
    mode='lines',
    line=dict(color='black', width=3),
    showlegend=False,
    hoverinfo='skip'
))
fig.add_annotation(x=2, y=1.2, ax=2, ay=1.8, arrowhead=2, arrowsize=2, arrowwidth=3, arrowcolor='black', showarrow=True, text='')

# Update layout
fig.update_layout(
    title='LangGraph SQE Agent Architecture',
    xaxis=dict(range=[0.5, 3.5], showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(range=[0.5, 5.5], showgrid=False, showticklabels=False, zeroline=False),
    plot_bgcolor='white',
    showlegend=False
)

# Save the chart
fig.write_image('langgraph_architecture.png')
fig.write_image('langgraph_architecture.svg', format='svg')
print("Flowchart saved as langgraph_architecture.png and langgraph_architecture.svg")