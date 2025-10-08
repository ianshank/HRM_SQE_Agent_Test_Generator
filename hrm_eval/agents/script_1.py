# Create a comparison table showing key differences between CrewAI and LangGraph implementations
import pandas as pd

comparison_data = {
    'Aspect': [
        'Architecture', 
        'State Management',
        'Tool Integration',
        'Agent Definition',
        'Workflow Control',
        'Routing Logic',
        'Error Handling',
        'Memory/Persistence',
        'Debugging/Visualization',
        'Scalability'
    ],
    'CrewAI Approach': [
        'Agent-based with roles and tasks',
        'Implicit state through agent interactions',
        'Tools assigned to agents directly',
        'Agent class with role, goal, backstory',
        'Sequential or hierarchical task execution',
        'Based on agent capabilities and assignments',
        'Agent-level error handling',
        'Limited built-in persistence',
        'Agent interaction logs',
        'Horizontal scaling through more agents'
    ],
    'LangGraph Approach': [
        'Graph-based nodes and edges',
        'Explicit TypedDict state container',
        'Tools as nodes or called within nodes',
        'Node functions with typed state input/output',
        'Conditional edges and dynamic routing',
        'Explicit routing functions with conditions',
        'Graph-level checkpointing and recovery',
        'Built-in state persistence and checkpointing',
        'Visual graph representation and Studio IDE',
        'Vertical scaling through complex workflows'
    ]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('crewai_vs_langgraph_comparison.csv', index=False)

print("CrewAI vs LangGraph Comparison:")
print("=" * 60)
for i, row in comparison_df.iterrows():
    print(f"\n{row['Aspect']}:")
    print(f"  CrewAI: {row['CrewAI Approach']}")
    print(f"  LangGraph: {row['LangGraph Approach']}")

print(f"\nComparison saved to: crewai_vs_langgraph_comparison.csv")