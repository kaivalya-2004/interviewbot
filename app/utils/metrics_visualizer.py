# app/utils/metrics_visualizer.py
"""
Optional: Advanced visualization component for behavioral metrics
Add this as an optional enhancement to visualize metrics with charts
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any


def create_radar_chart(metrics: Dict[str, Any]) -> go.Figure:
    """Create a radar chart for overall behavioral metrics."""
    
    categories = [
        'Confidence',
        'Emotional Stability',
        'Eye Contact',
        'Posture',
        'Engagement'
    ]
    
    values = [
        metrics.get('overall_confidence_score', 0),
        metrics.get('emotion_analysis', {}).get('emotional_stability_score', 0),
        metrics.get('eye_contact', {}).get('quality_score', 0),
        metrics.get('posture_composure', {}).get('posture_score', 0),
        metrics.get('posture_composure', {}).get('engagement_level', 0)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Candidate Performance',
        line_color='rgb(99, 110, 250)',
        fillcolor='rgba(99, 110, 250, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        showlegend=False,
        title="Overall Performance Radar",
        height=400
    )
    
    return fig


def create_emotion_timeline_chart(metrics: Dict[str, Any]) -> go.Figure:
    """Create a timeline visualization of emotions throughout the interview."""
    
    emotion_timeline = metrics.get('emotion_analysis', {}).get('emotion_timeline', {})
    
    if not emotion_timeline:
        return None
    
    stages = ['Beginning', 'Middle', 'End']
    emotions = [
        emotion_timeline.get('beginning', 'Unknown'),
        emotion_timeline.get('middle', 'Unknown'),
        emotion_timeline.get('end', 'Unknown')
    ]
    
    # Color mapping for emotions
    emotion_colors = {
        'confident': '#2ecc71',
        'nervous': '#e74c3c',
        'engaged': '#3498db',
        'anxious': '#e67e22',
        'calm': '#1abc9c',
        'stressed': '#c0392b',
        'enthusiastic': '#f39c12',
        'uncertain': '#95a5a6'
    }
    
    colors = [emotion_colors.get(emotion.lower(), '#7f8c8d') for emotion in emotions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=stages,
            y=[1, 1, 1],
            text=emotions,
            textposition='auto',
            marker_color=colors,
            showlegend=False
        )
    ])
    
    fig.update_layout(
        title="Emotional Journey Throughout Interview",
        xaxis_title="Interview Stage",
        yaxis_visible=False,
        height=300
    )
    
    return fig


def create_score_comparison_chart(metrics: Dict[str, Any]) -> go.Figure:
    """Create a horizontal bar chart comparing different behavioral scores."""
    
    scores = {
        'Confidence': metrics.get('overall_confidence_score', 0),
        'Interview Readiness': metrics.get('interview_readiness_score', 0),
        'Emotional Stability': metrics.get('emotion_analysis', {}).get('emotional_stability_score', 0),
        'Eye Contact': metrics.get('eye_contact', {}).get('quality_score', 0),
        'Posture': metrics.get('posture_composure', {}).get('posture_score', 0),
        'Engagement': metrics.get('posture_composure', {}).get('engagement_level', 0),
        'Professional Presentation': metrics.get('posture_composure', {}).get('professional_presentation', 0)
    }
    
    categories = list(scores.keys())
    values = list(scores.values())
    
    # Color based on score
    colors = []
    for val in values:
        if val >= 8:
            colors.append('#2ecc71')  # Green
        elif val >= 6:
            colors.append('#f39c12')  # Orange
        else:
            colors.append('#e74c3c')  # Red
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=categories,
        x=values,
        orientation='h',
        text=values,
        texttemplate='%{text}/10',
        textposition='auto',
        marker_color=colors,
        showlegend=False
    ))
    
    fig.update_layout(
        title="Detailed Score Breakdown",
        xaxis_title="Score (out of 10)",
        xaxis=dict(range=[0, 10]),
        height=400
    )
    
    return fig


def create_concern_gauge(metrics: Dict[str, Any]) -> go.Figure:
    """Create gauge charts for anxiety and dishonesty indicators."""
    
    anxiety = metrics.get('emotion_analysis', {}).get('anxiety_level', 0)
    dishonesty = metrics.get('gaze_behavior', {}).get('dishonesty_indicators', 0)
    fidgeting = metrics.get('body_movement', {}).get('fidgeting_score', 0)
    
    fig = go.Figure()
    
    # Anxiety gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=anxiety,
        domain={'x': [0, 0.32], 'y': [0, 1]},
        title={'text': "Anxiety Level"},
        gauge={
            'axis': {'range': [0, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 3], 'color': "lightgreen"},
                {'range': [3, 7], 'color': "yellow"},
                {'range': [7, 10], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 7
            }
        }
    ))
    
    # Dishonesty gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=dishonesty,
        domain={'x': [0.34, 0.66], 'y': [0, 1]},
        title={'text': "Dishonesty Signals"},
        gauge={
            'axis': {'range': [0, 10]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 3], 'color': "lightgreen"},
                {'range': [3, 7], 'color': "yellow"},
                {'range': [7, 10], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 7
            }
        }
    ))
    
    # Fidgeting gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=fidgeting,
        domain={'x': [0.68, 1], 'y': [0, 1]},
        title={'text': "Fidgeting"},
        gauge={
            'axis': {'range': [0, 10]},
            'bar': {'color': "darkorange"},
            'steps': [
                {'range': [0, 3], 'color': "lightgreen"},
                {'range': [3, 7], 'color': "yellow"},
                {'range': [7, 10], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 7
            }
        }
    ))
    
    fig.update_layout(
        title="Concern Indicators",
        height=300
    )
    
    return fig


def display_metrics_dashboard(analysis_result: Dict[str, Any]):
    """
    Main function to display the complete metrics dashboard with visualizations.
    Call this function from main.py after analysis is complete.
    """
    
    if not analysis_result or not analysis_result.get('metrics'):
        st.warning("No metrics available for visualization")
        return
    
    metrics = analysis_result['metrics']
    
    st.header("📊 Interactive Metrics Dashboard")
    
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Overall Confidence",
            f"{metrics.get('overall_confidence_score', 0)}/10",
            delta=None
        )
    with col2:
        st.metric(
            "Interview Readiness",
            f"{metrics.get('interview_readiness_score', 0)}/10",
            delta=None
        )
    with col3:
        anxiety = metrics.get('emotion_analysis', {}).get('anxiety_level', 0)
        st.metric(
            "Anxiety Level",
            f"{anxiety}/10",
            delta=None,
            delta_color="inverse"
        )
    with col4:
        enthusiasm = metrics.get('emotion_analysis', {}).get('enthusiasm_level', 0)
        st.metric(
            "Enthusiasm",
            f"{enthusiasm}/10",
            delta=None
        )
    
    st.markdown("---")
    
    # Visualizations
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Radar chart
        radar_fig = create_radar_chart(metrics)
        st.plotly_chart(radar_fig, use_container_width=True)
        
        # Emotion timeline
        emotion_fig = create_emotion_timeline_chart(metrics)
        if emotion_fig:
            st.plotly_chart(emotion_fig, use_container_width=True)
    
    with col_right:
        # Score comparison
        score_fig = create_score_comparison_chart(metrics)
        st.plotly_chart(score_fig, use_container_width=True)
    
    # Concern indicators
    st.markdown("---")
    concern_fig = create_concern_gauge(metrics)
    st.plotly_chart(concern_fig, use_container_width=True)
    
    # Additional insights
    st.markdown("---")
    col_insight1, col_insight2 = st.columns(2)
    
    with col_insight1:
        st.subheader("🎯 Key Strengths")
        strengths = metrics.get('key_strengths', [])
        for i, strength in enumerate(strengths, 1):
            st.success(f"{i}. {strength}")
    
    with col_insight2:
        st.subheader("📈 Areas for Improvement")
        improvements = metrics.get('areas_for_improvement', [])
        for i, area in enumerate(improvements, 1):
            st.info(f"{i}. {area}")


# ============================================================================
# HOW TO USE IN MAIN.PY:
# ============================================================================
# Add this import at the top of main.py:
# from app.utils.metrics_visualizer import display_metrics_dashboard
#
# Then in the analysis section, after showing the tabbed metrics, add:
#
# st.markdown("---")
# st.markdown("### 📈 Visual Analytics")
# display_metrics_dashboard(analysis_result)
# ============================================================================