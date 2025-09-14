import streamlit as st
import json
import datetime
import hashlib
import requests
from typing import Dict, Any
import time
import platform
import os

class AnalyticsTracker:
    """
    Anonymous analytics tracker for Streamlit dashboard
    Tracks user interactions while respecting privacy
    """

    def __init__(self):
        """Initialize the analytics tracker"""
        if 'analytics_session_id' not in st.session_state:
            # Create anonymous session ID
            st.session_state.analytics_session_id = self._generate_session_id()
            st.session_state.session_start_time = time.time()
            st.session_state.page_views = {}
            st.session_state.interactions = []
            st.session_state.analytics_sent = False

    def _generate_session_id(self) -> str:
        """Generate anonymous session ID"""
        # Create hash from timestamp and random value
        timestamp = str(datetime.datetime.now())
        session_string = f"{timestamp}_{id(st)}"
        return hashlib.md5(session_string.encode()).hexdigest()[:12]

    def track_page_view(self, page_name: str):
        """Track when a user views a specific tab/page"""
        if page_name not in st.session_state.page_views:
            st.session_state.page_views[page_name] = {
                'first_visit': datetime.datetime.now().isoformat(),
                'visit_count': 0,
                'total_time': 0
            }

        st.session_state.page_views[page_name]['visit_count'] += 1
        st.session_state.page_views[page_name]['last_visit'] = datetime.datetime.now().isoformat()

        # Log interaction
        self._log_interaction({
            'type': 'page_view',
            'page': page_name,
            'timestamp': datetime.datetime.now().isoformat()
        })

    def track_filter_change(self, filter_name: str, value: Any):
        """Track when users change filters"""
        self._log_interaction({
            'type': 'filter_change',
            'filter': filter_name,
            'value': str(value)[:50],  # Limit value length
            'timestamp': datetime.datetime.now().isoformat()
        })

    def track_button_click(self, button_name: str):
        """Track button interactions"""
        self._log_interaction({
            'type': 'button_click',
            'button': button_name,
            'timestamp': datetime.datetime.now().isoformat()
        })

    def track_chart_interaction(self, chart_type: str, action: str):
        """Track chart interactions"""
        self._log_interaction({
            'type': 'chart_interaction',
            'chart': chart_type,
            'action': action,
            'timestamp': datetime.datetime.now().isoformat()
        })

    def _log_interaction(self, interaction: Dict):
        """Log an interaction to session state"""
        if len(st.session_state.interactions) < 100:  # Limit stored interactions
            st.session_state.interactions.append(interaction)

    def get_session_duration(self) -> float:
        """Get current session duration in seconds"""
        return time.time() - st.session_state.session_start_time

    def prepare_analytics_report(self) -> Dict:
        """Prepare analytics data for sending"""
        return {
            'session_id': st.session_state.analytics_session_id,
            'session_start': datetime.datetime.fromtimestamp(st.session_state.session_start_time).isoformat(),
            'session_duration_seconds': self.get_session_duration(),
            'page_views': st.session_state.page_views,
            'total_interactions': len(st.session_state.interactions),
            'interaction_summary': self._summarize_interactions(),
            'metadata': {
                'timestamp': datetime.datetime.now().isoformat(),
                'platform': platform.system(),
                'streamlit_version': st.__version__,
                'app_url': 'https://paulnian-datascience-life-expectancy-dashboard-bgpxv6.streamlit.app/'
            }
        }

    def _summarize_interactions(self) -> Dict:
        """Create summary of interactions"""
        summary = {
            'page_views': 0,
            'filter_changes': 0,
            'button_clicks': 0,
            'chart_interactions': 0
        }

        for interaction in st.session_state.interactions:
            interaction_type = interaction.get('type', '')
            if interaction_type == 'page_view':
                summary['page_views'] += 1
            elif interaction_type == 'filter_change':
                summary['filter_changes'] += 1
            elif interaction_type == 'button_click':
                summary['button_clicks'] += 1
            elif interaction_type == 'chart_interaction':
                summary['chart_interactions'] += 1

        return summary

    def send_analytics_email(self, report: Dict) -> bool:
        """
        Send analytics report via email using a webhook service
        Currently using Google Analytics for tracking
        """
        try:
            # Google Analytics is handling the tracking
            # This method can be used for additional custom tracking if needed

            # For development/debugging, log the report
            print("Analytics Report:", json.dumps(report, indent=2))
            return True

        except Exception as e:
            print(f"Failed to send analytics: {e}")
            return False

    def should_send_report(self) -> bool:
        """Check if we should send a report"""
        # Send report when session ends or after significant activity
        duration = self.get_session_duration()
        interactions = len(st.session_state.interactions)

        # Send if: session > 5 minutes, or > 20 interactions, and not already sent
        return (
            not st.session_state.analytics_sent and
            (duration > 300 or interactions > 20)
        )

    def display_privacy_notice(self):
        """Display minimal privacy notice"""
        # This creates a small, unobtrusive notice
        st.markdown(
            """
            <div style='position: fixed; bottom: 10px; right: 10px;
                        background: rgba(255,255,255,0.9); padding: 5px 10px;
                        border-radius: 5px; font-size: 11px; color: #666;
                        z-index: 1000; max-width: 200px;'>
                📊 Anonymous analytics enabled for improvement purposes
            </div>
            """,
            unsafe_allow_html=True
        )


# Simpler alternative using Google Analytics
def add_google_analytics():
    """
    Add Google Analytics tracking code
    This is the standard, legal way to track usage
    """
    ga_code = """
    <!-- Google Analytics -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'GA_MEASUREMENT_ID');

        // Track Streamlit specific events
        document.addEventListener('DOMContentLoaded', function() {
            // Track page views
            gtag('event', 'page_view', {
                'page_title': document.title,
                'page_location': window.location.href
            });
        });
    </script>
    """

    # You need to replace GA_MEASUREMENT_ID with your actual Google Analytics ID
    # Get one free at: https://analytics.google.com/

    st.markdown(ga_code, unsafe_allow_html=True)


# Usage example for the main dashboard
def integrate_analytics(page_name: str = None):
    """
    Easy integration function for your dashboard
    Call this at the beginning of your main() function
    """
    tracker = AnalyticsTracker()

    # Track page view if provided
    if page_name:
        tracker.track_page_view(page_name)

    # Display privacy notice (required for legal compliance)
    tracker.display_privacy_notice()

    # Check if should send report
    if tracker.should_send_report():
        report = tracker.prepare_analytics_report()
        tracker.send_analytics_email(report)
        st.session_state.analytics_sent = True

    return tracker