"""
Simple Analytics Setup for Streamlit Dashboard
This script helps you set up email notifications for your dashboard analytics
"""

import streamlit as st
import json
import datetime
import requests
import hashlib

# EASY SETUP: Use Google Analytics (Recommended)
GOOGLE_ANALYTICS_CODE = """
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-YOUR_ID_HERE"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-YOUR_ID_HERE');

  // Track custom events for Streamlit
  gtag('event', 'page_view', {
    'page_title': document.title,
    'page_location': window.location.href,
    'page_path': window.location.pathname
  });
</script>
"""

def add_google_analytics_to_dashboard():
    """
    Add this to your main dashboard file after st.set_page_config()

    Steps:
    1. Go to https://analytics.google.com
    2. Create a new property for your Streamlit app
    3. Get your Measurement ID (starts with G-)
    4. Replace G-YOUR_ID_HERE with your actual ID
    5. Add this code to your dashboard
    """

    # Add this to your dashboard
    st.components.v1.html(GOOGLE_ANALYTICS_CODE, height=0)


# ALTERNATIVE: Simple visitor counter with email notifications
def simple_visitor_tracker():
    """
    Tracks visitors and sends daily summaries
    Uses free webhook service
    """

    # Initialize session tracking
    if 'visitor_tracked' not in st.session_state:
        st.session_state.visitor_tracked = False

    if not st.session_state.visitor_tracked:
        # Create visitor record
        visitor_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'session_id': hashlib.md5(str(datetime.datetime.now()).encode()).hexdigest()[:8],
            'app_url': 'https://paulnian-datascience-life-expectancy-dashboard-bgpxv6.streamlit.app/'
        }

        # Option 1: Use webhook.site for testing
        # Go to https://webhook.site to get a free webhook URL
        webhook_url = "https://webhook.site/YOUR-UNIQUE-ID"

        # Option 2: Use IFTTT (If This Then That)
        # 1. Create account at https://ifttt.com
        # 2. Create applet: Webhook → Email
        # 3. Get your webhook URL
        ifttt_url = "https://maker.ifttt.com/trigger/dashboard_visit/with/key/YOUR_KEY"

        # Send notification (uncomment one)
        try:
            # For testing with webhook.site:
            # response = requests.post(webhook_url, json=visitor_data, timeout=2)

            # For IFTTT email notifications:
            # response = requests.post(ifttt_url, json={"value1": visitor_data['session_id']}, timeout=2)

            st.session_state.visitor_tracked = True
        except:
            pass  # Fail silently to not disrupt user experience


# PRIVACY-COMPLIANT NOTICE
def add_privacy_notice():
    """
    Add a minimal, compliant privacy notice
    """
    st.markdown("""
    <style>
    .privacy-notice {
        position: fixed;
        bottom: 10px;
        left: 10px;
        background: rgba(255,255,255,0.95);
        padding: 8px 12px;
        border-radius: 5px;
        font-size: 11px;
        color: #666;
        border: 1px solid #ddd;
        z-index: 1000;
    }
    </style>
    <div class="privacy-notice">
        📊 We use analytics to improve this dashboard |
        <a href="#" onclick="this.parentElement.style.display='none'">Dismiss</a>
    </div>
    """, unsafe_allow_html=True)


# SETUP INSTRUCTIONS
def print_setup_instructions():
    print("""
    =====================================
    ANALYTICS SETUP INSTRUCTIONS
    =====================================

    OPTION 1: Google Analytics (Professional - Recommended)
    -------------------------------------------------------
    1. Go to https://analytics.google.com
    2. Create new property for your app
    3. Get Measurement ID (G-XXXXXXXXXX)
    4. Add to your dashboard:

       st.components.v1.html(GOOGLE_ANALYTICS_CODE.replace('G-YOUR_ID_HERE', 'G-XXXXXXXXXX'), height=0)

    Benefits:
    - Real-time visitor tracking
    - Geographic data
    - User behavior flows
    - Professional dashboard
    - 100% legal and standard


    OPTION 2: IFTTT Email Notifications (Simple)
    --------------------------------------------
    1. Sign up at https://ifttt.com (free)
    2. Create New Applet:
       - IF: Webhooks (receive web request)
       - Event Name: dashboard_visit
       - THEN: Email
       - Subject: Dashboard Visit {{Value1}}
       - Body: New visitor at {{OccurredAt}}
    3. Get your webhook key from IFTTT
    4. Add to your code


    OPTION 3: Webhook.site (For Testing)
    ------------------------------------
    1. Go to https://webhook.site
    2. Copy your unique URL
    3. Use for testing analytics data


    PRIVACY COMPLIANCE
    ------------------
    Always include a privacy notice:
    - Add add_privacy_notice() to your dashboard
    - Be transparent about data collection
    - Follow local privacy laws


    WHAT TO ADD TO YOUR DASHBOARD
    -----------------------------
    from simple_analytics_setup import add_google_analytics_to_dashboard, add_privacy_notice

    # In your main() function:
    add_privacy_notice()  # Shows privacy notice
    add_google_analytics_to_dashboard()  # Adds GA tracking

    =====================================
    """)

if __name__ == "__main__":
    print_setup_instructions()