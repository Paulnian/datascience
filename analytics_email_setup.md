# Analytics Email Setup Guide

## Option 1: Using Webhook.site + Zapier (Recommended - Free)

### Step 1: Set up Webhook.site
1. Go to https://webhook.site
2. Copy your unique webhook URL
3. This will be your temporary endpoint for testing

### Step 2: Set up Zapier Email Integration
1. Create free account at https://zapier.com
2. Create a new Zap:
   - Trigger: Webhooks by Zapier → Catch Hook
   - Action: Email by Zapier → Send Outbound Email
3. Configure email:
   - To: najarianpaul@gmail.com
   - Subject: "Dashboard Analytics Report - {{session_id}}"
   - Body: Format the JSON data nicely

### Step 3: Update the Code
Replace `YOUR_WEBHOOK_URL_HERE` in analytics_tracker.py with your Zapier webhook URL

## Option 2: Using EmailJS (Direct from Browser)

### Step 1: Set up EmailJS
1. Go to https://www.emailjs.com and create free account
2. Add Gmail as email service
3. Create email template with variables:
   - Subject: "Dashboard Analytics - {{session_id}}"
   - Body: Include {{session_duration}}, {{page_views}}, etc.

### Step 2: Get Your Credentials
- Service ID
- Template ID
- Public Key

### Step 3: Add to Your Dashboard
```python
import streamlit.components.v1 as components

def send_via_emailjs(data):
    emailjs_html = f"""
    <script src="https://cdn.jsdelivr.net/npm/@emailjs/browser@3/dist/email.min.js"></script>
    <script>
        emailjs.init("YOUR_PUBLIC_KEY");
        emailjs.send("YOUR_SERVICE_ID", "YOUR_TEMPLATE_ID", {{
            session_id: "{data['session_id']}",
            duration: "{data['session_duration_seconds']}",
            page_views: "{json.dumps(data['page_views'])}"
        }});
    </script>
    """
    components.html(emailjs_html, height=0)
```

## Option 3: Using Google Analytics (Most Professional)

### Step 1: Create Google Analytics Account
1. Go to https://analytics.google.com
2. Create new property for your app
3. Get Measurement ID (G-XXXXXXXXXX)

### Step 2: Add to Dashboard
```python
# In your main dashboard file
st.markdown(f"""
    <!-- Google Analytics -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){{dataLayer.push(arguments);}}
        gtag('js', new Date());
        gtag('config', 'G-XXXXXXXXXX');
    </script>
""", unsafe_allow_html=True)
```

### Benefits:
- Professional analytics dashboard
- Real-time visitor tracking
- Geographic data
- User flow visualization
- Completely legal and standard practice

## Privacy Compliance

### Minimal Notice (Current Implementation)
The code includes a small notice at bottom-right of the page.

### For Full Compliance, Consider:
1. Cookie consent banner
2. Privacy policy page
3. Opt-out mechanism
4. Data retention policy

## Testing the Analytics

1. Open your dashboard
2. Navigate through different tabs
3. Change some filters
4. Wait 5 minutes or interact 20+ times
5. Check your email for the report

## Important Notes

- **Never collect personally identifiable information (PII)**
- **Always inform users about tracking**
- **Comply with local privacy laws**
- **Use anonymous session IDs only**
- **Limit data retention period**