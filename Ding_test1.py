import streamlit as st
import yfinance as yf
import pandas as pd
from urllib.parse import quote, unquote
import numpy as np
import json
import re
from streamlit_javascript import st_javascript
import anthropic
import openai
import os
import logging
import warnings
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from datetime import datetime, timedelta
# Set up logging
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*st.experimental_get_query_params.*")

# Initialize session state variables
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = {}
if 'sent_alerts' not in st.session_state:
    st.session_state.sent_alerts = set()

COOLDOWN_PERIOD = timedelta(minutes=5)  # 5 minutes cooldown

# Twilio setup for WhatsApp
twilio_account_sid = st.secrets.get("twilio_account_sid") or os.getenv("TWILIO_ACCOUNT_SID")
twilio_auth_token = st.secrets.get("twilio_auth_token") or os.getenv("TWILIO_AUTH_TOKEN")
twilio_whatsapp_number = st.secrets.get("twilio_whatsapp_number") or os.getenv("TWILIO_WHATSAPP_NUMBER")

if not all([twilio_account_sid, twilio_auth_token, twilio_whatsapp_number]):
    st.error("Twilio credentials not found. Please set them in your secrets or as environment variables.")
    logging.error("Twilio credentials not found")
else:
    twilio_client = Client(twilio_account_sid, twilio_auth_token)
    logging.info("Twilio client created successfully")

    
# Initialize tickers using URL parameters
def get_tickers_from_url():
    params = st.experimental_get_query_params()
    return [unquote(ticker) for ticker in params.get("tickers", ["AAPL", "GOOGL", "MSFT", "NVDA"])]

def set_tickers_in_url(tickers):
    st.experimental_set_query_params(tickers=[quote(ticker) for ticker in tickers])

# Get initial tickers
tickers = get_tickers_from_url()

def fetch_data(ticker):
    """Fetch historical stock data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='5y')  # Fetch last 5 years of data
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame on error

def calculate_metrics(hist):
    """Calculate required financial metrics from historical data."""
    if hist.empty or 'Close' not in hist.columns:
        return {key: None for key in [
            'Current Price', 'Price 5 Years Ago', 'Price 3 Years Ago',
            'Peak Price Last 5 Years', 'Drops_by (%)', '1 Year Return (%)',
            '3 Year Return (%)', '5 Year Return (%)', '1 Day Change (%)',
            '3 Day Change (%)', 'Weekly Change (%)', 'Bi-Weekly Change (%)',
            '1 Month Change (%)', '2 Month Change (%)'
        ]}

    def safe_price(df, index):
        try:
            return df['Close'].iloc[index]
        except IndexError:
            return None

    def safe_return(current, past):
        if current is not None and past is not None and past != 0:
            return ((current - past) / past) * 100
        return None

    current_price = safe_price(hist, -1)
    price_5y_ago = safe_price(hist, 0)
    price_3y_ago = safe_price(hist, -3*252) if len(hist) >= 3*252 else None
    price_1y_ago = safe_price(hist, -252) if len(hist) >= 252 else None
    peak_price = hist['High'].max() if not hist['High'].empty else None

    metrics = {
        'Current Price': round(current_price, 2) if current_price is not None else None,
        'Price 5 Years Ago': round(price_5y_ago, 2) if price_5y_ago is not None else None,
        'Price 3 Years Ago': round(price_3y_ago, 2) if price_3y_ago is not None else None,
        'Peak Price Last 5 Years': round(peak_price, 2) if peak_price is not None else None,
        'Drops_by (%)': round(((peak_price - current_price) / peak_price) * 100, 2) if all(v is not None for v in [peak_price, current_price]) and peak_price != 0 else None,
        '1 Year Return (%)': round(safe_return(current_price, price_1y_ago), 2) if price_1y_ago is not None else None,
        '3 Year Return (%)': round(safe_return(current_price, price_3y_ago), 2) if price_3y_ago is not None else None,
        '5 Year Return (%)': round(safe_return(current_price, price_5y_ago), 2) if price_5y_ago is not None else None,
    }

    for period, days in [('1 Day', 1), ('3 Day', 3), ('Weekly', 5), ('Bi-Weekly', 10), ('1 Month', 21), ('2 Month', 42)]:
        if len(hist) > days:
            change = hist['Close'].pct_change(periods=days).iloc[-1] * 100
            metrics[f'{period} Change (%)'] = round(change, 2)
        else:
            metrics[f'{period} Change (%)'] = None

    return metrics

def create_sparkline(data, width=100, height=20):
    """Create a sparkline using SVG for the entire 5-year period."""
    if data.empty or 'Close' not in data.columns:
        return ""  # Return an empty string if there's no data

    values = data['Close'].tolist()
    if not values:
        return ""  # Return an empty string if the list is empty

    if len(set(values)) == 1:  # All values are the same
        return f"""
        <svg width="{width}" height="{height}" style="stroke: #006400; stroke-width: 1; fill: none;">
            <line x1="0" y1="{height/2}" x2="{width}" y2="{height/2}" />
        </svg>
        """

    try:
        min_val, max_val = min(values), max(values)
    except ValueError:  # This shouldn't happen given the previous checks, but just in case
        return ""

    if min_val == max_val:
        range_val = 1  # Avoid division by zero
    else:
        range_val = max_val - min_val

    points = []
    for i, val in enumerate(values):
        x = i * (width / (len(values) - 1)) if len(values) > 1 else width / 2
        y = height - ((val - min_val) / range_val) * height if range_val != 0 else height / 2
        points.append(f"{x},{y}")

    return f"""
    <svg width="{width}" height="{height}" style="stroke: #006400; stroke-width: 1; fill: none;">
        <polyline points="{' '.join(points)}" />
    </svg>
    """

def style_dataframe(df):
    """Apply styling to the dataframe."""
    def color_negative_red(val):
        if isinstance(val, (int, float)):
            color = 'red' if val < 0 else 'green'
            return f'color: {color}'
        return ''
    
    def highlight_Drops_by(val):
        return 'background-color: #90EE90'
    
    # Apply the color styling
    styled = df.style.applymap(color_negative_red, subset=[col for col in df.columns if '%' in col and col != 'Drops_by (%)'])
    styled = styled.applymap(highlight_Drops_by, subset=['Drops_by (%)'])
    
    return styled


# Add custom CSS to make it look more like a Streamlit dataframe
custom_css = """
<style>
table {
    border-collapse: collapse;
    margin: 25px 0;
    font-size: 0.9em;
    font-family: "Source Sans Pro", sans-serif;
    min-width: 400px;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    background-color: white;
}
table thead tr {
    background-color: #f0f2f6;
    color: black;
    text-align: left;
    font-weight: bold;
}
table th {
    font-size: 0.85em;  /* Reduced font size for headers */
    padding: 8px 12px;  /* Slightly reduced padding */
}
table td {
    padding: 8px 12px;
    border-bottom: 1px solid #dddddd;
}
table tbody tr {
    background-color: white;
}
table tbody tr:last-of-type {
    border-bottom: 2px solid #009879;
}
a {
    color: #0000EE;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}
</style>
"""
                
    
# Set up Anthropic API (make sure to handle this securely in a production environment)

# Try to get the API key from different sources
api_key = st.secrets.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    st.error("Anthropic API key not found. Please set it in your secrets or as an environment variable.")
    logging.error("Anthropic API key not found")
else:
    logging.info("API key found and set")


# Set up Anthropic client
try:
    anthropic_client = anthropic.Anthropic(api_key=api_key)
    logging.info("Anthropic client created successfully")

except Exception as e:
    st.error(f"Error creating Anthropic client: {str(e)}")
    logging.error(f"Error creating Anthropic client: {str(e)}")


def interpret_prompt_with_claude(prompt):
    try:
        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=300,
            temperature=0,
            system="You are a financial alert interpreter. Extract the stock ticker, alert type (increase or decrease), and percentage from the user's prompt. Respond in JSON format with keys: ticker, alert_type, percentage.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        logging.info(f"Claude response: {response.content[0].text}")
        return json.loads(response.content[0].text)
    except Exception as e:
        st.error(f"Error interpreting prompt: {str(e)}")
        logging.error(f"Error interpreting prompt: {str(e)}")
        return None
    
def create_alert(prompt):
    interpretation = interpret_prompt_with_claude(prompt)
    if interpretation:
        alert_type = "decrease" if interpretation['alert_type'] == "decrease" or "Drops_by" in prompt.lower() else "increase"
        try:
            # Remove '%' sign if present and convert to float
            percentage = float(interpretation['percentage'].rstrip('%'))
            return {
                "ticker": interpretation['ticker'],
                "percentage": percentage,
                "type": alert_type,
                "prompt": prompt,
                "active": True  # New alerts are active by default
            }
        except ValueError as e:
            st.error(f"Error creating alert: Invalid percentage value. {str(e)}")
            return None
    return None

# Function to check if alert is triggered
def check_alert(alert, df):
    row = df[df['Ticker'] == alert['ticker']]
    if not row.empty:
        if alert['type'] == "decrease":
            current_Drops_by = float(row['Drops_by (%)'].values[0].strip('%'))
            return current_Drops_by >= alert['percentage'], current_Drops_by
        elif alert['type'] == "increase":
            # Implement increase logic based on your data
            return False, 0
    return False, 0

def send_whatsapp_alert(message, to_number):
    try:
        message = twilio_client.messages.create(
            body=message,
            from_=f"whatsapp:{twilio_whatsapp_number}",
            to=f"whatsapp:{to_number}"
        )
        return True, message.sid
    except TwilioRestException as e:
        logging.error(f"Failed to send WhatsApp message: {str(e)}")
        return False, str(e)
    
def reset_alerts():
    """Reset all alerts and related session state variables."""
    st.session_state.alerts = []
    st.session_state.last_alert_time = {}
    st.session_state.sent_alerts = set()
    st.success("All alerts have been reset.")
    
# Add this new section for displaying and managing alerts
def display_alert_management():
    st.subheader("Manage Alerts")
    
    # Add a reset button
    if st.button("Reset All Alerts"):
        reset_alerts()
        st.experimental_rerun()
    
    # Create a DataFrame for the alerts
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    
    alert_df = pd.DataFrame(st.session_state.alerts)
    
    if alert_df.empty:
        # If there are no alerts, create an empty DataFrame with the correct columns
        alert_df = pd.DataFrame(columns=['Alert', 'Active'])
    else:
        # Determine which columns to display based on what's available
        display_columns = []
        if 'prompt' in alert_df.columns:
            display_columns.append('prompt')
        elif 'ticker' in alert_df.columns and 'percentage' in alert_df.columns and 'type' in alert_df.columns:
            # If 'prompt' is not available, create a description from other fields
            alert_df['description'] = alert_df.apply(
                lambda row: f"{row['ticker']} {row['type']}s by {row['percentage']}%",
                axis=1
            )
            display_columns.append('description')
        else:
            # Fallback if neither 'prompt' nor the combination of 'ticker', 'percentage', and 'type' are available
            st.warning("Alert structure is not recognized. Displaying all available information.")
            display_columns = list(alert_df.columns)

        # Always include 'active' column if it exists
        if 'active' in alert_df.columns:
            display_columns.append('active')
        else:
            # If 'active' doesn't exist, add it with default value True
            alert_df['active'] = True
            display_columns.append('active')

        # Select only the columns we want to display
        alert_df = alert_df[display_columns]

        # Rename columns for display
        column_names = {
            'prompt': 'Alert',
            'description': 'Alert',
            'active': 'Active'
        }
        alert_df.rename(columns=column_names, inplace=True)

    # Use st.data_editor to create an editable table
    edited_df = st.data_editor(alert_df, hide_index=True, num_rows="dynamic")

    # Update the session state alerts based on edited values
    new_alerts = []
    for i, row in edited_df.iterrows():
        if i < len(st.session_state.alerts):
            alert = st.session_state.alerts[i].copy()
            alert['active'] = row['Active']
            new_alerts.append(alert)
        elif not pd.isna(row['Alert']):  # This is a new alert added by the user
            new_alerts.append({
                'prompt': row['Alert'],
                'active': row['Active'],
                'ticker': '',  # You might want to parse the Alert text to extract these
                'percentage': 0,
                'type': 'decrease'
            })
    
    st.session_state.alerts = new_alerts

# Modify the alert checking and sending logic
def check_and_send_alerts(df, recipient_number):
    triggered_alerts = []
    new_alerts_to_send = []
    current_time = datetime.now()

    for alert in st.session_state.alerts:
        if alert['active']:  # Only process active alerts
            is_triggered, current_value = check_alert(alert, df)
            if is_triggered:
                alert_message = f"Alert triggered: {alert['prompt']} (Current value: {current_value:.2f}%)"
                alert_key = f"{alert['ticker']}_{alert['type']}_{alert['percentage']}"
                last_alert_time = st.session_state.last_alert_time.get(alert_key, datetime.min)
                
                if current_time - last_alert_time > COOLDOWN_PERIOD:
                    if alert_message not in st.session_state.sent_alerts:
                        new_alerts_to_send.append(alert_message)
                        st.session_state.sent_alerts.add(alert_message)
                        st.session_state.last_alert_time[alert_key] = current_time
                
                triggered_alerts.append(alert_message)

    # Send consolidated alert if there are new alerts
    if new_alerts_to_send:
        consolidated_message = "New Alerts:\n" + "\n".join(new_alerts_to_send)
        whatsapp_sent, whatsapp_result = send_whatsapp_alert(consolidated_message, recipient_number)
        if whatsapp_sent:
            st.markdown(f'<div style="background-color: #90EE90; padding: 10px; border-radius: 5px;">Alerts sent to WhatsApp number {recipient_number}</div>', unsafe_allow_html=True)
        else:
            st.error(f"Failed to send WhatsApp message: {whatsapp_result}")

    return triggered_alerts

            
# Function to validate and format the phone number
def validate_phone_number(number):
    # Remove any non-digit characters except for the leading '+'
    cleaned_number = re.sub(r'[^\d+]', '', number)
    
    # Ensure the number starts with '+'
    if not cleaned_number.startswith('+'):
        cleaned_number = '+' + cleaned_number
    
    # Ensure there's at least a country code and some digits
    if len(cleaned_number) < 5:
        return None
    
    return cleaned_number
    
# Main Streamlit app
st.title('Ding!')
st.markdown("Portfolio View")
st.sidebar.header("Alert Settings")
user_phone_number = st.sidebar.text_input("Recipient WhatsApp number (with country code)", value="+1")
default_recipient_number = "+16308536614"

# Validate and set the recipient number
recipient_number = validate_phone_number(user_phone_number) if user_phone_number != "+1" else default_recipient_number

if recipient_number:
    st.sidebar.success(f"Alerts will be sent to: {recipient_number}")
else:
    st.sidebar.error("Invalid phone number. Using default number.")
    recipient_number = default_recipient_number



# Input for adding new tickers
new_ticker = st.text_input("Add a new ticker").upper()
if st.button("Add Ticker"):
    if new_ticker and new_ticker not in tickers:
        tickers.append(new_ticker)
        set_tickers_in_url(tickers)
        st.experimental_rerun()

# Fetch data and calculate metrics for all tickers
all_metrics = []
for ticker in tickers:
    data = fetch_data(ticker)
    metrics = calculate_metrics(data)
    metrics['Ticker'] = ticker
    metrics['Sparkline'] = create_sparkline(data)
    all_metrics.append(metrics)

# Create a DataFrame from all metrics
df = pd.DataFrame(all_metrics).reset_index(drop=True)

# Reorder columns to have Ticker and Sparkline first
columns = ['Ticker', 'Sparkline'] + [col for col in df.columns if col not in ['Ticker', 'Sparkline']]
df = df[columns]

# Apply formatting
df['Current Price'] = df['Current Price'].map('${:,.2f}'.format)
df['Price 5 Years Ago'] = df['Price 5 Years Ago'].map('${:,.2f}'.format)
df['Price 3 Years Ago'] = df['Price 3 Years Ago'].map('${:,.2f}'.format)
df['Peak Price Last 5 Years'] = df['Peak Price Last 5 Years'].map('${:,.2f}'.format)

percentage_columns = [col for col in df.columns if '%' in col]
for col in percentage_columns:
    df[col] = df[col].map('{:,.2f}%'.format)

# Function to create hyperlink
def create_hyperlink(ticker):
    return f'<a href="https://seekingalpha.com/symbol/{ticker}" target="_blank">{ticker}</a>'

# Apply styling and create hyperlinks
styled_df = style_dataframe(df)
styled_df.format({'Ticker': create_hyperlink})

# Convert to HTML and preserve the HTML content in the Sparkline column
html = styled_df.to_html(escape=False, index=False)

# Display the custom HTML table
st.markdown(custom_css + styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)

# Reset button to clear added tickers
if st.button('Reset Tickers'):
    set_tickers_in_url(["AAPL", "GOOGL", "MSFT", "NVDA"])
    st.experimental_rerun()


# Alert Creation 
st.subheader("Create Alert")
user_prompt = st.text_input("Enter your alert condition:")
if st.button("Create Alert"):
    with st.spinner("Interpreting your alert..."):
        alert = create_alert(user_prompt)
    if alert:
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        st.session_state.alerts.append(alert)
        st.success(f"Alert created successfully for {alert['ticker']}!")
        st.json(alert)  # Display the created alert details
        
        # Display relevant data from the table
        relevant_data = df[df['Ticker'] == alert['ticker']]
        if not relevant_data.empty:
            st.write("Current data for this stock:")
            st.dataframe(relevant_data)
    else:
        st.error("Could not create alert. Please check your input and try again.")


# Display and manage alerts
display_alert_management()

# Check for triggered alerts
triggered_alerts = check_and_send_alerts(df, recipient_number)
        
# # Display current alerts
# if 'alerts' in st.session_state and st.session_state.alerts:
#     st.subheader("Current Alerts")
#     for i, alert in enumerate(st.session_state.alerts):
#         st.write(f"{i+1}. {alert['prompt']}")

# Display triggered alerts
if triggered_alerts:
    st.warning("Alerts Triggered:")
    for alert in triggered_alerts:
        st.write(alert)
        
# Check for triggered alerts and send SMS

triggered_alerts = []
if 'alerts' in st.session_state:
    for alert in st.session_state.alerts:
        is_triggered, current_value = check_alert(alert, df)
        if is_triggered:
            alert_message = f"Ding! {alert['prompt']} (Current value: {current_value:.2f}%)"
            triggered_alerts.append(alert_message)
            
            # Send WhatsApp message
            whatsapp_sent, whatsapp_result = send_whatsapp_alert(alert_message, recipient_number)
            if whatsapp_sent:
                st.markdown(f'<div style="background-color: #90EE90; padding: 10px; border-radius: 5px;">Alert sent to WhatsApp number {recipient_number}</div>', unsafe_allow_html=True)
            else:
                st.error(f"Failed to send WhatsApp message: {whatsapp_result}")




# JavaScript for custom alert
st.markdown("""
<script>
function createCustomAlert(message) {
    var alertDiv = document.createElement('div');
    alertDiv.style.position = 'fixed';
    alertDiv.style.top = '20px';
    alertDiv.style.left = '50%';
    alertDiv.style.transform = 'translateX(-50%)';
    alertDiv.style.backgroundColor = '#f44336';
    alertDiv.style.color = 'white';
    alertDiv.style.padding = '20px';
    alertDiv.style.borderRadius = '5px';
    alertDiv.style.zIndex = '9999';
    alertDiv.innerHTML = message;
    document.body.appendChild(alertDiv);
    setTimeout(function() {
        alertDiv.style.display = 'none';
    }, 5000);
}
</script>
""", unsafe_allow_html=True)

            
# Trigger custom alert for triggered alerts
if triggered_alerts:
    alert_message = "Alerts triggered:\\n" + "\\n".join(triggered_alerts)
    st_javascript(f"createCustomAlert('{alert_message}')")
    