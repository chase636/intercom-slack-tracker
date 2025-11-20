import sys
import os
import requests
import time
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import seaborn as sns 
import numpy as np 
import json

# ==========================================
# 🎨 CHART STYLING FOR DARK MODE
# ==========================================
# Use a dark, modern theme for Matplotlib
plt.style.use('dark_background') 
sns.set_theme(style="darkgrid", palette="viridis")

# Define custom colors
SLACK_GREEN = "#2eb67d" # Current Period
SLACK_GRAY = "#808080"  # Previous Period
TEXT_COLOR = "#e0e0e0"  # Lighter text for dark background
TITLE_COLOR = "#ffffff" # White for titles

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================

def get_config():
    """Retrieves configuration securely from environment variables."""
    config = {
        "INTERCOM_TOKEN": os.environ.get("INTERCOM_TOKEN"),
        "SLACK_BOT_TOKEN": os.environ.get("SLACK_BOT_TOKEN"),
        "SLACK_CHANNEL_ID": os.environ.get("SLACK_CHANNEL_ID")
    }
    if not all(config.values()):
        raise EnvironmentError("Missing required environment variables (tokens or channel ID). Please check GCP settings.")
    return config

def get_time_ranges(mode):
    """Returns (current_start_ts, current_end_ts, previous_start_ts, label)"""
    now = datetime.now()
    
    if mode == 'daily':
        today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday_midnight = today_midnight - timedelta(days=1)
        
        return (
            int(today_midnight.timestamp()), int(now.timestamp()), 
            int(yesterday_midnight.timestamp()), "Yesterday"
        )
        
    elif mode == 'weekly':
        seven_days_ago = now - timedelta(days=7)
        fourteen_days_ago = now - timedelta(days=14)
        return (
            int(seven_days_ago.timestamp()), int(now.timestamp()),
            int(fourteen_days_ago.timestamp()), "Previous 7 Days"
        )

    elif mode == 'monthly':
        thirty_days_ago = now - timedelta(days=30)
        sixty_days_ago = now - timedelta(days=60)
        return (
            int(thirty_days_ago.timestamp()), int(now.timestamp()),
            int(sixty_days_ago.timestamp()), "Previous 30 Days"
        )
    
    raise ValueError("Invalid mode. Use: daily, weekly, or monthly")

def fetch_intercom_data(start_ts, intercom_token):
    """Fetches ALL closed conversations since start_ts"""
    print(f"   ⏳ Fetching data since {datetime.fromtimestamp(start_ts)}...")
    
    headers = {
        "Authorization": f"Bearer {intercom_token}",
        "Accept": "application/json",
        "Intercom-Version": "2.11"
    }
    
    search_url = "https://api.intercom.io/conversations/search"
    query = {
        "query": {
            "operator": "AND",
            "value": [
                {"field": "updated_at", "operator": ">", "value": start_ts},
                {"field": "state", "operator": "=", "value": "closed"}
            ]
        },
        "pagination": {"per_page": 150}
    }
    
    conversations = []
    has_more = True
    
    while has_more:
        try:
            resp = requests.post(search_url, headers=headers, json=query)
            resp.raise_for_status()
            data = resp.json()
            conversations.extend(data.get('conversations', []))
            
            if 'pages' in data and 'next' in data['pages']:
                query['pagination']['starting_after'] = data['pages']['next']['starting_after']
            else:
                has_more = False
        except Exception as e:
            print(f"   ❌ API Error: {e}")
            raise e

    return conversations

def generate_chart(df, mode, comparison_label):
    """Generates a combined chart for volume and a line chart for CSAT."""
    
    has_volume_data = not df.empty and (df['current'].sum() > 0 or df['previous'].sum() > 0)
    has_csat_data = not df.empty and df['csat_count'].sum() > 0

    # Create a figure with two subplots: one for volume, one for CSAT
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False, 
                             gridspec_kw={'height_ratios': [2, 1]})

    if not has_volume_data and not has_csat_data:
        # --- Handle the "No Data" Case ---
        axes[0].axis('off')
        axes[1].axis('off')
        
        axes[0].text(0.5, 0.5, 'NO ACTIVITY DURING THIS PERIOD', 
                 horizontalalignment='center', verticalalignment='center', 
                 transform=axes[0].transAxes, fontsize=20, color=SLACK_GRAY)
        axes[0].set_title(f"{mode.capitalize()} Report", color=TITLE_COLOR)
        
    else:
        # --- Volume Chart (Top Subplot) ---
        ax1 = axes[0]
        if has_volume_data:
            df_volume = df.copy()
            df_volume['total'] = df_volume['current'] + df_volume['previous']
            df_volume = df_volume.sort_values('total', ascending=True) 
            
            y_pos = np.arange(len(df_volume))
            
            ax1.barh(y_pos + 0.2, df_volume['current'], height=0.4, label='Current Period', color=SLACK_GREEN)
            ax1.barh(y_pos - 0.2, df_volume['previous'], height=0.4, label=comparison_label, color=SLACK_GRAY)
            
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(df_volume.index, color=TEXT_COLOR)
            ax1.set_xlabel('Cases Closed', color=TEXT_COLOR)
            ax1.tick_params(axis='x', colors=TEXT_COLOR)
            ax1.tick_params(axis='y', colors=TEXT_COLOR)
            ax1.legend(facecolor='#2c2c2c', edgecolor=SLACK_GRAY, labelcolor=TEXT_COLOR)
            ax1.set_title(f'Agent Volume: {mode.capitalize()} Comparison', color=TITLE_COLOR)
        else:
            ax1.text(0.5, 0.5, 'NO VOLUME DATA', horizontalalignment='center', 
                     verticalalignment='center', transform=ax1.transAxes, 
                     fontsize=16, color=SLACK_GRAY)
            ax1.set_title(f'Agent Volume: {mode.capitalize()} Comparison', color=TITLE_COLOR)
            ax1.axis('off')


        # --- CSAT Chart (Bottom Subplot) ---
        ax2 = axes[1]
        if has_csat_data:
            df_csat = df[df['csat_count'] > 0].copy()
            if not df_csat.empty:
                df_csat['avg_csat'] = df_csat['csat_sum'] / df_csat['csat_count']
                df_csat = df_csat.sort_values('avg_csat', ascending=False)
                
                ax2.plot(df_csat.index, df_csat['avg_csat'], marker='o', linestyle='-', color=SLACK_GREEN)
                
                ax2.set_ylabel('Avg CSAT (1-5)', color=TEXT_COLOR)
                ax2.set_xlabel('Agent', color=TEXT_COLOR)
                ax2.set_ylim(1, 5)
                ax2.tick_params(axis='x', colors=TEXT_COLOR, rotation=45, ha='right')
                ax2.tick_params(axis='y', colors=TEXT_COLOR)
                ax2.set_title('Agent CSAT (Current Period)', color=TITLE_COLOR)
                ax2.grid(True, linestyle='--', alpha=0.6, color=SLACK_GRAY)
            else:
                ax2.text(0.5, 0.5, 'NO CSAT RATINGS', horizontalalignment='center', 
                         verticalalignment='center', transform=ax2.transAxes, 
                         fontsize=16, color=SLACK_GRAY)
                ax2.set_title('Agent CSAT (Current Period)', color=TITLE_COLOR)
                ax2.axis('off')
        else:
            ax2.text(0.5, 0.5, 'NO CSAT RATINGS', horizontalalignment='center', 
                     verticalalignment='center', transform=ax2.transAxes, 
                     fontsize=16, color=SLACK_GRAY)
            ax2.set_title('Agent CSAT (Current Period)', color=TITLE_COLOR)
            ax2.axis('off')

    plt.tight_layout()
    filename = "report_chart.png"
    # Save the file. facecolor ensures the surrounding area is also dark.
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    return filename

def send_slack(msg, file_path=None, is_error=False, slack_token=None, channel_id=None):
    client = WebClient(token=slack_token)
    try:
        if is_error:
            client.chat_postMessage(channel=channel_id, text=f"🚨 *System Error:* {msg}")
        elif file_path:
            client.files_upload_v2(
                channel=channel_id,
                file=file_path,
                title="Performance Chart",
                initial_comment=msg
            )
        else:
            client.chat_postMessage(channel=channel_id, text=msg)
        print("   ✅ Sent to Slack.")
    except SlackApiError as e:
        print(f"   ❌ Slack Error: {e.response['error']}")
        # Send a final error message if the failure was during image upload
        if not is_error:
             client.chat_postMessage(channel=channel_id, text=f"🚨 *Critical Error: Slack API Failure*\nDetails: {e.response['error']}")


# ==========================================
# 🚀 MAIN LOGIC (Runs the report)
# ==========================================
def main(mode):
    """The core logic for fetching and processing data."""
    config = get_config()
    intercom_token = config["INTERCOM_TOKEN"]
    slack_token = config["SLACK_BOT_TOKEN"]
    channel_id = config["SLACK_CHANNEL_ID"]

    print(f"🚀 Starting {mode.upper()} report...")

    try:
        # 1. Time Calculation
        curr_start, curr_end, prev_start, comp_label = get_time_ranges(mode)
        
        # 2. Fetch Data (Fetches last 48h/2 weeks/2 months of data)
        all_data = fetch_intercom_data(prev_start, intercom_token)
        
        # 3. Process Data
        stats = {}
        
        for conv in all_data:
            assignee = conv.get('assignee', {})
            if assignee.get('type') != 'admin': continue
            
            agent = assignee.get('name', 'Unknown')
            closed_at = conv.get('statistics', {}).get('last_close_at', 0)
            
            if agent not in stats:
                stats[agent] = {'current': 0, 'previous': 0, 'csat_sum': 0, 'csat_count': 0}
            
            # Bucket data and track CSAT
            if closed_at >= curr_start:
                stats[agent]['current'] += 1
                rating = conv.get('conversation_rating', {}).get('rating')
                if rating:
                    stats[agent]['csat_sum'] += rating
                    stats[agent]['csat_count'] += 1
            elif closed_at >= prev_start:
                stats[agent]['previous'] += 1

        # 4. Create DataFrame and clean data types
        df = pd.DataFrame.from_dict(stats, orient='index')
        df = df.reindex(columns=['current', 'previous', 'csat_sum', 'csat_count']).fillna(0)
        
        # Ensure CSAT sum/count are numeric
        df['csat_sum'] = pd.to_numeric(df['csat_sum'], errors='coerce').fillna(0)
        df['csat_count'] = pd.to_numeric(df['csat_count'], errors='coerce').fillna(0)

        # 5. Text Report Construction
        if df.empty or (df['current'].sum() == 0 and df['previous'].sum() == 0 and df['csat_count'].sum() == 0):
            report_msg = f"*📊 {mode.capitalize()} Report*\n\nNo conversations were closed or rated during this period. 💤"
        else:
            report_msg = f"*📊 Support {mode.capitalize()} Report*\n"
            report_msg += f"_(Comparing against {comp_label})_\n\n"
            
            total_closed = df['current'].sum()
            prev_closed = df['previous'].sum()
            
            if total_closed > 0:
                diff = total_closed - prev_closed
                trend = "📈" if diff >= 0 else "📉"
                report_msg += f"*Team Total:* {int(total_closed)} Closed ({trend} {diff:+d})\n\n"
            else:
                report_msg += "*Team Total:* No closed conversations in current period.\n\n"
            
            # Agent Breakdown (sort by current period's closed count)
            df = df.sort_values('current', ascending=False)
            for agent, row in df.iterrows():
                if row['current'] == 0 and row['previous'] == 0: continue 
                
                csat_txt = "N/A"
                if row['csat_count'] > 0:
                    score = round(row['csat_sum'] / row['csat_count'], 1)
                    csat_txt = f"⭐ {score}"
                
                report_msg += f"• *{agent}*: {int(row['current'])} Closed (Prev: {int(row['previous'])}) | CSAT: {csat_txt}\n"

        # 6. Chart & Send
        chart_file = generate_chart(df, mode, comp_label)
        send_slack(report_msg, chart_file, slack_token=slack_token, channel_id=channel_id)

    except Exception as e:
        # 7. Error Handling (Post to Slack)
        error_msg = f"The {mode} report script crashed.\nError details: `{str(e)}`"
        print(error_msg)
        send_slack(error_msg, is_error=True, slack_token=slack_token, channel_id=channel_id)


# ==========================================
# ☁️ CLOUD FUNCTION ENTRY POINT
# ==========================================

def intercom_report(request):
    """
    Main entry point for Google Cloud Function.
    Determines the mode from the scheduler payload.
    """
    try:
        # Attempt to parse JSON payload from Cloud Scheduler
        request_json = request.get_json(silent=True)
        
        if request_json and 'mode' in request_json:
            mode = request_json['mode']
        else:
            # Fallback for manual testing or unexpected trigger
            mode = 'daily'
    except Exception as e:
        print(f"Could not parse request JSON: {e}")
        mode = 'daily' # Default safely
    
    # Execute the core logic
    main(mode) 
    
    return 'Report run successfully!', 200

# Optional: Keep for local testing if needed, but remove for final GCP deployment simplicity
# if __name__ == "__main__":
#     # Usage: python main.py daily
#     selected_mode = sys.argv[1] if len(sys.argv) > 1 else 'daily'
#     # Set dummy environment variables for local testing
#     os.environ['INTERCOM_TOKEN'] = "dG9r:..."
#     os.environ['SLACK_BOT_TOKEN'] = "xoxb:..."
#     os.environ['SLACK_CHANNEL_ID'] = "C12345..."
#     main(selected_mode)