import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import textwrap

def connect_db():
    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "recipe_feedback.db"
    return sqlite3.connect(str(db_path))

def wrap_labels(labels, max_width=20):
    return [textwrap.fill(label, max_width) for label in labels]

def generate_performance_report():
    conn = connect_db()

    # Get the date range for the report
    date_range_query = """
    SELECT MIN(timestamp) as start_date, MAX(timestamp) as end_date
    FROM (
        SELECT timestamp FROM user_feedback
        UNION ALL
        SELECT timestamp FROM bad_image_feedback
    )
    """
    date_range_df = pd.read_sql_query(date_range_query, conn)
    start_date = pd.to_datetime(date_range_df['start_date'][0]).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(date_range_df['end_date'][0]).strftime('%Y-%m-%d')

    # Set up the plot
    plt.figure(figsize=(12, 10))
    plt.suptitle(f"Recipe Recommendation App Performance Report\n{start_date} to {end_date}", fontsize=16)

    # Use a blue color palette
    blue_palette = sns.color_palette("Blues", n_colors=4)

    # 1. Number of likes, dislikes, and bad image reports
    feedback_query = """
    SELECT 
        SUM(CASE WHEN feedback = 'Liked' THEN 1 ELSE 0 END) as likes,
        SUM(CASE WHEN feedback = 'Disliked' THEN 1 ELSE 0 END) as dislikes
    FROM user_feedback
    """
    feedback_df = pd.read_sql_query(feedback_query, conn)
    
    bad_image_query = "SELECT COUNT(*) as bad_image_reports FROM bad_image_feedback"
    bad_image_df = pd.read_sql_query(bad_image_query, conn)

    plt.subplot(2, 2, 1)
    feedback_data = {
        'Metric': ['Likes', 'Dislikes', 'Bad Image Reports'],
        'Count': [feedback_df['likes'][0], feedback_df['dislikes'][0], bad_image_df['bad_image_reports'][0]]
    }
    feedback_df = pd.DataFrame(feedback_data)
    
    ax = plt.gca()
    bars = ax.barh(feedback_df['Metric'], feedback_df['Count'], color=blue_palette[1:])
    
    # Add labels to the end of each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:,.0f}', 
                ha='left', va='center', fontweight='bold')
    
    plt.title("User Feedback Distribution")
    plt.xlabel("Count")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 2. Distribution of relevance scores for retrieved recipes
    relevance_query = "SELECT relevance_score FROM assessed_recipes WHERE relevance_score IS NOT NULL"
    relevance_df = pd.read_sql_query(relevance_query, conn)

    plt.subplot(2, 2, 2)
    sns.histplot(relevance_df['relevance_score'], kde=True, color=blue_palette[2])
    plt.title("Distribution of Relevance Scores")
    plt.xlabel("Relevance Score")
    plt.ylabel("Frequency")

    # 3 & 4. Top 3 most common dishes liked and disliked
    top_liked_query = """
    SELECT recipe_title, COUNT(*) as count
    FROM user_feedback
    WHERE feedback = 'Liked'
    GROUP BY recipe_title
    ORDER BY count DESC
    LIMIT 3
    """
    top_liked_df = pd.read_sql_query(top_liked_query, conn)

    top_disliked_query = """
    SELECT recipe_title, COUNT(*) as count
    FROM user_feedback
    WHERE feedback = 'Disliked'
    GROUP BY recipe_title
    ORDER BY count DESC
    LIMIT 3
    """
    top_disliked_df = pd.read_sql_query(top_disliked_query, conn)

    plt.subplot(2, 2, 3)
    ax = plt.gca()
    bars = ax.barh(wrap_labels(top_liked_df['recipe_title']), top_liked_df['count'], color=blue_palette[2])
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:,.0f}', 
                ha='left', va='center', fontweight='bold')
    plt.title("Top 3 Most Liked Dishes")
    plt.xlabel("Number of Likes")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.subplot(2, 2, 4)
    ax = plt.gca()
    bars = ax.barh(wrap_labels(top_disliked_df['recipe_title']), top_disliked_df['count'], color=blue_palette[2])
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:,.0f}', 
                ha='left', va='center', fontweight='bold')
    plt.title("Top 3 Most Disliked Dishes")
    plt.xlabel("Number of Dislikes")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('performance_report.png')
    plt.close()

    conn.close()
    print("Performance report generated and saved as 'performance_report.png'")

if __name__ == "__main__":
    generate_performance_report()