import sqlite3
import pandas as pd
from datetime import datetime
import uuid
import argparse

DB_NAME = 'recipe_feedback.db'

def connect_db():
    return sqlite3.connect(DB_NAME)

def reset_database():
    conn = connect_db()
    c = conn.cursor()
    
    # Drop all existing tables
    c.execute("DROP TABLE IF EXISTS user_feedback")
    c.execute("DROP TABLE IF EXISTS user_sessions")
    c.execute("DROP TABLE IF EXISTS user_queries")
    c.execute("DROP TABLE IF EXISTS returned_recipes")
    c.execute("DROP TABLE IF EXISTS user_clicks")
    
    conn.commit()
    conn.close()
    print("All tables have been dropped.")
    
    # Recreate all tables
    setup_database()
    print("Database has been reset. All tables dropped and recreated.")

def setup_database():
    conn = connect_db()
    c = conn.cursor()
    
    # Create user_feedback table
    c.execute('''CREATE TABLE IF NOT EXISTS user_feedback
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  recipe_title TEXT,
                  feedback TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create user_sessions table
    c.execute('''CREATE TABLE IF NOT EXISTS user_sessions
                 (session_id TEXT PRIMARY KEY,
                  username TEXT,
                  start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                  last_activity DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create user_queries table
    c.execute('''CREATE TABLE IF NOT EXISTS user_queries
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT,
                  query TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (session_id) REFERENCES user_sessions(session_id))''')
    
    # Create returned_recipes table with cosine_similarity column
    c.execute('''CREATE TABLE IF NOT EXISTS returned_recipes
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT,
                  query_id INTEGER,
                  recipe_title TEXT,
                  position INTEGER,
                  cosine_similarity REAL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (session_id) REFERENCES user_sessions(session_id),
                  FOREIGN KEY (query_id) REFERENCES user_queries(id))''')
    
    # Create user_clicks table
    c.execute('''CREATE TABLE IF NOT EXISTS user_clicks
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT,
                  recipe_title TEXT,
                  click_type TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (session_id) REFERENCES user_sessions(session_id))''')
    
    conn.commit()
    conn.close()
    print("Database setup complete.")

def create_session(username):
    session_id = str(uuid.uuid4())
    conn = connect_db()
    c = conn.cursor()
    c.execute("INSERT INTO user_sessions (session_id, username) VALUES (?, ?)", (session_id, username))
    conn.commit()
    conn.close()
    return session_id

def update_session_activity(session_id):
    conn = connect_db()
    c = conn.cursor()
    c.execute("UPDATE user_sessions SET last_activity = CURRENT_TIMESTAMP WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

def store_user_query(session_id, query):
    conn = connect_db()
    c = conn.cursor()
    c.execute("INSERT INTO user_queries (session_id, query) VALUES (?, ?)", (session_id, query))
    query_id = c.lastrowid
    conn.commit()
    conn.close()
    return query_id

def store_returned_recipes(session_id, query_id, recipes):
    conn = connect_db()
    c = conn.cursor()
    for position, recipe in enumerate(recipes, start=1):
        c.execute("""
            INSERT INTO returned_recipes 
            (session_id, query_id, recipe_title, position, cosine_similarity) 
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, query_id, recipe['Title'], position, recipe['cosine_similarity']))
    conn.commit()
    conn.close()

def store_user_click(session_id, recipe_title, click_type):
    conn = connect_db()
    c = conn.cursor()
    c.execute("INSERT INTO user_clicks (session_id, recipe_title, click_type) VALUES (?, ?, ?)",
              (session_id, recipe_title, click_type))
    conn.commit()
    conn.close()

def save_feedback(username, recipe_title, feedback):
    conn = connect_db()
    c = conn.cursor()
    c.execute("INSERT INTO user_feedback (username, recipe_title, feedback) VALUES (?, ?, ?)",
              (username, recipe_title, feedback))
    conn.commit()
    conn.close()

def view_all_feedback():
    conn = connect_db()
    query = "SELECT * FROM user_feedback"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_feedback_by_user(username):
    conn = connect_db()
    query = "SELECT * FROM user_feedback WHERE username = ?"
    df = pd.read_sql_query(query, conn, params=(username,))
    conn.close()
    return df

def get_feedback_by_recipe(recipe_title):
    conn = connect_db()
    query = "SELECT * FROM user_feedback WHERE recipe_title = ?"
    df = pd.read_sql_query(query, conn, params=(recipe_title,))
    conn.close()
    return df

def get_most_liked_recipes(limit=10):
    conn = connect_db()
    query = """
    SELECT recipe_title, COUNT(*) as like_count
    FROM user_feedback
    WHERE feedback = 'Liked'
    GROUP BY recipe_title
    ORDER BY like_count DESC
    LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df

def get_feedback_stats():
    conn = connect_db()
    query = """
    SELECT 
        COUNT(*) as total_feedback,
        SUM(CASE WHEN feedback = 'Liked' THEN 1 ELSE 0 END) as likes,
        SUM(CASE WHEN feedback = 'Disliked' THEN 1 ELSE 0 END) as dislikes,
        SUM(CASE WHEN feedback NOT IN ('Liked', 'Disliked') THEN 1 ELSE 0 END) as detailed_feedback
    FROM user_feedback
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_user_queries(session_id=None, username=None):
    conn = connect_db()
    if session_id:
        query = "SELECT * FROM user_queries WHERE session_id = ? ORDER BY timestamp DESC"
        params = (session_id,)
    elif username:
        query = """
        SELECT uq.* 
        FROM user_queries uq
        JOIN user_sessions us ON uq.session_id = us.session_id
        WHERE us.username = ?
        ORDER BY uq.timestamp DESC
        """
        params = (username,)
    else:
        query = "SELECT * FROM user_queries ORDER BY timestamp DESC"
        params = ()
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def get_returned_recipes(session_id=None):
    conn = connect_db()
    if session_id:
        query = """
        SELECT rr.*, uq.query
        FROM returned_recipes rr
        JOIN user_queries uq ON rr.query_id = uq.id
        WHERE rr.session_id = ?
        ORDER BY rr.timestamp DESC, rr.position ASC
        """
        params = (session_id,)
    else:
        query = """
        SELECT rr.*, uq.query
        FROM returned_recipes rr
        JOIN user_queries uq ON rr.query_id = uq.id
        ORDER BY rr.timestamp DESC, rr.position ASC
        """
        params = ()
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def get_user_clicks(session_id=None):
    conn = connect_db()
    if session_id:
        query = "SELECT * FROM user_clicks WHERE session_id = ? ORDER BY timestamp DESC"
        params = (session_id,)
    else:
        query = "SELECT * FROM user_clicks ORDER BY timestamp DESC"
        params = ()
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Database management for Recipe Recommendation System")
    parser.add_argument('--reset', action='store_true', help='Reset the database (drop all tables and recreate)')
    args = parser.parse_args()

    if args.reset:
        reset_database()
    else:
        setup_database()

    print("\nViewing all feedback:")
    print(view_all_feedback())

    print("\nMost liked recipes:")
    print(get_most_liked_recipes())

    print("\nFeedback stats:")
    print(get_feedback_stats())

    print("\nUser queries:")
    print(get_user_queries())

    print("\nReturned recipes:")
    print(get_returned_recipes())

    print("\nUser clicks:")
    print(get_user_clicks())