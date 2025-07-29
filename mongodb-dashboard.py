import streamlit as st
import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
from typing import List, Dict
import os
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="MongoDB Search Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Initialize connection with security and SSL fix
@st.cache_resource
def init_connection():
    """
    Initialize MongoDB connection securely with SSL handling.
    """
    # Method 1: Try Streamlit secrets (for Streamlit Cloud)
    try:
        connection_string = st.secrets["MONGODB_URI"]
        # Add SSL parameters if not present
        if "tls=true" not in connection_string:
            if "?" in connection_string:
                connection_string += "&tls=true&tlsAllowInvalidCertificates=true"
            else:
                connection_string += "?tls=true&tlsAllowInvalidCertificates=true"
        
        client = MongoClient(
            connection_string,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=10000
        )
        # Test connection
        client.server_info()
        return client
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
    
    # Method 2: Try environment variable (for other deployments)
    connection_string = os.getenv("MONGODB_URI")
    if connection_string:
        try:
            return MongoClient(connection_string)
        except Exception as e:
            st.error(f"Connection error: {str(e)}")
    
    # Method 3: Manual input (for development/testing)
    st.error("⚠️ No MongoDB connection configured!")
    st.info("""
    **For Streamlit Cloud deployment:**
    1. Go to your app settings
    2. Add MONGODB_URI to Secrets
    
    **For local development:**
    1. Create `.streamlit/secrets.toml` file
    2. Add: `MONGODB_URI = "your-connection-string"`
    3. Make sure this file is in `.gitignore`
    
    **SSL Error Fix:**
    Add these parameters to your connection string:
    `?retryWrites=true&w=majority&tls=true&tlsAllowInvalidCertificates=true`
    """)
    
    # Allow manual connection for testing
    with st.expander("🔧 Manual Connection (Development Only)"):
        st.warning("⚠️ Only use this for testing. Never share apps with credentials visible!")
        manual_uri = st.text_input(
            "MongoDB URI", 
            type="password",
            help="This will not be saved and is only for testing"
        )
        if manual_uri and st.button("Connect"):
            try:
                client = MongoClient(manual_uri)
                # Test connection
                client.server_info()
                st.success("✅ Connected successfully!")
                return client
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")
    
    return None

# Search with pagination and date filtering
def search_with_pagination(collection_name, database_name, _client, query, sort_field, sort_order, 
                          start_date, end_date, page=0, page_size=50):
    """Search and load data with pagination and date filtering"""
    try:
        db = _client[database_name]
        collection = db[collection_name]
        
        # Build query
        search_filter = {}
        
        # Add date range filter
        if start_date or end_date:
            date_filter = {}
            if start_date:
                date_filter["$gte"] = pd.Timestamp(start_date).to_pydatetime()
            if end_date:
                date_filter["$lte"] = pd.Timestamp(end_date).to_pydatetime()
            search_filter["datetime"] = date_filter
        
        # Add text search
        if query:
            search_filter["Body"] = {"$regex": query, "$options": "i"}
        
        # Get total count for this search
        total_count = collection.count_documents(search_filter)
        
        # Sort direction
        sort_dir = -1 if sort_order == "Descending" else 1
        
        # Load one page of results
        skip = page * page_size
        cursor = collection.find(search_filter)
        
        if sort_field:
            cursor = cursor.sort(sort_field, sort_dir)
        
        data = list(cursor.skip(skip).limit(page_size))
        
        return pd.DataFrame(data), total_count
    
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return pd.DataFrame(), 0

# Perform vector search
def vector_search(collection, query_vector, index_name, embedding_field, limit=50):
    """Perform vector similarity search using MongoDB Atlas Vector Search"""
    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": embedding_field,
                    "queryVector": query_vector,
                    "numCandidates": limit * 2,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "datetime": 1,
                    "URL": 1,
                    "Engagement": 1,
                    "Shares": 1,
                    "Quotes": 1,
                    "Likes": 1,
                    "Replies": 1,
                    "Reposts": 1,
                    "Reactions": 1,
                    "Views": 1,
                    "Body": 1,
                    "Clean_Text": 1,
                    "embeddings": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(collection.aggregate(pipeline))
        return pd.DataFrame(results)
    
    except Exception as e:
        st.error(f"Vector search error: {str(e)}")
        return pd.DataFrame()

# Function to render data table with search buttons
def render_data_table_with_search(df, display_columns, is_semantic_result=False):
    """Render data table with integrated search buttons"""
    # Create HTML for the table with embedded buttons
    html_rows = []
    
    # Add header row
    header_cols = ["<th style='text-align: center; width: 50px;'>Search</th>"]
    for col in display_columns:
        header_cols.append(f"<th>{col}</th>")
    if is_semantic_result and 'score' in df.columns:
        header_cols.append("<th>Score</th>")
    
    html_rows.append("<tr>" + "".join(header_cols) + "</tr>")
    
    # Add data rows
    for idx, row in df.iterrows():
        row_cells = [f"<td style='text-align: center;'><button onclick='window.location.href=\"?search_idx={idx}\"' style='background: none; border: none; cursor: pointer; font-size: 16px;'>🔍</button></td>"]
        
        for col in display_columns:
            value = row[col]
            # Format datetime
            if col == 'datetime' and pd.notna(value):
                value = pd.to_datetime(value).strftime('%Y-%m-%d %H:%M')
            # Truncate long text
            elif col == 'Body' and isinstance(value, str) and len(value) > 200:
                value = value[:200] + "..."
            # Format numbers
            elif isinstance(value, (int, float)) and col in ['Engagement', 'Shares', 'Likes', 'Views', 'Replies']:
                value = f"{value:,.0f}"
            
            row_cells.append(f"<td>{value}</td>")
        
        # Add score if semantic search
        if is_semantic_result and 'score' in df.columns:
            row_cells.append(f"<td>{row[['score']].values[0]:.3f}</td>")
        
        html_rows.append("<tr>" + "".join(row_cells) + "</tr>")
    
    # Create the complete HTML table
    html_table = f"""
    <style>
        .dataframe {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        .dataframe th {{
            background-color: #f0f2f6;
            padding: 10px;
            text-align: left;
            font-weight: bold;
            border-bottom: 2px solid #ddd;
        }}
        .dataframe td {{
            padding: 8px;
            border-bottom: 1px solid #eee;
        }}
        .dataframe tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
    <table class="dataframe">
        {"".join(html_rows)}
    </table>
    """
    
    # Display the HTML table
    st.markdown(html_table, unsafe_allow_html=True)
    
    # Check if a search button was clicked
    query_params = st.query_params
    if "search_idx" in query_params:
        search_idx = int(query_params["search_idx"])
        if search_idx in df.index:
            st.session_state.semantic_search_idx = search_idx
            st.session_state.is_semantic_search = True
            st.session_state.page = 0
            # Clear the query parameter
            st.query_params.clear()
            st.rerun()

# Alternative approach using st.data_editor
def render_data_table_with_buttons(df, display_columns, is_semantic_result=False):
    """Render data table with search buttons using st.data_editor"""
    # Create a copy of the dataframe for display
    display_df = df[display_columns].copy()
    
    # Format datetime column if present
    if 'datetime' in display_df.columns:
        display_df['datetime'] = pd.to_datetime(display_df['datetime']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Add score column if this is a semantic search result
    if is_semantic_result and 'score' in df.columns:
        display_df['score'] = df['score'].round(3)
    
    # Create a container for the table
    container = st.container()
    
    with container:
        # Create columns for each row
        for i, (idx, row) in enumerate(display_df.iterrows()):
            cols = st.columns([0.5, 2, 8, 2, 1, 1, 1, 2])  # Adjust ratios based on your needs
            
            # Search button
            with cols[0]:
                if st.button("🔍", key=f"search_{idx}", help="Find similar"):
                    st.session_state.semantic_search_idx = idx
                    st.session_state.is_semantic_search = True
                    st.session_state.page = 0
                    st.rerun()
            
            # Display data fields
            field_idx = 1
            for col_name in display_columns[:7]:  # Limit columns shown
                if col_name in row:
                    with cols[field_idx]:
                        if col_name == 'Body':
                            st.text(str(row[col_name])[:50] + "..." if len(str(row[col_name])) > 50 else str(row[col_name]))
                        else:
                            st.text(str(row[col_name]))
                    field_idx += 1
            
            # Add score if semantic result
            if is_semantic_result and 'score' in row:
                with cols[-1]:
                    st.text(f"{row['score']:.3f}")
            
            # Add separator
            if i < len(display_df) - 1:
                st.markdown("---")

# Main app
def main():
    st.title("📊 MongoDB Semantic Search Dashboard")
    st.markdown("Search and explore your social media data with text and semantic search")
    
    # Initialize client
    client = init_connection()
    if not client:
        st.stop()
    
    # Test connection
    try:
        # List available databases
        databases = client.list_database_names()
    except Exception as e:
        st.error(f"Cannot connect to MongoDB: {str(e)}")
        st.stop()
    
    # Sidebar configuration with defaults
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Database selection
        database_name = st.selectbox(
            "Select Database",
            options=databases,
            help="Choose from available databases"
        )
        
        # Collection selection
        if database_name:
            try:
                collections = client[database_name].list_collection_names()
                collection_name = st.selectbox(
                    "Select Collection",
                    options=collections,
                    help="Choose from available collections"
                )
            except:
                collection_name = st.text_input("Collection Name")
        else:
            collection_name = st.text_input("Collection Name")
        
        # Pre-configured field mapping
        st.subheader("Field Configuration")
        text_field = "Body"
        embedding_field = "embeddings"
        vector_index_name = "grok_vector_index"
        
        st.info(f"""
        **Configured Fields:**
        - Text Field: {text_field}
        - Embedding Field: {embedding_field}
        - Vector Index: {vector_index_name}
        """)
        
        # Engagement metrics for sorting
        sort_fields = ["Engagement", "Shares", "Quotes", "Likes", "Replies", "Reposts", "Reactions", "Views"]
        
        # Load data button
        load_button = st.button("Load Data", type="primary", use_container_width=True)
        
        # Add a button to clear semantic search and go back to regular search
        if 'is_semantic_search' in st.session_state and st.session_state.is_semantic_search:
            if st.button("← Back to Regular Search", use_container_width=True):
                st.session_state.is_semantic_search = False
                st.session_state.page = 0
                st.rerun()
    
    # Main content area
    if load_button or 'data_loaded' in st.session_state:
        st.session_state.data_loaded = True
        
        # Initialize session state variables
        if 'page' not in st.session_state:
            st.session_state.page = 0
        if 'is_semantic_search' in st.session_state is None:
            st.session_state.is_semantic_search = False
        
        # Get database and collection
        db = client[database_name]
        collection = db[collection_name]
        
        # Check if we're in semantic search mode
        if st.session_state.get('is_semantic_search', False) and 'semantic_search_idx' in st.session_state:
            # Semantic search mode
            st.info("🧠 Showing semantic search results. Use the sidebar button to return to regular search.")
            
            # Get the embedding for semantic search
            try:
                # Retrieve the document with the embedding
                if 'current_df' in st.session_state:
                    source_doc = st.session_state.current_df.loc[st.session_state.semantic_search_idx]
                    query_embedding = source_doc[embedding_field]
                    
                    # Convert to list if numpy array
                    if isinstance(query_embedding, np.ndarray):
                        query_embedding = query_embedding.tolist()
                    
                    # Perform vector search
                    df = vector_search(
                        collection,
                        query_embedding,
                        vector_index_name,
                        embedding_field,
                        limit=50
                    )
                    
                    total_count = len(df)
                    
                    # Show source document info
                    st.success(f"Found {total_count} similar documents to: \"{source_doc['Body'][:100]}...\"")
                else:
                    st.error("Source document not found. Returning to regular search.")
                    st.session_state.is_semantic_search = False
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error in semantic search: {str(e)}")
                st.session_state.is_semantic_search = False
                st.rerun()
        
        else:
            # Regular search mode
            # Search and filter section
            st.subheader("🔍 Search and Filter")
            
            # First row: Text search and date range
            col1, col2, col3 = st.columns([3, 1.5, 1.5])
            
            with col1:
                search_query = st.text_input("Search in Body text", placeholder="Enter keywords to search...")
            
            with col2:
                start_date = st.date_input("Start Date", value=None)
            
            with col3:
                end_date = st.date_input("End Date", value=None)
            
            # Second row: Sort options
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                sort_field = st.selectbox("Sort by", options=sort_fields)
            
            with col2:
                sort_order = st.selectbox("Order", ["Descending", "Ascending"])
            
            with col3:
                if st.button("Search", type="primary", use_container_width=True):
                    st.session_state.page = 0  # Reset to first page on new search
            
            # Load data with pagination
            with st.spinner("Loading data..."):
                df, total_count = search_with_pagination(
                    collection_name, database_name, client, 
                    search_query, sort_field, sort_order,
                    start_date, end_date,
                    st.session_state.page, 50
                )
        
        # Store current dataframe in session state for semantic search
        st.session_state.current_df = df
        
        if df.empty and st.session_state.page == 0:
            st.warning("No data found with the current filters. Try adjusting your search criteria.")
            return
        
        # Display results count and pagination (only for regular search)
        if not st.session_state.get('is_semantic_search', False):
            total_pages = max(1, (total_count + 49) // 50)  # Round up
            
            col1, col2, col3 = st.columns([2, 3, 2])
            with col1:
                st.markdown(f"**Total results: {total_count:,}**")
            
            with col2:
                # Pagination controls
                page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
                with page_col1:
                    if st.button("◀ Previous", disabled=(st.session_state.page == 0)):
                        st.session_state.page -= 1
                        st.rerun()
                
                with page_col2:
                    st.markdown(f"<center>Page {st.session_state.page + 1} of {total_pages}</center>", unsafe_allow_html=True)
                
                with page_col3:
                    if st.button("Next ▶", disabled=(st.session_state.page >= total_pages - 1)):
                        st.session_state.page += 1
                        st.rerun()
            
            with col3:
                # Jump to page
                page_number = st.number_input(
                    "Go to page", 
                    min_value=1, 
                    max_value=total_pages, 
                    value=st.session_state.page + 1,
                    step=1
                )
                if page_number - 1 != st.session_state.page:
                    st.session_state.page = page_number - 1
                    st.rerun()
        
        # Display data table
        if not df.empty:
            # Create display columns (exclude embeddings and _id)
            display_columns = ['datetime', 'Body', 'Engagement', 'Shares', 'Likes', 
                             'Replies', 'Views', 'URL']
            display_columns = [col for col in display_columns if col in df.columns]
            
            st.subheader("📄 Results")
            
            # Method 1: Create a more integrated table display
            # First, show column headers
            header_cols = st.columns([0.5, 2, 4, 1.5, 1.5, 1.5, 1.5, 1.5, 2])
            with header_cols[0]:
                st.markdown("**🔍**")
            with header_cols[1]:
                st.markdown("**Date/Time**")
            with header_cols[2]:
                st.markdown("**Body**")
            with header_cols[3]:
                st.markdown("**Engagement**")
            with header_cols[4]:
                st.markdown("**Shares**")
            with header_cols[5]:
                st.markdown("**Likes**")
            with header_cols[6]:
                st.markdown("**Views**")
            with header_cols[7]:
                st.markdown("**Replies**")
            with header_cols[8]:
                st.markdown("**URL**")
            
            # Add score header if semantic search
            if st.session_state.get('is_semantic_search', False) and 'score' in df.columns:
                score_col = st.columns([14.5, 1.5])[1]
                with score_col:
                    st.markdown("**Score**")
            
            st.divider()
            
            # Create a container for the scrollable table
            with st.container():
                # Display each row
                for idx in df.index:
                    row = df.loc[idx]
                    
                    # Create columns for this row
                    if st.session_state.get('is_semantic_search', False) and 'score' in df.columns:
                        cols = st.columns([0.5, 2, 4, 1.5, 1.5, 1.5, 1.5, 1.5, 2, 1.5])
                    else:
                        cols = st.columns([0.5, 2, 4, 1.5, 1.5, 1.5, 1.5, 1.5, 2])
                    
                    # Search button
                    with cols[0]:
                        if st.button("🔍", key=f"search_{idx}", help="Find similar"):
                            st.session_state.semantic_search_idx = idx
                            st.session_state.is_semantic_search = True
                            st.session_state.page = 0
                            st.rerun()
                    
                    # Date/Time
                    with cols[1]:
                        if 'datetime' in row and pd.notna(row['datetime']):
                            st.text(pd.to_datetime(row['datetime']).strftime('%Y-%m-%d %H:%M'))
                        else:
                            st.text("")
                    
                    # Body
                    with cols[2]:
                        if 'Body' in row and pd.notna(row['Body']):
                            body_text = str(row['Body'])[:150] + "..." if len(str(row['Body'])) > 150 else str(row['Body'])
                            st.text(body_text)
                        else:
                            st.text("")
                    
                    # Engagement
                    with cols[3]:
                        if 'Engagement' in row and pd.notna(row['Engagement']):
                            st.text(f"{int(row['Engagement']):,}")
                        else:
                            st.text("0")
                    
                    # Shares
                    with cols[4]:
                        if 'Shares' in row and pd.notna(row['Shares']):
                            st.text(f"{int(row['Shares']):,}")
                        else:
                            st.text("0")
                    
                    # Likes
                    with cols[5]:
                        if 'Likes' in row and pd.notna(row['Likes']):
                            st.text(f"{int(row['Likes']):,}")
                        else:
                            st.text("0")
                    
                    # Views
                    with cols[6]:
                        if 'Views' in row and pd.notna(row['Views']):
                            st.text(f"{int(row['Views']):,}")
                        else:
                            st.text("0")
                    
                    # Replies
                    with cols[7]:
                        if 'Replies' in row and pd.notna(row['Replies']):
                            st.text(f"{int(row['Replies']):,}")
                        else:
                            st.text("0")
                    
                    # URL
                    with cols[8]:
                        if 'URL' in row and pd.notna(row['URL']):
                            st.markdown(f"[Link]({row['URL']})")
                        else:
                            st.text("")
                    
                    # Score (if semantic search)
                    if st.session_state.get('is_semantic_search', False) and 'score' in df.columns:
                        with cols[9]:
                            st.text(f"{row['score']:.3f}")
                    
                    # Add subtle separator
                    st.markdown(
                        """<hr style="margin: 2px 0; border: none; border-top: 1px solid #eee;">""",
                        unsafe_allow_html=True
                    )
            
            # Summary statistics (only for regular search)
            if not st.session_state.get('is_semantic_search', False):
                with st.expander("📊 Summary Statistics"):
                    if 'Engagement' in df.columns:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Avg Engagement", f"{df['Engagement'].mean():.0f}")
                        with col2:
                            st.metric("Total Views", f"{df['Views'].sum():,.0f}" if 'Views' in df.columns else "N/A")
                        with col3:
                            st.metric("Total Shares", f"{df['Shares'].sum():,.0f}" if 'Shares' in df.columns else "N/A")
                        with col4:
                            st.metric("Total Likes", f"{df['Likes'].sum():,.0f}" if 'Likes' in df.columns else "N/A")
        
        else:
            st.info("No results found. Try adjusting your filters.")
    
    else:
        st.info("👈 Please configure your database settings in the sidebar and click 'Load Data'")
        
        # Instructions
        with st.expander("📖 Quick Start Guide", expanded=True):
            st.markdown("""
            ### Your Dashboard is Pre-Configured for:
            
            **Fields:**
            - **Text Search**: Body field
            - **Sort Options**: Engagement, Shares, Quotes, Likes, Replies, Reposts, Reactions, Views
            - **Date Filter**: datetime field
            - **Vector Search**: embeddings field with grok_vector_index
            
            ### Features:
            
            1. **Text Search**: Search within the Body text
            2. **Date Range Filter**: Filter by datetime range
            3. **Flexible Sorting**: Sort by any engagement metric
            4. **Pagination**: Browse through large datasets 50 records at a time
            5. **Semantic Search**: Click the 🔍 button next to any row to find similar documents
            
            ### How Semantic Search Works:
            
            - Click the 🔍 button on any row to find similar documents
            - The table will be replaced with 50 similar results
            - Similar results include a "score" column showing similarity (higher = more similar)
            - You can click 🔍 on any result to search again
            - Use "← Back to Regular Search" in the sidebar to return
            
            ### To Get Started:
            
            1. Select your database and collection from the sidebar
            2. Click "Load Data"
            3. Use the search and filter options to explore your data
            4. Click 🔍 on any row to find semantically similar content
            """)

if __name__ == "__main__":
    main()