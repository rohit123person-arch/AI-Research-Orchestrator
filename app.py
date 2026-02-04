"""
🧠 INTELLIGENT RESEARCH ORCHESTRATOR - WEB INTERFACE
Streamlit-powered web UI for the multi-agent research system

Features:
- Beautiful, interactive web interface
- Real-time progress tracking
- Source visualization
- Report download
- Search history
"""

import streamlit as st
from intelligent_research_orchestrator_tavily import ResearchOrchestrator, ResearchState
import json
from datetime import datetime
import os
import time
import requests

# Page configuration
st.set_page_config(
    page_title="AI Research Orchestrator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    .source-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .agent-status {
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
    .agent-active {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .agent-complete {
        background-color: #e7f3ff;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'current_research' not in st.session_state:
    st.session_state.current_research = None
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None

def initialize_orchestrator():
    """Initialize the research orchestrator"""
    # Get API key from session state (user input) or environment variable
    tavily_key = st.session_state.get('tavily_api_key') or os.getenv("TAVILY_API_KEY")
    base_url = st.session_state.get('ollama_url', 'http://localhost:11434')
    
    try:
        orchestrator = ResearchOrchestrator(
            base_url=base_url,
            tavily_api_key=tavily_key
        )
        return orchestrator, None
    except Exception as e:
        return None, str(e)

def render_header():
    """Render the app header"""
    st.markdown('<h1 class="main-header">🧠 AI Research Orchestrator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Agent Research System powered by Ollama + Tavily</p>', unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar with configuration"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Ollama settings
        st.subheader("🤖 Ollama Settings")
        ollama_url = st.text_input(
            "Ollama URL",
            value=st.session_state.get('ollama_url', 'http://localhost:11434'),
            help="URL where Ollama is running (e.g., http://localhost:11434)",
            key="ollama_url_input"
        )
        st.session_state.ollama_url = ollama_url
        
        # Test Ollama connection
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Test Connection", use_container_width=True):
                try:
                    import requests
                    response = requests.get(f"{ollama_url}/api/tags", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ Connected!")
                    else:
                        st.error("❌ Connection failed")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)[:50]}")
        
        # Tavily settings
        st.subheader("🔍 Tavily API Settings")
        
        # Check for environment variable first
        env_tavily_key = os.getenv("TAVILY_API_KEY")
        
        # Use session state to persist user input
        if 'tavily_api_key' not in st.session_state:
            st.session_state.tavily_api_key = env_tavily_key or ""
        
        # Input field for API key
        tavily_key_input = st.text_input(
            "Tavily API Key",
            value=st.session_state.tavily_api_key,
            type="password",
            help="Enter your Tavily API key (starts with 'tvly-')",
            placeholder="tvly-xxxxxxxxxxxxxxxx",
            key="tavily_key_input"
        )
        
        # Update session state
        if tavily_key_input:
            st.session_state.tavily_api_key = tavily_key_input
        
        # Display status
        if st.session_state.tavily_api_key:
            st.success(f"✅ API Key: {st.session_state.tavily_api_key[:10]}...")
            # Reset orchestrator when key changes
            if 'last_tavily_key' in st.session_state:
                if st.session_state.last_tavily_key != st.session_state.tavily_api_key:
                    st.session_state.orchestrator = None
            st.session_state.last_tavily_key = st.session_state.tavily_api_key
        else:
            st.warning("⚠️ No Tavily API key provided")
            st.info("💡 System will use mock data without an API key")
        
        # Quick link to get API key
        st.markdown("🔗 [Get Free API Key](https://tavily.com) (1,000 searches/month)")
        
        # Model info
        st.subheader("📊 Model Information")
        st.info("""
        **Model:** llama3.2:latest
        **Context:** 4096 tokens
        **Temperature:** 0.7
        """)
        
        # Research history
        st.subheader("📚 Research History")
        if st.session_state.research_history:
            st.write(f"Total researches: {len(st.session_state.research_history)}")
            if st.button("Clear History"):
                st.session_state.research_history = []
                st.rerun()
        else:
            st.write("No research history yet")
        
        st.markdown("---")
        
        # Reset configuration
        st.subheader("🔄 Reset & Settings")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Config", use_container_width=True):
                st.session_state.tavily_api_key = ""
                st.session_state.skip_tavily = False
                st.session_state.orchestrator = None
                st.rerun()
        with col2:
            if st.button("📋 Show Setup", use_container_width=True):
                if 'skip_tavily' in st.session_state:
                    del st.session_state.skip_tavily
                if 'tavily_api_key' in st.session_state:
                    st.session_state.tavily_api_key = ""
                st.rerun()
        
        # Links
        st.subheader("🔗 Quick Links")
        st.markdown("""
        - [Get Tavily API Key](https://tavily.com)
        - [Ollama Documentation](https://ollama.com)
        - [GitHub Repository](#)
        """)

def render_agent_status(messages):
    """Render agent execution status"""
    st.subheader("🤖 Agent Execution Status")
    
    agents = ["planner", "researcher", "analyzer", "fact_checker", "writer", "visualizer"]
    completed_agents = set(msg['agent'] for msg in messages)
    
    cols = st.columns(3)
    for idx, agent in enumerate(agents):
        with cols[idx % 3]:
            if agent in completed_agents:
                st.markdown(f"""
                <div class="agent-status agent-complete">
                    ✅ {agent.upper().replace('_', ' ')}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="agent-status">
                    ⏳ {agent.upper().replace('_', ' ')}
                </div>
                """, unsafe_allow_html=True)

def render_initial_setup():
    """Render initial setup screen for first-time users"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 1rem; color: white; margin: 2rem 0;">
            <h2>🎉 Welcome to AI Research Orchestrator!</h2>
            <p>Let's get you set up in just a few steps</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Setup Checklist")
        
        # Step 1: Ollama
        st.markdown("#### 1️⃣ Ollama Setup")
        st.info("""
        **Ollama** is the local AI engine that powers the research system.
        
        ✅ **Already installed?** Great! Make sure it's running.  
        ❌ **Not installed?** Visit [ollama.com](https://ollama.com) to install.
        """)
        
        if st.button("🔍 Test Ollama Connection", use_container_width=True):
            try:
                import requests
                ollama_url = st.session_state.get('ollama_url', 'http://localhost:11434')
                response = requests.get(f"{ollama_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    st.success("✅ Ollama is running and ready!")
                    st.balloons()
                else:
                    st.error("❌ Ollama is not responding. Please start Ollama and try again.")
            except Exception as e:
                st.error(f"❌ Cannot connect to Ollama: {str(e)}")
                st.info("💡 Make sure Ollama is running: `ollama serve`")
        
        st.markdown("---")
        
        # Step 2: Tavily API Key
        st.markdown("#### 2️⃣ Tavily API Key (Free)")
        st.info("""
        **Tavily** provides real web search capabilities.
        
        🎁 **Free tier:** 1,000 searches per month  
        ⏱️ **Setup time:** 2 minutes
        """)
        
        st.markdown("**Get your free API key:**")
        st.markdown("1. Visit [tavily.com](https://tavily.com)")
        st.markdown("2. Sign up for a free account")
        st.markdown("3. Copy your API key")
        st.markdown("4. Paste it in the sidebar → ")
        
        st.markdown("---")
        
        # Step 3: Start Research
        st.markdown("#### 3️⃣ Start Researching!")
        
        if st.session_state.get('tavily_api_key'):
            st.success("✅ All set! You can now start researching.")
            if st.button("🚀 Go to Research", type="primary", use_container_width=True):
                st.rerun()
        else:
            st.warning("⚠️ Enter your Tavily API key in the sidebar to continue")
            st.markdown("""
            💡 **Don't have an API key yet?** 
            - You can still test the system with mock data
            - Click "Skip for now" to try it out
            """)
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("⏭️ Skip for now (use mock data)", use_container_width=True):
                    st.session_state.tavily_api_key = None
                    st.session_state.skip_tavily = True
                    st.rerun()
        
        st.markdown("---")
        
        # Quick Tips
        with st.expander("💡 Quick Tips for Best Results"):
            st.markdown("""
            - **Be specific** in your research questions
            - Include timeframes (e.g., "in 2024")
            - Mention specific technologies or domains
            - Review the sources provided in results
            - Download reports for future reference
            
            **Example questions:**
            - "What are the latest developments in quantum computing in 2024?"
            - "How is AI being used in cancer detection?"
            - "What are the main challenges in electric vehicle adoption?"
            """)
        
        # Troubleshooting
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **Ollama not connecting?**
            - Make sure Ollama is installed: [ollama.com](https://ollama.com)
            - Start Ollama: `ollama serve`
            - Pull the model: `ollama pull llama3.2:latest`
            - Check the URL in the sidebar
            
            **Tavily API not working?**
            - Verify your API key starts with 'tvly-'
            - Check you haven't exceeded the free tier (1,000/month)
            - Visit [tavily.com](https://tavily.com) to check your account
            
            **Other issues?**
            - Check the documentation files
            - Review error messages carefully
            - Try restarting Streamlit
            """)

def render_sources(search_results):
    """Render search sources"""
    st.subheader("📚 Sources Used")
    
    if not search_results:
        st.info("No sources available yet")
        return
    
    # Group by query
    sources_by_query = {}
    for result in search_results:
        query = result.get('query', 'Unknown')
        if query not in sources_by_query:
            sources_by_query[query] = []
        sources_by_query[query].append(result)
    
    # Display sources
    for query, sources in sources_by_query.items():
        with st.expander(f"🔍 Query: {query} ({len(sources)} results)", expanded=False):
            for idx, source in enumerate(sources, 1):
                st.markdown(f"""
                <div class="source-card">
                    <strong>{idx}. {source.get('title', 'Untitled')}</strong><br>
                    <small>🔗 <a href="{source.get('url', '#')}" target="_blank">{source.get('url', 'N/A')}</a></small><br>
                    <small>⭐ Score: {source.get('score', 'N/A')}</small><br>
                    <p style="margin-top: 0.5rem; color: #666;">
                    {source.get('content', 'No content available')[:200]}...
                    </p>
                </div>
                """, unsafe_allow_html=True)

def render_metrics(results):
    """Render result metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{results.get('confidence', 0):.0f}%</h3>
            <p>Confidence Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{results.get('sources', 0)}</h3>
            <p>Sources Found</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{results.get('iterations', 0)}</h3>
            <p>Research Iterations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{results.get('agents_used', 0)}</h3>
            <p>Agents Used</p>
        </div>
        """, unsafe_allow_html=True)

def run_research(query, orchestrator):
    """Run research with real-time updates"""
    
    # Create placeholders for dynamic updates
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    agent_status_placeholder = st.empty()
    
    try:
        status_placeholder.info("🚀 Initializing research workflow...")
        progress_bar.progress(10)
        
        # Run research
        status_placeholder.info("🔬 Research in progress...")
        results = orchestrator.research(query)
        
        progress_bar.progress(100)
        status_placeholder.success("✅ Research completed successfully!")
        
        return results, None
        
    except Exception as e:
        status_placeholder.error(f"❌ Error during research: {str(e)}")
        return None, str(e)

def main():
    """Main application"""
    render_header()
    render_sidebar()
    
    # Check if initial setup is needed (unless user chose to skip)
    if not st.session_state.get('tavily_api_key') and not os.getenv("TAVILY_API_KEY") and not st.session_state.get('skip_tavily'):
        render_initial_setup()
        return
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔍 New Research", "📊 Results", "📚 History"])
    
    with tab1:
        st.header("Start New Research")
        
        # Quick configuration status
        with st.expander("⚙️ Configuration Status", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🤖 Ollama Status**")
                ollama_status = st.empty()
                try:
                    import requests
                    ollama_url = st.session_state.get('ollama_url', 'http://localhost:11434')
                    response = requests.get(f"{ollama_url}/api/tags", timeout=2)
                    if response.status_code == 200:
                        ollama_status.success("✅ Connected")
                    else:
                        ollama_status.error("❌ Not connected")
                except:
                    ollama_status.warning("⚠️ Cannot connect")
            
            with col2:
                st.markdown("**🔍 Tavily Status**")
                if st.session_state.get('tavily_api_key'):
                    st.success("✅ API key configured")
                elif st.session_state.get('skip_tavily'):
                    st.info("ℹ️ Using mock data")
                else:
                    st.warning("⚠️ No API key (mock data)")
        
        # Research input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_area(
                "Enter your research question:",
                height=100,
                placeholder="Example: What are the latest developments in quantum computing?",
                help="Be specific for best results"
            )
        
        with col2:
            st.write("**Suggested Topics:**")
            suggestions = [
                "Latest AI developments",
                "Quantum computing news",
                "Climate change solutions",
                "Healthcare AI innovations",
                "Electric vehicle trends"
            ]
            for suggestion in suggestions:
                if st.button(suggestion, key=f"suggest_{suggestion}", use_container_width=True):
                    query = suggestion
        
        # Example queries
        with st.expander("💡 Example Research Questions"):
            st.markdown("""
            - What are the latest developments in quantum computing?
            - How is AI transforming healthcare in 2024?
            - What are the environmental impacts of electric vehicles?
            - Recent breakthroughs in renewable energy technology
            - Current trends in cybersecurity threats
            - Impact of remote work on productivity
            - Latest discoveries in space exploration
            - Advances in cancer treatment research
            """)
        
        # Research button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            start_research = st.button(
                "🚀 Start Research",
                type="primary",
                use_container_width=True,
                disabled=not query
            )
        
        # Run research
        if start_research and query:
            # Initialize orchestrator if needed
            if st.session_state.orchestrator is None:
                with st.spinner("Initializing AI system..."):
                    orchestrator, error = initialize_orchestrator()
                    if error:
                        st.error(f"❌ Failed to initialize: {error}")
                        st.stop()
                    st.session_state.orchestrator = orchestrator
                    st.success("✅ AI system initialized!")
            
            # Run research
            with st.spinner("Conducting research..."):
                results, error = run_research(query, st.session_state.orchestrator)
                
                if error:
                    st.error(f"Research failed: {error}")
                else:
                    # Store results
                    st.session_state.current_research = {
                        'query': query,
                        'results': results,
                        'timestamp': datetime.now().isoformat()
                    }
                    st.session_state.research_history.append(st.session_state.current_research)
                    
                    st.success("✅ Research completed! Check the Results tab.")
                    st.balloons()
    
    with tab2:
        st.header("Research Results")
        
        if st.session_state.current_research:
            research = st.session_state.current_research
            results = research['results']
            
            # Display query
            st.subheader("📝 Research Query")
            st.info(research['query'])
            
            # Display metrics
            st.subheader("📊 Research Metrics")
            render_metrics(results)
            
            st.markdown("---")
            
            # Agent status
            if 'full_state' in results:
                render_agent_status(results['full_state'].get('messages', []))
            
            st.markdown("---")
            
            # Research report
            st.subheader("📄 Comprehensive Report")
            report = results.get('report', 'No report available')
            st.markdown(report)
            
            # Download button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # Sources
            if 'search_results' in results:
                render_sources(results['search_results'])
            
            st.markdown("---")
            
            # Agent workflow
            with st.expander("🔄 Detailed Agent Workflow"):
                if 'full_state' in results:
                    for idx, msg in enumerate(results['full_state'].get('messages', []), 1):
                        st.markdown(f"**{idx}. {msg['agent'].upper()}**")
                        st.write(f"⏰ Time: {msg.get('timestamp', 'N/A')}")
                        if 'results_found' in msg:
                            st.write(f"📊 Results found: {msg['results_found']}")
                        with st.expander("View output"):
                            st.code(msg.get('content', '')[:500] + "...")
                        st.markdown("---")
        else:
            st.info("👈 Start a new research in the 'New Research' tab to see results here")
    
    with tab3:
        st.header("Research History")
        
        if st.session_state.research_history:
            st.write(f"Total researches: {len(st.session_state.research_history)}")
            
            # Display history in reverse chronological order
            for idx, research in enumerate(reversed(st.session_state.research_history)):
                with st.expander(
                    f"📄 {research['query'][:60]}... - {research['timestamp'][:19]}",
                    expanded=False
                ):
                    results = research['results']
                    
                    # Mini metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence", f"{results.get('confidence', 0):.0f}%")
                    with col2:
                        st.metric("Sources", results.get('sources', 0))
                    with col3:
                        st.metric("Iterations", results.get('iterations', 0))
                    
                    # Report preview
                    st.markdown("**Report Preview:**")
                    st.write(results.get('report', '')[:300] + "...")
                    
                    # View full button
                    if st.button("View Full Report", key=f"view_{idx}"):
                        st.session_state.current_research = research
                        st.rerun()
        else:
            st.info("No research history yet. Conduct your first research to build history!")

if __name__ == "__main__":
    main()