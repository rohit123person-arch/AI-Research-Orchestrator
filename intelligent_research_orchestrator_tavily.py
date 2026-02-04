"""
🧠 INTELLIGENT RESEARCH ORCHESTRATOR
A Multi-Agent AI System powered by LangGraph

This system uses multiple specialized AI agents working together to:
1. Research topics using web search (Tavily API)
2. Analyze and synthesize information
3. Generate comprehensive reports
4. Fact-check and validate claims
5. Create visual summaries

Built with LangGraph for sophisticated agent orchestration
Modified to use Ollama with llama3.2:latest model and Tavily for web search
"""

from typing import TypedDict, Annotated, List, Dict, Optional
import operator
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from tavily import TavilyClient
import json
import os
from datetime import datetime

# ==============================================================================
# STATE DEFINITION
# ==============================================================================

class ResearchState(TypedDict):
    """State that gets passed between agents"""
    query: str
    search_results: List[Dict]
    analyzed_data: Dict
    fact_checks: List[Dict]
    report: str
    visual_summary: Dict
    messages: Annotated[List, operator.add]
    next_agent: str
    iteration: int
    confidence_score: float

# ==============================================================================
# AGENT DEFINITIONS
# ==============================================================================

class ResearchOrchestrator:
    """Main orchestrator that manages the research workflow"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 tavily_api_key: Optional[str] = None):
        """
        Initialize the orchestrator with Ollama and Tavily
        
        Args:
            base_url: The base URL for Ollama API (default: http://localhost:11434)
            tavily_api_key: Tavily API key for web search. If None, will try to read from TAVILY_API_KEY env var
        """
        self.llm = ChatOllama(
            model="llama3.2:latest",
            temperature=0.7,
            base_url=base_url,
            num_ctx=4096  # Context window size
        )
        
        # Initialize Tavily client
        api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        if not api_key:
            print("⚠️  WARNING: No Tavily API key found. Web search will use mock data.")
            print("   Set TAVILY_API_KEY environment variable or pass tavily_api_key parameter.")
            print("   Get your free API key at: https://tavily.com")
            self.tavily_client = None
        else:
            self.tavily_client = TavilyClient(api_key=api_key)
            print("✅ Tavily client initialized successfully")
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        workflow = StateGraph(ResearchState)
        
        # Add nodes (agents)
        workflow.add_node("planner", self.planner_agent)
        workflow.add_node("researcher", self.researcher_agent)
        workflow.add_node("analyzer", self.analyzer_agent)
        workflow.add_node("fact_checker", self.fact_checker_agent)
        workflow.add_node("writer", self.writer_agent)
        workflow.add_node("visualizer", self.visualizer_agent)
        
        # Define the workflow
        workflow.set_entry_point("planner")
        
        # Planner decides what to do first
        workflow.add_conditional_edges(
            "planner",
            self.route_after_planning,
            {
                "research": "researcher",
                "end": END
            }
        )
        
        # Researcher -> Analyzer
        workflow.add_edge("researcher", "analyzer")
        
        # Analyzer can loop back or continue
        workflow.add_conditional_edges(
            "analyzer",
            self.route_after_analysis,
            {
                "fact_check": "fact_checker",
                "more_research": "researcher",
                "write": "writer"
            }
        )
        
        # Fact checker -> Writer
        workflow.add_edge("fact_checker", "writer")
        
        # Writer -> Visualizer
        workflow.add_edge("writer", "visualizer")
        
        # Visualizer -> End
        workflow.add_edge("visualizer", END)
        
        return workflow.compile()
    
    # ==========================================================================
    # AGENT IMPLEMENTATIONS
    # ==========================================================================
    
    def planner_agent(self, state: ResearchState) -> ResearchState:
        """Plans the research strategy"""
        print("\n🎯 PLANNER AGENT: Creating research strategy...")
        
        messages = [
            SystemMessage(content="""You are a research planning expert. 
            Analyze the query and create a strategic research plan.
            Determine:
            1. What information is needed
            2. What sources to consult
            3. How to structure the research
            4. Success criteria
            
            Respond in JSON format with: strategy, key_questions, sources, success_criteria"""),
            HumanMessage(content=f"Create a research plan for: {state['query']}")
        ]
        
        response = self.llm.invoke(messages)
        
        state["messages"].append({
            "agent": "planner",
            "content": response.content,
            "timestamp": datetime.now().isoformat()
        })
        
        state["next_agent"] = "researcher"
        state["iteration"] = 0
        
        print(f"   ✓ Strategy created: {len(response.content)} characters")
        return state
    
    def researcher_agent(self, state: ResearchState) -> ResearchState:
        """Conducts research using Tavily web search"""
        print(f"\n🔍 RESEARCHER AGENT: Gathering information (iteration {state['iteration']})...")
        
        # First, ask LLM what to search for
        messages = [
            SystemMessage(content="""You are an expert web researcher.
            Based on the query and current progress, determine what information to search for.
            Generate 2-3 specific search queries that will help answer the research question.
            Make queries concise and focused.
            
            Respond with ONLY a JSON object in this exact format:
            {"search_queries": ["query1", "query2", "query3"], "reasoning": "why these queries"}"""),
            HumanMessage(content=f"""
            Original query: {state['query']}
            Current iteration: {state['iteration']}
            Previous findings count: {len(state.get('search_results', []))}
            
            What should we search for next?
            """)
        ]
        
        response = self.llm.invoke(messages)
        
        # Parse search queries from LLM response
        search_queries = []
        try:
            # Try to extract JSON from response
            content = response.content.strip()
            # Remove markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(content)
            search_queries = parsed.get("search_queries", [])
            reasoning = parsed.get("reasoning", "")
            print(f"   📋 Planning to search: {len(search_queries)} queries")
            print(f"   💡 Reasoning: {reasoning[:100]}...")
        except Exception as e:
            print(f"   ⚠️  Could not parse LLM response, using default query")
            search_queries = [state['query']]
        
        # Perform actual web searches using Tavily
        all_results = []
        
        if self.tavily_client:
            # Use real Tavily search
            for query in search_queries[:3]:  # Limit to 3 queries
                try:
                    print(f"   🌐 Searching Tavily: '{query}'")
                    
                    # Perform Tavily search
                    tavily_response = self.tavily_client.search(
                        query=query,
                        max_results=3,  # Get top 3 results per query
                        search_depth="advanced",  # Use advanced search for better quality
                        include_domains=[],
                        exclude_domains=[]
                    )
                    
                    # Extract results
                    for result in tavily_response.get("results", []):
                        all_results.append({
                            "query": query,
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "content": result.get("content", ""),
                            "score": result.get("score", 0),
                            "published_date": result.get("published_date", ""),
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    print(f"      ✓ Found {len(tavily_response.get('results', []))} results")
                    
                except Exception as e:
                    print(f"      ✗ Search failed: {str(e)}")
        else:
            # Fallback to mock results if no Tavily API key
            print("   ⚠️  Using mock data (no Tavily API key)")
            for query in search_queries[:3]:
                all_results.extend([
                    {
                        "query": query,
                        "title": f"Mock Result for: {query}",
                        "content": "This is mock content. Please configure Tavily API key for real search results.",
                        "url": f"https://example.com/mock-{state['iteration']}",
                        "score": 0.8,
                        "timestamp": datetime.now().isoformat()
                    }
                ])
        
        if "search_results" not in state:
            state["search_results"] = []
        
        state["search_results"].extend(all_results)
        state["messages"].append({
            "agent": "researcher",
            "content": response.content,
            "results_found": len(all_results),
            "timestamp": datetime.now().isoformat()
        })
        
        state["iteration"] += 1
        
        print(f"   ✓ Total results gathered: {len(all_results)}")
        return state
    
    def analyzer_agent(self, state: ResearchState) -> ResearchState:
        """Analyzes and synthesizes research findings"""
        print("\n📊 ANALYZER AGENT: Synthesizing information...")
        
        # Prepare search results summary
        results_summary = []
        for result in state['search_results'][-10:]:  # Last 10 results
            results_summary.append({
                "title": result.get("title", ""),
                "content": result.get("content", "")[:300],  # First 300 chars
                "url": result.get("url", "")
            })
        
        messages = [
            SystemMessage(content="""You are a data analysis expert.
            Analyze the research findings and extract:
            1. Key insights and patterns
            2. Main themes and arguments
            3. Data quality assessment
            4. Knowledge gaps
            5. Confidence level (0-100)
            
            Respond with ONLY a JSON object in this format:
            {
                "insights": ["insight1", "insight2"],
                "themes": ["theme1", "theme2"],
                "quality_score": 85,
                "gaps": ["gap1", "gap2"],
                "confidence": 80,
                "needs_more_research": false
            }"""),
            HumanMessage(content=f"""
            Query: {state['query']}
            Research results found: {len(state['search_results'])}
            Latest results: {json.dumps(results_summary, indent=2)}
            
            Analyze these findings.
            """)
        ]
        
        response = self.llm.invoke(messages)
        
        # Parse confidence and determine next step
        try:
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(content)
            confidence = analysis.get("confidence", 70)
            needs_more = analysis.get("needs_more_research", False)
        except Exception as e:
            print(f"   ⚠️  Could not parse analysis: {e}")
            confidence = 75
            needs_more = False
        
        state["analyzed_data"] = {
            "analysis": response.content,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        state["confidence_score"] = confidence
        
        state["messages"].append({
            "agent": "analyzer",
            "content": response.content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Determine next step
        if needs_more and state["iteration"] < 3:
            state["next_agent"] = "more_research"
        elif confidence < 70 and state["iteration"] < 3:
            state["next_agent"] = "more_research"
        elif confidence >= 80:
            state["next_agent"] = "write"
        else:
            state["next_agent"] = "fact_check"
        
        print(f"   ✓ Analysis complete (confidence: {confidence}%)")
        return state
    
    def fact_checker_agent(self, state: ResearchState) -> ResearchState:
        """Validates claims and checks facts"""
        print("\n✅ FACT CHECKER AGENT: Verifying information...")
        
        messages = [
            SystemMessage(content="""You are a fact-checking expert.
            Review the analyzed data and verify key claims.
            Identify:
            1. Claims that need verification
            2. Potential biases or inconsistencies
            3. Missing sources or citations
            4. Reliability assessment
            
            Respond in JSON format with: verified_claims, concerns, reliability_score"""),
            HumanMessage(content=f"""
            Query: {state['query']}
            Analysis: {state['analyzed_data']['analysis']}
            Number of sources: {len(state['search_results'])}
            
            Verify the key claims.
            """)
        ]
        
        response = self.llm.invoke(messages)
        
        state["fact_checks"] = [{
            "checks": response.content,
            "timestamp": datetime.now().isoformat()
        }]
        
        state["messages"].append({
            "agent": "fact_checker",
            "content": response.content,
            "timestamp": datetime.now().isoformat()
        })
        
        print("   ✓ Fact-checking complete")
        return state
    
    def writer_agent(self, state: ResearchState) -> ResearchState:
        """Generates the final research report"""
        print("\n✍️ WRITER AGENT: Creating comprehensive report...")
        
        # Prepare source citations
        sources = []
        for i, result in enumerate(state['search_results'][:10], 1):
            sources.append(f"[{i}] {result.get('title', 'Untitled')} - {result.get('url', '')}")
        
        messages = [
            SystemMessage(content="""You are an expert technical writer.
            Create a comprehensive, well-structured research report.
            
            Include:
            1. Executive Summary
            2. Key Findings
            3. Detailed Analysis
            4. Supporting Evidence
            5. Conclusions
            6. Recommendations
            
            Use clear headings, bullet points, and professional formatting.
            Make it engaging and informative.
            Include source citations where relevant."""),
            HumanMessage(content=f"""
            Query: {state['query']}
            
            Number of sources researched: {len(state['search_results'])}
            
            Analysis summary: {state['analyzed_data']['analysis'][:1000]}
            
            Fact checks: {json.dumps(state.get('fact_checks', []))}
            
            Available sources:
            {chr(10).join(sources)}
            
            Create the final report.
            """)
        ]
        
        response = self.llm.invoke(messages)
        
        state["report"] = response.content
        state["messages"].append({
            "agent": "writer",
            "content": response.content,
            "word_count": len(response.content.split()),
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"   ✓ Report generated ({len(response.content.split())} words)")
        return state
    
    def visualizer_agent(self, state: ResearchState) -> ResearchState:
        """Creates visual summary and data representation"""
        print("\n📈 VISUALIZER AGENT: Creating visual summary...")
        
        messages = [
            SystemMessage(content="""You are a data visualization expert.
            Create a visual summary of the research including:
            1. Key statistics and metrics
            2. Important relationships
            3. Timeline of findings
            4. Confidence breakdown
            
            Respond in JSON format with structured data for visualization."""),
            HumanMessage(content=f"""
            Query: {state['query']}
            Report length: {len(state['report'])} characters
            Confidence score: {state['confidence_score']}
            Sources used: {len(state['search_results'])}
            
            Create visual summary data.
            """)
        ]
        
        response = self.llm.invoke(messages)
        
        state["visual_summary"] = {
            "data": response.content,
            "timestamp": datetime.now().isoformat()
        }
        
        state["messages"].append({
            "agent": "visualizer",
            "content": response.content,
            "timestamp": datetime.now().isoformat()
        })
        
        print("   ✓ Visual summary created")
        return state
    
    # ==========================================================================
    # ROUTING FUNCTIONS
    # ==========================================================================
    
    def route_after_planning(self, state: ResearchState) -> str:
        """Decide what to do after planning"""
        return "research"
    
    def route_after_analysis(self, state: ResearchState) -> str:
        """Decide what to do after analysis"""
        return state.get("next_agent", "write")
    
    # ==========================================================================
    # MAIN EXECUTION
    # ==========================================================================
    
    def research(self, query: str) -> Dict:
        """Execute the full research workflow"""
        print("\n" + "="*80)
        print("🧠 INTELLIGENT RESEARCH ORCHESTRATOR")
        print("   Powered by Ollama (llama3.2) + Tavily Search")
        print("="*80)
        print(f"\n📝 Research Query: {query}\n")
        
        initial_state = ResearchState(
            query=query,
            search_results=[],
            analyzed_data={},
            fact_checks=[],
            report="",
            visual_summary={},
            messages=[],
            next_agent="planner",
            iteration=0,
            confidence_score=0.0
        )
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        print("\n" + "="*80)
        print("✨ RESEARCH COMPLETE")
        print("="*80)
        
        return {
            "query": final_state["query"],
            "report": final_state["report"],
            "confidence": final_state["confidence_score"],
            "sources": len(final_state["search_results"]),
            "iterations": final_state["iteration"],
            "agents_used": len(final_state["messages"]),
            "visual_summary": final_state["visual_summary"],
            "search_results": final_state["search_results"],
            "full_state": final_state
        }

# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

def main():
    """Example usage of the research orchestrator"""
    
    # Initialize the orchestrator with Ollama and Tavily
    # Make sure:
    # 1. Ollama is running locally (default: http://localhost:11434)
    # 2. TAVILY_API_KEY environment variable is set, or pass it directly
    
    print("Initializing Research Orchestrator...")
    print("- Ollama model: llama3.2:latest")
    print("- Web Search: Tavily API")
    print()
    
    # Option 1: Use environment variable
    orchestrator = ResearchOrchestrator()
    
    # Option 2: Pass API key directly
    # orchestrator = ResearchOrchestrator(tavily_api_key="your-api-key-here")
    
    # Example research queries
    queries = [
        "What are the latest developments in quantum computing?",
        "How is AI transforming healthcare in 2024?",
        "What are the environmental impacts of electric vehicles?",
    ]
    
    # Run research on the first query
    query = queries[0]
    results = orchestrator.research(query)
    
    # Display results
    print("\n" + "="*80)
    print("📊 RESEARCH RESULTS")
    print("="*80)
    print(f"\n🎯 Query: {results['query']}")
    print(f"📈 Confidence: {results['confidence']}%")
    print(f"📚 Sources: {results['sources']}")
    print(f"🔄 Iterations: {results['iterations']}")
    print(f"🤖 Agents Used: {results['agents_used']}")
    
    print("\n" + "-"*80)
    print("📄 FINAL REPORT")
    print("-"*80)
    print(results['report'])
    
    print("\n" + "-"*80)
    print("🔗 SOURCES USED")
    print("-"*80)
    for i, source in enumerate(results['search_results'][:10], 1):
        print(f"{i}. {source.get('title', 'Untitled')}")
        print(f"   URL: {source.get('url', 'N/A')}")
        print(f"   Score: {source.get('score', 'N/A')}")
        print()
    
    print("\n" + "-"*80)
    print("🔄 AGENT WORKFLOW")
    print("-"*80)
    for i, msg in enumerate(results['full_state']['messages'], 1):
        print(f"\n{i}. {msg['agent'].upper()}")
        print(f"   Time: {msg['timestamp']}")
        if 'results_found' in msg:
            print(f"   Results found: {msg['results_found']}")

if __name__ == "__main__":
    main()
