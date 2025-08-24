#!/usr/bin/env python3
"""
Multi-Domain Gradio UI for Sanskrit Tutor.
Provides specialized interfaces for different Sanskrit knowledge domains.
"""

import os
import time
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("ERROR: Gradio not available. Install with: pip install gradio")

from rag import SanskritRAG, RAGResponse
from domain_manager import MultiDomainManager, SanskritDomain
from ingest import DataIngester, QAPair
from utils.config_validator import ConfigValidator


class MultiDomainSanskritUI:
    """Enhanced Gradio UI with multi-domain support for Sanskrit knowledge."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.rag_system = None
        self.domain_manager = None
        self.qa_pairs = []
        self.chat_history = []
        
        # UI state
        self.current_domain = SanskritDomain.GENERAL
        self.domain_chat_histories = {}
        
        # Initialize domain-specific chat histories
        for domain in SanskritDomain:
            self.domain_chat_histories[domain] = []
    
    def initialize(self) -> bool:
        """Initialize the multi-domain RAG system."""
        try:
            print("Initializing Multi-Domain Sanskrit Tutor UI...")
            
            # Validate configuration first
            validator = ConfigValidator()
            if not validator.validate_all():
                return False
            
            # Initialize RAG system with multi-domain support
            self.rag_system = SanskritRAG(self.config_path)
            if not self.rag_system.initialize():
                return False
            
            # Initialize domain manager
            self.domain_manager = MultiDomainManager(self.config_path)
            
            # Load QA pairs for exercise mode
            self._load_qa_pairs()
            
            print("Multi-Domain Sanskrit Tutor UI initialized successfully!")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to initialize Multi-Domain UI: {str(e)}")
            return False
    
    def _load_qa_pairs(self):
        """Load QA pairs for exercise mode."""
        try:
            ingester = DataIngester()
            data = ingester.load_all_data(self.config_path)
            self.qa_pairs = data['qa_pairs']
            print(f"Loaded {len(self.qa_pairs)} exercise questions")
        except Exception as e:
            print(f"WARNING: Failed to load QA pairs: {str(e)}")
            self.qa_pairs = []
    
    def domain_chat_response(self, message: str, history: List[List[str]], domain_name: str) -> Tuple[str, List[List[str]]]:
        """Handle chat messages with domain-specific responses."""
        if not self.rag_system:
            return "System not initialized. Please check the configuration.", history
        
        if not message.strip():
            return "Please enter a question.", history
        
        try:
            # Convert domain name to enum
            domain = SanskritDomain(domain_name)
            
            # Set domain context
            self.rag_system.set_domain(domain)
            
            # Get domain-specific response
            response, detected_domain = self.rag_system.answer_with_domain_detection(message)
            
            # Format response with domain information
            domain_config = self.rag_system.get_domain_config(detected_domain)
            response_text = response.answer
            
            # Add metadata
            if response.retrieved_passages:
                response_text += f"\n\n📚 Retrieved {len(response.retrieved_passages)} relevant passages"
                
            if response.citations:
                response_text += f"\n📎 Citations: {', '.join(response.citations)}"
            
            response_text += f"\n⏱️ Processing time: {response.processing_time:.2f}s"
            response_text += f"\n🎯 Domain: {domain_config['expert_name']}"
            
            # Update history
            new_history = history + [[message, response_text]]
            
            return "", new_history
            
        except Exception as e:
            error_response = f"Sorry, I encountered an error: {str(e)}"
            new_history = history + [[message, error_response]]
            return "", new_history
    
    def auto_detect_domain_response(self, message: str, history: List[List[str]]) -> Tuple[str, List[List[str]], str]:
        """Handle chat with automatic domain detection."""
        if not self.rag_system or not message.strip():
            return "", history, "general"
        
        try:
            # Auto-detect domain and get response
            response, detected_domain = self.rag_system.answer_with_domain_detection(message)
            
            # Format response
            response_text = response.answer
            
            # Add metadata
            if response.retrieved_passages:
                response_text += f"\n\n📚 Retrieved {len(response.retrieved_passages)} relevant passages"
            
            if response.citations:
                response_text += f"\n📎 Citations: {', '.join(response.citations)}"
            
            response_text += f"\n⏱️ Processing time: {response.processing_time:.2f}s"
            
            # Update history
            new_history = history + [[message, response_text]]
            
            return "", new_history, detected_domain.value
            
        except Exception as e:
            error_response = f"Sorry, I encountered an error: {str(e)}"
            new_history = history + [[message, error_response]]
            return "", new_history, "general"
    
    def get_domain_info(self, domain_name: str) -> str:
        """Get information about a specific domain."""
        try:
            domain = SanskritDomain(domain_name)
            config = self.rag_system.get_domain_config(domain)
            
            info = f"""
            {config['icon']} **{config['display_name']}**
            
            **Expert:** {config['expert_name']}
            
            **Description:** {config['description']}
            
            **Specialized Features:**
            {chr(10).join(f"• {feature}" for feature in config['features'])}
            
            **Primary Sources:** {', '.join(self.domain_manager.get_domain_config(domain).primary_sources)}
            """
            
            return info.strip()
            
        except Exception as e:
            return f"Error loading domain info: {str(e)}"
    
    def get_domain_example_questions(self, domain_name: str) -> List[str]:
        """Get example questions for a domain."""
        examples = {
            "philosophy": [
                "What does Krishna teach about dharma?",
                "Explain the concept of moksha in the Bhagavad Gita",
                "What is the relationship between karma and dharma?"
            ],
            "grammar": [
                "What is the sandhi rule for vowel combination?",
                "How do you form the past participle of √गम्?",
                "Explain the vibhakti system in Sanskrit"
            ],
            "ayurveda": [
                "How do I treat a vata imbalance?",
                "What are the three doshas in Ayurveda?",
                "Which herbs are good for pitta constitution?"
            ],
            "mathematics": [
                "Solve this arithmetic problem from Lilavati",
                "Explain the chakravala method",
                "How did ancient Indians calculate square roots?"
            ],
            "yoga": [
                "Explain the eight limbs of yoga",
                "What is the difference between dharana and dhyana?",
                "How do I practice pranayama?"
            ],
            "general": [
                "What is Sanskrit?",
                "How many Sanskrit texts are there?",
                "What is the importance of Sanskrit in Indian culture?"
            ]
        }
        
        return examples.get(domain_name, ["Ask me anything about Sanskrit!"])
    
    def create_interface(self):
        """Create the multi-domain Gradio interface."""
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio is required for the UI")
        
        # Custom CSS for domain-specific styling
        custom_css = """
        .domain-philosophy { border-left: 4px solid #FF6B35; }
        .domain-grammar { border-left: 4px solid #2E8B57; }
        .domain-ayurveda { border-left: 4px solid #228B22; }
        .domain-mathematics { border-left: 4px solid #4169E1; }
        .domain-yoga { border-left: 4px solid #9370DB; }
        .domain-general { border-left: 4px solid #708090; }
        
        .domain-header {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 5px;
        }
        """
        
        with gr.Blocks(title="Multi-Domain Sanskrit Tutor", css=custom_css) as demo:
            gr.Markdown(
                """
                # 🕉️ Multi-Domain Sanskrit AI Tutor
                
                **Specialized AI assistants for different Sanskrit knowledge domains**
                
                Choose your expert: Philosophy AI • Panini AI • Charak AI • Lilavati AI • Yoga AI
                """
            )
            
            with gr.Tabs():
                # Auto-Detection Tab
                with gr.TabItem("🎯 Auto-Detect Domain"):
                    gr.Markdown("Ask any Sanskrit question and I'll automatically detect the appropriate domain expert!")
                    
                    with gr.Row():
                        with gr.Column(scale=3):
                            auto_chatbot = gr.Chatbot(
                                label="Auto-Domain Sanskrit Chat",
                                height=400
                            )
                            auto_msg = gr.Textbox(
                                label="Your Question",
                                placeholder="Ask anything about Sanskrit - grammar, philosophy, Ayurveda, etc.",
                                lines=2
                            )
                            auto_submit = gr.Button("Ask", variant="primary")
                        
                        with gr.Column(scale=1):
                            detected_domain = gr.Textbox(
                                label="Detected Domain",
                                value="general",
                                interactive=False
                            )
                            gr.Markdown(
                                """
                                **Domain Detection Examples:**
                                - "What is sandhi?" → Grammar
                                - "How to treat headaches?" → Ayurveda  
                                - "Solve 2x + 3 = 7" → Mathematics
                                - "What is meditation?" → Yoga
                                - "What is dharma?" → Philosophy
                                """
                            )
                    
                    auto_submit.click(
                        self.auto_detect_domain_response,
                        inputs=[auto_msg, auto_chatbot],
                        outputs=[auto_msg, auto_chatbot, detected_domain]
                    )
                    
                    auto_msg.submit(
                        self.auto_detect_domain_response,
                        inputs=[auto_msg, auto_chatbot],
                        outputs=[auto_msg, auto_chatbot, detected_domain]
                    )
                
                # Domain-Specific Tabs
                domains = [
                    ("🕉️ Philosophy AI", "philosophy"),
                    ("📝 Panini AI", "grammar"),
                    ("🌿 Charak AI", "ayurveda"),
                    ("🔢 Lilavati AI", "mathematics"),
                    ("🧘 Yoga AI", "yoga"),
                    ("🎓 General", "general")
                ]
                
                for display_name, domain_key in domains:
                    with gr.TabItem(display_name):
                        self._create_domain_tab(domain_key)
                
                # Exercise Mode Tab
                with gr.TabItem("📚 Exercise Mode"):
                    self._create_exercise_tab()
                
                # Tools Tab
                with gr.TabItem("🔧 Sanskrit Tools"):
                    self._create_tools_tab()
        
        return demo
    
    def _create_domain_tab(self, domain_key: str):
        """Create a tab for a specific domain."""
        domain_config = self.rag_system.get_domain_config(SanskritDomain(domain_key))
        
        gr.Markdown(f"""
        <div class="domain-header domain-{domain_key}">
        {domain_config['icon']} **{domain_config['expert_name']}**
        </div>
        
        **Specialization:** {domain_config['description']}
        
        **Features:** {', '.join(domain_config['features'])}
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                domain_chatbot = gr.Chatbot(
                    label=f"{domain_config['expert_name']} Chat",
                    height=400
                )
                domain_msg = gr.Textbox(
                    label="Your Question",
                    placeholder=f"Ask {domain_config['expert_name']} anything about {domain_config['description']}",
                    lines=2
                )
                domain_submit = gr.Button("Ask Expert", variant="primary")
            
            with gr.Column(scale=1):
                gr.Markdown("**Example Questions:**")
                examples = self.get_domain_example_questions(domain_key)
                for example in examples:
                    gr.Button(
                        example,
                        size="sm"
                    ).click(
                        lambda x=example: x,
                        outputs=domain_msg
                    )
        
        # Wire up the chat functionality
        domain_submit.click(
            self.domain_chat_response,
            inputs=[domain_msg, domain_chatbot, gr.State(domain_key)],
            outputs=[domain_msg, domain_chatbot]
        )
        
        domain_msg.submit(
            self.domain_chat_response,
            inputs=[domain_msg, domain_chatbot, gr.State(domain_key)],
            outputs=[domain_msg, domain_chatbot]
        )
    
    def _create_exercise_tab(self):
        """Create the exercise/quiz tab."""
        gr.Markdown("## 📚 Sanskrit Knowledge Exercises")
        gr.Markdown("Practice with curated questions from Sanskrit texts and get detailed feedback.")
        
        with gr.Row():
            with gr.Column():
                exercise_question = gr.Textbox(
                    label="Question",
                    interactive=False,
                    lines=3
                )
                exercise_difficulty = gr.Textbox(
                    label="Difficulty",
                    interactive=False
                )
                
                get_question_btn = gr.Button("Get New Exercise", variant="primary")
                
            with gr.Column():
                user_answer = gr.Textbox(
                    label="Your Answer",
                    placeholder="Type your answer here...",
                    lines=4
                )
                submit_answer_btn = gr.Button("Submit Answer", variant="secondary")
        
        exercise_feedback = gr.Textbox(
            label="Feedback & Explanation",
            interactive=False,
            lines=8
        )
        
        with gr.Row():
            exercise_score = gr.Number(
                label="Score",
                value=0,
                interactive=False
            )
            exercise_total = gr.Number(
                label="Total Questions",
                value=0,
                interactive=False
            )
        
        # Exercise functionality would be implemented here
        # (keeping it simple for now)
    
    def _create_tools_tab(self):
        """Create the Sanskrit tools tab."""
        gr.Markdown("## 🔧 Sanskrit Language Tools")
        
        with gr.Tabs():
            with gr.TabItem("Grammar Checker"):
                gr.Markdown("*Feature coming soon: Check Sanskrit grammar and sandhi rules*")
                
            with gr.TabItem("Pronunciation Guide"):
                gr.Markdown("*Feature coming soon: IAST to audio pronunciation*")
                
            with gr.TabItem("Text Converter"):
                gr.Markdown("*Feature coming soon: Devanagari ↔ IAST conversion*")
            
            with gr.TabItem("Meter Analysis"):
                gr.Markdown("*Feature coming soon: Analyze Sanskrit poetic meters*")
    
    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        if not self.initialize():
            raise RuntimeError("Failed to initialize the system")
        
        demo = self.create_interface()
        return demo.launch(**kwargs)


def main():
    """Main entry point for the multi-domain UI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Domain Sanskrit Tutor UI")
    parser.add_argument(
        "--config",
        default="user_assets/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link"
    )
    
    args = parser.parse_args()
    
    try:
        ui = MultiDomainSanskritUI(args.config)
        
        print(f"Launching Multi-Domain Sanskrit Tutor...")
        print(f"- Config: {args.config}")
        print(f"- Port: {args.port}")
        print(f"- Share: {args.share}")
        
        ui.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share,
            show_error=True
        )
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
