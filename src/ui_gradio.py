#!/usr/bin/env python3
"""
Gradio UI for Sanskrit Tutor RAG system.
Provides chat interface, exercise mode, and optional audio upload functionality.
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
from ingest import DataIngester, QAPair
from utils.config_validator import ConfigValidator


class SanskritTutorUI:
    """Gradio-based UI for the Sanskrit Tutor system."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.rag_system = None
        self.qa_pairs = []
        self.chat_history = []
        
        # UI state
        self.current_exercise = None
        self.exercise_score = 0
        self.exercise_total = 0
        
    def initialize(self) -> bool:
        """Initialize the RAG system and load QA pairs."""
        try:
            print("Initializing Sanskrit Tutor UI...")
            
            # Validate configuration first
            validator = ConfigValidator()
            if not validator.validate_all():
                return False
            
            # Initialize RAG system
            self.rag_system = SanskritRAG(self.config_path)
            if not self.rag_system.initialize():
                return False
            
            # Load QA pairs for exercise mode
            self._load_qa_pairs()
            
            print("Sanskrit Tutor UI initialized successfully!")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to initialize UI: {str(e)}")
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
    
    def chat_response(self, message: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
        """
        Handle chat messages in the chat interface.
        
        Args:
            message: User message
            history: Chat history
            
        Returns:
            Tuple of (response, updated_history)
        """
        if not self.rag_system:
            return "System not initialized. Please check the configuration.", history
        
        if not message.strip():
            return "Please enter a question.", history
        
        try:
            # Get RAG response
            rag_response = self.rag_system.answer_question(message)
            
            # Format response with metadata
            response_text = rag_response.answer
            
            # Add retrieval information if available
            if rag_response.retrieved_passages:
                response_text += f"\n\n📚 **Retrieved {len(rag_response.retrieved_passages)} relevant passages**"
                
            # Add citations if available
            if rag_response.citations:
                response_text += f"\n📎 **Citations:** {', '.join(rag_response.citations)}"
            
            # Add processing info
            response_text += f"\n⏱️ *Processing time: {rag_response.processing_time:.2f}s*"
            
            # Update history
            new_history = history + [[message, response_text]]
            
            return "", new_history
            
        except Exception as e:
            error_response = f"Sorry, I encountered an error: {str(e)}"
            new_history = history + [[message, error_response]]
            return "", new_history
    
    def get_random_exercise(self) -> Tuple[str, str, str, str]:
        """
        Get a random exercise question.
        
        Returns:
            Tuple of (question, difficulty, expected_answer, question_id)
        """
        if not self.qa_pairs:
            return "No exercise questions available", "N/A", "Please load QA pairs first", "none"
        
        qa_pair = random.choice(self.qa_pairs)
        self.current_exercise = qa_pair
        
        return qa_pair.question, qa_pair.difficulty, qa_pair.answer, qa_pair.id
    
    def check_exercise_answer(self, user_answer: str, expected_answer: str, question_id: str) -> Tuple[str, int, int]:
        """
        Check user's answer against expected answer and provide feedback.
        
        Args:
            user_answer: User's submitted answer
            expected_answer: Expected answer
            question_id: Question ID for tracking
            
        Returns:
            Tuple of (feedback_text, score, total_questions)
        """
        if not user_answer.strip():
            return "Please provide an answer.", self.exercise_score, self.exercise_total
        
        self.exercise_total += 1
        
        # Simple scoring based on key term matching
        user_lower = user_answer.lower()
        expected_lower = expected_answer.lower()
        
        # Extract key terms (very basic approach)
        import re
        expected_terms = re.findall(r'\b[a-zA-Zāīūṛṝḷḹēōṃḥṅñṭḍṇtdnpbmyrlvśṣshkgcjñṭḍṇ]+\b', expected_lower)
        
        matches = sum(1 for term in expected_terms if term in user_lower)
        score_ratio = matches / max(1, len(expected_terms))
        
        if score_ratio > 0.7:
            self.exercise_score += 1
            feedback = f"✅ **Excellent!** Your answer shows good understanding.\n\n"
        elif score_ratio > 0.4:
            self.exercise_score += 0.5
            feedback = f"✓ **Good attempt!** You got some key points.\n\n"
        else:
            feedback = f"📚 **Keep learning!** Here's what to focus on:\n\n"
        
        # Add the expected answer and explanation
        feedback += f"**Expected answer:** {expected_answer}\n\n"
        
        # Add related passage information if available
        if self.current_exercise and self.current_exercise.related_passage_ids:
            feedback += "**Related passages:** " + ", ".join(self.current_exercise.related_passage_ids) + "\n\n"
        
        # Use RAG system to provide additional explanation
        try:
            explanation_query = f"Explain in detail: {self.current_exercise.question}"
            rag_response = self.rag_system.answer_question(explanation_query, retrieval_k=3)
            feedback += f"**Detailed explanation:**\n{rag_response.answer}"
        except Exception as e:
            feedback += f"Could not generate additional explanation: {str(e)}"
        
        return feedback, int(self.exercise_score), self.exercise_total
    
    def reset_exercise_score(self) -> Tuple[int, int]:
        """Reset exercise scoring."""
        self.exercise_score = 0
        self.exercise_total = 0
        return 0, 0
    
    def process_audio(self, audio_file) -> str:
        """
        Process uploaded audio file (placeholder for ASR functionality).
        
        Args:
            audio_file: Uploaded audio file
            
        Returns:
            Processing result message
        """
        if audio_file is None:
            return "No audio file uploaded."
        
        # Placeholder implementation
        # In a full implementation, this would:
        # 1. Use ASR to transcribe Sanskrit speech
        # 2. Compare with expected pronunciation
        # 3. Provide feedback on pronunciation accuracy
        
        return f"Audio file received: {audio_file}. ASR functionality not yet implemented. This feature would analyze Sanskrit pronunciation and provide feedback."
    
    def get_passage_details(self, passage_id: str) -> str:
        """Get detailed information about a specific passage."""
        if not self.rag_system:
            return "System not initialized."
            
        passage = self.rag_system.get_passage_by_id(passage_id)
        if not passage:
            return f"Passage {passage_id} not found."
        
        details = f"""**Passage Details: {passage['id']}**

**Work:** {passage['work']}
**Chapter:** {passage['chapter']}
**Verse:** {passage['verse']}

**Devanagari:** {passage['text_devanagari']}
**IAST:** {passage['text_iast']}

**Source:** {passage['source_url']}
"""
        
        if passage.get('notes'):
            details += f"\n**Notes:** {passage['notes']}"
        
        return details
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        if not GRADIO_AVAILABLE:
            raise RuntimeError("Gradio not available. Install with: pip install gradio")
        
        # Custom CSS for better styling
        css = """
        .sanskrit-title {
            text-align: center;
            color: #d35400;
            font-size: 2em;
            margin-bottom: 1em;
        }
        .tab-content {
            padding: 1em;
        }
        .exercise-score {
            background-color: #e8f5e8;
            padding: 1em;
            border-radius: 5px;
            margin: 1em 0;
        }
        """
        
        with gr.Blocks(css=css, title="Sanskrit Tutor") as interface:
            gr.Markdown("# 🕉️ Sanskrit Tutor - RAG-powered Learning System", elem_classes=["sanskrit-title"])
            
            gr.Markdown("""
            Welcome to the Sanskrit Tutor! This system uses Retrieval-Augmented Generation (RAG) to help you learn Sanskrit language, literature, and philosophy.
            
            **Features:**
            - 💬 **Chat Mode**: Ask questions about Sanskrit texts, grammar, and philosophy
            - 📝 **Exercise Mode**: Practice with guided questions and get detailed feedback
            - 🔍 **Passage Lookup**: Explore specific text passages in detail
            - 🎵 **Audio Practice**: Upload audio for pronunciation practice (experimental)
            """)
            
            with gr.Tabs():
                # Chat Tab
                with gr.Tab("💬 Chat Mode"):
                    gr.Markdown("Ask any question about Sanskrit! The system will find relevant passages and provide detailed answers with citations.")
                    
                    chatbot = gr.Chatbot(
                        height=500,
                        label="Sanskrit Tutor Chat",
                        placeholder="Ask me anything about Sanskrit..."
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Type your question here... (e.g., 'What is dharma?', 'Explain the concept of karma')",
                            label="Your Question",
                            scale=4
                        )
                        submit_btn = gr.Button("Ask", variant="primary", scale=1)
                    
                    # Example questions
                    gr.Markdown("**Example questions:**")
                    example_questions = [
                        "What does 'dharma' mean in Sanskrit philosophy?",
                        "Explain the concept of karma in Hindu texts",
                        "What is the significance of Om/AUM?",
                        "Translate this verse: धर्मे चार्थे च कामे च मोक्षे च भरतर्षभ",
                        "What are the four purusharthas?"
                    ]
                    
                    for question in example_questions:
                        gr.Button(question, variant="secondary").click(
                            lambda q=question: (q, []), outputs=[msg, chatbot]
                        )
                    
                    # Chat functionality
                    submit_btn.click(
                        self.chat_response,
                        inputs=[msg, chatbot],
                        outputs=[msg, chatbot]
                    )
                    
                    msg.submit(
                        self.chat_response,
                        inputs=[msg, chatbot],
                        outputs=[msg, chatbot]
                    )
                
                # Exercise Tab
                with gr.Tab("📝 Exercise Mode"):
                    gr.Markdown("Practice with structured questions and get detailed feedback on your answers!")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            exercise_question = gr.Textbox(
                                label="Question",
                                interactive=False,
                                lines=3
                            )
                            
                            exercise_difficulty = gr.Textbox(
                                label="Difficulty",
                                interactive=False
                            )
                            
                            user_answer = gr.Textbox(
                                label="Your Answer",
                                placeholder="Type your answer here...",
                                lines=4
                            )
                            
                            with gr.Row():
                                new_question_btn = gr.Button("New Question", variant="secondary")
                                submit_answer_btn = gr.Button("Submit Answer", variant="primary")
                        
                        with gr.Column(scale=1):
                            with gr.Group():
                                gr.Markdown("### Score", elem_classes=["exercise-score"])
                                score_display = gr.Textbox(
                                    label="Current Score",
                                    value="0 / 0",
                                    interactive=False
                                )
                                reset_score_btn = gr.Button("Reset Score", variant="secondary")
                    
                    feedback_area = gr.Markdown("Click 'New Question' to start practicing!")
                    
                    # Hidden components to store exercise state
                    expected_answer = gr.Textbox(visible=False)
                    question_id = gr.Textbox(visible=False)
                    current_score = gr.Number(visible=False, value=0)
                    total_questions = gr.Number(visible=False, value=0)
                    
                    # Exercise functionality
                    new_question_btn.click(
                        self.get_random_exercise,
                        outputs=[exercise_question, exercise_difficulty, expected_answer, question_id]
                    )
                    
                    submit_answer_btn.click(
                        self.check_exercise_answer,
                        inputs=[user_answer, expected_answer, question_id],
                        outputs=[feedback_area, current_score, total_questions]
                    ).then(
                        lambda score, total: f"{score} / {total}",
                        inputs=[current_score, total_questions],
                        outputs=[score_display]
                    ).then(
                        lambda: "",
                        outputs=[user_answer]
                    )
                    
                    reset_score_btn.click(
                        self.reset_exercise_score,
                        outputs=[current_score, total_questions]
                    ).then(
                        lambda: "0 / 0",
                        outputs=[score_display]
                    ).then(
                        lambda: "Score reset! Click 'New Question' to start practicing.",
                        outputs=[feedback_area]
                    )
                
                # Passage Lookup Tab
                with gr.Tab("🔍 Passage Lookup"):
                    gr.Markdown("Look up specific passages by their ID to see detailed information.")
                    
                    with gr.Row():
                        passage_id_input = gr.Textbox(
                            label="Passage ID",
                            placeholder="e.g., gretil_shloka_001",
                            scale=3
                        )
                        lookup_btn = gr.Button("Look Up", variant="primary", scale=1)
                    
                    passage_details = gr.Markdown("Enter a passage ID and click 'Look Up' to see details.")
                    
                    lookup_btn.click(
                        self.get_passage_details,
                        inputs=[passage_id_input],
                        outputs=[passage_details]
                    )
                    
                    passage_id_input.submit(
                        self.get_passage_details,
                        inputs=[passage_id_input],
                        outputs=[passage_details]
                    )
                
                # Audio Practice Tab
                with gr.Tab("🎵 Audio Practice"):
                    gr.Markdown("""
                    **Audio Practice (Experimental Feature)**
                    
                    Upload audio recordings of Sanskrit pronunciation for analysis and feedback.
                    *Note: This feature requires additional setup and is not fully implemented yet.*
                    """)
                    
                    with gr.Row():
                        audio_input = gr.Audio(
                            label="Upload Audio Recording",
                            type="filepath"
                        )
                    
                    audio_feedback = gr.Textbox(
                        label="Analysis Result",
                        interactive=False,
                        lines=5
                    )
                    
                    process_audio_btn = gr.Button("Analyze Audio", variant="primary")
                    
                    process_audio_btn.click(
                        self.process_audio,
                        inputs=[audio_input],
                        outputs=[audio_feedback]
                    )
            
            # Footer
            gr.Markdown("""
            ---
            **About**: This Sanskrit Tutor uses RAG (Retrieval-Augmented Generation) to provide accurate, 
            citation-backed responses about Sanskrit texts and concepts. The system requires user-supplied 
            text corpora and can work with local GGUF models or hosted APIs.
            """)
            
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        if not self.rag_system:
            print("ERROR: System not initialized. Cannot launch interface.")
            return
            
        interface = self.create_interface()
        
        # Default launch parameters
        launch_params = {
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "share": False,
            "show_error": True,
            "inbrowser": True
        }
        launch_params.update(kwargs)
        
        print(f"Launching Sanskrit Tutor UI...")
        print(f"- Server: http://{launch_params['server_name']}:{launch_params['server_port']}")
        print(f"- Index contains: {self.rag_system.index.ntotal} passages")
        print(f"- Exercise questions: {len(self.qa_pairs)}")
        print(f"- LLM backend: {self.rag_system.llm_manager.get_current_backend_info()['backend']}")
        
        interface.launch(**launch_params)


def main():
    """Command-line interface for launching the UI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Launch Sanskrit Tutor Gradio UI"
    )
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
        "--host",
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )
    
    args = parser.parse_args()
    
    try:
        # Create and initialize UI
        ui = SanskritTutorUI(args.config)
        
        if not ui.initialize():
            print("Failed to initialize Sanskrit Tutor UI.")
            exit(1)
        
        # Launch interface
        ui.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            inbrowser=not args.no_browser
        )
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
