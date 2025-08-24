#!/usr/bin/env python3
"""
Sanskrit Conversational Chat UI.
Complete interface for conversing in Sanskrit with AI assistance.
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

from sanskrit_conversation import SanskritConversationalBot, SanskritLanguageProcessor
from utils.config_validator import ConfigValidator


class SanskritChatUI:
    """Complete Sanskrit conversational interface."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.sanskrit_bot = None
        self.language_processor = SanskritLanguageProcessor()
        self.conversation_mode = "bilingual"
        
        # Sample conversations for demo
        self.sample_conversations = {
            "basic": [
                ("नमस्ते!", "namaste!", "Greetings!"),
                ("कः अस्ति धर्मः?", "kaḥ asti dharmaḥ?", "What is dharma?"),
                ("अहम् संस्कृतम् शिक्षितुम् इच्छामि।", "aham saṃskṛtam śikṣitum icchāmi", "I want to learn Sanskrit."),
                ("गुरुः कः?", "guruḥ kaḥ?", "Who is the guru?"),
                ("योगः किम्?", "yogaḥ kim?", "What is yoga?")
            ],
            "philosophy": [
                ("आत्मा किम्?", "ātmā kim?", "What is the soul?"),
                ("मोक्षः कथम् प्राप्यते?", "mokṣaḥ katham prāpyate?", "How is liberation attained?"),
                ("ब्रह्म सत्यम् वा?", "brahma satyam vā?", "Is Brahman real?"),
                ("कर्म फलम् किम्?", "karma phalam kim?", "What is the fruit of action?")
            ],
            "daily": [
                ("कथम् अस्ति?", "katham asti?", "How are you?"),
                ("किम् करोषि?", "kim karoṣi?", "What are you doing?"),
                ("अहम् गच्छामि।", "aham gacchāmi", "I am going."),
                ("शुभ रात्रिः!", "śubha rātriḥ!", "Good night!")
            ]
        }
    
    def initialize(self) -> bool:
        """Initialize the Sanskrit conversational system."""
        try:
            print("Initializing Sanskrit Chat UI...")
            
            # Validate configuration first
            validator = ConfigValidator()
            if not validator.validate_all():
                return False
            
            # Initialize Sanskrit bot
            self.sanskrit_bot = SanskritConversationalBot(self.config_path)
            if not self.sanskrit_bot.initialize():
                return False
            
            print("Sanskrit Chat UI initialized successfully!")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to initialize Sanskrit Chat UI: {str(e)}")
            return False
    
    def process_sanskrit_chat(self, user_input: str, history: List[List[str]], mode: str) -> Tuple[str, List[List[str]]]:
        """Process Sanskrit conversation with different display modes."""
        
        if not self.sanskrit_bot:
            return "System not initialized. Please check the configuration.", history
        
        if not user_input.strip():
            return "", history
        
        try:
            # Set conversation mode
            self.conversation_mode = mode
            
            # Process the conversation
            conversation = self.sanskrit_bot.process_sanskrit_conversation(user_input)
            
            # Format response based on selected mode
            if mode == "sanskrit_only":
                response_text = f"""{conversation.bot_response}
                
*{conversation.bot_response_iast}*"""
            
            elif mode == "learning":
                script = self.language_processor.detect_script(user_input)
                vocab = self.language_processor.extract_sanskrit_vocabulary(user_input)
                
                response_text = f"""**📝 Input Analysis:**
• Text: {conversation.user_input}
• Script: {script.title()}
• Translation: {conversation.translation}
• Vocabulary: {', '.join(vocab)}

**🤖 Sanskrit Response:**
{conversation.bot_response}

**📖 IAST:** {conversation.bot_response_iast}

**🔤 English:** {conversation.bot_translation}

**📚 Context:** Used {len(conversation.context_passages)} passages from Sanskrit texts"""
            
            else:  # bilingual (default)
                response_text = f"""**संस्कृतम्:** {conversation.bot_response}

**IAST:** {conversation.bot_response_iast}

**English:** {conversation.bot_translation}

📚 *Referenced {len(conversation.context_passages)} passages from Sanskrit texts*"""
            
            # Update history
            new_history = history + [[user_input, response_text]]
            return "", new_history
            
        except Exception as e:
            error_response = f"""क्षम्यताम्! (kṣamyatām!)
Sorry, I encountered an error: {str(e)}

Please try:
• Simple Sanskrit questions like: कः अस्ति धर्मः?
• English questions about Sanskrit concepts
• Basic greetings like: नमस्ते!"""
            new_history = history + [[user_input, error_response]]
            return "", new_history
    
    def get_sample_questions(self, category: str) -> List[str]:
        """Get sample questions for different categories."""
        if category in self.sample_conversations:
            return [f"{skt} ({eng})" for skt, iast, eng in self.sample_conversations[category]]
        return ["नमस्ते! (Hello!)", "कः अस्ति धर्मः? (What is dharma?)"]
    
    def create_interface(self):
        """Create the Sanskrit conversational interface."""
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio is required for the UI")
        
        with gr.Blocks(title="Sanskrit Conversational AI", theme=gr.themes.Soft()) as demo:
            
            gr.Markdown("""
            # 🕉️ Sanskrit Conversational AI
            
            **Talk WITH the AI in Sanskrit language!**
            
            This chatbot can understand and respond in Sanskrit (Devanagari and IAST).
            Ask questions in Sanskrit and get responses in Sanskrit with English explanations.
            """)
            
            with gr.Tabs():
                # Main Conversation Tab
                with gr.TabItem("💬 Sanskrit Conversation"):
                    
                    with gr.Row():
                        mode_selector = gr.Radio(
                            choices=["bilingual", "sanskrit_only", "learning"],
                            value="bilingual",
                            label="Conversation Mode",
                            info="Choose how you want to interact"
                        )
                    
                    chatbot = gr.Chatbot(
                        label="Sanskrit AI Conversation",
                        height=500,
                        placeholder="Start a conversation in Sanskrit or English..."
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="Your Message (Sanskrit or English)",
                            placeholder="Type in Sanskrit: कः अस्ति धर्मः? or English: What is dharma?",
                            lines=2,
                            scale=4
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    # Wire up the conversation
                    send_btn.click(
                        self.process_sanskrit_chat,
                        inputs=[msg_input, chatbot, mode_selector],
                        outputs=[msg_input, chatbot]
                    )
                    
                    msg_input.submit(
                        self.process_sanskrit_chat,
                        inputs=[msg_input, chatbot, mode_selector],
                        outputs=[msg_input, chatbot]
                    )
                
                # Sample Conversations Tab
                with gr.TabItem("📚 Learn by Example"):
                    gr.Markdown("## Sample Sanskrit Conversations")
                    gr.Markdown("Click any example to try it in the conversation tab!")
                    
                    with gr.Tabs():
                        for category, samples in self.sample_conversations.items():
                            with gr.TabItem(category.title()):
                                gr.Markdown(f"### {category.title()} Sanskrit Conversations")
                                
                                for sanskrit, iast, english in samples:
                                    with gr.Row():
                                        gr.Textbox(
                                            value=f"**Sanskrit:** {sanskrit}\n**IAST:** {iast}\n**English:** {english}",
                                            lines=3,
                                            interactive=False,
                                            scale=3
                                        )
                                        gr.Button(
                                            "Try This →",
                                            scale=1
                                        ).click(
                                            lambda x=sanskrit: x,
                                            outputs=msg_input  # This would need proper connection
                                        )
                
                # Grammar Helper Tab
                with gr.TabItem("📝 Sanskrit Grammar"):
                    gr.Markdown("## Sanskrit Grammar Assistant")
                    gr.Markdown("*Get help with Sanskrit grammar, vocabulary, and sentence construction*")
                    
                    with gr.Row():
                        with gr.Column():
                            grammar_input = gr.Textbox(
                                label="Sanskrit Text",
                                placeholder="Enter Sanskrit text for analysis...",
                                lines=3
                            )
                            analyze_btn = gr.Button("Analyze", variant="secondary")
                        
                        with gr.Column():
                            grammar_output = gr.Textbox(
                                label="Grammar Analysis",
                                lines=6,
                                interactive=False
                            )
                    
                    # Placeholder for grammar analysis
                    analyze_btn.click(
                        lambda x: f"Grammar analysis for: {x}\n\n*Feature coming soon - will analyze Sanskrit grammar, sandhi, and word formation*",
                        inputs=grammar_input,
                        outputs=grammar_output
                    )
                
                # Pronunciation Tab
                with gr.TabItem("🎵 Pronunciation Practice"):
                    gr.Markdown("## Sanskrit Pronunciation Guide")
                    
                    with gr.Row():
                        with gr.Column():
                            pronunciation_input = gr.Textbox(
                                label="Sanskrit Text",
                                placeholder="Enter Sanskrit text for pronunciation guide...",
                                lines=2
                            )
                            pronounce_btn = gr.Button("Get Pronunciation", variant="secondary")
                        
                        with gr.Column():
                            pronunciation_output = gr.Textbox(
                                label="Pronunciation Guide",
                                lines=4,
                                interactive=False
                            )
                    
                    # Placeholder for pronunciation
                    pronounce_btn.click(
                        lambda x: f"Pronunciation guide for: {x}\n\n*Feature coming soon - will provide detailed pronunciation instructions*",
                        inputs=pronunciation_input,
                        outputs=pronunciation_output
                    )
            
            # Instructions section
            gr.Markdown("""
            ## 🎯 How to Use
            
            ### Conversation Modes:
            - **Bilingual**: Sanskrit response + IAST + English explanation (recommended)
            - **Sanskrit Only**: Pure Sanskrit conversation with IAST
            - **Learning**: Detailed analysis of your input + Sanskrit response
            
            ### Sample Questions to Try:
            ```
            Sanskrit Input Examples:
            कः अस्ति धर्मः?           (What is dharma?)
            नमस्ते गुरो!              (Greetings teacher!)
            किम् भवान् शिक्षयति?      (What do you teach?)
            योगः किम्?               (What is yoga?)
            अहम् संस्कृतम् इच्छामि।    (I want Sanskrit.)
            
            English Input Examples:
            What does Krishna teach about karma?
            How do I learn Sanskrit grammar?
            Explain the concept of dharma
            ```
            
            ### Features:
            - 🔤 **Script Detection**: Automatically detects Devanagari, IAST, or English
            - 📖 **Vocabulary Analysis**: Extracts and explains Sanskrit words
            - 🎯 **Context-Aware**: Responses based on 719 passages from Bhagavad Gita
            - 📚 **Citation-Backed**: All responses include references to authentic Sanskrit texts
            """)
        
        return demo
    
    def launch(self, **kwargs):
        """Launch the Sanskrit chat interface."""
        if not self.initialize():
            raise RuntimeError("Failed to initialize the Sanskrit chat system")
        
        demo = self.create_interface()
        return demo.launch(**kwargs)


def main():
    """Main entry point for Sanskrit conversational UI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sanskrit Conversational AI")
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
        ui = SanskritChatUI(args.config)
        
        print("🕉️ Launching Sanskrit Conversational AI...")
        print(f"- Can understand Sanskrit (Devanagari & IAST)")
        print(f"- Can respond in Sanskrit with explanations")
        print(f"- Based on {719} authentic Sanskrit passages")
        print(f"- Port: {args.port}")
        print(f"- Share: {args.share}")
        print()
        print("Sample conversation starters:")
        print("• नमस्ते! (Hello!)")
        print("• कः अस्ति धर्मः? (What is dharma?)")
        print("• अहम् संस्कृतम् इच्छामि। (I want Sanskrit.)")
        print()
        
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
