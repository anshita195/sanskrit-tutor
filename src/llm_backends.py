#!/usr/bin/env python3
"""
LLM backends for Sanskrit Tutor RAG system.
Supports user-supplied GGUF models with graceful fallback to hosted APIs.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Iterator
from abc import ABC, abstractmethod
import warnings

# Try importing llama-cpp-python for local GGUF inference
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None

# Try importing requests for hosted APIs
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 500, 
                temperature: float = 0.7, stop_sequences: List[str] = None) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available and ready."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model."""
        pass


class LocalGGUFBackend(LLMBackend):
    """Local GGUF model backend using llama-cpp-python."""
    
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = 0):
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.model = None
        self._model_info = {}
        
    def load_model(self) -> bool:
        """
        Load the GGUF model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not LLAMA_CPP_AVAILABLE:
            print("ERROR: llama-cpp-python is required for local GGUF inference.")
            print("Install with: pip install llama-cpp-python")
            return False
            
        if not self.model_path.exists():
            print(f"ERROR: GGUF model file not found: {self.model_path.absolute()}")
            print("Please place your GGUF model file at the specified path.")
            return False
            
        try:
            print(f"Loading GGUF model: {self.model_path.name}")
            print(f"Context length: {self.n_ctx}, GPU layers: {self.n_gpu_layers}")
            
            # Initialize model with error handling
            self.model = Llama(
                model_path=str(self.model_path.absolute()),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            
            # Store model info
            self._model_info = {
                'backend': 'Local GGUF',
                'model_path': str(self.model_path),
                'context_length': str(self.n_ctx),
                'gpu_layers': str(self.n_gpu_layers)
            }
            
            print("Local GGUF model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load GGUF model: {str(e)}")
            print("Check that the model file is valid and you have sufficient memory.")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 500, 
                temperature: float = 0.7, stop_sequences: List[str] = None) -> str:
        """Generate text using the local GGUF model."""
        if not self.is_available():
            raise RuntimeError("Local GGUF model is not available")
            
        try:
            # Prepare stop sequences
            stop = stop_sequences or []
            
            # Generate response
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                echo=False
            )
            
            # Extract generated text
            generated_text = response['choices'][0]['text'].strip()
            return generated_text
            
        except Exception as e:
            print(f"ERROR: GGUF generation failed: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """Check if the local GGUF model is loaded and ready."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model."""
        return self._model_info.copy()


class HuggingFaceBackend(LLMBackend):
    """Hugging Face Inference API backend."""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1", 
                 api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv('HF_API_KEY')
        self.base_url = "https://api-inference.huggingface.co/models"
        self._model_info = {
            'backend': 'Hugging Face API',
            'model_name': model_name
        }
        
    def is_available(self) -> bool:
        """Check if the HF API is available."""
        if not REQUESTS_AVAILABLE:
            return False
        if not self.api_key:
            return False
        return True
    
    def generate(self, prompt: str, max_tokens: int = 500, 
                temperature: float = 0.7, stop_sequences: List[str] = None) -> str:
        """Generate text using Hugging Face Inference API."""
        if not self.is_available():
            raise RuntimeError("Hugging Face API is not available. Check API key and internet connection.")
            
        url = f"{self.base_url}/{self.model_name}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False,
                "do_sample": True
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '').strip()
            else:
                generated_text = str(result)
                
            return generated_text
            
        except requests.RequestException as e:
            print(f"ERROR: Hugging Face API request failed: {str(e)}")
            raise
        except Exception as e:
            print(f"ERROR: Hugging Face API generation failed: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the model."""
        return self._model_info.copy()


class OpenAIBackend(LLMBackend):
    """OpenAI API backend (fallback only)."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1"
        self._model_info = {
            'backend': 'OpenAI API',
            'model_name': model_name
        }
        
    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        if not REQUESTS_AVAILABLE:
            return False
        if not self.api_key:
            return False
        return True
    
    def generate(self, prompt: str, max_tokens: int = 500, 
                temperature: float = 0.7, stop_sequences: List[str] = None) -> str:
        """Generate text using OpenAI API."""
        if not self.is_available():
            raise RuntimeError("OpenAI API is not available. Check API key and internet connection.")
            
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop_sequences
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result['choices'][0]['message']['content'].strip()
            return generated_text
            
        except requests.RequestException as e:
            print(f"ERROR: OpenAI API request failed: {str(e)}")
            raise
        except Exception as e:
            print(f"ERROR: OpenAI API generation failed: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the model."""
        return self._model_info.copy()


class LLMManager:
    """
    Manages multiple LLM backends with fallback logic.
    Prioritizes user-supplied local models, falls back to hosted APIs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.primary_backend = None
        self.fallback_backends = []
        self.current_backend = None
        
    def initialize_backends(self) -> bool:
        """
        Initialize LLM backends based on configuration.
        
        Returns:
            True if at least one backend is available, False otherwise
        """
        print("Initializing LLM backends...")
        
        # Try to initialize local GGUF backend first
        if self.config.get('gguf_local', False):
            model_path = self.config.get('model_path')
            if model_path:
                print("Attempting to load local GGUF model...")
                gguf_backend = LocalGGUFBackend(
                    model_path=model_path,
                    n_ctx=self.config.get('n_ctx', 4096),
                    n_gpu_layers=self.config.get('n_gpu_layers', 0)
                )
                
                if gguf_backend.load_model():
                    self.primary_backend = gguf_backend
                    print("Local GGUF model will be used as primary backend.")
                else:
                    print("Failed to load local GGUF model. Will try fallback backends.")
        
        # Initialize fallback backends
        print("Initializing fallback backends...")
        
        # Hugging Face API
        hf_backend = HuggingFaceBackend(
            model_name=self.config.get('hf_model', 'mistralai/Mistral-7B-Instruct-v0.1')
        )
        if hf_backend.is_available():
            self.fallback_backends.append(hf_backend)
            print("Hugging Face API backend available.")
        else:
            if not os.getenv('HF_API_KEY'):
                print("Hugging Face API not available: HF_API_KEY environment variable not set.")
            else:
                print("Hugging Face API not available: requests library missing.")
        
        # OpenAI API (last resort)
        openai_backend = OpenAIBackend(
            model_name=self.config.get('openai_model', 'gpt-3.5-turbo')
        )
        if openai_backend.is_available():
            self.fallback_backends.append(openai_backend)
            print("OpenAI API backend available.")
        else:
            if not os.getenv('OPENAI_API_KEY'):
                print("OpenAI API not available: OPENAI_API_KEY environment variable not set.")
            else:
                print("OpenAI API not available: requests library missing.")
        
        # Set current backend
        if self.primary_backend:
            self.current_backend = self.primary_backend
        elif self.fallback_backends:
            self.current_backend = self.fallback_backends[0]
            print(f"Using fallback backend: {self.current_backend.get_model_info()['backend']}")
        else:
            print("ERROR: No LLM backends available!")
            self._print_setup_instructions()
            return False
        
        print(f"LLM backend initialized: {self.current_backend.get_model_info()['backend']}")
        return True
    
    def generate(self, prompt: str, max_tokens: int = 500, 
                temperature: float = 0.7, stop_sequences: List[str] = None) -> str:
        """
        Generate text using the current backend with fallback logic.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Stop sequences for generation
            
        Returns:
            Generated text
        """
        if not self.current_backend:
            raise RuntimeError("No LLM backend available. Please check your configuration.")
        
        # Try current backend
        try:
            return self.current_backend.generate(
                prompt, max_tokens, temperature, stop_sequences
            )
        except Exception as e:
            print(f"WARNING: Current backend failed: {str(e)}")
            
            # Try fallback backends
            for backend in self.fallback_backends:
                if backend == self.current_backend:
                    continue
                    
                try:
                    print(f"Trying fallback backend: {backend.get_model_info()['backend']}")
                    result = backend.generate(prompt, max_tokens, temperature, stop_sequences)
                    
                    # Update current backend if successful
                    self.current_backend = backend
                    print(f"Switched to fallback backend: {backend.get_model_info()['backend']}")
                    return result
                    
                except Exception as fallback_error:
                    print(f"Fallback backend also failed: {str(fallback_error)}")
                    continue
            
            # All backends failed
            raise RuntimeError("All LLM backends failed. Please check your configuration and network connection.")
    
    def get_current_backend_info(self) -> Dict[str, str]:
        """Get information about the current backend."""
        if self.current_backend:
            return self.current_backend.get_model_info()
        return {'backend': 'None', 'status': 'No backend available'}
    
    def is_local_backend(self) -> bool:
        """Check if current backend is local (not API-based)."""
        if not self.current_backend:
            return False
        return isinstance(self.current_backend, LocalGGUFBackend)
    
    def _print_setup_instructions(self):
        """Print setup instructions when no backends are available."""
        print("\n" + "="*80)
        print("LLM BACKEND SETUP REQUIRED")
        print("="*80)
        print("No LLM backends are available. To use the Sanskrit Tutor, you need either:")
        print()
        print("OPTION 1: Local GGUF Model (Recommended)")
        print("  1. Download a GGUF model (e.g., Mistral-7B-Instruct)")
        print("  2. Place it in: user_assets/models/<model_name>.gguf")
        print("  3. Update user_assets/config.yaml:")
        print("     model_path: \"user_assets/models/<model_name>.gguf\"")
        print("     gguf_local: true")
        print("  4. Install: pip install llama-cpp-python")
        print()
        print("OPTION 2: Hosted API (Requires API Key)")
        print("  1. Set gguf_local: false in user_assets/config.yaml")
        print("  2. Set environment variable:")
        print("     export HF_API_KEY=\"hf_your_key_here\"  # Hugging Face")
        print("     OR")
        print("     export OPENAI_API_KEY=\"sk_your_key_here\"  # OpenAI")
        print("  3. Install: pip install requests")
        print()
        print("For more details, see the README file.")
        print("="*80)


def create_llm_manager(config_path: str) -> Optional[LLMManager]:
    """
    Create and initialize LLM manager from configuration.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Initialized LLMManager or None if failed
    """
    try:
        import yaml
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        manager = LLMManager(config)
        
        if manager.initialize_backends():
            return manager
        else:
            return None
            
    except Exception as e:
        print(f"ERROR: Failed to create LLM manager: {str(e)}")
        return None


def main():
    """Command-line interface for testing LLM backends."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test LLM backends for Sanskrit Tutor"
    )
    parser.add_argument(
        "--config",
        default="user_assets/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--prompt",
        default="Translate this Sanskrit text: धर्मे चार्थे च कामे च मोक्षे च भरतर्षभ।",
        help="Test prompt to generate"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    try:
        print("Testing LLM backends...")
        print(f"Config: {args.config}")
        print(f"Test prompt: {args.prompt}")
        print()
        
        # Create LLM manager
        manager = create_llm_manager(args.config)
        if not manager:
            print("Failed to initialize LLM manager.")
            exit(1)
        
        # Print backend info
        info = manager.get_current_backend_info()
        print(f"Current backend: {info['backend']}")
        for key, value in info.items():
            if key != 'backend':
                print(f"  {key}: {value}")
        print()
        
        # Generate response
        print("Generating response...")
        print("-" * 50)
        
        response = manager.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=0.7
        )
        
        print(response)
        print("-" * 50)
        print("Generation completed successfully!")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        exit(1)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
