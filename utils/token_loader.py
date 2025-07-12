#!/usr/bin/env python3
"""
Token Loader - Load tất cả tokens từ file token.env
"""

import os
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class TokenLoader:
    def __init__(self, token_file: str = "token.env"):
        self.token_file = token_file
        self.tokens = {}
        self.gemini_keys = []
        self.hf_token = None
        self.github_token = None
        self.load_all_tokens()
        
    def load_all_tokens(self):
        """Load tất cả tokens từ file"""
        try:
            if not os.path.exists(self.token_file):
                logger.warning(f"Token file {self.token_file} not found")
                return
                
            with open(self.token_file, "r") as f:
                content = f.read()
                
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                # Parse different token types
                if line.startswith('HF_TOKEN='):
                    self.hf_token = line.split('=', 1)[1].strip()
                    self.tokens['hf_token'] = self.hf_token
                    logger.info("Loaded HuggingFace token")
                    
                elif line.startswith('Gemini_API'):
                    # Extract Gemini API key from "Gemini_API = KEY" format
                    api_key = line.split('=', 1)[1].strip()
                    if api_key and api_key not in self.gemini_keys:
                        self.gemini_keys.append(api_key)
                        self.tokens[f'gemini_key_{len(self.gemini_keys)}'] = api_key
                        
                elif line.startswith('PAT_git='):
                    self.github_token = line.split('=', 1)[1].strip()
                    self.tokens['github_token'] = self.github_token
                    logger.info("Loaded GitHub token")
                    
                elif 'AIzaSy' in line:
                    # Direct Gemini API key
                    if line and line not in self.gemini_keys:
                        self.gemini_keys.append(line)
                        self.tokens[f'gemini_key_{len(self.gemini_keys)}'] = line
                        
            logger.info(f"Loaded {len(self.gemini_keys)} Gemini API keys")
            logger.info(f"Total tokens loaded: {len(self.tokens)}")
            
        except Exception as e:
            logger.error(f"Error loading tokens: {e}")
            
    def get_gemini_keys(self) -> List[str]:
        """Lấy danh sách Gemini API keys"""
        return self.gemini_keys.copy()
        
    def get_hf_token(self) -> Optional[str]:
        """Lấy HuggingFace token"""
        return self.hf_token
        
    def get_github_token(self) -> Optional[str]:
        """Lấy GitHub token"""
        return self.github_token
        
    def get_all_tokens(self) -> Dict[str, str]:
        """Lấy tất cả tokens"""
        return self.tokens.copy()
        
    def validate_tokens(self) -> Dict[str, bool]:
        """Validate các tokens"""
        validation = {
            'hf_token': bool(self.hf_token),
            'github_token': bool(self.github_token),
            'gemini_keys': len(self.gemini_keys) > 0
        }
        
        logger.info(f"Token validation: {validation}")
        return validation

# Global instance
token_loader = TokenLoader() 