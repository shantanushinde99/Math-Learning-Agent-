"""
History Manager - Manages problem solving history
"""
import os
import json
from datetime import datetime
from typing import List, Dict, Any
from config.settings import settings
from src.utils.logger import app_logger


class HistoryManager:
    """Manages the history of solved problems"""
    
    def __init__(self):
        """Initialize history manager"""
        self.history_dir = settings.HISTORY_DIR
        os.makedirs(self.history_dir, exist_ok=True)
        self.history_file = os.path.join(self.history_dir, "problem_history.json")
        
    def add_to_history(self, result: Dict[str, Any]):
        """
        Add a problem result to history
        
        Args:
            result: Problem solving result
        """
        history = self._load_history()
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "problem": result["problem"],
            "category": result["routing"]["category"],
            "solution": result["solution"]["solution"],
            "model_used": result["solution"]["model_used"]
        }
        
        history.append(entry)
        self._save_history(history)
        app_logger.info("Added to history")
        
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent history
        
        Args:
            limit: Number of recent entries to return
            
        Returns:
            List of history entries
        """
        history = self._load_history()
        return history[-limit:][::-1]  # Most recent first
    
    def search_history(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Search history by keyword
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            Matching history entries
        """
        history = self._load_history()
        keyword_lower = keyword.lower()
        
        results = [
            entry for entry in history
            if keyword_lower in entry["problem"].lower() or
               keyword_lower in entry["category"].lower()
        ]
        
        return results[::-1]  # Most recent first
    
    def clear_history(self):
        """Clear all history"""
        self._save_history([])
        app_logger.info("History cleared")
        
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load history from file"""
        if not os.path.exists(self.history_file):
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            app_logger.error(f"Error loading history: {str(e)}")
            return []
    
    def _save_history(self, history: List[Dict[str, Any]]):
        """Save history to file"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            app_logger.error(f"Error saving history: {str(e)}")
