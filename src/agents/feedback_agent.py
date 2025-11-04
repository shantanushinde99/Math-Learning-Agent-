"""
Feedback Agent - Handles human feedback and learning
"""
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.utils.logger import app_logger
from config.settings import settings


class FeedbackAgent:
    """Agent responsible for collecting and processing human feedback"""
    
    def __init__(self):
        """Initialize feedback agent"""
        self.feedback_dir = settings.FEEDBACK_DIR
        os.makedirs(self.feedback_dir, exist_ok=True)
        self.feedback_file = os.path.join(self.feedback_dir, "feedback_log.json")
        
    def collect_feedback(
        self,
        problem: str,
        category: str,
        solution: str,
        rating: int,
        comments: str = "",
        correct_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Collect and store human feedback
        
        Args:
            problem: The math problem
            category: Problem category
            solution: The solution provided
            rating: Rating from 1-5
            comments: Optional feedback comments
            correct_answer: Optional correct answer if solution was wrong
            
        Returns:
            Feedback record
        """
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "problem": problem,
            "category": category,
            "solution": solution,
            "rating": rating,
            "comments": comments,
            "correct_answer": correct_answer,
            "approved": rating >= 4
        }
        
        # Load existing feedback
        feedback_data = self._load_feedback()
        
        # Add new feedback
        feedback_data.append(feedback_entry)
        
        # Save feedback
        self._save_feedback(feedback_data)
        
        app_logger.info(f"Feedback collected: Rating {rating}/5")
        return feedback_entry
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about collected feedback"""
        feedback_data = self._load_feedback()
        
        if not feedback_data:
            return {
                "total_feedback": 0,
                "average_rating": 0,
                "approved_count": 0,
                "category_stats": {}
            }
        
        total = len(feedback_data)
        ratings = [f["rating"] for f in feedback_data]
        approved = len([f for f in feedback_data if f.get("approved", False)])
        
        # Category statistics
        category_stats = {}
        for entry in feedback_data:
            cat = entry["category"]
            if cat not in category_stats:
                category_stats[cat] = {
                    "count": 0,
                    "total_rating": 0,
                    "approved": 0
                }
            category_stats[cat]["count"] += 1
            category_stats[cat]["total_rating"] += entry["rating"]
            if entry.get("approved", False):
                category_stats[cat]["approved"] += 1
        
        # Calculate averages
        for cat in category_stats:
            stats = category_stats[cat]
            stats["average_rating"] = stats["total_rating"] / stats["count"]
        
        return {
            "total_feedback": total,
            "average_rating": sum(ratings) / total,
            "approved_count": approved,
            "approval_rate": approved / total,
            "category_stats": category_stats
        }
    
    def get_learning_insights(self) -> List[Dict[str, Any]]:
        """Get insights for improving the system based on feedback"""
        feedback_data = self._load_feedback()
        
        # Find low-rated solutions
        low_rated = [
            f for f in feedback_data
            if f["rating"] < 3
        ]
        
        # Group by category
        insights = []
        category_issues = {}
        
        for entry in low_rated:
            cat = entry["category"]
            if cat not in category_issues:
                category_issues[cat] = []
            category_issues[cat].append({
                "problem": entry["problem"],
                "comments": entry["comments"],
                "correct_answer": entry.get("correct_answer")
            })
        
        for category, issues in category_issues.items():
            insights.append({
                "category": category,
                "issue_count": len(issues),
                "examples": issues[:3]  # Show top 3 examples
            })
        
        return insights
    
    def _load_feedback(self) -> List[Dict[str, Any]]:
        """Load feedback from file"""
        if not os.path.exists(self.feedback_file):
            return []
        
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            app_logger.error(f"Error loading feedback: {str(e)}")
            return []
    
    def _save_feedback(self, feedback_data: List[Dict[str, Any]]):
        """Save feedback to file"""
        try:
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(feedback_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            app_logger.error(f"Error saving feedback: {str(e)}")
