from typing import Dict, Any

class CallStatsManager:
    def __init__(self):
        # In-memory storage: call_id -> stats_dict
        self.active_calls: Dict[str, Dict[str, Any]] = {}

    def update_stats(self, call_id: str, label: str, confidence: float) -> Dict[str, Any]:
        if call_id not in self.active_calls:
            self.active_calls[call_id] = {
                "total_chunks": 0,
                "ai_chunks": 0,
                "human_chunks": 0,
                "risk_score": 0.0,
                "verdict": "HUMAN"
            }
        
        stats = self.active_calls[call_id]
        stats["total_chunks"] += 1
        
        # User Logic: AI (FAKE) vs HUMAN (REAL)
        if label == "FAKE":
            stats["ai_chunks"] += 1
        else:
            stats["human_chunks"] += 1
            
        # Compute Risk Score: ai_chunks / total_chunks
        if stats["total_chunks"] > 0:
            stats["risk_score"] = float(stats["ai_chunks"]) / stats["total_chunks"]
            
        # Final Verdict Logic as per requirements
        # risk < 0.3 -> Human, risk >= 0.3 -> Likely AI
        stats["verdict"] = "LIKELY AI" if stats["risk_score"] >= 0.3 else "HUMAN"
        
        return stats

    def get_stats(self, call_id: str):
        return self.active_calls.get(call_id, {})

# Global instance
call_stats = CallStatsManager()
