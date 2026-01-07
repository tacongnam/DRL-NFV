class Statistics:
    @staticmethod
    def calculate(completed_history, dropped_history, request_cursor):
        total_completed = len(completed_history)
        total_dropped = len(dropped_history)
        total_processed = total_completed + total_dropped
        
        acc_ratio = (total_completed / total_processed * 100.0) if total_processed > 0 else 0.0
        
        avg_delay = 0.0
        if total_completed > 0:
            avg_delay = sum(r.elapsed_time for r in completed_history) / total_completed
            
        return {
            'acceptance_ratio': acc_ratio,
            'avg_e2e_delay': avg_delay,
            'total_generated': request_cursor,
            'active_count': 0,
            'completed_count': total_completed,
            'dropped_count': total_dropped
        }