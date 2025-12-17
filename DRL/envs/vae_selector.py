import numpy as np
from envs.observer import Observer

class VAEDCSelector:
    """Chọn DC dựa trên GenAI model"""
    
    def __init__(self, genai_model):
        """
        Args:
            genai_model: VAEModel instance (trained)
        """
        self.model = genai_model
    
    def select_dc(self, dcs, sfc_manager):
        """
        Chọn DC có value cao nhất
        
        Args:
            dcs: List of DataCenter objects
            sfc_manager: SFC_Manager object
        
        Returns:
            int: DC ID được chọn
        """
        # Get all DC states
        dc_states = Observer.get_all_dc_states(dcs, sfc_manager)
        
        # Predict values
        values = self.model.predict_dc_values(dc_states)
        
        # Select DC with max value
        selected_idx = np.argmax(values)
        
        return selected_idx
    
    def get_dc_ranking(self, dcs, sfc_manager):
        """
        Lấy ranking của tất cả DC
        
        Returns:
            List[int]: DC IDs sorted by value (descending)
        """
        dc_states = Observer.get_all_dc_states(dcs, sfc_manager)
        values = self.model.predict_dc_values(dc_states)
        
        # Sort indices by value
        ranked_indices = np.argsort(values)[::-1]  # Descending
        
        return ranked_indices.tolist()