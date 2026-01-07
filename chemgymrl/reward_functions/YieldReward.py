"""
Custom yield-based reward function for Diels-Alder reaction
"""

class YieldReward:
    """
    Reward function based on product yield.
    Tracks how much product is formed relative to the limiting reactant.
    """
    
    def __init__(self, target_material="Cyanocyclohexene", limiting_reactant="Butadiene"):
        """
        Args:
            target_material: Name of the product to maximize
            limiting_reactant: Name of the limiting reactant (usually has fewer moles)
        """
        self.target_material = target_material
        self.limiting_reactant = limiting_reactant
        self.initial_limiting_moles = None
        self.previous_yield = 0.0
    
    def reset(self):
        """Reset tracking variables at episode start"""
        self.initial_limiting_moles = None
        self.previous_yield = 0.0
    
    def __call__(self, vessels, target_material):
        """
        Calculate reward based on yield increase.
        
        Args:
            vessels: Tuple of vessels (from shelf.get_working_vessels())
            target_material: Target material name (from GenBench)
            
        Returns:
            float: Reward value
        """
        # Get the reaction vessel (first vessel in the tuple)
        vessel = vessels[0]
        
        materials = vessel.get_material_dataframe()
        
        # Keep trying to initialize until we have the limiting reactant with a real amount
        if self.initial_limiting_moles is None:
            if self.limiting_reactant in materials.index:
                amount = materials.loc[self.limiting_reactant, 'Amount']
                if amount > 1e-12:
                    self.initial_limiting_moles = amount
                    print(f"[YieldReward] Initialized with {self.limiting_reactant}: {amount:.6f} mol")
            
            # If we still don't have it, return 0 reward
            if self.initial_limiting_moles is None:
                return 0.0
        
        # Get current product moles (may not exist yet)
        if self.target_material in materials.index:
            product_moles = materials.loc[self.target_material, 'Amount']
        else:
            product_moles = 0.0
        
        # Calculate yield (0 to 1 scale)
        current_yield = product_moles / self.initial_limiting_moles
        
        # Dense reward: give reward for increase in yield
        reward = current_yield - self.previous_yield
        
        # Update previous yield
        self.previous_yield = current_yield
        
        return reward


class YieldRewardWithBonus:
    """
    Yield-based reward with terminal bonus for high conversion.
    """
    
    def __init__(self, target_material="Cyanocyclohexene", 
                 limiting_reactant="Butadiene",
                 high_yield_threshold=0.8,
                 terminal_bonus=1.0):
        """
        Args:
            target_material: Name of the product
            limiting_reactant: Name of the limiting reactant
            high_yield_threshold: Yield fraction to trigger bonus (e.g., 0.8 = 80%)
            terminal_bonus: Bonus reward for achieving high yield
        """
        self.target_material = target_material
        self.limiting_reactant = limiting_reactant
        self.initial_limiting_moles = None
        self.previous_yield = 0.0
        self.high_yield_threshold = high_yield_threshold
        self.terminal_bonus = terminal_bonus
        self.bonus_given = False
    
    def reset(self):
        """Reset tracking variables"""
        self.initial_limiting_moles = None
        self.previous_yield = 0.0
        self.bonus_given = False
    
    def __call__(self, vessels, target_material):
        """
        Calculate reward with bonus.
        
        Args:
            vessels: Tuple of vessels (from shelf.get_working_vessels())
            target_material: Target material name (from GenBench)
            
        Returns:
            float: Reward value
        """
        # Get the reaction vessel (first vessel)
        vessel = vessels[0]
        
        materials = vessel.get_material_dataframe()
        
        # Keep trying to initialize until we have the limiting reactant with a real amount
        if self.initial_limiting_moles is None:
            if self.limiting_reactant in materials.index:
                amount = materials.loc[self.limiting_reactant, 'Amount']
                if amount > 1e-12:
                    self.initial_limiting_moles = amount
                    print(f"[YieldRewardWithBonus] Initialized with {self.limiting_reactant}: {amount:.6f} mol")
            
            # If we still don't have it, return 0 reward
            if self.initial_limiting_moles is None:
                return 0.0
        
        # Calculate yield (product may not exist yet)
        if self.target_material in materials.index:
            product_moles = materials.loc[self.target_material, 'Amount']
        else:
            product_moles = 0.0
            
        current_yield = product_moles / self.initial_limiting_moles
        
        # Base reward: increase in yield
        reward = current_yield - self.previous_yield
        
        # Add bonus if threshold crossed
        if current_yield >= self.high_yield_threshold and not self.bonus_given:
            reward += self.terminal_bonus
            self.bonus_given = True
        
        # Update tracking
        self.previous_yield = current_yield
        
        return reward
    