import re
import random
from datetime import datetime
from typing import Dict, Any, Optional
import discord
import logging

logger = logging.getLogger("red.gpt5assistant.variables")


class VariableProcessor:
    """Processes dynamic variables in text strings, similar to aiuser functionality"""
    
    def __init__(self):
        self.variable_patterns = {
            'botname': r'\{botname\}',
            'username': r'\{username\}',
            'displayname': r'\{displayname\}', 
            'authorname': r'\{authorname\}',
            'servername': r'\{servername\}',
            'channelname': r'\{channelname\}',
            'serveremojis': r'\{serveremojis\}',
            'date': r'\{date\}',
            'time': r'\{time\}',
            'timestamp': r'\{timestamp\}',
            'random': r'\{random\}',
            'randomnumber': r'\{randomnumber\}',  # alias for compatibility
        }
    
    async def process_variables(
        self,
        text: str,
        bot: Optional[discord.Client] = None,
        guild: Optional[discord.Guild] = None,
        channel: Optional[discord.TextChannel] = None,
        user: Optional[discord.User] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process all dynamic variables in the given text"""
        
        if not text or '{' not in text:
            return text
        
        # Create variable values
        variables = await self._build_variable_dict(bot, guild, channel, user, context)
        
        # Replace variables in text
        processed_text = text
        for var_name, pattern in self.variable_patterns.items():
            if var_name in variables:
                try:
                    processed_text = re.sub(pattern, str(variables[var_name]), processed_text, flags=re.IGNORECASE)
                except Exception as e:
                    logger.warning(f"Error processing variable {var_name}: {e}")
        
        return processed_text
    
    async def _build_variable_dict(
        self,
        bot: Optional[discord.Client] = None,
        guild: Optional[discord.Guild] = None,
        channel: Optional[discord.TextChannel] = None,
        user: Optional[discord.User] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build dictionary of variable name -> value mappings"""
        
        variables = {}
        now = datetime.now()
        
        # Bot-related variables
        if bot:
            variables['botname'] = bot.user.display_name if bot.user else "Assistant"
        
        # User-related variables
        if user:
            variables['username'] = user.name
            variables['displayname'] = user.display_name
            variables['authorname'] = user.display_name  # alias for displayname
        
        # Server-related variables
        if guild:
            variables['servername'] = guild.name
            
            # Get random server emoji
            if guild.emojis:
                random_emoji = random.choice(guild.emojis)
                variables['serveremojis'] = str(random_emoji)
            else:
                variables['serveremojis'] = "😊"  # fallback
        
        # Channel-related variables
        if channel:
            variables['channelname'] = channel.name
        
        # Time-related variables
        variables['date'] = now.strftime("%Y-%m-%d")
        variables['time'] = now.strftime("%H:%M:%S")
        variables['timestamp'] = str(int(now.timestamp()))
        
        # Random variables
        variables['random'] = str(random.randint(1, 1000))
        variables['randomnumber'] = variables['random']  # alias
        
        # Context variables (custom additions)
        if context:
            variables.update(context)
        
        return variables
    
    def get_available_variables(self) -> Dict[str, str]:
        """Get list of available variables with descriptions"""
        return {
            'botname': 'Bot\'s display name',
            'username': 'User\'s username',
            'displayname': 'User\'s display name in the server',
            'authorname': 'Message author\'s name (alias for displayname)',
            'servername': 'Server/guild name',
            'channelname': 'Current channel name',
            'serveremojis': 'Random emoji from the server',
            'date': 'Current date (YYYY-MM-DD)',
            'time': 'Current time (HH:MM:SS)',
            'timestamp': 'Unix timestamp',
            'random': 'Random number 1-1000',
            'randomnumber': 'Random number 1-1000 (alias for random)',
        }
    
    def has_variables(self, text: str) -> bool:
        """Check if text contains any variables"""
        if not text:
            return False
        
        for pattern in self.variable_patterns.values():
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def extract_variables(self, text: str) -> list[str]:
        """Extract all variables found in text"""
        if not text:
            return []
        
        found_variables = []
        for var_name, pattern in self.variable_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                found_variables.append(var_name)
        
        return found_variables
    
    def validate_variables(self, text: str) -> tuple[bool, list[str]]:
        """Validate that all variables in text are supported"""
        if not text:
            return True, []
        
        # Find all variable-like patterns
        all_vars = re.findall(r'\{([^}]+)\}', text, re.IGNORECASE)
        
        unsupported = []
        for var in all_vars:
            if var.lower() not in [name.lower() for name in self.variable_patterns.keys()]:
                unsupported.append(var)
        
        return len(unsupported) == 0, unsupported


# Global instance
variable_processor = VariableProcessor()