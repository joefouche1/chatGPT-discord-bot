"""
Context management commands for Discord bot.
Provides slash commands for managing per-channel conversation contexts.
"""

import discord
from discord import app_commands
from discord.ext import commands
from datetime import datetime
import json
from typing import Optional
from utils.log import logger


class ContextCommands(commands.Cog):
    """Cog for context management commands"""
    
    def __init__(self, bot):
        self.bot = bot
    
    @app_commands.command(name="context_info", description="Get information about this channel's conversation context")
    async def context_info(self, interaction: discord.Interaction):
        """Display information about the current channel's conversation context"""
        channel_id = str(interaction.channel_id)
        
        try:
            stats = await self.bot.conversation_manager.get_channel_stats(channel_id)
            
            if stats:
                embed = discord.Embed(
                    title="📊 Conversation Context Info",
                    description=f"Context details for {stats.get('channel_name', 'this channel')}",
                    color=discord.Color.blue()
                )
                
                embed.add_field(name="Channel", value=f"<#{channel_id}>", inline=True)
                embed.add_field(name="Guild", value=stats.get('guild_name', 'N/A'), inline=True)
                embed.add_field(name="Messages", value=stats.get('message_count', 0), inline=True)
                
                created_at = stats.get('created_at')
                if isinstance(created_at, datetime):
                    created_str = created_at.strftime("%Y-%m-%d %H:%M")
                else:
                    created_str = str(created_at)[:16]
                
                last_activity = stats.get('last_activity')
                if isinstance(last_activity, datetime):
                    last_str = last_activity.strftime("%Y-%m-%d %H:%M")
                else:
                    last_str = str(last_activity)[:16]
                
                embed.add_field(name="Context Created", value=created_str, inline=True)
                embed.add_field(name="Last Activity", value=last_str, inline=True)
                embed.add_field(name="History Length", value=stats.get('history_length', 0), inline=True)
                
                if stats.get('learned_preferences', 0) > 0:
                    embed.add_field(name="Learned Preferences", value=stats.get('learned_preferences', 0), inline=True)
                if stats.get('important_facts', 0) > 0:
                    embed.add_field(name="Important Facts", value=stats.get('important_facts', 0), inline=True)
                if stats.get('known_users', 0) > 0:
                    embed.add_field(name="Known Users", value=stats.get('known_users', 0), inline=True)
                
                await interaction.response.send_message(embed=embed, ephemeral=True)
            else:
                await interaction.response.send_message(
                    "No conversation context exists for this channel yet. Start chatting to create one!",
                    ephemeral=True
                )
        except Exception as e:
            logger.error(f"Error getting context info: {e}")
            await interaction.response.send_message(
                "An error occurred while fetching context information.",
                ephemeral=True
            )
    
    @app_commands.command(name="context_clear", description="Clear this channel's conversation history")
    async def context_clear(self, interaction: discord.Interaction):
        """Clear the conversation context for the current channel"""
        channel_id = str(interaction.channel_id)
        
        try:
            # Check if user has manage_messages permission
            if not interaction.user.guild_permissions.manage_messages:
                await interaction.response.send_message(
                    "You need the 'Manage Messages' permission to clear conversation context.",
                    ephemeral=True
                )
                return
            
            await self.bot.conversation_manager.clear_context(channel_id)
            
            embed = discord.Embed(
                title="🗑️ Context Cleared",
                description=f"Conversation history for <#{channel_id}> has been cleared.",
                color=discord.Color.green()
            )
            embed.set_footer(text=f"Cleared by {interaction.user.name}")
            
            await interaction.response.send_message(embed=embed)
            logger.info(f"Context cleared for channel {channel_id} by {interaction.user.name}")
            
        except Exception as e:
            logger.error(f"Error clearing context: {e}")
            await interaction.response.send_message(
                "An error occurred while clearing the context.",
                ephemeral=True
            )
    
    @app_commands.command(name="context_list", description="List all active conversation contexts")
    @app_commands.default_permissions(administrator=True)
    async def context_list(self, interaction: discord.Interaction):
        """List all active conversation contexts (Admin only)"""
        try:
            contexts = self.bot.conversation_manager.get_all_contexts_info()
            
            if not contexts:
                await interaction.response.send_message(
                    "No active conversation contexts found.",
                    ephemeral=True
                )
                return
            
            embed = discord.Embed(
                title="📋 Active Conversation Contexts",
                description=f"Found {len(contexts)} active context(s)",
                color=discord.Color.blue()
            )
            
            for ctx in contexts[:25]:  # Discord embed field limit
                channel_id = ctx.get('channel_id')
                channel_name = ctx.get('channel_name', 'Unknown')
                message_count = ctx.get('message_count', 0)
                last_activity = ctx.get('last_activity', 'Unknown')
                
                if isinstance(last_activity, str):
                    last_str = last_activity[:16]
                else:
                    last_str = str(last_activity)[:16]
                
                embed.add_field(
                    name=f"#{channel_name}",
                    value=f"ID: {channel_id}\nMessages: {message_count}\nLast: {last_str}",
                    inline=True
                )
            
            if len(contexts) > 25:
                embed.set_footer(text=f"Showing first 25 of {len(contexts)} contexts")
            
            await interaction.response.send_message(embed=embed, ephemeral=True)
            
        except Exception as e:
            logger.error(f"Error listing contexts: {e}")
            await interaction.response.send_message(
                "An error occurred while listing contexts.",
                ephemeral=True
            )
    
    @app_commands.command(name="context_export", description="Export this channel's conversation history")
    async def context_export(self, interaction: discord.Interaction):
        """Export the conversation history for the current channel"""
        channel_id = str(interaction.channel_id)
        
        try:
            # Check if user has appropriate permissions
            if not interaction.user.guild_permissions.manage_messages:
                await interaction.response.send_message(
                    "You need the 'Manage Messages' permission to export conversation context.",
                    ephemeral=True
                )
                return
            
            # Defer the response as this might take a moment
            await interaction.response.defer(ephemeral=True)
            
            # Get the conversation history
            history = await self.bot.conversation_manager.get_history(channel_id)
            stats = await self.bot.conversation_manager.get_channel_stats(channel_id)
            
            if not history:
                await interaction.followup.send(
                    "No conversation history found for this channel.",
                    ephemeral=True
                )
                return
            
            # Create export data
            export_data = {
                "channel_id": channel_id,
                "channel_name": stats.get('channel_name', 'Unknown') if stats else 'Unknown',
                "guild_name": stats.get('guild_name', 'Unknown') if stats else 'Unknown',
                "export_timestamp": datetime.now().isoformat(),
                "exported_by": str(interaction.user),
                "message_count": len(history),
                "conversation_history": history
            }
            
            # Convert to JSON
            json_data = json.dumps(export_data, indent=2, default=str)
            
            # Create a file
            filename = f"conversation_{channel_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            file = discord.File(
                io.BytesIO(json_data.encode()),
                filename=filename
            )
            
            embed = discord.Embed(
                title="📤 Conversation Exported",
                description=f"Exported {len(history)} messages from <#{channel_id}>",
                color=discord.Color.green()
            )
            embed.set_footer(text=f"Exported by {interaction.user.name}")
            
            await interaction.followup.send(
                embed=embed,
                file=file,
                ephemeral=True
            )
            
            logger.info(f"Context exported for channel {channel_id} by {interaction.user.name}")
            
        except Exception as e:
            logger.error(f"Error exporting context: {e}")
            await interaction.followup.send(
                "An error occurred while exporting the context.",
                ephemeral=True
            )
    
    @app_commands.command(name="context_cleanup", description="Clean up inactive conversation contexts")
    @app_commands.default_permissions(administrator=True)
    async def context_cleanup(self, interaction: discord.Interaction):
        """Clean up inactive conversation contexts (Admin only)"""
        try:
            # Perform cleanup
            removed_count = await self.bot.conversation_manager.cleanup_inactive_contexts()
            
            embed = discord.Embed(
                title="🧹 Context Cleanup Complete",
                description=f"Removed {removed_count} inactive context(s)",
                color=discord.Color.green()
            )
            
            if removed_count > 0:
                embed.add_field(
                    name="Note",
                    value="Contexts inactive for more than 7 days have been removed.",
                    inline=False
                )
            
            embed.set_footer(text=f"Cleanup performed by {interaction.user.name}")
            
            await interaction.response.send_message(embed=embed, ephemeral=True)
            logger.info(f"Context cleanup performed by {interaction.user.name}, removed {removed_count} contexts")
            
        except Exception as e:
            logger.error(f"Error during context cleanup: {e}")
            await interaction.response.send_message(
                "An error occurred during context cleanup.",
                ephemeral=True
            )


async def setup(bot):
    """Setup function to add the cog to the bot"""
    await bot.add_cog(ContextCommands(bot))


# Add missing import for io
import io
