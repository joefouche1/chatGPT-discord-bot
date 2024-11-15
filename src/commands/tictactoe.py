import discord
from discord import app_commands
from discord.ext import commands

class TicTacToe(commands.Cog):
    def __init__(self, client):
        self.client = client
        # Initialize game state variables

    @app_commands.command(name="tictactoe", description="Start a game of Tic-Tac-Toe")
    async def tictactoe(self, interaction: discord.Interaction, opponent: discord.Member):
        # Logic to start and manage the game
        await interaction.response.send_message(f"Starting Tic-Tac-Toe between {interaction.user.mention} and {opponent.mention}!")
        
def setup(client):
    client.add_cog(TicTacToe(client)) 