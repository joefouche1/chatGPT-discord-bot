import discord
from discord import app_commands
from discord.ext import commands

class Meme(commands.Cog):
    def __init__(self, client):
        self.client = client

    @app_commands.command(name="meme", description="Generate a meme based on your prompt")
    async def meme(self, interaction: discord.Interaction, top_text: str, bottom_text: str):
        prompt = f"Meme with top text '{top_text}' and bottom text '{bottom_text}'"
        image_url = await self.client.draw(prompt)
        embed = discord.Embed(title="Your Meme")
        embed.set_image(url=image_url)
        await interaction.response.send_message(embed=embed)

def setup(client):
    client.add_cog(Meme(client)) 