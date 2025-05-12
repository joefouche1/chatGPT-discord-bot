import discord
from discord import app_commands
from discord.ext import commands
from PIL import Image, ImageDraw, ImageFont
import io
import aiohttp
import textwrap

class Meme(commands.Cog):
    def __init__(self, client):
        self.client = client
        self.meme_group = app_commands.Group(name="meme", description="Meme generation commands")

    @app_commands.command(name="generate", description="Generate a meme based on your prompt")
    async def generate_meme(self, interaction: discord.Interaction, top_text: str, bottom_text: str):
        await interaction.response.defer(thinking=True)
        
        try:
            # Generate base image with DALL-E
            prompt = f"A meme-worthy image of {top_text.lower()} and {bottom_text.lower()}, funny, high contrast, good for memes"
            image_url = await self.client.draw(prompt)
            
            # Download the generated image
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    image_data = await resp.read()
            
            # Create PIL Image from bytes
            img = Image.open(io.BytesIO(image_data))
            draw = ImageDraw.Draw(img)
            
            # Load a meme font (you'll need to add this to your project)
            font_size = int(img.width / 8)
            try:
                font = ImageFont.truetype("impact.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Add top text
            top_lines = textwrap.wrap(top_text.upper(), width=20)
            y = 10
            for line in top_lines:
                line_width = draw.textlength(line, font=font)
                x = (img.width - line_width) / 2
                # Draw black outline
                for adj in range(-2, 3):
                    for adj2 in range(-2, 3):
                        draw.text((x+adj, y+adj2), line, font=font, fill="black")
                # Draw white text
                draw.text((x, y), line, font=font, fill="white")
                y += font_size
            
            # Add bottom text
            bottom_lines = textwrap.wrap(bottom_text.upper(), width=20)
            y = img.height - (len(bottom_lines) * font_size) - 10
            for line in bottom_lines:
                line_width = draw.textlength(line, font=font)
                x = (img.width - line_width) / 2
                # Draw black outline
                for adj in range(-2, 3):
                    for adj2 in range(-2, 3):
                        draw.text((x+adj, y+adj2), line, font=font, fill="black")
                # Draw white text
                draw.text((x, y), line, font=font, fill="white")
                y += font_size
            
            # Convert back to bytes for Discord
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Create Discord file and embed
            file = discord.File(img_byte_arr, filename="meme.png")
            embed = discord.Embed(title="Your Generated Meme")
            embed.set_image(url="attachment://meme.png")
            
            await interaction.followup.send(file=file, embed=embed)
            
        except Exception as e:
            await interaction.followup.send(f"Error generating meme: {str(e)}")

    async def cog_load(self):
        self.client.tree.add_command(self.meme_group)

def setup(client):
    client.add_cog(Meme(client)) 