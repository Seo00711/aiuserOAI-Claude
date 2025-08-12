from .cog import GPT5Assistant

async def setup(bot):
    await bot.add_cog(GPT5Assistant(bot))
EOF < /dev/null