# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os

from dotenv import dotenv_values, load_dotenv

load_dotenv()

# %%
config = dotenv_values(".env")
os.environ["ANTHROPIC_API_KEY"] = config["ANTHROPIC_API_KEY"]

# %%
# %load_ext jupyter_ai_magics

# %%
# %config AiMagics.default_language_model = "anthropic-chat:claude-3-5-sonnet-20241022"

# %%
# %ai help
