import os
import re
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Fix typo: LANGCHAIN, not LANCHAIN
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "").strip()
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "").strip()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

# Setup LangChain caption generator
def setup_llm_chain(topic="travel"):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a short and catchy caption for the given topic. "),
        ("user", "Topic: {topic}")
    ])

    llm = ChatGroq(
        model="Gemma2-9b-It",
        groq_api_key=groq_api_key
    )

    return prompt | llm | StrOutputParser()

# Start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! Mention me with a topic like '@CaptionGenBot beach' to get a catchy caption!")

# Help command
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Mention this bot with a topic. Example: '@CaptionGenBot fashion'.")

# Caption generation handler
async def generate_caption(update: Update, context: ContextTypes.DEFAULT_TYPE, topic: str):
    await update.message.reply_text(f"Creating a caption for {topic}...")
    caption = setup_llm_chain(topic).invoke({"topic": topic}).strip()
    await update.message.reply_text(caption)

# Handle messages where bot is mentioned
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message.text
    bot_username = context.bot.username

    if f'@{bot_username}' in msg:
        match = re.search(f'@{bot_username}\\s+(.*)', msg)
        if match and match.group(1).strip():
            await generate_caption(update, context, match.group(1).strip())
        else:
            await update.message.reply_text("Please specify a topic after mentioning me.")

# Main function
def main():
    token = os.getenv("TELEGRAM_API_KEY")
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()