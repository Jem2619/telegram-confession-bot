import os
import logging
import sys
from aiohttp import web
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

# Добавляем путь к текущей директории
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bot import dp, bot, main
from dotenv import load_dotenv

# Настройка логов
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

load_dotenv()

# Настройки Webhook
WEBHOOK_HOST = os.getenv("WEBHOOK_HOST")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")
WEBHOOK_PATH = f"/webhook/{WEBHOOK_SECRET}"
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

# Настройки сервера
WEB_SERVER_HOST = "0.0.0.0"
WEB_SERVER_PORT = int(os.getenv("PORT", 8080))

async def on_startup(app):
    """Действия при запуске приложения"""
    logger.info("Инициализация базы данных и настроек...")
    # Запускаем инициализацию из bot.py (создание таблиц и т.д.)
    # Важно: убедись, что main() в bot.py НЕ запускает polling!
    await main()

    logger.info(f"Установка Webhook: {WEBHOOK_URL}")
    await bot.set_webhook(
        url=WEBHOOK_URL,
        secret_token=WEBHOOK_SECRET,
        drop_pending_updates=True
    )

async def on_shutdown(app):
    """Действия при остановке"""
    logger.info("Удаление Webhook")
    await bot.delete_webhook()

def start_webhook_server():
    # 1. Создаем веб-приложение
    app = web.Application()

    # 2. Настраиваем обработчик запросов от Telegram
    webhook_requests_handler = SimpleRequestHandler(
        dispatcher=dp,
        bot=bot,
        secret_token=WEBHOOK_SECRET,
    )
    
    # 3. Регистрируем путь для вебхука
    webhook_requests_handler.register(app, path=WEBHOOK_PATH)

    # 4. Настраиваем приложение (добавляем диспетчер и бота в контекст)
    setup_application(app, dp, bot=bot)

    # 5. Добавляем функции старта и остановки
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    # 6. Запускаем сервер
    logger.info(f"Запуск сервера aiohttp на порту {WEB_SERVER_PORT}...")
    web.run_app(app, host=WEB_SERVER_HOST, port=WEB_SERVER_PORT)

if __name__ == "__main__":
    try:
        start_webhook_server()
    except KeyboardInterrupt:
        pass
