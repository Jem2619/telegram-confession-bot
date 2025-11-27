import os
import asyncio
import logging
import sys

# Добавляем текущую директорию в путь, чтобы Python видел модули
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bot import dp, bot, main
from dotenv import load_dotenv

# Настройка логирования для Render
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

load_dotenv()

# Получаем переменные окружения
WEBHOOK_HOST = os.getenv("WEBHOOK_HOST")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")
# Если переменные не заданы, логируем предупреждение (но не падаем сразу, чтобы видеть логи)
if not WEBHOOK_HOST or not WEBHOOK_SECRET:
    logger.error("ОШИБКА: Не заданы WEBHOOK_HOST или WEBHOOK_SECRET в Environment Variables!")

WEBHOOK_PATH = f"/webhook/{WEBHOOK_SECRET}"
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"
WEB_SERVER_PORT = int(os.getenv("PORT", 8080))
WEB_SERVER_HOST = "0.0.0.0"

async def on_startup(bot_instance, webhook_url):
    """Устанавливает вебхук при старте"""
    logger.info(f"Установка Webhook по URL: {webhook_url}")
    # ИСПРАВЛЕНИЕ: Вызываем метод у bot, а не у dispatcher.bot
    await bot_instance.set_webhook(
        url=webhook_url,
        secret_token=WEBHOOK_SECRET,
        drop_pending_updates=True
    )

async def on_shutdown(bot_instance):
    """Удаляет вебхук при остановке"""
    logger.info("Удаление Webhook")
    await bot_instance.delete_webhook()

async def start_webhook():
    # 1. Инициализация БД (вызов функции main из bot.py)
    # Важно: убедитесь, что в bot.py в функции main() НЕТ запуска dp.start_polling()
    await main()

    # 2. Установка вебхука
    # Передаем объект 'bot' напрямую
    await on_startup(bot, WEBHOOK_URL)

    try:
        logger.info(f"Запуск Webhook-сервера на порту {WEB_SERVER_PORT}...")
        
        # 3. Запуск сервера
        # Метод start_webhook сам запускает бесконечный цикл aiohttp
        await dp.start_webhook(
            bot=bot, # Передаем объект бота сюда
            listen=WEB_SERVER_HOST,
            port=WEB_SERVER_PORT,
            url_path=WEBHOOK_PATH,
            secret_token=WEBHOOK_SECRET
        )
    except Exception as e:
        logger.error(f"Критическая ошибка запуска Webhook-сервера: {e}")
    finally:
        # 4. Очистка при выходе
        await on_shutdown(bot)

if __name__ == "__main__":
    try:
        asyncio.run(start_webhook())
    except KeyboardInterrupt:
        logger.info("Бот остановлен")
