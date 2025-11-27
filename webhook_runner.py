import os
import asyncio
import logging
from bot import dp, bot, main # Импортируем dp, bot, и main из твоего основного файла
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Основной URL твоего сервиса на Render, который будет предоставлен позже
WEBHOOK_HOST = os.getenv("WEBHOOK_HOST") 
# Секретный токен для проверки, что запрос пришел именно от Telegram
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET") 
# Путь, по которому Telegram будет отправлять обновления (может быть любым)
WEBHOOK_PATH = f"/webhook/{WEBHOOK_SECRET}"
# Полный URL для установки вебхука
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}" 
# Локальный порт, на котором будет слушать сервер Render
WEB_SERVER_PORT = int(os.getenv("PORT", 8080))
WEB_SERVER_HOST = "0.0.0.0"

logger = logging.getLogger(__name__)

async def on_startup(dispatcher, webhook_url):
    # При запуске устанавливаем Webhook
    logging.info(f"Установка Webhook по URL: {webhook_url}")
    await dispatcher.bot.set_webhook(
        url=webhook_url, 
        secret_token=WEBHOOK_SECRET,
        drop_pending_updates=True
    )

async def on_shutdown(dispatcher):
    # При остановке удаляем Webhook
    logging.info("Удаление Webhook")
    await dispatcher.bot.delete_webhook()

async def start_webhook():
    # Инициализируем БД
    # Твоя функция initialize_database() находится в bot.py в main(), 
    # ее нужно вызывать, или просто вызвать main(), но тогда нужно 
    # убрать dp.start_polling оттуда. Лучше вынеси инициализацию БД 
    # и main() в отдельные функции или импортируй main.
    
    # Для простоты, вызовем основную функцию (если main содержит только setup)
    # Если main содержит asyncio.run(main()), то вызываем его без run, или просто 
    # вызываем dp.start_polling, как у тебя было.
    
    # ПРЕДПОЛОЖИМ, что инициализация БД произойдет при первом импорте bot.py
    
    # Вызываем функцию setup (если она есть) или просто настраиваем Webhook
    await on_startup(dp, WEBHOOK_URL)
    
    # Запускаем Webhook
    try:
        logging.info("Запуск Webhook-сервера...")
        await dp.start_webhook(
            listen=WEB_SERVER_HOST,
            port=WEB_SERVER_PORT,
            url=WEBHOOK_PATH,
            secret_token=WEBHOOK_SECRET
        )
    except Exception as e:
        logger.error(f"Ошибка запуска Webhook-сервера: {e}")
    finally:
        await on_shutdown(dp)

if __name__ == "__main__":
    # Вызываем основную функцию твоего bot.py для инициализации 
    # всего (например, БД), но без asyncio.run()!
    # Твой main() содержит asyncio.run(dp.start_polling()), поэтому 
    # мы будем использовать эту обертку для запуска.
    try:
        asyncio.run(start_webhook())
    except KeyboardInterrupt:
        logging.info("Бот остановлен")