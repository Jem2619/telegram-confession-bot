import os
import sqlite3
import logging
import re
import time
import textwrap
import asyncio
import traceback
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Убраны неиспользуемые импорты, в том числе прокси
if '/usr/lib/python3.10/lib-dynload' not in sys.path:
    sys.path.append('/usr/lib/python3.10/lib-dynload')
from aiogram.client.default import DefaultBotProperties
from aiogram.types import FSInputFile
from datetime import datetime, timedelta
from contextlib import contextmanager
from PIL import Image, ImageDraw, ImageFont
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardRemove, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder
from aiogram.enums import ParseMode
from dotenv import load_dotenv
from cachetools import TTLCache
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError

# Загрузка переменных окружения
load_dotenv()

# Проверка токена
if not os.getenv("BOT_TOKEN"):
    print("BOT_TOKEN не найден в .env файле! Завершение работы.")
    sys.exit(1)

# Настройки
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_CHAT_ID = int(os.getenv("ADMIN_CHAT_ID", -1002887237992))
CHANNEL_ID = int(os.getenv("CHANNEL_ID", -1002649253605))
INSTAGRAM_GROUP_ID = int(os.getenv("INSTAGRAM_GROUP_ID", -1002620842581))

# Инициализация кешей
USER_CACHE = TTLCache(maxsize=1000, ttl=300)
BAN_CACHE = TTLCache(maxsize=500, ttl=600)
STATS_CACHE = TTLCache(maxsize=10, ttl=3600)
# НОВЫЙ КЕШ для предотвращения двойных предупреждений при альбомах
REJECTED_MEDIA_GROUPS = TTLCache(maxsize=100, ttl=5)

# Инициализация бота и диспетчера
bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher()

# Настройка логирования (консолидировано)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "bot.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Создаем папки для временных файлов
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs("backgrounds", exist_ok=True)
os.makedirs("fonts", exist_ok=True)


# ============================= Вспомогательные функции =============================

async def safe_edit_message(
        message: types.Message,
        text: str = None,
        caption: str = None,
        reply_markup: InlineKeyboardMarkup = None,
        parse_mode: ParseMode = None
):
    """Безопасное редактирование сообщения с обработкой ошибки 'message not modified'"""
    try:
        if message.photo or message.video:
            # Для медиа-сообщений используем edit_caption, если он предоставлен
            if caption is not None:
                return await message.edit_caption(caption=caption, reply_markup=reply_markup, parse_mode=parse_mode)
            # Иначе используем текущий caption
            return await message.edit_caption(caption=message.caption, reply_markup=reply_markup, parse_mode=parse_mode)
        else:
            # Для текстовых сообщений используем edit_text
            if text is not None:
                return await message.edit_text(text=text, reply_markup=reply_markup, parse_mode=parse_mode)
            # Иначе используем текущий текст
            return await message.edit_text(text=message.text, reply_markup=reply_markup, parse_mode=parse_mode)
    except TelegramBadRequest as e:
        if "message is not modified" in str(e):
            logger.warning("Сообщение не изменено, пропускаем редактирование")
            return message
        raise


# --- НОВАЯ ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ ВОССТАНОВЛЕНИЯ СООБЩЕНИЯ (ФИНАЛЬНАЯ ВЕРСИЯ) ---
async def restore_admin_message(callback: CallbackQuery, post_id: int):
    """Восстанавливает исходное сообщение модерации с кнопками Одобрить/Отклонить,
    получая чистый контент из БД и добавляя невидимый символ для обхода бага API."""
    # 1. Получаем пост из БД
    post = get_post(post_id)
    if not post:
        logger.error(f"Post not found for restore: {post_id}")
        await callback.answer("Ошибка: Пост не найден.", show_alert=True)
        return

    # 2. Получаем контент
    original_content = post.get('caption') or "<code>[ОРИГИНАЛЬНЫЙ ТЕКСТ ИЗ БД ОТСУТСТВУЕТ]</code>"
    # Добавляем невидимый символ (\u200b) для принудительного обновления текста в Telegram
    content_to_restore = original_content + "\u200b"

    # 3. Создаем кнопки
    builder = InlineKeyboardBuilder()
    builder.row(InlineKeyboardButton(text="✅ Одобрить", callback_data=f"approve_{post_id}"))
    builder.row(InlineKeyboardButton(text="❌ Отклонить", callback_data=f"reject_{post_id}"))

    message = callback.message
    is_media = post['media_type'] in ['photo', 'video']
    reply_markup = builder.as_markup()

    # 4. Редактируем сообщение
    try:
        if is_media:
            await message.edit_caption(
                caption=content_to_restore,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup
            )
        else:
            await message.edit_text(
                text=content_to_restore,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup
            )

        await callback.answer("Выбор отменен. Сообщение восстановлено.")
    except TelegramBadRequest as e:
        if "message is not modified" in str(e):
            await callback.answer("Выбор отменен.")
        else:
            logger.error(f"Ошибка редактирования при восстановлении: {e}")
            await callback.answer("Ошибка восстановления!", show_alert=True)
    except Exception as e:
        logger.error(f"Критическая ошибка при восстановлении: {e}")
        await callback.answer("Критическая ошибка восстановления!", show_alert=True)

# ============================= Функции для генерации Instagram-постов =============================


def split_text_for_instagram(text: str, max_chunk=800, min_chunk=200):
    """Разбивает текст на части для Instagram"""
    text = text.strip()
    n = len(text)

    if n <= max_chunk:
        return [text]

    if n <= max_chunk + min_chunk:
        mid = n // 2
        for i in range(0, 100):
            left_pos = mid - i
            right_pos = mid + i
            for pos in sorted({left_pos, right_pos}, reverse=True):
                if 0 < pos < n and text[pos] in (' ', '\n', '.', ',', '!', '?', ':', ';', '-'):
                    return [text[:pos + 1].strip(), text[pos + 1:].strip()]
        return [text[:mid], text[mid:]]

    parts = []
    current = text

    while current:
        if len(current) <= max_chunk:
            parts.append(current)
            break

        chunk = current[:max_chunk]
        split_index = -1
        separators = ['\n', '. ', '! ', '? ', ', ', ': ', '; ', ' - ', ' ']

        for sep in separators:
            index = chunk.rfind(sep)
            if index > max_chunk * 0.7:
                split_index = index + len(sep) - 1
                break

        if split_index == -1:
            for sep in separators:
                index = chunk.rfind(sep)
                if index > 0:
                    split_index = index + len(sep) - 1
                    break

        if split_index == -1:
            parts.append(chunk)
            current = current[max_chunk:].lstrip()
        else:
            parts.append(current[:split_index + 1].strip())
            current = current[split_index + 1:].lstrip()

    i = len(parts) - 1
    while i > 0:
        if len(parts[i]) < min_chunk:
            parts[i - 1] = parts[i - 1] + "\n" + parts[i]
            parts.pop(i)
        i -= 1

    if len(parts) > 1:
        avg_length = sum(len(p) for p in parts) / len(parts)
        for i in range(len(parts)):
            if len(parts[i]) < avg_length * 0.7:
                if i > 0 and len(parts[i - 1]) + len(parts[i]) <= max_chunk * 1.2:
                    parts[i - 1] = parts[i - 1] + "\n" + parts[i]
                    parts.pop(i)
                    break
                elif i < len(parts) - 1 and len(parts[i]) + len(parts[i + 1]) <= max_chunk * 1.2:
                    parts[i] = parts[i] + "\n" + parts[i + 1]
                    parts.pop(i + 1)
                    break

    return parts


def remove_emojis(text: str) -> str:
    """Удаляет эмодзи из текста"""
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


def generate_instagram_image(text: str, theme: str, part: int = None, total_parts: int = None):
    """Генерирует изображение для Instagram (без эмодзи)"""
    try:
        # Удаляем эмодзи из текста перед обработкой
        text = remove_emojis(text)

        # Остальной код остается без изменений
        width, height = 1080, 1080
        bg_path = os.path.join(BASE_DIR, "backgrounds", f"{theme}_background.jpg")
        bg_color = (255, 255, 255) if theme == "white" else (0, 0, 0)

        try:
            if os.path.exists(bg_path):
                image = Image.open(bg_path).convert("RGB")
                image = image.resize((width, height), Image.LANCZOS)
            else:
                image = Image.new('RGB', (width, height), bg_color)
        except Exception:
            image = Image.new('RGB', (width, height), bg_color)

        draw = ImageDraw.Draw(image)
        text_color = (0, 0, 0) if theme == "white" else (255, 255, 255)
        font_path = os.path.join(BASE_DIR, "fonts", "Akrobat-Bold.otf")
        base_font_size = 70

        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, base_font_size)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        padding_x = int(width * 0.12)
        padding_y = int(height * 0.12)
        max_width = width - 2 * padding_x
        max_height = height - 2 * padding_y

        lines = []
        for line in text.split('\n'):
            wrapped = textwrap.wrap(line, width=60)
            lines.extend(wrapped)

        min_font_size = 18
        selected_size = base_font_size

        for size in range(base_font_size, min_font_size - 1, -2):
            try:
                if os.path.exists(font_path):
                    test_font = ImageFont.truetype(font_path, size)
                else:
                    test_font = ImageFont.load_default()
            except:
                test_font = ImageFont.load_default()

            line_height = size * 1.2
            total_text_height = len(lines) * line_height

            if total_text_height > max_height:
                continue

            fits = True
            for line_text in lines:
                bbox = draw.textbbox((0, 0), line_text, font=test_font)
                text_width = bbox[2] - bbox[0]
                if text_width > max_width:
                    fits = False
                    break

            if fits:
                selected_size = size
                font = test_font
                break

        if selected_size < min_font_size:
            selected_size = min_font_size
            try:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, selected_size)
                else:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()

        line_height = selected_size * 1.2
        total_text_height = len(lines) * line_height
        y = (height - total_text_height) / 2

        for line_text in lines:
            bbox = draw.textbbox((0, 0), line_text, font=font)
            text_width = bbox[2] - bbox[0]
            x = (width - text_width) / 2
            draw.text((x, y), line_text, fill=text_color, font=font)
            y += line_height

        if total_parts and total_parts > 1:
            footer_text = f"{part}/{total_parts}"
            try:
                if os.path.exists(font_path):
                    footer_font = ImageFont.truetype(font_path, 30)
                else:
                    footer_font = ImageFont.load_default()
            except:
                footer_font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), footer_text, font=footer_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = width - text_width - 50
            y = height - text_height - 50
            draw.text((x, y), footer_text, fill=text_color, font=footer_font)

        img_path = os.path.join(TEMP_DIR, f"instagram_{int(time.time())}_{part if part else 1}.jpg")
        image.save(img_path, quality=90)
        return img_path

    except Exception as e:
        logger.error(f"Ошибка генерации: {e}\n{traceback.format_exc()}")
        return None


async def send_instagram_post(post: dict):
    """Создаёт и отправляет изображение для Instagram"""
    if not post.get('caption'):
        logger.warning("Пустой текст поста, пропуск генерации")
        return

    try:
        text_parts = split_text_for_instagram(post['caption'])
        logger.info(f"Текст разбит на {len(text_parts)} частей")

        image_paths = []
        for i, part_text in enumerate(text_parts):
            img_path = generate_instagram_image(
                text=part_text,
                theme=post['theme'],
                part=i + 1,
                total_parts=len(text_parts)
            )
            if img_path:
                image_paths.append(img_path)

        if not image_paths:
            logger.error("Не создано ни одного изображения!")
            await bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text=f"❌ Ошибка генерации Instagram-поста для ID {post['post_id']}"
            )
            return

        for i, img_path in enumerate(image_paths):
            caption = f"Пост для Instagram (ID: {post['post_id']})\nТема: {'Светлая' if post['theme'] == 'white' else 'Тёмная'}"
            if len(image_paths) > 1:
                caption += f"\nЧасть {i + 1}/{len(image_paths)}"

            try:
                await bot.send_photo(
                    chat_id=INSTAGRAM_GROUP_ID,
                    photo=FSInputFile(img_path),
                    caption=caption
                )
            except Exception as e:
                logger.error(f"Ошибка отправки: {e}")
                await asyncio.sleep(1)
                try:
                    await bot.send_photo(
                        chat_id=INSTAGRAM_GROUP_ID,
                        photo=FSInputFile(img_path),
                        caption=caption
                    )
                except Exception as e2:
                    logger.error(f"Повторная ошибка: {e2}")
            finally:
                try:
                    os.remove(img_path)
                except:
                    pass
            await asyncio.sleep(0.5)

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}\n{traceback.format_exc()}")
        await bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text=f"❌ Ошибка Instagram-поста ID {post['post_id']}:\n{str(e)}"
        )


# ============================= Команды бота (для админов) =============================

async def handle_make_ig_post(message: Message, theme: str):
    """Общий обработчик для создания Instagram-поста в чате админов."""
    if message.chat.id != INSTAGRAM_GROUP_ID and message.chat.id != ADMIN_CHAT_ID:
        return

    command = f"/m{'w' if theme == 'white' else 'b'}"
    text = message.text.replace(command, '').strip()

    if not text:
        await message.answer(f"Пожалуйста, укажите текст поста после команды {command}")
        return

    fake_post = {
        'caption': text,
        'theme': theme,
        'post_id': f"manual_{message.message_id}"
    }

    try:
        await send_instagram_post(fake_post)
        # Убрано сообщение "Пост для Instagram успешно создан!"
    except Exception as e:
        logger.error(f"Ошибка при создании поста: {e}")
        await message.answer(f"Ошибка: {str(e)}")


@dp.message(Command("mw"))
async def cmd_make_white(message: Message):
    """Создает пост для Instagram из текста сообщения (Белая тема)"""
    await handle_make_ig_post(message, "white")


@dp.message(Command("mb"))
async def cmd_make_dark(message: Message):
    """Создает пост для Instagram из текста сообщения (Тёмная тема)"""
    await handle_make_ig_post(message, "dark")


# ============================= Работа с базой данных =============================

@contextmanager
def db_connection():
    conn = None
    try:
        conn = sqlite3.connect('confessions.db', check_same_thread=False)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA cache_size = -10000")
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def initialize_database():
    """Инициализирует таблицы базы данных и добавляет недостающие столбцы."""
    try:
        with db_connection() as cursor:
            # === Схема таблицы users ===
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    full_name TEXT,
                    banned INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # --- МИГРАЦИЯ: Добавление столбцов created_at и updated_at (для исправления предыдущей ошибки) ---
            try:
                cursor.execute("SELECT created_at FROM users LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute("ALTER TABLE users ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP")
                logger.info("Добавлен столбец users.created_at")

            try:
                cursor.execute("SELECT updated_at FROM users LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute("ALTER TABLE users ADD COLUMN updated_at DATETIME DEFAULT CURRENT_TIMESTAMP")
                logger.info("Добавлен столбец users.updated_at")

            # === Схема таблицы posts ===
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS posts (
                    post_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    theme TEXT NOT NULL,
                    anonymous BOOLEAN NOT NULL,
                    media_type TEXT,
                    media_id TEXT,
                    caption TEXT,
                    status TEXT DEFAULT 'pending',
                    published_in TEXT,
                    moderated_by INTEGER,
                    admin_chat_msg_id INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bans (
                    ban_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    reason TEXT,
                    banned_by INTEGER,
                    banned_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS warnings (
                    warning_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    post_id INTEGER,
                    warned_by INTEGER,
                    warned_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (post_id) REFERENCES posts(post_id)
                )
            ''')

            # Создание индексов
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_posts_status ON posts (status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_banned ON users (banned)")

        logger.info("Таблицы и индексы базы данных успешно созданы")

    except Exception as e:
        logger.error(f"Критическая ошибка инициализации базы данных: {e}")
        sys.exit(1)


def save_user(user: types.User):
    with db_connection() as cursor:
        cursor.execute(
            "INSERT OR IGNORE INTO users (user_id, username, full_name, banned) VALUES (?, ?, ?, COALESCE((SELECT banned FROM users WHERE user_id = ?), 0))",
            (user.id, user.username, user.full_name, user.id)
        )
        cursor.execute(
            "UPDATE users SET username = ?, full_name = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
            (user.username, user.full_name, user.id)
        )

    USER_CACHE[user.id] = {
        'user_id': user.id,
        'username': user.username,
        'full_name': user.full_name,
        'banned': is_user_banned(user.id)
    }


def is_user_banned(user_id: int) -> bool:
    if user_id in BAN_CACHE:
        return BAN_CACHE[user_id]

    with db_connection() as cursor:
        cursor.execute("SELECT banned FROM users WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
        banned = result and result[0] == 1

    BAN_CACHE[user_id] = banned
    return banned


def get_user(user_id: int) -> dict:
    if user_id in USER_CACHE:
        return USER_CACHE[user_id]

    with db_connection() as cursor:
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        columns = [col[0] for col in cursor.description]
        row = cursor.fetchone()

    if row:
        user_data = dict(zip(columns, row))
        USER_CACHE[user_id] = user_data
        return user_data
    else:
        logger.warning(f"User {user_id} not found in database, creating stub")
        with db_connection() as cursor:
            cursor.execute(
                "INSERT OR IGNORE INTO users (user_id, username, full_name) VALUES (?, ?, ?)",
                (user_id, f"unknown_{user_id}", f"User #{user_id}")
            )
        user_data = {
            'user_id': user_id,
            'username': f"unknown_{user_id}",
            'full_name': f"User #{user_id}"
        }
        USER_CACHE[user_id] = user_data
        return user_data


def save_post(user_id: int, theme: str, anonymous: bool, media_type: str, media_id: str, caption: str) -> int:
    logger.info(f"Saving post for user: {user_id}")
    with db_connection() as cursor:
        cursor.execute(
            "INSERT INTO posts (user_id, theme, anonymous, media_type, media_id, caption) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, theme, anonymous, media_type, media_id, caption)
        )
        return cursor.lastrowid


def get_post(post_id: int) -> dict:
    with db_connection() as cursor:
        cursor.execute("SELECT * FROM posts WHERE post_id = ?", (post_id,))
        columns = [col[0] for col in cursor.description]
        row = cursor.fetchone()
        return dict(zip(columns, row)) if row else None


def update_post_status(post_id: int, status: str, published_in: str = None, moderator_id: int = None):
    query = "UPDATE posts SET status = ?"
    params = [status]

    if published_in:
        query += ", published_in = ?"
        params.append(published_in)

    if moderator_id is not None:
        query += ", moderated_by = ?"
        params.append(moderator_id)

    query += " WHERE post_id = ?"
    params.append(post_id)

    with db_connection() as cursor:
        cursor.execute(query, tuple(params))


def update_admin_message_id(post_id: int, message_id: int):
    with db_connection() as cursor:
        cursor.execute("UPDATE posts SET admin_chat_msg_id = ? WHERE post_id = ?", (message_id, post_id))


def ban_user(user_id: int, reason: str, banned_by: int) -> bool:
    try:
        with db_connection() as cursor:
            cursor.execute("INSERT INTO bans (user_id, reason, banned_by) VALUES (?, ?, ?)",
                           (user_id, reason, banned_by))
            cursor.execute("UPDATE users SET banned = 1 WHERE user_id = ?", (user_id,))

        BAN_CACHE[user_id] = True
        if user_id in USER_CACHE:
            USER_CACHE[user_id]['banned'] = True

        return True
    except Exception as e:
        logger.error(f"Error banning user: {e}")
        return False


def unban_user(user_id: int) -> bool:
    with db_connection() as cursor:
        cursor.execute("UPDATE users SET banned = 0 WHERE user_id = ?", (user_id,))

    BAN_CACHE[user_id] = False
    if user_id in USER_CACHE:
        USER_CACHE[user_id]['banned'] = False

    return True


def get_active_bans() -> list:
    with db_connection() as cursor:
        cursor.execute("SELECT * FROM bans WHERE user_id IN (SELECT user_id FROM users WHERE banned = 1)")
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def warn_user(user_id: int, post_id: int, warned_by: int) -> bool:
    try:
        with db_connection() as cursor:
            cursor.execute(
                "INSERT INTO warnings (user_id, post_id, warned_by) VALUES (?, ?, ?)",
                (user_id, post_id, warned_by)
            )
        return True
    except Exception as e:
        logger.error(f"Error saving warning: {e}")
        return False


# ============================= Вспомогательные функции =============================

def format_admin_caption(post: dict, user_info: dict) -> str:
    theme = "белая" if post['theme'] == 'white' else "тёмная"
    anonymity = "Анонимно" if post['anonymous'] else "Открыто"

    user_link = f"<a href='tg://user?id={user_info['user_id']}'>{user_info['full_name']}</a>"

    if user_info.get('username'):
        user_link += f" (@{user_info['username']})"

    caption = (
        f"<b>Автор:</b> {user_link}\n"
        f"<b>Тип:</b> {anonymity}\n"
        f"<b>Тема:</b> {theme}\n"
        f"<b>Дата:</b> {post['created_at']}\n"
        f"<b>ID поста:</b> {post['post_id']}\n"
    )

    if post.get('status') != 'pending':
        status_map = {
            'approved': 'Одобрено',
            'rejected_ad': 'Отклонено (Реклама)',
            'rejected_rules': 'Отклонено (Нарушение правил)',
            'rejected_warning': 'Отклонено + Предупреждение',
            'rejected_ban': 'Отклонено + Бан'
        }
        status_text = status_map.get(post['status'], post['status'])

        if post.get('moderated_by'):
            moderator = get_user(post['moderated_by'])
            moderator_name = moderator['full_name'] if moderator else f"ID:{post['moderated_by']}"
            caption += f"<b>Модератор:</b> {moderator_name}\n"

        caption += f"<b>Статус:</b> {status_text}\n"

        if post.get('published_in'):
            publish_map = {
                'tg': 'Только ТГК',
                'both': 'ТГК и Instagram'
            }
            caption += f"<b>Публикация:</b> {publish_map.get(post['published_in'], post['published_in'])}"

    return caption


async def update_admin_message(post_id: int, message: types.Message):
    post = get_post(post_id)
    if not post:
        logger.error(f"Post not found for update: {post_id}")
        return

    # Форматируем информацию о модерации
    moderator_info = ""
    if post.get('moderated_by'):
        moderator = get_user(post['moderated_by'])
        moderator_name = moderator['full_name'] if moderator else f"ID:{post['moderated_by']}"
        moderator_info = f"\n\n<b>Модератор:</b> {moderator_name}"

    status_info = ""
    if post.get('status'):
        status_map = {
            'approved': '✅ Одобрено',
            'rejected_ad': '❌ Отклонено (Реклама)',
            'rejected_rules': '❌ Отклонено (Нарушение правил)',
            'rejected_warning': '❌ Отклонено + Предупреждение',
            'rejected_ban': '❌ Отклонено + Бан'
        }
        status_text = status_map.get(post['status'], post['status'])
        status_info = f"\n<b>Статус:</b> {status_text}"

    publish_info = ""
    if post.get('published_in'):
        if post['published_in'] == 'both':
            publish_info = "\n<b>Публикация:</b> ТГК и Instagram"
        elif post['published_in'] == 'tg':
            publish_info = "\n<b>Публикация:</b> Только ТГК"

    # Собираем полный текст для контентного сообщения
    full_text = ""
    if message.caption:
        # Для медиа-сообщений (у которых есть подпись)
        full_text = f"{message.caption}{moderator_info}{status_info}{publish_info}"
    elif message.text:
        # Для текстовых сообщений
        full_text = f"{message.text}{moderator_info}{status_info}{publish_info}"

    # Определяем, что нужно редактировать (текст или подпись)
    edit_method = message.edit_caption if message.photo or message.video else message.edit_text

    try:
        await edit_method(
            caption=full_text,
            text=full_text,  # text/caption будет использован в зависимости от edit_method
            parse_mode=ParseMode.HTML,
            reply_markup=None
        )
    except Exception as e:
        logger.error(
            f"Ошибка при обновлении сообщения админа (пост ID: {post_id}, сообщение ID: {message.message_id}): {e}")


def format_channel_caption(post: dict, user_info: dict) -> str:
    caption = post['caption'] or ""

    if not post['anonymous'] and user_info:
        username = f"@{user_info['username']}" if user_info.get('username') else user_info['full_name']
        caption += f"\n\nавтор поста - {username}"

    return caption


def get_monthly_stats():
    if 'stats' in STATS_CACHE:
        return STATS_CACHE['stats']

    thirty_days_ago = datetime.now() - timedelta(days=30)
    date_str = thirty_days_ago.strftime("%Y-%m-%d %H:%M:%S")

    stats = {}

    try:
        with db_connection() as cursor:
            cursor.execute("SELECT COUNT(*) FROM posts WHERE created_at >= ?", (date_str,))
            stats['total_posts'] = cursor.fetchone()[0] or 0

            cursor.execute("SELECT anonymous, COUNT(*) FROM posts WHERE created_at >= ? GROUP BY anonymous",
                           (date_str,))
            stats['anonymous'] = 0
            stats['non_anonymous'] = 0
            for row in cursor.fetchall():
                if row[0]:
                    stats['anonymous'] = row[1]
                else:
                    stats['non_anonymous'] = row[1]

            cursor.execute("SELECT media_type, COUNT(*) FROM posts WHERE created_at >= ? GROUP BY media_type",
                           (date_str,))
            stats['photo'] = 0
            stats['video'] = 0
            stats['text'] = 0
            for row in cursor.fetchall():
                if row[0] == 'photo':
                    stats['photo'] = row[1]
                elif row[0] == 'video':
                    stats['video'] = row[1]
                else:
                    stats['text'] += row[1]

            cursor.execute("SELECT status, COUNT(*) FROM posts WHERE created_at >= ? GROUP BY status", (date_str,))
            stats['approved'] = 0
            stats['rejected'] = 0
            stats['pending'] = 0
            stats['rejected_details'] = {}
            for row in cursor.fetchall():
                status = row[0]
                count = row[1]

                if status == 'approved':
                    stats['approved'] = count
                elif status == 'pending':
                    stats['pending'] = count
                else:
                    stats['rejected'] += count
                    stats['rejected_details'][status] = count

            cursor.execute("""
                SELECT
                    SUM(CASE WHEN reason = 'Реклама' THEN 1 ELSE 0 END) as ads,
                    SUM(CASE WHEN reason = 'Нарушение правил' THEN 1 ELSE 0 END) as rules,
                    SUM(CASE WHEN reason LIKE '%Бан%' THEN 1 ELSE 0 END) as bans
                FROM bans
                WHERE banned_at >= ?
            """, (date_str,))
            row = cursor.fetchone()
            stats['rejected_reason'] = {
                'ads': row[0] or 0,
                'rules': row[1] or 0,
                'bans': row[2] or 0
            }

    except Exception as e:
        logger.error(f"Ошибка при получении статистики: {e}")
        stats = {'error': f"Ошибка при получении статистики: {str(e)}"}

    STATS_CACHE['stats'] = stats
    return stats


# ============================= Классы состояний =============================

class PostCreation(StatesGroup):
    choosing_theme = State()
    choosing_anonymity = State()
    getting_text = State()
    media_choice = State()
    getting_media = State()


# ============================= Обработчики команд =============================

@dp.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()

    # Проверка на бан
    user_id = message.from_user.id
    if is_user_banned(user_id):
        await message.answer("Вы забанены и не можете использовать бота.")
        return

    save_user(message.from_user)

    builder = ReplyKeyboardBuilder()
    builder.row(types.KeyboardButton(text="Создать пост"))
    builder.row(types.KeyboardButton(text="Правила"))

    welcome_message = (
        f"Добро пожаловать, {message.from_user.full_name}!\n\n"
        "С помощью этого бота вы можете анонимно или открыто поделиться "
        "своим признанием, вопросом или историей. После модерации, мы "
        "опубликуем его в нашем ТГК и Instagram аккаунте.\n\n"
        "Доступные команды:\n"
        "/create - Создать новое признание\n"
        "/start - Начать работу\n"
        "/help - Помощь\n"
        "/rules - Правила бота"
    )
    await message.answer(
        text=welcome_message,
        reply_markup=builder.as_markup(resize_keyboard=True)
    )


@dp.message(F.text == "Создать пост")
async def create_post_button(message: Message, state: FSMContext):
    await cmd_create(message, state)


@dp.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "Помощь по боту:\n\n"
        "1. Чтобы создать признание - используйте /create\n"
        "2. После создания пост проходит модерацию\n"
        "3. Вы получите уведомление о статусе модерации\n"
        "4. Запрещено размещать рекламу и нарушать правила\n\n"
        "По проблемам: @shiebeid"
    )


# ИСПРАВЛЕНИЕ ОШИБКИ: Разбиваем комбинированный фильтр на два отдельных декоратора
@dp.message(Command("rules"))
@dp.message(F.text == "Правила")
async def cmd_rules(message: Message):
    await message.answer(
        "Правила сообщества:\n\n"
        "1. Запрещены оскорбления и дискриминация\n"
        "2. Не публикуйте личную информацию\n"
        "3. Запрещена реклама и спам\n"
        "4. Контент должен иметь смысловую нагрузку\n"
        "5. Максимальная длина видео - 10 секунд\n"
        "6. Нельзя использовать в постах номера телефонов\n"
        "7. Нельзя представляться другим человеком (даже отмечая свой аккаунт)\n"
        "8. Нельзя публиковать своих знакомых, друзей\n"
        "9. Не публикуем посты с анкетами, с поиском друзей\парней\девушек, связанное с ДВ, сайт знакомств\n\n"
        "Нарушение правил ведет к бану!"
    )


@dp.message(Command("create"))
async def cmd_create(message: Message, state: FSMContext):
    user_id = message.from_user.id
    if is_user_banned(user_id):
        await message.answer("Вы забанены и не можете создавать посты!")
        return

    save_user(message.from_user)

    await state.clear()
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="Белая тема", callback_data="theme_white"))
    builder.add(InlineKeyboardButton(text="Тёмная тема", callback_data="theme_dark"))
    builder.add(InlineKeyboardButton(text="Отмена", callback_data="cancel"))
    builder.adjust(2)

    sent_message = await message.answer(
        "Шаг 1 из 4: Выберите тему оформления\n"
        "• Белая тема – поиск конкретного человека (с фото или описанием)\n"
        "• Тёмная тема – признания, вопросы, истории, обсуждения",
        reply_markup=builder.as_markup()
    )
    # Сохраняем ID сообщения для последующего редактирования
    await state.update_data(step_message_id=sent_message.message_id)
    await state.set_state(PostCreation.choosing_theme)


# ============================= Обработчики состояний =============================

@dp.callback_query(PostCreation.choosing_theme, F.data.startswith("theme_"))
async def theme_choice(callback: CallbackQuery, state: FSMContext):
    choice = callback.data.split("_")[1]
    await state.update_data(theme=choice)

    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="Анонимно", callback_data="anon_yes"))
    builder.add(InlineKeyboardButton(text="Открыто (с указанием автора)", callback_data="anon_no"))
    builder.add(InlineKeyboardButton(text="Отмена", callback_data="cancel"))
    builder.adjust(1)

    await safe_edit_message(
        message=callback.message,
        text=f"Шаг 2 из 4: Выбрана {'Белая' if choice == 'white' else 'Тёмная'} тема.\n"
             "Выберите тип публикации:",
        reply_markup=builder.as_markup()
    )
    await state.set_state(PostCreation.choosing_anonymity)
    await callback.answer()


@dp.callback_query(PostCreation.choosing_anonymity, F.data.startswith("anon_"))
async def anonymity_choice(callback: CallbackQuery, state: FSMContext):
    choice = callback.data.split("_")[1]
    anonymous = choice == "yes"
    await state.update_data(anonymous=anonymous)

    await safe_edit_message(
        message=callback.message,
        text=f"Шаг 3 из 4: Выбрана {'Анонимная' if anonymous else 'Открытая'} публикация.\n\n"
             "Напишите текст вашего признания. "
             "Максимальная длина текста без медиа — 3600 символов, с медиа — 1024 символа.",
        reply_markup=None  # Убираем кнопки
    )

    # Отправляем отдельное сообщение с инструкцией для шага 4 (чтобы пользователь не "завис")
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="Отмена", callback_data="cancel"))

    sent_message = await callback.message.answer(
        "После отправки текста, вам будет предложено добавить фото или видео "
        "(максимум 10 секунд), либо опубликовать только текст.",
        reply_markup=builder.as_markup()
    )
    await state.update_data(step_message_id=sent_message.message_id)
    await state.set_state(PostCreation.getting_text)
    await callback.answer()


@dp.message(PostCreation.getting_text)
async def process_text(message: Message, state: FSMContext):
    save_user(message.from_user)

    if not message.text and not message.caption:
        await message.answer("Пожалуйста, напишите текст для поста!")
        return

    # Используем message.text для текстовых постов, message.caption для медиа-постов
    text_content = message.text or message.caption or ""
    await state.update_data(caption=text_content)

    # Удаляем сообщение с инструкцией для шага 3
    data = await state.get_data()
    step_message_id = data.get("step_message_id")
    if step_message_id:
        try:
            await bot.delete_message(chat_id=message.chat.id, message_id=step_message_id)
        except:
            pass

    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="Добавить фото", callback_data="media_photo"))
    builder.add(InlineKeyboardButton(text="Добавить видео", callback_data="media_video"))
    builder.add(InlineKeyboardButton(text="Только текст", callback_data="media_none"))
    builder.adjust(2, 1)

    sent_message = await message.answer(
        "Шаг 4 из 4: Медиа-контент\n\n"
        "Вы можете добавить фото или видео (не более 10 секунд), "
        "либо опубликовать пост только с текстом.",
        reply_markup=builder.as_markup()
    )
    # Обновляем ID сообщения в состоянии
    await state.update_data(step_message_id=sent_message.message_id)
    await state.set_state(PostCreation.media_choice)


@dp.callback_query(PostCreation.media_choice, F.data.startswith("media_"))
async def media_choice(callback: CallbackQuery, state: FSMContext):
    choice = callback.data.split("_")[1]
    user_data = await state.get_data()
    caption = user_data.get('caption', '')

    # Проверка длины текста для постов с медиа (макс. 1024)
    if choice != "none":
        if len(caption) > 1024:
            excess = len(caption) - 1024
            await safe_edit_message(
                message=callback.message,
                text=f"Ошибка! Если пост содержит фото/видео, "
                     f"текст не может превышать 1024 символов.\n\n"
                     f"Ваш текст: {len(caption)} символов.\n"
                     f"Сократите текст минимум на {excess} символов и создайте пост заново.",
                reply_markup=None
            )
            await state.clear()
            await callback.answer()
            return
    # Проверка длины текста для постов только с текстом (макс. 3600)
    else:
        if len(caption) > 3600:
            excess = len(caption) - 3600
            await safe_edit_message(
                message=callback.message,
                text=f"Ошибка! Текст поста не может превышать 3600 символов.\n\n"
                     f"Ваш текст: {len(caption)} символов.\n"
                     f"Сократите текст минимум на {excess} символов и создайте пост заново.",
                reply_markup=None
            )
            await state.clear()
            await callback.answer()
            return

    if choice == "none":
        # Удаляем сообщение с кнопками
        try:
            await callback.message.delete()
        except:
            pass

        # Создаем пост сразу
        await create_post_from_data(callback.from_user, callback.message.chat.id, state, user_data)
    else:
        # Установка состояния ожидания медиа
        media_type = "photo" if choice == "photo" else "video"
        await state.update_data(media_type=media_type)

        # Обновление сообщения
        builder = InlineKeyboardBuilder()
        builder.add(InlineKeyboardButton(text="Отмена", callback_data="cancel"))

        await safe_edit_message(
            message=callback.message,
            text=f"Отправьте {'фото' if choice == 'photo' else 'видео'}.",
            reply_markup=builder.as_markup()
        )
        await state.set_state(PostCreation.getting_media)

    await callback.answer()


@dp.message(PostCreation.getting_media)
async def process_media(message: Message, state: FSMContext):
    # 1. ИСПРАВЛЕНИЕ #1: Проверка на медиагруппу (не более 1 файла)
    if message.media_group_id:

        if message.media_group_id in REJECTED_MEDIA_GROUPS:
            # Если ID медиагруппы уже в кеше, значит, мы уже отправили предупреждение
            return

        # 1.1. Отправляем предупреждение (только один раз)
        REJECTED_MEDIA_GROUPS[message.media_group_id] = True

        await message.answer(
            "❌ Ошибка: Вы можете добавить к посту только <b>один</b> медиафайл (фото или видео). "
            "Альбомы (группы фото/видео) не поддерживаются. "
            "Пожалуйста, создайте пост заново и отправьте только один файл."
        )

        # 1.2. Очищаем состояние
        await state.clear()
        return

    user_data = await state.get_data()
    media_type = user_data.get('media_type')

    # 2. Проверка соответствия типа и длительности видео
    if media_type == "video" and message.video and message.video.duration > 10:
        await message.answer("Видео должно быть не длиннее 10 секунд!")
        return

    if media_type == "photo" and not message.photo:
        await message.answer("Пожалуйста, отправьте фото!")
        return

    if media_type == "video" and not message.video:
        await message.answer("Пожалуйста, отправьте видео!")
        return

    # 3. Сохранение медиа ID
    media_id = message.photo[-1].file_id if message.photo else (message.video.file_id if message.video else None)

    if not media_id:
        await message.answer("Ошибка: не удалось получить медиафайл. Пожалуйста, попробуйте снова.")
        await state.clear()
        return

    await state.update_data(media_id=media_id)
    user_data = await state.get_data()

    # 4. Создание поста
    await create_post_from_data(message.from_user, message.chat.id, state, user_data)


async def create_post_from_data(user: types.User, chat_id: int, state: FSMContext, user_data: dict):
    try:
        save_user(user)
        post_id = save_post(
            user_id=user.id,
            theme=user_data['theme'],
            anonymous=user_data['anonymous'],
            media_type=user_data.get('media_type'),
            media_id=user_data.get('media_id'),
            caption=user_data['caption']
        )
        await send_to_moderation(post_id)
        await bot.send_message(
            chat_id=chat_id,
            text="Пост отправлен на модерацию! Вы получите уведомление о результате. "
                 "Обычно посты проверяются поздно ночью, пожалуйста ожидайте!",
            reply_markup=ReplyKeyboardRemove()
        )
    except Exception as e:
        logger.error(f"Ошибка создания поста: {e}\n{traceback.format_exc()}")
        await bot.send_message(
            chat_id=chat_id,
            text="Произошла ошибка при создании поста. Пожалуйста, попробуйте позже."
        )
    finally:
        await state.clear()


async def send_to_moderation(post_id: int):
    post = get_post(post_id)
    if not post:
        logger.error(f"Post not found: {post_id}")
        return

    user_info = get_user(post['user_id'])
    admin_caption = format_admin_caption(post, user_info)

    # Отправляем метаданные отдельным сообщением
    meta_msg = await bot.send_message(
        chat_id=1052106591,
        text=admin_caption,
        parse_mode=ParseMode.HTML
    )

    content = post['caption'] or ""
    builder = InlineKeyboardBuilder()
    builder.row(InlineKeyboardButton(text="✅ Одобрить", callback_data=f"approve_{post_id}"))
    builder.row(InlineKeyboardButton(text="❌ Отклонить", callback_data=f"reject_{post_id}"))

    # Отправляем контент с кнопками отдельным сообщением
    try:
        if post['media_type'] == 'photo':
            content_msg = await bot.send_photo(
                chat_id=ADMIN_CHAT_ID,
                photo=post['media_id'],
                caption=content,
                reply_markup=builder.as_markup()
            )
        elif post['media_type'] == 'video':
            content_msg = await bot.send_video(
                chat_id=ADMIN_CHAT_ID,
                video=post['media_id'],
                caption=content,
                reply_markup=builder.as_markup()
            )
        else:
            content_msg = await bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text=content,
                reply_markup=builder.as_markup()
            )

        # Сохраняем ID сообщения с контентом для дальнейшего редактирования
        update_admin_message_id(post_id, content_msg.message_id)

    except Exception as e:
        logger.error(f"Ошибка при отправке контента на модерацию (ID: {post_id}): {e}")
        # Отправляем админам предупреждение, если контент не отправился
        await bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text=f"❌ ВНИМАНИЕ: Не удалось отправить контент поста ID {post_id} (автор {user_info['full_name']}). Пожалуйста, проверьте в логах."
        )


# ================== ИСПРАВЛЕННЫЙ БЛОК ==================
@dp.callback_query(F.data == "cancel")
async def handle_cancel(callback: CallbackQuery, state: FSMContext):
    """Общий обработчик отмены (только для точного совпадения 'cancel')"""
    await state.clear()

    # Удаляем сообщение, если оно не было удалено ранее
    try:
        await callback.message.delete()
    except TelegramBadRequest as e:
        if "message can't be deleted" not in str(e):
            # Редактируем, если удалить не удалось
            await callback.message.edit_text("Операция отменена", reply_markup=None)
    except Exception:
        pass

    # Пытаемся удалить сообщение с ID, сохраненным в state
    data = await state.get_data()
    step_message_id = data.get("step_message_id")
    if step_message_id:
        try:
            await bot.delete_message(chat_id=callback.message.chat.id, message_id=step_message_id)
        except:
            pass

    await callback.answer("Создание поста отменено.")
# =======================================================


@dp.callback_query(F.data.startswith("approve_"))
async def approve_post(callback: CallbackQuery):
    if callback.message.chat.id != ADMIN_CHAT_ID:
        return

    post_id = int(callback.data.split("_")[1])

    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="Опубликовать (Только ТГК)", callback_data=f"publish_tg_{post_id}"))
    builder.add(InlineKeyboardButton(text="Опубликовать (ТГК + Insta)", callback_data=f"publish_both_{post_id}"))
    builder.add(InlineKeyboardButton(text="Отмена", callback_data=f"cancel_approve_{post_id}"))
    builder.adjust(1)

    post = get_post(post_id)
    if not post:
        await callback.answer("Ошибка: Пост не найден.", show_alert=True)
        return

    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Явно передаем контент из БД при смене кнопок
    message = callback.message
    is_media = post['media_type'] in ['photo', 'video']
    current_content = post.get('caption') or "<code>[ТЕКСТ ИЗ БД ПОТЕРЯН НА ШАГЕ APPROVE]</code>"
    content_to_edit = current_content + "\u200b"  # Невидимый символ для принудительного обновления

    try:
        if is_media:
            await message.edit_caption(
                caption=content_to_edit,
                reply_markup=builder.as_markup(),
                parse_mode=ParseMode.HTML
            )
        else:
            await message.edit_text(
                text=content_to_edit,
                reply_markup=builder.as_markup(),
                parse_mode=ParseMode.HTML
            )
    except TelegramBadRequest as e:
        if "message is not modified" not in str(e):
            logger.error(f"Ошибка сохранения контента в approve_post: {e}")
            await callback.answer("Ошибка при изменении кнопок!", show_alert=True)

    await callback.answer()


async def publish_post(post: dict):
    user_info = get_user(post['user_id'])
    channel_caption = format_channel_caption(post, user_info)

    try:
        if post['media_type'] == 'photo':
            await bot.send_photo(
                chat_id=CHANNEL_ID,
                photo=post['media_id'],
                caption=channel_caption,
                parse_mode=ParseMode.HTML
            )
        elif post['media_type'] == 'video':
            await bot.send_video(
                chat_id=CHANNEL_ID,
                video=post['media_id'],
                caption=channel_caption,
                parse_mode=ParseMode.HTML
            )
        else:
            await bot.send_message(
                chat_id=CHANNEL_ID,
                text=channel_caption,
                parse_mode=ParseMode.HTML
            )
    except Exception as e:
        logger.error(f"Ошибка публикации в ТГК (ID: {post['post_id']}): {e}")
        raise


async def notify_user_approved(post: dict, publish_type: str):
    user = get_user(post['user_id'])
    if user:
        try:
            post_text = (post['caption'] or "")[:20]
            if len(post['caption'] or "") > 20: post_text += "..."
            publish_text = "ТГК и Instagram" if publish_type == "both" else "ТГК"
            await bot.send_message(
                chat_id=user['user_id'],
                text=f"Ваш пост «{post_text}» одобрен и будет опубликован в {publish_text}!"
            )
        except TelegramForbiddenError:
            logger.warning(f"Пользователь {user['user_id']} заблокировал бота.")
        except Exception as e:
            logger.error(f"Не удалось уведомить пользователя об одобрении: {e}")


@dp.callback_query(F.data.startswith("publish_"))
async def publish_choice(callback: CallbackQuery):
    if callback.message.chat.id != ADMIN_CHAT_ID:
        return
    data = callback.data.split("_")
    publish_type = data[1]
    post_id = int(data[2])
    post = get_post(post_id)
    moderator_id = callback.from_user.id

    try:
        # 1. Публикация в ТГК
        await publish_post(post)

        # 2. Публикация в Instagram, если выбрано
        if publish_type == "both":
            await send_instagram_post(post)

    except Exception as e:
        logger.error(f"Ошибка публикации поста: {e}")
        await callback.answer("Ошибка публикации!", show_alert=True)
        return

    published_in = "both" if publish_type == "both" else "tg"
    update_post_status(
        post_id=post_id,
        status="approved",
        published_in=published_in,
        moderator_id=moderator_id
    )

    # 3. Обновляем сообщение в чате админа (заменяет кнопки на статус)
    await update_admin_message(post_id, callback.message)

    # 4. Уведомление пользователя
    user = get_user(post['user_id'])
    if user:
        try:
            post_text = (post['caption'] or "")[:20]
            if len(post['caption'] or "") > 20: post_text += "..."
            publish_text = "ТГК и Instagram" if publish_type == "both" else "ТГК"
            await bot.send_message(
                chat_id=user['user_id'],
                text=f"Ваш пост «{post_text}» одобрен и будет опубликован в {publish_text}!"
            )
        except Exception as e:
            logger.error(f"Не удалось уведомить пользователя: {e}")

    await callback.answer("Пост опубликован!")


@dp.callback_query(F.data.startswith("reject_"))
async def reject_post(callback: CallbackQuery):
    if callback.message.chat.id != ADMIN_CHAT_ID:
        return

    post_id = int(callback.data.split("_")[1])
    post = get_post(post_id)

    if not post:
        await callback.answer("Ошибка: Пост не найден в БД.", show_alert=True)
        return

    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="Реклама", callback_data=f"reason_ads_{post_id}"))
    builder.add(InlineKeyboardButton(text="Нарушение правил", callback_data=f"reason_rules_{post_id}"))
    builder.add(InlineKeyboardButton(text="Предупредить", callback_data=f"warning_{post_id}"))
    builder.add(InlineKeyboardButton(text="Забанить", callback_data=f"reason_ban_{post_id}"))
    builder.add(InlineKeyboardButton(text="Отмена", callback_data=f"cancel_reject_{post_id}"))
    builder.adjust(2, 2, 1)

    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: ЯВНО ПЕРЕДАЕМ КОНТЕНТ ИЗ БД
    message = callback.message
    is_media = post['media_type'] in ['photo', 'video']
    current_content = post.get('caption') or "<code>[ТЕКСТ ИЗ БД ПОТЕРЯН НА ШАГЕ REJECT]</code>"
    content_to_edit = current_content + "\u200b"  # Невидимый символ

    try:
        if is_media:
            await message.edit_caption(
                caption=content_to_edit,
                reply_markup=builder.as_markup(),
                parse_mode=ParseMode.HTML
            )
        else:
            await message.edit_text(
                text=content_to_edit,
                reply_markup=builder.as_markup(),
                parse_mode=ParseMode.HTML
            )
    except TelegramBadRequest as e:
        if "message is not modified" not in str(e):
            logger.error(f"Ошибка сохранения контента в reject_post: {e}")
            await callback.answer("Ошибка при изменении кнопок!", show_alert=True)

    await callback.answer()


@dp.callback_query(F.data.startswith("reason_"))
async def reason_reject(callback: CallbackQuery):
    if callback.message.chat.id != ADMIN_CHAT_ID:
        return

    data = callback.data.split("_")
    reason_type = data[1]
    post_id = int(data[2])
    moderator_id = callback.from_user.id
    post = get_post(post_id)

    status = "rejected_unknown"
    reason_text = "Неизвестная причина"

    if reason_type == 'ads':
        status = "rejected_ad"
        reason_text = "Реклама"
    elif reason_type == 'rules':
        status = "rejected_rules"
        reason_text = "Нарушение правил"
    elif reason_type == 'ban':
        status = "rejected_ban"
        reason_text = "Нарушение правил + Бан"

    update_post_status(
        post_id=post_id,
        status=status,
        moderator_id=moderator_id
    )

    await update_admin_message(post_id, callback.message)

    # Уведомление пользователя
    user = get_user(post['user_id'])
    if user:
        try:
            post_text = (post['caption'] or "")[:20]
            if len(post['caption'] or "") > 20: post_text += "..."

            if reason_type == 'ads':
                message_text = (
                    f"Ваш пост «{post_text}» отклонен по причине: Реклама/услуги\n\n"
                    "Размещение рекламных постов возможно только на платной основе.\n"
                    "Для сотрудничества обратитесь к администрации - @shiebeid"
                )
            elif reason_type == 'rules':
                message_text = (
                    f"Ваш пост «{post_text}» отклонен по причине: Нарушение правил. "
                    "Ознакомьтесь с правилами публикации:\n\n"
                    "1. Запрещены оскорбления и дискриминация\n"
                    "2. Не публикуйте личную информацию\n"
                    "3. Запрещена реклама и спам\n"
                    "4. Контент должен иметь смысловую нагрузку\n"
                    "5. Максимальная длина видео - 10 секунд\n"
                    "6. Нельзя использовать в постах номера телефонов\n"
                    "7. Нельзя представляться другим человеком (даже отмечая свой аккаунт)\n"
                    "8. Нельзя публиковать своих знакомых, друзей\n"
                    "9. Не публикуем посты с анкетами, с поиском друзей\парней\девушек, связанное с ДВ, сайт знакомств\n\n"
                    "Нарушение правил ведет к бану!"
                )
            else:
                message_text = f"Ваш пост «{post_text}» отклонен по причине: {reason_text}"

            if reason_type == 'ban':
                message_text += "\n\nВы забанены в этом боте!"
                ban_user(user['user_id'], reason_text, moderator_id)

            await bot.send_message(
                chat_id=user['user_id'],
                text=message_text
            )
        except Exception as e:
            logger.error(f"Не удалось уведомить пользователя: {e}")

    await callback.answer("Пост отклонен.")


@dp.callback_query(F.data.startswith("warning_"))
async def warning_reject(callback: CallbackQuery):
    if callback.message.chat.id != ADMIN_CHAT_ID:
        return

    post_id = int(callback.data.split("_")[1])
    moderator_id = callback.from_user.id
    post = get_post(post_id)

    status = "rejected_warning"
    reason_text = "Предупреждение"

    update_post_status(
        post_id=post_id,
        status=status,
        moderator_id=moderator_id
    )

    warn_user(post['user_id'], post_id, moderator_id)

    await update_admin_message(post_id, callback.message)

    # Уведомление пользователя
    user = get_user(post['user_id'])
    if user:
        try:
            post_text = (post['caption'] or "")[:20]
            if len(post['caption'] or "") > 20: post_text += "..."

            message_text = (
                f"Ваш пост «{post_text}» отклонен.\n\n"
                "Вынесено предупреждение. Будьте внимательны при публикации постов. "
                "Следующее нарушение может привести к бану."
            )

            await bot.send_message(
                chat_id=user['user_id'],
                text=message_text
            )
        except Exception as e:
            logger.error(f"Не удалось уведомить пользователя: {e}")

    await callback.answer("Пост отклонен, пользователю вынесено предупреждение.")


@dp.callback_query(F.data.startswith("cancel_approve_"))
async def handle_cancel_approve(callback: CallbackQuery):
    if callback.message.chat.id != ADMIN_CHAT_ID:
        return
    post_id = int(callback.data.split("_")[2])
    # Вызываем функцию восстановления
    await restore_admin_message(callback, post_id)


@dp.callback_query(F.data.startswith("cancel_reject_"))
async def handle_cancel_reject(callback: CallbackQuery):
    if callback.message.chat.id != ADMIN_CHAT_ID:
        return
    post_id = int(callback.data.split("_")[2])
    # Вызываем функцию восстановления
    await restore_admin_message(callback, post_id)


@dp.message(Command("ban"))
async def cmd_ban(message: Message):
    if message.chat.id != ADMIN_CHAT_ID:
        return

    try:
        match = re.match(r'/ban\s+(@?\w+)\s*(.*)', message.text)
        if not match:
            await message.answer("Использование: /ban <user_id|@username> <причина>")
            return

        identifier = match.group(1)
        reason = match.group(2) or "Причина не указана"

        user_id = None
        if identifier.startswith('@'):
            username = identifier[1:]
            with db_connection() as cursor:
                cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
                result = cursor.fetchone()
                if not result:
                    await message.answer(f"Пользователь @{username} не найден")
                    return
                user_id = result[0]
        else:
            try:
                user_id = int(identifier)
            except ValueError:
                await message.answer("Некорректный идентификатор. Используйте ID или @username")
                return

        if ban_user(user_id, reason, message.from_user.id):
            await message.answer(f"Пользователь {identifier} забанен!")
            user = get_user(user_id)
            if user:
                try:
                    await bot.send_message(user_id, f"Вы забанены в боте. Причина: {reason}")
                except:
                    pass
        else:
            await message.answer("Ошибка бана")

    except Exception as e:
        logger.error(f"Ошибка в команде /ban: {e}")
        await message.answer("Ошибка выполнения команды")


@dp.message(Command("unban"))
async def cmd_unban(message: Message):
    if message.chat.id != ADMIN_CHAT_ID:
        return

    try:
        match = re.match(r'/unban\s+(@?\w+)', message.text)
        if not match:
            await message.answer("Использование: /unban <user_id|@username>")
            return

        identifier = match.group(1)

        user_id = None
        if identifier.startswith('@'):
            username = identifier[1:]
            with db_connection() as cursor:
                cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
                result = cursor.fetchone()
                if not result:
                    await message.answer(f"Пользователь @{username} не найден")
                    return
                user_id = result[0]
        else:
            try:
                user_id = int(identifier)
            except ValueError:
                await message.answer("Некорректный идентификатор. Используйте ID или @username")
                return

        if unban_user(user_id):
            await message.answer(f"Пользователь {identifier} разбанен!")
            user = get_user(user_id)
            if user:
                try:
                    await bot.send_message(user_id, "Вы разблокированы в боте! Пожалуйста соблюдайте правила.")
                except:
                    pass
        else:
            await message.answer("Пользователь не найден или не был забанен")

    except Exception as e:
        logger.error(f"Ошибка в команде /unban: {e}")
        await message.answer("Ошибка выполнения команды")


@dp.message(Command("stats"))
async def cmd_stats(message: Message):
    if message.chat.id != ADMIN_CHAT_ID:
        return

    try:
        stats = get_monthly_stats()

        if stats.get('error'):
            await message.answer(stats['error'])
            return

        response = "<b>Статистика за последние 30 дней:</b>\n\n"
        response += f"<b>Всего постов:</b> {stats['total_posts']}\n\n"

        response += "<b>Тип анонимности:</b>\n"
        response += f" • Анонимно: {stats['anonymous']}\n"
        response += f" • Открыто: {stats['non_anonymous']}\n\n"

        response += "<b>Тип контента:</b>\n"
        response += f" • Фото: {stats['photo']}\n"
        response += f" • Видео: {stats['video']}\n"
        response += f" • Текст: {stats['text']}\n\n"

        response += "<b>Статусы постов:</b>\n"
        response += f" • Одобрено: {stats['approved']}\n"
        response += f" • Отклонено: {stats['rejected']}\n"
        response += f" • Ожидает решения: {stats['pending']}\n\n"

        response += "<b>Причины отклонений:</b>\n"
        response += f" • Реклама: {stats['rejected_reason']['ads']}\n"
        response += f" • Нарушение правил: {stats['rejected_reason']['rules']}\n"
        response += f" • Бан: {stats['rejected_reason']['bans']}\n\n"

        if stats.get('rejected_details'):
            response += "<b>Детали отклонений (статус):</b>\n"
            for status, count in stats['rejected_details'].items():
                status_name = {
                    'rejected_ad': 'Реклама',
                    'rejected_rules': 'Нарушение правил',
                    'rejected_warning': 'Предупреждение',
                    'rejected_ban': 'Бан'
                }.get(status, status)
                response += f" • {status_name}: {count}\n"

        await message.answer(response, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.error(f"Ошибка при формировании статистики: {e}")
        await message.answer(f"Произошла ошибка при формировании статистики: {str(e)}")


@dp.errors()
async def errors_handler(event: types.ErrorEvent):
    logger.error(f"Ошибка в обработчике: {event.exception}\nВ обновлении: {event.update}")
    return True


# ============================= Запуск бота =============================

async def main():
    """Главная функция запуска бота, без логики прокси."""
    initialize_database()

    # Проверка соединения
    try:
        bot_info = await bot.get_me()
        logger.info(f"Успешное соединение с Telegram API! Бот: @{bot_info.username}")
    except Exception as e:
        logger.critical(
            f"Критическая ошибка: Не удалось подключиться к Telegram API. Проверьте BOT_TOKEN и сетевое соединение. Ошибка: {e}")
        sys.exit(1)

    logger.info("Удаление активного вебхука...")
    try:
        # Устанавливаем более долгий таймаут для стабильности
        await bot.delete_webhook(drop_pending_updates=True, request_timeout=15)
        logger.info("Вебхук успешно удален")
    except Exception as e:
        logger.error(f"Ошибка при удалении вебхука: {e}")

    logger.info("Бот запущен в режиме polling")
    # Установим таймаут для стабильности
    await dp.start_polling(bot, polling_timeout=30)


if __name__ == "__main__":
    logger.info("Запуск бота...")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:

        logger.info("Бот остановлен вручную")
