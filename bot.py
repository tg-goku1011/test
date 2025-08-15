# bot.py
"""
Telegram Video Credits Bot — JSON storage edition (reply keyboard + robust media index)
Requirements:
  - Python 3.10+
  - python-telegram-bot >= 20
  - pydantic

Quick start:
  1) pip install python-telegram-bot==20.7 pydantic
  2) Edit BOT_TOKEN / OWNER_ID / ADMIN_IDS below
  3) python bot.py
"""
import json
import random
from typing import Optional
import os
import json
import logging
import asyncio
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from telegram import (
    Update, InlineKeyboardMarkup, InlineKeyboardButton, Message, LabeledPrice,
    InputFile, constants, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
)
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters, PreCheckoutQueryHandler
)

# ====================== CONFIG ======================
# >>>> FILL THESE WITH YOUR REAL VALUES <<<<
BOT_TOKEN = "8147391663:AAGIUGlv1YnS8DT7BmXaNRZy3m4kCGss0RY"
OWNER_ID = 8074331297                    # e.g., 8074331297
ADMIN_IDS = [OWNER_ID, 8114114957]       # Add more admin IDs here
DATA_DIR = "."                          # Or a custom directory

# ---------------- BASIC LOGGING ----------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("video-credits-bot-json")

# ---------------- FILENAMES ----------------
DATA_FILE = os.path.join(DATA_DIR, "data.json")
WATCHED_FILE = os.path.join(DATA_DIR, "watched.json")

# ---------------- DEFAULTS & SETTINGS ----------------
FREE_WELCOME_CREDITS = 10
CREDIT_PACKS = [(10, 0), (30, 5), (60, 10), (100, 15)]
CHECKING_SECONDS = 8
PAYMENT_REMINDER_MINUTES = 30

# Default categories (label, price, code)
DEFAULT_CATEGORIES = [
    ("Dark", 5, "dark"),
    ("RP", 4, "rp"),
    ("MOM & SO", 4, "momso"),
    ("Desi", 3, "desi"),
    ("Snapchat", 3, "snap"),
    ("Smart", 2, "smart"),
    ("Boxes", 4, "boxes"),
    ("Mix", 1, "mix"),
]

# ---------------- DATA MODELS ----------------
class User(BaseModel):
    user_id: int
    username: Optional[str] = None
    first_name: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    credits: int = FREE_WELCOME_CREDITS
    is_banned: bool = False
    joined_once: bool = False

class Category(BaseModel):
    code: str
    label: str
    price: int
    channel_id: Optional[int] = None  # -100... id expected

class MediaEntry(BaseModel):
    channel_id: int
    message_id: int
    file_id: Optional[str] = None
    type: Optional[str] = None  # "video"/"document"/"photo"
    caption: Optional[str] = ""
    message_link: Optional[str] = None
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# ---------------- STORAGE & LOCKS ----------------
# data.json structure:
# {
#   "users": { "<user_id>": { ... } },
#   "categories": [ {code,label,price,channel_id}, ... ],
#   "media": { "<channel_id>": [ {message_id,file_id,type,caption,message_link,ts}, ... ] },
#   "orders": [ {user_id,cat_code,channel_id,message_id,price,ts}, ... ],
#   "admins": { "owner": OWNER_ID, "admins": [id,id,...] },
#   "cfg": {
#       "welcome_text": "...",               # plain text fallback
#       "welcome_payload": {...},            # stored media/text for /start
#       "support_username": null,
#       "force_sub_enabled": false,
#       "force_sub_channels": [],
#       "upi_qr_by_pack": { "10": {...}, ...}
#   }
# }
STORE_LOCK = asyncio.Lock()
WATCH_LOCK = asyncio.Lock()

def atomic_write_json(path: str, data: dict):
    dirn = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dirn, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise

async def load_data() -> Dict[str, Any]:
    async with STORE_LOCK:
        if not os.path.exists(DATA_FILE):
            seed = {
                "users": {},
                "categories": [],
                "media": {},
                "orders": [],
                "admins": {"owner": OWNER_ID, "admins": list(set(ADMIN_IDS))},
                "cfg": {
                    "welcome_text": "👋 Welcome! (welcome not configured)\n\nOwner: set the welcome message by running /setwelcome and forwarding/sending the message to me.",
                    "welcome_payload": None,
                    "support_username": None,
                    "force_sub_enabled": False,
                    "force_sub_channels": [],
                    "upi_qr_by_pack": {}
                }
            }
            for label, price, code in DEFAULT_CATEGORIES:
                seed["categories"].append({"code": code, "label": label, "price": price, "channel_id": None})
            atomic_write_json(DATA_FILE, seed)
            return seed
        with open(DATA_FILE, "r", encoding="utf8") as f:
            return json.load(f)

async def save_data(d: Dict[str, Any]):
    async with STORE_LOCK:
        atomic_write_json(DATA_FILE, d)

async def load_watched():
    if not os.path.exists(WATCHED_FILE):
        return {}
    with open(WATCHED_FILE, "r") as f:
        return json.load(f)
    
async def save_watched(data):
    with open(WATCHED_FILE, "w") as f:
        json.dump(data, f, indent=2)


async def sample_unseen_media_for_user(user_id: int, channel_id: int, sample_size: int = 20) -> Optional[dict]:
    """
    Return a random media item from the given channel that the user hasn't watched yet.
    """
    # Load channel messages
    data_file = f"channel_{channel_id}.json"
    if not os.path.exists(data_file):
        return None  # No media collected yet

    with open(data_file, "r") as f:
        items = json.load(f)

    # Load watched data
    watched = await load_watched()
    user_watched = set(watched.get(str(user_id), {}).get(str(channel_id), []))

    # Filter unseen items
    unseen = [item for item in items if str(item["message_id"]) not in user_watched]
    if not unseen:
        return None

    # Sample one randomly (or first, up to sample_size)
    sample_pool = unseen[:sample_size]
    return random.choice(sample_pool)


async def mark_watched(user_id: int, channel_id: int, message_id: int):
    data = await load_watched()
    uid = str(user_id); cid = str(channel_id); mid = int(message_id)
    data.setdefault(uid, {})
    data[uid].setdefault(cid, [])
    if mid not in data[uid][cid]:
        data[uid][cid].append(mid)
    await save_watched(data)

async def get_watched_set(user_id: int, channel_id: int) -> set:
    data = await load_watched()
    return set(int(x) for x in data.get(str(user_id), {}).get(str(channel_id), []))

# ---------------- HELPERS ----------------
def is_owner(uid: int) -> bool:
    return uid == OWNER_ID

async def is_admin(uid: int) -> bool:
    d = await load_data()
    admins = set(d.get("admins", {}).get("admins", []))
    return uid in admins

async def add_admin(uid: int):
    d = await load_data()
    admins = d.setdefault("admins", {}).setdefault("admins", [])
    if uid not in admins:
        admins.append(uid)
        await save_data(d)

async def remove_admin(uid: int):
    d = await load_data()
    admins = d.setdefault("admins", {}).setdefault("admins", [])
    if uid in admins and uid != d.get("admins", {}).get("owner"):
        admins.remove(uid)
        await save_data(d)

async def get_user(uid: int, udata: Optional[Update]=None) -> dict:
    d = await load_data()
    users = d.setdefault("users", {})
    u = users.get(str(uid))
    if not u:
        user_obj = User(
            user_id=uid,
            username=(udata.effective_user.username if udata and udata.effective_user else None),
            first_name=(udata.effective_user.first_name if udata and udata.effective_user else None),
        )
        u = user_obj.model_dump()
        users[str(uid)] = u
        await save_data(d)
    return u

async def adjust_credits(uid: int, delta: int):
    d = await load_data()
    users = d.setdefault("users", {})
    u = users.get(str(uid))
    if not u:
        u = User(user_id=uid).model_dump()
        users[str(uid)] = u
    u["credits"] = max(0, int(u.get("credits", 0)) + int(delta))
    await save_data(d)

async def set_user_credits(uid: int, value: int):
    d = await load_data()
    users = d.setdefault("users", {})
    u = users.get(str(uid))
    if not u:
        u = User(user_id=uid).model_dump()
        users[str(uid)] = u
    u["credits"] = max(0, int(value))
    await save_data(d)

async def create_order(record: dict):
    d = await load_data()
    d.setdefault("orders", []).append(record)
    await save_data(d)

async def list_categories() -> List[dict]:
    d = await load_data()
    return d.get("categories", [])

async def set_channel_for_category(index: int, channel_id: int):
    d = await load_data()
    cats = d.setdefault("categories", [])
    if 0 <= index < len(cats):
        cats[index]["channel_id"] = int(channel_id)
        await save_data(d)
        return True
    return False

def build_message_link(chat_id: int, message_id: int, username: Optional[str]) -> Optional[str]:
    # Public channel -> https://t.me/<username>/<message_id>
    # Private channel -> https://t.me/c/<abs(chat_id)[4:]>/<message_id>  (chat_id like -100xxxxxxxxx)
    if username:
        return f"https://t.me/{username}/{message_id}"
    s = str(chat_id)
    if s.startswith("-100"):
        return f"https://t.me/c/{s[4:]}/{message_id}"
    return None

# ---------------- HELPER: INDEX CHANNEL POST ----------------
async def index_channel_post(msg):
    """
    Save channel post media info (video, photo, document) into JSON for tracking.
    """
    if not msg or not msg.chat_id:
        return

    # Only index media posts
    media_type = None
    file_id = None
    caption = msg.caption or ""
    
    if msg.video:
        media_type = "video"
        file_id = msg.video.file_id
    elif msg.document:
        media_type = "document"
        file_id = msg.document.file_id
    elif msg.photo:
        media_type = "photo"
        file_id = msg.photo[-1].file_id  # largest resolution

    if not media_type or not file_id:
        return  # skip non-media messages

    d = await load_data()
    # Check if this channel is assigned to any category
    cats = d.get("categories", [])
    cat_idx = None
    for i, c in enumerate(cats):
        if str(c.get("channel_id")) == str(msg.chat_id):
            cat_idx = i
            break
    if cat_idx is None:
        return  # channel not linked to any category

    cat = cats[cat_idx]
    cat.setdefault("media", [])

    # Avoid duplicates
    if any(m["message_id"] == msg.message_id for m in cat["media"]):
        return

    cat["media"].append({
        "message_id": msg.message_id,
        "type": media_type,
        "file_id": file_id,
        "caption": caption
    })

    await save_data(d)
    log.info(f"Indexed post {msg.message_id} in category '{cat.get('label')}'")

# ---------------- KEYBOARDS ----------------
def kb_store_inline() -> InlineKeyboardMarkup:
    rows = []
    for qty, disc in CREDIT_PACKS:
        rows.append([InlineKeyboardButton(f"Buy {qty} credits ({disc}% off)", callback_data=f"buy:{qty}:{disc}")])
    rows.append([InlineKeyboardButton("🔙 Back", callback_data="back_main")])
    return InlineKeyboardMarkup(rows)

def kb_after_upi(pack_key: str, support_username: Optional[str]) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton("✅ Payment Done", callback_data=f"pd:upi:{pack_key}")]]
    if support_username:
        rows.append([InlineKeyboardButton("🆘 Contact Support", url=f"https://t.me/{support_username}")])
    rows.append([InlineKeyboardButton("🔙 Back", callback_data="store")])
    return InlineKeyboardMarkup(rows)

def reply_kb_main(categories: List[dict]) -> ReplyKeyboardMarkup:
    """Reply keyboard near typing area: 2 buttons per row."""
    # Each button text: "1. Dark", "2. RP", ...
    buttons: List[List[KeyboardButton]] = []
    row: List[KeyboardButton] = []
    for i, c in enumerate(categories):
        label = c.get("label", c.get("code"))
        row.append(KeyboardButton(f"{i+1}. {label}"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    # Add action row
    buttons.append([KeyboardButton("🛒 Buy Credits"), KeyboardButton("💳 Credits")])
    buttons.append([KeyboardButton("📂 Categories")])
    return ReplyKeyboardMarkup(buttons, resize_keyboard=True)

# ---------------- HANDLERS ----------------
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user:
        return
    await get_user(user.id, update)

    d = await load_data()
    cats = d.get("categories", [])
    kb = reply_kb_main(cats)

    # Send welcome payload if configured, else default helper text
    cfg = d.get("cfg", {})
    wp = cfg.get("welcome_payload")
    if wp:
        typ = wp.get("type")
        cap = wp.get("caption", "")
        try:
            if typ == "text":
                await context.bot.send_message(chat_id=update.effective_chat.id, text=cap)
            elif typ == "photo":
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=wp.get("file_id"), caption=cap)
            elif typ == "video":
                await context.bot.send_video(chat_id=update.effective_chat.id, video=wp.get("file_id"), caption=cap)
            elif typ == "document":
                await context.bot.send_document(chat_id=update.effective_chat.id, document=wp.get("file_id"), caption=cap)
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=cfg.get("welcome_text"))
        except Exception:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=cfg.get("welcome_text"))
    else:
        # explicit instruction when welcome is not configured
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="👋 Welcome! (welcome not configured)\n\nOwner: set the welcome message by running /setwelcome and forwarding/sending the message to me."
        )

    # Then send the reply keyboard near typing area
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Choose a category or open the store:",
        reply_markup=kb
    )

async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    base = (
        "/start — Main menu\n"
        "/credits — Show credits\n"
        "/categories — List categories & channel ids\n"
        "/buy — Open credit packs\n"
    )
    if await is_admin(uid):
        admin = (
            "\nAdmin commands:\n"
            "/setwelcome — reply with text/media to set welcome\n"
            "/setchannel <cat_number> <channel_id>\n"
            "/setcategory <cat_number> <price> <label>\n"
            "/setqr <pack_qty> (reply to UPI QR media)\n"
            "/promote <user_id>\n"
            "/demote <user_id>\n"
            "/addcredits <user_id> <amount>\n"
            "/users\n"
            "/broadcast (reply to message + /broadcast)\n"
            "/dm <user_id> <message>\n"
            "/stats\n"
            "/exportdata\n"
        )
        await update.message.reply_text(base + admin)
    else:
        await update.message.reply_text(base)

async def credits_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    u = await get_user(uid)
    credits = int(u.get("credits", 0))
    await update.message.reply_text(f"💳 Your credits: {credits}\nUse them to unlock videos. Tap 🛒 Buy Credits to add more.")

async def categories_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    d = await load_data()
    cats = d.get("categories", [])
    lines = ["📂 *Available Categories*", ""]
    for i, c in enumerate(cats):
        ch = c.get("channel_id")
        ch_str = f"`{ch}`" if ch else "_Not set_"
        lines.append(f"*{i+1}. {c.get('label')}*  ·  💵 *{c.get('price')}*  ·  {ch_str}")
    text = "\n".join(lines)
    await update.message.reply_text(text, parse_mode=constants.ParseMode.MARKDOWN)

# ---- set welcome (owner/admin) ----
async def setwelcome_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not await is_admin(uid):
        return
    msg = update.message
    # Use the message that this command replies to (if any), else the command message itself
    src = msg.reply_to_message or msg

    payload = None
    if src.photo:
        payload = {"type": "photo", "file_id": src.photo[-1].file_id, "caption": src.caption or ""}
    elif src.video:
        payload = {"type": "video", "file_id": src.video.file_id, "caption": src.caption or ""}
    elif src.document:
        payload = {"type": "document", "file_id": src.document.file_id, "caption": src.caption or ""}
    else:
        # treat as plain text (caption = text)
        txt = (src.text or src.caption or "").strip()
        if not txt:
            await update.message.reply_text("Send text or reply to a media with /setwelcome.")
            return
        payload = {"type": "text", "file_id": None, "caption": txt}

    d = await load_data()
    cfg = d.setdefault("cfg", {})
    cfg["welcome_payload"] = payload
    await save_data(d)
    await update.message.reply_text("✅ Welcome message saved.")

# ---- Admin helpers ----
async def adminhelp_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Kept for convenience; /help already shows admin commands if admin.
    await help_handler(update, context)

# ---------------- SET CHANNEL HANDLER ----------------
async def setchannel_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not await is_admin(uid):
        return
    parts = (update.message.text or "").split()
    if len(parts) != 3:
        await update.message.reply_text("Usage: /setchannel <category_number> <channel_id>")
        return
    try:
        idx = int(parts[1]) - 1
        ch = int(parts[2])
    except:
        await update.message.reply_text("Invalid numbers.")
        return

    # Verify bot admin status in channel
    try:
        bot_member = await context.bot.get_chat_member(chat_id=ch, user_id=context.bot.id)
        if bot_member.status not in ("administrator", "creator"):
            await update.message.reply_text(
                "❌ Bot is not an admin in this channel. Please make it admin first."
            )
            return
    except Exception as e:
        await update.message.reply_text(f"❌ Failed to verify bot status: {e}")
        return

    # Assign channel to category
    d = await load_data()
    cats = d.get("categories", [])
    if idx < 0 or idx >= len(cats):
        await update.message.reply_text("❌ Invalid category number.")
        return

    cats[idx]["channel_id"] = ch
    cats[idx].setdefault("media", [])
    await save_data(d)
    await update.message.reply_text(
        "✅ Channel updated for category.\n• Bot is admin ✔️\n• New posts will be indexed automatically."
    )


async def setcategory_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not await is_admin(uid):
        return
    parts = (update.message.text or "").split()
    if len(parts) < 4:
        await update.message.reply_text("Usage: /setcategory <category_number> <price> <label...>")
        return
    try:
        idx = int(parts[1]) - 1
        price = int(parts[2])
    except:
        await update.message.reply_text("Invalid number inputs.")
        return
    label = " ".join(parts[3:])
    d = await load_data()
    cats = d.setdefault("categories", [])
    if idx < 0 or idx >= len(cats):
        await update.message.reply_text("Invalid category number.")
        return
    cats[idx]["price"] = price
    cats[idx]["label"] = label
    await save_data(d)
    await update.message.reply_text("✅ Category updated.")

async def setqr_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not await is_admin(uid):
        return
    parts = (update.message.text or "").split()
    if len(parts) != 2 or not update.message.reply_to_message:
        await update.message.reply_text("Usage: reply to a media with /setqr <pack_qty>")
        return
    pack = parts[1]
    src = update.message.reply_to_message
    payload = {}
    if src.photo:
        payload = {"type": "photo", "file_id": src.photo[-1].file_id, "caption": src.caption or ""}
    elif src.document:
        payload = {"type": "document", "file_id": src.document.file_id, "caption": src.caption or ""}
    elif src.video:
        payload = {"type": "video", "file_id": src.video.file_id, "caption": src.caption or ""}
    else:
        await update.message.reply_text("Reply must contain photo/document/video of UPI QR.")
        return
    d = await load_data()
    cfg = d.setdefault("cfg", {})
    upi = cfg.setdefault("upi_qr_by_pack", {})
    upi[str(pack)] = payload
    await save_data(d)
    await update.message.reply_text(f"✅ UPI QR saved for pack {pack}.")

async def promote_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not await is_admin(uid):
        return
    parts = (update.message.text or "").split()
    if len(parts) != 2:
        await update.message.reply_text("Usage: /promote <user_id>")
        return
    try:
        target = int(parts[1])
    except:
        await update.message.reply_text("Invalid user id.")
        return
    await add_admin(target)
    await update.message.reply_text("✅ Promoted as admin.")

async def demote_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not await is_admin(uid):
        return
    parts = (update.message.text or "").split()
    if len(parts) != 2:
        await update.message.reply_text("Usage: /demote <user_id>")
        return
    try:
        target = int(parts[1])
    except:
        await update.message.reply_text("Invalid user id.")
        return
    await remove_admin(target)
    await update.message.reply_text("✅ Demoted (if not owner).")

async def addcredits_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not await is_admin(uid):
        return
    parts = (update.message.text or "").split()
    if len(parts) != 3:
        await update.message.reply_text("Usage: /addcredits <user_id> <amount>")
        return
    try:
        target = int(parts[1]); amount = int(parts[2])
    except:
        await update.message.reply_text("Invalid args.")
        return
    await adjust_credits(target, amount)
    await update.message.reply_text("✅ Credits adjusted.")

async def users_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not await is_admin(uid):
        return
    d = await load_data()
    users = d.get("users", {})
    total = len(users)
    sample_lines = []
    for i, (k, v) in enumerate(users.items()):
        if i >= 10:
            break
        sample_lines.append(f"- {k} @{v.get('username','')} credits={v.get('credits',0)}")
    await update.message.reply_text(f"Total users: {total}\n" + ("\n".join(sample_lines) if sample_lines else ""))

async def dm_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not await is_admin(uid):
        return
    parts = (update.message.text or "").split(maxsplit=2)
    if len(parts) < 3:
        await update.message.reply_text("Usage: /dm <user_id> <message>")
        return
    try:
        target = int(parts[1])
    except:
        await update.message.reply_text("Invalid user id.")
        return
    msg = parts[2]
    try:
        await context.bot.send_message(chat_id=target, text=msg)
        await update.message.reply_text("✅ Message sent.")
    except Exception as e:
        await update.message.reply_text(f"Failed to send: {e}")

BROADCAST_TASKS_LOCK = asyncio.Lock()
async def broadcast_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not await is_admin(uid):
        return
    if not update.message.reply_to_message:
        await update.message.reply_text("Reply to the message you want to broadcast, then send /broadcast.")
        return
    src = update.message.reply_to_message
    d = await load_data()
    users = list(d.get("users", {}).keys())
    total = len(users)
    await update.message.reply_text(f"📣 Broadcast starting to {total} users...")
    succ = fail = 0
    for k in users:
        try:
            target_id = int(k)
            await context.bot.copy_message(chat_id=target_id, from_chat_id=src.chat_id, message_id=src.message_id)
            succ += 1
        except Exception:
            fail += 1
        await asyncio.sleep(0.05)
    await update.message.reply_text(f"✅ Broadcast finished. Sent: {succ}, Fail: {fail}")

async def stats_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not await is_admin(uid):
        return
    d = await load_data()
    total_users = len(d.get("users", {}))
    total_orders = len(d.get("orders", []))
    total_credits_sold = sum([int(o.get("price", 0)) for o in d.get("orders", [])])
    total_videos_sent = total_orders
    await update.message.reply_text(
        f"👥 Users: {total_users}\n🎬 Videos delivered: {total_videos_sent}\n🧾 Credits recorded in orders: {total_credits_sold}"
    )

async def exportdata_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not await is_admin(uid):
        return
    files = [p for p in [DATA_FILE, WATCHED_FILE] if os.path.exists(p)]
    if not files:
        await update.message.reply_text("No files found.")
        return
    for p in files:
        try:
            await context.bot.send_document(chat_id=update.effective_chat.id, document=InputFile(p))
        except Exception as e:
            await update.message.reply_text(f"Failed to send {p}: {e}")

# ---------------- STORE / PAYMENTS ----------------
async def store_open(update: Update, context: ContextTypes.DEFAULT_TYPE, via_button=False):
    if via_button and update.callback_query:
        await update.callback_query.message.edit_text("🛒 Choose a credit pack:", reply_markup=kb_store_inline())
    else:
        await update.message.reply_text("🛒 Choose a credit pack:", reply_markup=kb_store_inline())

async def precheckout_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.pre_checkout_query.answer(ok=True)

async def successful_payment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sp = update.message.successful_payment
    uid = update.effective_user.id
    qty = int(sp.invoice_payload or "0")
    if qty <= 0:
        return
    await adjust_credits(uid, qty)
    await create_order({
        "user_id": uid, "cat_code": None, "channel_id": None, "message_id": None,
        "price": qty, "ts": datetime.now(timezone.utc).isoformat()
    })
    await update.message.reply_text(f"✅ Added {qty} credits. Use /start.")

# ---------------- CALLBACKS ----------------
async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""
    uid = q.from_user.id
    d = await load_data()

    if data == "back_main":
        cats = d.get("categories", [])
        await q.message.edit_text("Back to store.", reply_markup=kb_store_inline())
        return

    if data == "store":
        await store_open(update, context, via_button=True)
        return

    if data.startswith("buy:"):
        _, qty, disc = data.split(":")
        qty_i = int(qty)
        rows = [[InlineKeyboardButton("⭐ Pay with Telegram Stars", callback_data=f"stars:{qty}")]]
        if d.get("cfg", {}).get("upi_qr_by_pack", {}).get(str(qty)):
            rows.append([InlineKeyboardButton("📤 Pay with UPI (QR)", callback_data=f"upi:{qty}")])
        rows.append([InlineKeyboardButton("🔙 Back", callback_data="back_main")])
        await q.message.edit_text(
            f"Pack: *{qty_i} credits* — discount *{disc}%*.\nChoose payment:",
            parse_mode=constants.ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(rows)
        )
        return

    if data.startswith("stars:"):
        qty = int(data.split(":")[1])
        prices = [LabeledPrice(label=f"{qty} Credits", amount=qty)]
        try:
            await context.bot.send_invoice(
                chat_id=uid,
                title=f"{qty} Credits",
                description="Credit pack purchase",
                payload=str(qty),
                provider_token="",   # set if you have a token; empty often works for Stars
                currency="XTR",
                prices=prices,
                start_parameter=f"buy_{qty}",
            )
        except Exception as e:
            await q.message.reply_text(f"Failed to send invoice: {e}")
        return

    if data.startswith("upi:"):
        qty = data.split(":")[1]
        upi = d.get("cfg", {}).get("upi_qr_by_pack", {}).get(str(qty))
        if not upi:
            await q.message.reply_text("UPI not configured for this pack.")
            return
        kb = kb_after_upi(str(qty), d.get("cfg", {}).get("support_username"))
        typ = upi.get("type"); fid = upi.get("file_id"); cap = upi.get("caption", "")
        try:
            if typ == "photo":
                await q.message.reply_photo(photo=fid, caption=cap, reply_markup=kb, protect_content=True)
            elif typ == "video":
                await q.message.reply_video(video=fid, caption=cap, reply_markup=kb, protect_content=True)
            elif typ == "document":
                await q.message.reply_document(document=fid, caption=cap, reply_markup=kb, protect_content=True)
            else:
                await q.message.reply_text("Invalid UPI payload.")
        except Exception:
            await q.message.reply_text("Failed to send UPI media.")
        context.application.job_queue.run_once(
            lambda ctx: ctx.bot.send_message(uid, "💡 Need help completing your payment? If you already paid, tap “✅ Payment Done”."),
            when=timedelta(minutes=PAYMENT_REMINDER_MINUTES)
        )
        return

    if data.startswith("pd:"):
        _, method, pack = data.split(":")
        checking = await q.message.reply_text("⏳ Checking payment...")
        await asyncio.sleep(CHECKING_SECONDS)
        try:
            await checking.delete()
        except:
            pass
        text = "❌ Payment not received."
        kb = InlineKeyboardMarkup(
            [[InlineKeyboardButton("🆘 Contact Support", url=f"https://t.me/{d.get('cfg', {}).get('support_username')}")]]
        ) if d.get('cfg', {}).get('support_username') else None
        await q.message.reply_text(text, reply_markup=kb)
        return

# ---------------- CHANNEL POST INDEXER ----------------
async def channel_post_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.channel_post
    if not msg:
        return
    try:
        await index_channel_post(msg)
    except Exception:
        log.exception("Indexing failed")

# ---------------- MESSAGE ROUTER (for reply keyboard taps) ----------------
async def main_message_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle taps from the ReplyKeyboard and general messages (non-commands)."""
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip()

    # Reply keyboard actions
    if text == "🛒 Buy Credits":
        await store_open(update, context, via_button=False)
        return
    if text == "💳 Credits":
        await credits_handler(update, context)
        return
    if text == "📂 Categories":
        await categories_handler(update, context)
        return

    # Category selection via "N. Label"
    if text[:1].isdigit() and "." in text:
        num_part = text.split(".", 1)[0].strip()
        try:
            idx = int(num_part) - 1
        except:
            return
        d = await load_data()
        cats = d.get("categories", [])
        if idx < 0 or idx >= len(cats):
            await update.message.reply_text("Invalid category.")
            return
        category = cats[idx]
        await handle_category_click(update, context, category)
        return

# ---------------- CATEGORY FLOW ----------------
async def handle_category_click(update: Update, context: ContextTypes.DEFAULT_TYPE, category: dict):
    uid = update.effective_user.id
    price = int(category.get("price", 1))
    channel_id = category.get("channel_id")

    # Force-sub (optional)
    d = await load_data()
    cfg = d.get("cfg", {})
    if cfg.get("force_sub_enabled"):
        missing = []
        for ch in cfg.get("force_sub_channels", []):
            try:
                cm = await context.bot.get_chat_member(chat_id=ch, user_id=uid)
                if cm.status not in ("member", "administrator", "creator"):
                    missing.append(ch)
            except Exception:
                missing.append(ch)
        if missing:
            rows = []
            for m in missing:
                if str(m).startswith("-100"):
                    rows.append([InlineKeyboardButton("➡️ Open Channel", url=f"https://t.me/c/{str(m)[4:]}")])
                else:
                    rows.append([InlineKeyboardButton("➡️ Open Channel", url=f"https://t.me/{m}")])
            rows.append([InlineKeyboardButton("✅ I Joined", callback_data="verify_join")])
            await update.message.reply_text("🔒 Please join required channels first.", reply_markup=InlineKeyboardMarkup(rows))
            return

    # Credits check
    u = await get_user(uid)
    have = int(u.get("credits", 0))
    if have < price:
        if have == 0:
            await update.message.reply_text(
                "⚠️ You have 0 credits. Buy credits to continue.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🛒 Buy Credits", callback_data="store")]])
            )
        else:
            await update.message.reply_text(
                f"⚠️ You need {price} credits for this category but you have less. Buy more to continue.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🛒 Buy Credits", callback_data="store")]])
            )
        return

    if not channel_id:
        await update.message.reply_text("This category does not have a linked channel yet. Admin can set it with /setchannel.")
        return

    # Pick unseen media
    item = await sample_unseen_media_for_user(uid, int(channel_id), sample_size=20)
    if not item:
        await update.message.reply_text("🎉 You have watched all available videos in this category. New videos coming soon!")
        return

    # Debit & record
    await adjust_credits(uid, -price)
    await mark_watched(uid, int(channel_id), int(item["message_id"]))
    await create_order({
        "user_id": uid,
        "cat_code": category.get("code"),
        "channel_id": int(channel_id),
        "message_id": int(item["message_id"]),
        "price": price,
        "ts": datetime.now(timezone.utc).isoformat()
    })

    # Deliver: prefer file_id; if it fails, fallback to copy_message from channel
    delivered = False
    try:
        if item.get("type") == "video" and item.get("file_id"):
            await update.message.reply_video(item["file_id"], caption=item.get("caption", ""), protect_content=True)
            delivered = True
        elif item.get("type") == "document" and item.get("file_id"):
            await update.message.reply_document(item["file_id"], caption=item.get("caption", ""), protect_content=True)
            delivered = True
        elif item.get("type") == "photo" and item.get("file_id"):
            await update.message.reply_photo(item["file_id"], caption=item.get("caption", ""), protect_content=True)
            delivered = True
    except Exception:
        delivered = False

    if not delivered:
        try:
            await context.bot.copy_message(
                chat_id=update.effective_chat.id,
                from_chat_id=int(channel_id),
                message_id=int(item["message_id"])
            )
            delivered = True
        except Exception:
            delivered = False

    if delivered:
        newu = await get_user(uid)
        if int(newu.get("credits", 0)) == 0:
            await update.message.reply_text(
                "⚠️ You have used all your credits. Buy more to continue:",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🛒 Buy Credits", callback_data="store")]])
            )
        else:
            await update.message.reply_text("✅ Delivered. Enjoy the video!")
    else:
        await update.message.reply_text(
            "❌ Failed to deliver media (possibly missing channel admin rights or removed post).",
        )

# ---------------- BUY SHORTCUT ----------------
async def buy_cmd_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await store_open(update, context, via_button=False)

# ---------------- PAYMENTS ----------------
# (precheckout + successful_payment already defined above)

# ---------------- STARTUP CHECK ----------------
def check_env():
    if not BOT_TOKEN:
        raise SystemExit("Set BOT_TOKEN at the top of bot.py.")
    if not OWNER_ID:
        raise SystemExit("Set OWNER_ID at the top of bot.py.")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

# ---------------- APP START ----------------
def main():
    check_env()
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # User commands
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("credits", credits_handler))
    app.add_handler(CommandHandler("categories", categories_handler))
    app.add_handler(CommandHandler("buy", buy_cmd_handler))

    # Admin commands
    app.add_handler(CommandHandler("setwelcome", setwelcome_handler))
    app.add_handler(CommandHandler("adminhelp", adminhelp_handler))
    app.add_handler(CommandHandler("setchannel", setchannel_handler))
    app.add_handler(CommandHandler("setcategory", setcategory_handler))
    app.add_handler(CommandHandler("setqr", setqr_handler))
    app.add_handler(CommandHandler("promote", promote_handler))
    app.add_handler(CommandHandler("demote", demote_handler))
    app.add_handler(CommandHandler("addcredits", addcredits_handler))
    app.add_handler(CommandHandler("users", users_handler))
    app.add_handler(CommandHandler("broadcast", broadcast_handler))
    app.add_handler(CommandHandler("dm", dm_handler))
    app.add_handler(CommandHandler("stats", stats_handler))
    app.add_handler(CommandHandler("exportdata", exportdata_handler))

    # Payments
    app.add_handler(PreCheckoutQueryHandler(precheckout_handler))
    app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment))

    # Callbacks & channel posts
    app.add_handler(CallbackQueryHandler(callback_router))
    app.add_handler(MessageHandler(filters.UpdateType.CHANNEL_POST, channel_post_handler))

    # Reply keyboard taps and general non-command messages
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, main_message_router))

    log.info("Starting bot (JSON edition)...")

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
