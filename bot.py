"""Discord Q&A bot with a text knowledge base, answered by Groq LLM.

Knowledge base = .txt / .md files in the knowledge/ folder.
Manage it with /kb upload, /kb text, /kb list, /kb remove.
Ask questions with /ask or by mentioning the bot.
"""

import os
import re
import sys
import math
import logging
from pathlib import Path

import aiohttp
from aiohttp import web
import discord
from discord import app_commands

# ---------------------------------------------------------------- config

BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
KNOWLEDGE_DIR.mkdir(exist_ok=True)

MAX_FILE_BYTES = 2 * 1024 * 1024          # 2 MB per KB file
MAX_CONTEXT_CHARS = 9000                  # KB text sent to the LLM per question
CHUNK_TARGET_CHARS = 1200                 # size of one retrieval chunk

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("poseidon")


def load_env() -> dict:
    """Read KEY=VALUE pairs from .env (no external dependency needed)."""
    env = {}
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    # real environment variables win over the file
    for key in ("DISCORD_TOKEN", "GROQ_API_KEY", "GROQ_MODEL"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


ENV = load_env()
DISCORD_TOKEN = ENV.get("DISCORD_TOKEN", "")
GROQ_API_KEY = ENV.get("GROQ_API_KEY", "")
GROQ_MODEL = ENV.get("GROQ_MODEL", "llama-3.3-70b-versatile")
SUPPORT_CHANNEL = ENV.get("SUPPORT_CHANNEL", "ask-poseidon").lower()

if not DISCORD_TOKEN or not GROQ_API_KEY:
    print("Missing DISCORD_TOKEN or GROQ_API_KEY in .env — fill them in and restart.")
    sys.exit(1)

SYSTEM_PROMPT = (
    "You are Poseidon, a helpful support assistant for a Clash of Clans bot. "
    "Answer the user's question using ONLY the knowledge base excerpts provided. "
    "Be concise and direct. If the knowledge base does not contain the answer, "
    "say you don't have that information yet and suggest asking a server admin. "
    "Never invent commands, prices, or features that are not in the excerpts."
)

# ---------------------------------------------------------------- knowledge base

WORD_RE = re.compile(r"[a-z0-9]{2,}")
STOPWORDS = frozenset(
    "the a an is are was were be been do does did can could will would should "
    "how what when where which who why to of in on at for with and or not it "
    "its this that these those you your i my me we our they them he she his her "
    "there here have has had if then than so as by from about into over under".split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS]


class KnowledgeBase:
    """Loads text files, splits them into chunks, ranks chunks by TF-IDF overlap."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.chunks: list[tuple[str, str]] = []  # (source filename, chunk text)
        self.reload()

    def reload(self) -> int:
        self.chunks = []
        for path in sorted(self.directory.glob("*")):
            if path.suffix.lower() not in (".txt", ".md"):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for chunk in self._split(text):
                self.chunks.append((path.name, chunk))
        log.info("Knowledge base loaded: %d chunks from %d files",
                 len(self.chunks), len(self.files()))
        return len(self.chunks)

    def files(self) -> list[Path]:
        return [p for p in sorted(self.directory.glob("*"))
                if p.suffix.lower() in (".txt", ".md")]

    @staticmethod
    def _split(text: str) -> list[str]:
        """Merge paragraphs into chunks of roughly CHUNK_TARGET_CHARS."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks, current = [], ""
        for para in paragraphs:
            if current and len(current) + len(para) > CHUNK_TARGET_CHARS:
                chunks.append(current)
                current = para
            else:
                current = f"{current}\n\n{para}" if current else para
            # a single huge paragraph still gets hard-split
            while len(current) > CHUNK_TARGET_CHARS * 2:
                chunks.append(current[: CHUNK_TARGET_CHARS * 2])
                current = current[CHUNK_TARGET_CHARS * 2:]
        if current:
            chunks.append(current)
        return chunks

    def search(self, question: str, max_chars: int = MAX_CONTEXT_CHARS) -> list[tuple[str, str]]:
        """Return the most relevant chunks for the question, within a size budget."""
        query_terms = tokenize(question)
        if not query_terms or not self.chunks:
            return self.chunks[: max(1, max_chars // CHUNK_TARGET_CHARS)]

        # document frequency per term, for IDF weighting
        doc_freq: dict[str, int] = {}
        chunk_tokens = [tokenize(chunk) for _, chunk in self.chunks]
        for tokens in chunk_tokens:
            for term in set(tokens):
                doc_freq[term] = doc_freq.get(term, 0) + 1

        n = len(self.chunks)
        scored = []
        for (source, chunk), tokens in zip(self.chunks, chunk_tokens):
            if not tokens:
                continue
            score = 0.0
            for term in query_terms:
                tf = tokens.count(term)
                if tf:
                    idf = math.log(1 + n / doc_freq[term])
                    score += (1 + math.log(tf)) * idf
            if score > 0:
                scored.append((score / math.sqrt(len(tokens)), source, chunk))

        scored.sort(reverse=True)
        picked, used = [], 0
        for _, source, chunk in scored:
            if used + len(chunk) > max_chars:
                continue
            picked.append((source, chunk))
            used += len(chunk)
            if used > max_chars * 0.9:
                break
        return picked


KB = KnowledgeBase(KNOWLEDGE_DIR)

# ---------------------------------------------------------------- LLM

async def ask_llm(question: str, http: aiohttp.ClientSession) -> str:
    excerpts = KB.search(question)
    if excerpts:
        context = "\n\n".join(f"[from {src}]\n{chunk}" for src, chunk in excerpts)
    else:
        context = "(the knowledge base is empty)"

    payload = {
        "model": GROQ_MODEL,
        "temperature": 0.3,
        "max_tokens": 700,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Knowledge base excerpts:\n\n{context}\n\n---\nQuestion: {question}"},
        ],
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    async with http.post(GROQ_URL, json=payload, headers=headers,
                         timeout=aiohttp.ClientTimeout(total=60)) as resp:
        data = await resp.json()
        if resp.status != 200:
            err = data.get("error", {}).get("message", str(data))[:200]
            raise RuntimeError(f"Groq API error {resp.status}: {err}")
        return data["choices"][0]["message"]["content"].strip()


def in_support_channel(channel) -> bool:
    """True for the configured support channel (by name or ID) and its threads."""
    if isinstance(channel, discord.Thread):
        channel = channel.parent
    name = (getattr(channel, "name", "") or "").lower()
    return name == SUPPORT_CHANNEL or str(getattr(channel, "id", "")) == SUPPORT_CHANNEL


def split_message(text: str, limit: int = 2000) -> list[str]:
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    parts.append(text)
    return parts

# ---------------------------------------------------------------- discord bot

class PoseidonBot(discord.Client):
    def __init__(self, *, message_content: bool):
        intents = discord.Intents.default()
        intents.message_content = message_content
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.http_session: aiohttp.ClientSession | None = None

    async def setup_hook(self):
        self.http_session = aiohttp.ClientSession()
        await self._start_health_server()
        # Only sync slash commands when explicitly asked. Command definitions
        # persist on Discord's side between restarts, so syncing on every boot
        # is unnecessary and hammers Discord's most rate-limited endpoint —
        # frequent restarts (e.g. Render recycling the free instance) can trip
        # a global 429 that blocks the token. Set SYNC_COMMANDS=1 for one boot
        # after you add or change a command, then remove it.
        if os.environ.get("SYNC_COMMANDS", "").lower() in ("1", "true", "yes"):
            try:
                await self.tree.sync()
                log.info("Slash commands synced.")
            except Exception as e:
                log.warning("Command sync failed (continuing anyway): %s", e)
        else:
            log.info("Skipping command sync (set SYNC_COMMANDS=1 to force a sync).")

    async def _start_health_server(self):
        """Tiny HTTP server so hosts like Render see an open port and
        uptime pingers (GET/HEAD) can keep the service awake."""
        async def health(request: web.Request) -> web.Response:
            return web.Response(text="Poseidon is running.")

        app = web.Application()
        app.router.add_get("/", health)       # aiohttp answers HEAD for GET routes
        app.router.add_get("/health", health)
        port = int(os.environ.get("PORT", 8080))
        self._web_runner = web.AppRunner(app)
        await self._web_runner.setup()
        try:
            await web.TCPSite(self._web_runner, "0.0.0.0", port).start()
            log.info("Health endpoint listening on port %d (GET/HEAD / and /health)", port)
        except OSError as e:
            log.warning("Could not start health server on port %d: %s", port, e)

    async def close(self):
        if self.http_session:
            await self.http_session.close()
        if getattr(self, "_web_runner", None):
            await self._web_runner.cleanup()
        await super().close()

    async def on_ready(self):
        log.info("Logged in as %s (id %s)", self.user, self.user.id)
        log.info("Invite URL: https://discord.com/oauth2/authorize"
                 "?client_id=%s&scope=bot%%20applications.commands&permissions=274877975552",
                 self.user.id)

    async def on_message(self, message: discord.Message):
        if message.author.bot or not self.intents.message_content:
            return
        is_dm = message.guild is None
        mentioned = self.user in message.mentions
        if not is_dm:
            # only ever respond inside the support channel
            if not in_support_channel(message.channel):
                return
            if not mentioned:
                return
        question = re.sub(rf"<@!?{self.user.id}>", "", message.content).strip()
        if not question:
            await message.reply("Ask me a question about the CoC bot! You can also use `/ask`.")
            return
        async with message.channel.typing():
            try:
                answer = await ask_llm(question, self.http_session)
            except Exception as e:
                log.exception("LLM call failed")
                await message.reply(f"Sorry, something went wrong: {e}")
                return
        for part in split_message(answer):
            await message.reply(part)


def build_commands(bot: PoseidonBot):
    tree = bot.tree

    @tree.command(name="ask", description="Ask a question about the CoC bot")
    @app_commands.describe(question="Your question")
    async def ask(interaction: discord.Interaction, question: str):
        if interaction.guild and not in_support_channel(interaction.channel):
            await interaction.response.send_message(
                f"Please ask in the **#{SUPPORT_CHANNEL}** channel.", ephemeral=True)
            return
        await interaction.response.defer()
        try:
            answer = await ask_llm(question, bot.http_session)
        except Exception as e:
            log.exception("LLM call failed")
            await interaction.followup.send(f"Sorry, something went wrong: {e}")
            return
        parts = split_message(f"**Q:** {question[:300]}\n\n{answer}")
        await interaction.followup.send(parts[0])
        for part in parts[1:]:
            await interaction.followup.send(part)

    kb_group = app_commands.Group(
        name="kb",
        description="Manage the knowledge base",
        default_permissions=discord.Permissions(manage_guild=True),
    )

    @kb_group.command(name="upload", description="Add a .txt or .md file to the knowledge base")
    @app_commands.describe(file="Text file with info about the CoC bot")
    async def kb_upload(interaction: discord.Interaction, file: discord.Attachment):
        await interaction.response.defer(ephemeral=True)
        name = Path(file.filename).name
        if Path(name).suffix.lower() not in (".txt", ".md"):
            await interaction.followup.send("Only `.txt` and `.md` files are supported.")
            return
        if file.size > MAX_FILE_BYTES:
            await interaction.followup.send("File too large (max 2 MB).")
            return
        data = await file.read()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
        (KNOWLEDGE_DIR / name).write_text(text, encoding="utf-8")
        chunks = KB.reload()
        await interaction.followup.send(
            f"Added **{name}** ({len(text):,} chars). Knowledge base now has "
            f"{len(KB.files())} file(s), {chunks} chunks.")

    @kb_group.command(name="text", description="Add a short text snippet to the knowledge base")
    @app_commands.describe(name="Name for this entry (e.g. pricing)",
                           content="The information itself")
    async def kb_text(interaction: discord.Interaction, name: str, content: str):
        await interaction.response.defer(ephemeral=True)
        safe = re.sub(r"[^A-Za-z0-9_\- ]", "", name).strip() or "entry"
        path = KNOWLEDGE_DIR / f"{safe}.txt"
        if path.exists():  # append instead of overwrite
            content = path.read_text(encoding="utf-8") + "\n\n" + content
        path.write_text(content, encoding="utf-8")
        KB.reload()
        await interaction.followup.send(f"Saved to **{path.name}**.")

    @kb_group.command(name="list", description="List knowledge base files")
    async def kb_list(interaction: discord.Interaction):
        files = KB.files()
        if not files:
            await interaction.response.send_message(
                "Knowledge base is empty. Use `/kb upload` or `/kb text` to add info.",
                ephemeral=True)
            return
        lines = [f"• **{p.name}** — {p.stat().st_size:,} bytes" for p in files]
        await interaction.response.send_message(
            f"**Knowledge base** ({len(KB.chunks)} chunks):\n" + "\n".join(lines),
            ephemeral=True)

    @kb_group.command(name="remove", description="Remove a file from the knowledge base")
    @app_commands.describe(filename="Exact file name shown by /kb list")
    async def kb_remove(interaction: discord.Interaction, filename: str):
        path = KNOWLEDGE_DIR / Path(filename).name
        if not path.exists():
            await interaction.response.send_message(
                f"No file named **{filename}**. Check `/kb list`.", ephemeral=True)
            return
        path.unlink()
        KB.reload()
        await interaction.response.send_message(f"Removed **{path.name}**.", ephemeral=True)

    @kb_remove.autocomplete("filename")
    async def kb_remove_autocomplete(interaction: discord.Interaction, current: str):
        return [app_commands.Choice(name=p.name, value=p.name)
                for p in KB.files() if current.lower() in p.name.lower()][:25]

    tree.add_command(kb_group)


def main():
    for message_content in (True, False):
        bot = PoseidonBot(message_content=message_content)
        build_commands(bot)
        try:
            bot.run(DISCORD_TOKEN)
            return
        except discord.PrivilegedIntentsRequired:
            print(
                "\n[!] 'Message Content Intent' is not enabled for this bot.\n"
                "    Mention-based answers need it. Enable it at\n"
                "    https://discord.com/developers/applications -> your app -> Bot ->\n"
                "    Privileged Gateway Intents -> MESSAGE CONTENT INTENT.\n"
                "    Starting in slash-command-only mode (/ask still works)...\n")
        except discord.LoginFailure:
            print("[!] Discord rejected the token. Check DISCORD_TOKEN in .env.")
            return


if __name__ == "__main__":
    main()
