# Poseidon — Discord Q&A bot for your CoC bot

Answers user questions from a knowledge base of text files, using Groq's free LLM API.

## Run it

```
python bot.py
```

Tokens live in `.env` (already filled in from tokens.txt).

## One-time Discord setup

1. Go to https://discord.com/developers/applications → your app → **Bot** →
   enable **MESSAGE CONTENT INTENT** (needed so the bot can answer when @mentioned).
2. Invite the bot with the URL printed in the console when it starts.

## Commands

The bot only operates in the **#ask-poseidon** channel (change with
`SUPPORT_CHANNEL` in `.env` — a channel name or ID). DMs always work.

| Command | Who | What |
|---|---|---|
| `/ask <question>` | everyone | Ask a question, answered from the knowledge base |
| @mention the bot in #ask-poseidon, or DM it | everyone | Same as /ask |
| `/kb upload <file>` | admins (Manage Server) | Add a `.txt`/`.md` file to the knowledge base |
| `/kb text <name> <content>` | admins | Add/append a short snippet without a file |
| `/kb list` | admins | Show what's in the knowledge base |
| `/kb remove <filename>` | admins | Delete an entry (with autocomplete) |

## Knowledge base

Just files in the `knowledge/` folder — you can also drop `.txt`/`.md` files in there
by hand and restart the bot (or run `/kb list` after using any /kb command to reload).
A sample file is included; replace it with real info about your CoC bot.

## Deploying on Render

1. Push this folder to a GitHub repo (`.env` and `tokens.txt` are gitignored —
   make sure the `knowledge/` files ARE committed, they're the bot's brain).
2. On https://dashboard.render.com → New → Web Service → connect the repo.
   `render.yaml` preconfigures everything; just enter `DISCORD_TOKEN` and
   `GROQ_API_KEY` when prompted (Environment tab).
3. The bot serves `GET`/`HEAD` on `/` and `/health` (Render's `PORT`), so the
   port check passes. On the free plan the service sleeps after ~15 min idle —
   add an uptime pinger (e.g. UptimeRobot, HEAD request to your
   `https://<service>.onrender.com/` every 5 min) to keep it awake.

**Note:** Render's free disk is ephemeral — files added with `/kb upload` are
lost on every deploy/restart. Treat the GitHub repo as the source of truth:
commit knowledge changes and push (Render auto-redeploys).

## Config (.env)

- `DISCORD_TOKEN` — bot token
- `GROQ_API_KEY` — from https://console.groq.com
- `GROQ_MODEL` — default `llama-3.3-70b-versatile`
