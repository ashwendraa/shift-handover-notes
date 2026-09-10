#!/usr/bin/env python3
"""
Loop Support Shift Handover Bot
=================================

What this does, every time it runs:
1. Pulls every Intercom conversation currently in "open" or "snoozed" state.
2. Skips any conversation that already has a note and has not received
   at least 2 new customer messages since that note (so an automated
   bounce/auto-reply loop, or a conversation with only agent activity,
   does not get a near-duplicate note every run).
3. Builds a clean transcript of each conversation (assignment history,
   who has replied, what the customer has said).
4. Calls the Claude API to generate a shift handover summary in this
   format:
     - 2 line summary of ownership / handover status
     - Bullet list of the questions the customer has asked
     - Solved or Pending, with a one line reason
5. Posts that summary back into the Intercom conversation as an INTERNAL
   NOTE (never sent to the customer).

Run modes:
    --dry-run          Do everything except actually post the note.
                        Prints what WOULD be posted. Use this first.
    --limit-ids ID,ID  Only process these specific conversation IDs.
                        Use this for your first live test.
    --state open       Only process open conversations (default: both)
    --state snoozed     Only process snoozed conversations
    (no flags)         Full run: all open + snoozed conversations, live.

Required environment variables:
    INTERCOM_ACCESS_TOKEN   Your Intercom access token
    INTERCOM_ADMIN_ID       The Intercom admin ID the notes should be
                             posted as (a bot/app user works well)
    ANTHROPIC_API_KEY       Your Anthropic API key

Optional environment variables:
    ANTHROPIC_MODEL          Default: claude-haiku-4-5-20251001
    STATE_FILE               Default: shift_handover_state.json
                              (tracks which conversations were already
                              summarized at what updated_at, so re-runs
                              on an unchanged ticket are skipped)
    LOG_FILE                 Default: shift_handover_log.csv
"""

import os
import sys
import json
import time
import csv
import argparse
import re
import html
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

INTERCOM_API_BASE = "https://api.intercom.io"
ANTHROPIC_API_BASE = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
STATE_FILE_DEFAULT = "shift_handover_state.json"
LOG_FILE_DEFAULT = "shift_handover_log.csv"

# Admin email addresses that are not real human agents (shared/system
# inboxes etc). Conversations currently owned by one of these are
# skipped, same as conversations never escalated to a human at all.
EXCLUDED_ADMIN_EMAILS = {"product@loopwork.co"}

# A conversation that already has a note only gets a new one once at
# least this many NEW customer messages have arrived since the last
# note. This is what stops automated bounce/auto-reply loops and
# quiet agent-only activity from generating near-duplicate notes.
MIN_NEW_CUSTOMER_MESSAGES_FOR_UPDATE = 2

DISCLAIMER_TEXT = "Note: This summary is for reference only, cross check all details if needed to avoid escalation."
IST_OFFSET = timedelta(hours=5, minutes=30)

REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
SLEEP_BETWEEN_CONVERSATIONS = 0.5


# ---------------------------------------------------------------------
# Small HTTP helper with retries
# ---------------------------------------------------------------------

def http_request(method, url, headers=None, body=None):
    headers = headers or {}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {e.code}: {raw[:500]}"
            if e.code == 429 or e.code >= 500:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            raise RuntimeError(f"Request failed ({url}): {last_error}")
        except (urllib.error.URLError, TimeoutError) as e:
            last_error = str(e)
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue
    raise RuntimeError(f"Request failed after {MAX_RETRIES} attempts ({url}): {last_error}")


# ---------------------------------------------------------------------
# Intercom API
# ---------------------------------------------------------------------

class IntercomClient:
    def __init__(self, access_token):
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Intercom-Version": "2.13",
        }

    def search_conversations(self, state):
        """Yield every conversation in the given state ('open' or 'snoozed')."""
        starting_after = None
        while True:
            query = {
                "query": {
                    "operator": "AND",
                    "value": [
                        {"field": "state", "operator": "=", "value": state},
                    ],
                },
                "pagination": {"per_page": 150},
            }
            if starting_after:
                query["pagination"]["starting_after"] = starting_after

            result = http_request(
                "POST",
                f"{INTERCOM_API_BASE}/conversations/search",
                headers=self.headers,
                body=query,
            )
            conversations = result.get("conversations", [])
            for c in conversations:
                yield c

            pages = result.get("pages", {})
            next_page = pages.get("next")
            if not next_page:
                break
            starting_after = next_page.get("starting_after")
            if not starting_after:
                break

    def get_conversation(self, conversation_id):
        return http_request(
            "GET",
            f"{INTERCOM_API_BASE}/conversations/{conversation_id}",
            headers=self.headers,
        )

    def get_admin_email(self, admin_id, cache):
        """Resolves an Intercom admin ID to their email, with an in-memory
        cache so the same admin is only looked up once per run."""
        if admin_id is None:
            return None
        admin_id = str(admin_id)
        if admin_id in cache:
            return cache[admin_id]
        try:
            result = http_request(
                "GET",
                f"{INTERCOM_API_BASE}/admins/{admin_id}",
                headers=self.headers,
            )
            email = (result.get("email") or "").lower()
        except Exception:
            email = ""
        cache[admin_id] = email
        return email

    def post_note(self, conversation_id, admin_id, html_body):
        body = {
            "message_type": "note",
            "type": "admin",
            "admin_id": str(admin_id),
            "body": html_body,
        }
        return http_request(
            "POST",
            f"{INTERCOM_API_BASE}/conversations/{conversation_id}/reply",
            headers=self.headers,
            body=body,
        )


# ---------------------------------------------------------------------
# Anthropic API
# ---------------------------------------------------------------------

class AnthropicClient:
    def __init__(self, api_key, model):
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Accept": "application/json",
        }
        self.model = model

    def summarize(self, transcript_text, meta):
        system_prompt = (
            "You write extremely terse shift handover notes for a customer "
            "support team. A busy agent starting their shift reads hundreds "
            "of these in a row, so every extra word costs them time. Produce "
            "a summary strictly in this format, and nothing else:\n\n"
            "OWNER: <just the current owner's name, nothing else, no "
            "sentence, no punctuation beyond the name itself. This is "
            "who the next shift follows up with.>\n\n"
            "HANDOVER: <ONE short line, under 12 words if possible. Do "
            "NOT repeat the owner's name here, that is already shown "
            "separately. Just the current state of play in the fewest "
            "words possible. If the conversation moved between two or "
            "more human agents, show that as 'Name1 to Name2' or "
            "similar, only if it actually happened. Do NOT narrate how "
            "it got there (no 'Loop AI could not resolve X and handed "
            "off to Y', no explaining what the bot tried, no quoting "
            "what the agent said). State the current fact only, not "
            "the backstory.>\n\n"
            "QUESTIONS:\n"
            "- <question 1 the customer asked, paraphrased, as short as "
            "possible, ideally under 10 words>\n"
            "- <question 2, if any, same style>\n"
            "(list every distinct question or ask from the customer, do "
            "not invent questions that were not asked, do not merge "
            "distinct questions into one, but do not restate the same "
            "question twice either)\n\n"
            "STATUS: <Solved or Pending>\n"
            "REASON: <under 10 words explaining why, no restating the "
            "handover line>\n\n"
            "Rules: Be factual. Never state something as resolved unless "
            "the transcript actually shows resolution. If uncertain "
            "whether it is solved or pending, say Pending. Do not use "
            "em dashes or en dashes anywhere in your output. Never use "
            "full sentences with subject plus verb plus explanation where "
            "a fragment will do. Cut every word that does not change what "
            "the reader needs to do next. No pleasantries, no restating "
            "the customer's name and company (already shown elsewhere), "
            "no quoting message bodies verbatim."
        )

        user_prompt = (
            f"Conversation ID: {meta.get('id')}\n"
            f"Company: {meta.get('company_name') or 'Unknown'}\n"
            f"MRR: {meta.get('mrr')}\n"
            f"Current state: {meta.get('state')}\n"
            f"Currently assigned to: {meta.get('current_admin_name') or 'Unassigned'}\n"
            f"All admins who have replied in this conversation: "
            f"{', '.join(meta.get('admin_names', [])) or 'None yet'}\n\n"
            f"Transcript (oldest to newest):\n{transcript_text}"
        )

        body = {
            "model": self.model,
            "max_tokens": 500,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        result = http_request(
            "POST",
            ANTHROPIC_API_BASE,
            headers=self.headers,
            body=body,
        )

        blocks = result.get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return text.strip()


# ---------------------------------------------------------------------
# Transcript building
# ---------------------------------------------------------------------

TAG_RE = re.compile(r"<[^>]+>")
QUOTE_MARKERS = (
    "From:", "Sent:", "Date:", "To:", "Subject:", "wrote:",
    "Get Outlook", "On ", "-----Original Message-----",
)


def strip_html(raw):
    if not raw:
        return ""
    text = html.unescape(raw)
    text = TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def trim_quoted_reply(text):
    """Cut off text at the first sign of a quoted earlier email, so we do
    not feed the same message to Claude a dozen times over via email
    quote chains."""
    earliest_cut = len(text)
    for marker in QUOTE_MARKERS:
        idx = text.find(marker)
        if idx != -1 and idx < earliest_cut and idx > 0:
            earliest_cut = idx
    return text[:earliest_cut].strip()


def build_conversation_view(conversation, bot_admin_id=None):
    """Returns (transcript_text, meta_dict).

    bot_admin_id, when given, is this script's own posting admin ID.
    Notes authored by that account (our own previously posted shift
    handover notes) are excluded entirely, so they never get fed back in
    as "context" and never count as a human having replied.
    """
    parts = conversation.get("conversation_parts", {}).get("conversation_parts", [])
    source = conversation.get("source", {}) or {}
    bot_admin_id = str(bot_admin_id) if bot_admin_id is not None else None

    lines = []
    admin_names = []
    current_admin_name = None
    seen_authors = set()
    customer_message_count = 0

    initial_body = strip_html(source.get("body", ""))
    initial_body = trim_quoted_reply(initial_body)
    author = source.get("author", {}) or {}
    if initial_body:
        lines.append(f"[Customer - {author.get('name', 'unknown')}]: {initial_body}")
        if author.get("type") in ("user", "lead"):
            customer_message_count += 1

    for part in parts:
        part_type = part.get("part_type")
        part_author = part.get("author", {}) or {}
        author_type = part_author.get("type")
        author_name = part_author.get("name", "")
        author_id = str(part_author.get("id")) if part_author.get("id") is not None else None

        if bot_admin_id is not None and author_id == bot_admin_id:
            # This is one of our own automated notes, skip entirely.
            continue

        if part_type in ("comment", "note") and part.get("body"):
            text = strip_html(part["body"])
            text = trim_quoted_reply(text)
            if not text:
                continue
            if author_type == "admin":
                label = f"[Agent - {author_name}]"
                if part_type == "note":
                    label = f"[Internal note - {author_name}]"
                if author_name and author_name not in seen_authors:
                    admin_names.append(author_name)
                    seen_authors.add(author_name)
                current_admin_name = author_name or current_admin_name
            elif author_type == "user" or author_type == "lead":
                label = f"[Customer - {author_name}]"
                customer_message_count += 1
            elif author_type == "bot":
                label = "[Bot]"
            else:
                label = f"[{author_type}]"
            lines.append(f"{label}: {text}")

        elif part_type == "assignment" and part.get("assigned_to", {}).get("type") == "admin":
            pass  # admin id only, name captured via comments instead

        elif part_type == "close":
            lines.append("[System]: Conversation was closed.")
        elif part_type == "snoozed":
            until = part.get("event_details", {}).get("until", "")
            lines.append(f"[System]: Conversation snoozed {until}.")
        elif part_type in ("open", "unsnoozed", "timer_unsnooze"):
            lines.append("[System]: Conversation reopened.")

    transcript_text = "\n".join(lines) if lines else "(no readable message content)"

    company = conversation.get("company") or {}
    meta = {
        "id": conversation.get("id"),
        "state": conversation.get("state"),
        "company_name": company.get("name"),
        "mrr": (company.get("custom_attributes") or {}).get("MRR"),
        "current_admin_name": current_admin_name,
        "admin_names": admin_names,
        "updated_at": conversation.get("updated_at"),
        "customer_message_count": customer_message_count,
    }
    return transcript_text, meta


# ---------------------------------------------------------------------
# Note formatting
# ---------------------------------------------------------------------

def parse_claude_summary(raw_text):
    """Parses the structured text Claude returns into an HTML note body.
    Falls back to dumping the raw text if parsing fails, rather than
    silently posting nothing."""
    owner = ""
    handover = ""
    questions = []
    status = ""
    reason = ""

    owner_match = re.search(r"OWNER:\s*(.+?)(?=\n\s*HANDOVER:|\Z)", raw_text, re.S)
    handover_match = re.search(r"HANDOVER:\s*(.+?)(?=\n\s*QUESTIONS:|\Z)", raw_text, re.S)
    questions_match = re.search(r"QUESTIONS:\s*(.+?)(?=\n\s*STATUS:|\Z)", raw_text, re.S)
    status_match = re.search(r"STATUS:\s*(.+?)(?=\n\s*REASON:|\Z)", raw_text, re.S)
    reason_match = re.search(r"REASON:\s*(.+)", raw_text, re.S)

    if owner_match:
        owner = owner_match.group(1).strip().splitlines()[0].strip()
    if handover_match:
        handover = handover_match.group(1).strip()
    if questions_match:
        for line in questions_match.group(1).strip().splitlines():
            line = line.strip().lstrip("-").strip()
            if line:
                questions.append(line)
    if status_match:
        status = status_match.group(1).strip().splitlines()[0].strip()
    if reason_match:
        reason = reason_match.group(1).strip().splitlines()[0].strip()

    if not owner and not handover and not questions and not status:
        # Parsing failed, fall back to raw text so nothing is silently lost
        safe = html.escape(raw_text)
        return (
            f"<p><b>Shift handover summary</b></p><p>{safe}</p>"
            f"<p style=\"color:#999;font-size:10px;font-style:italic;margin:4px 0 0 0\">"
            f"{html.escape(DISCLAIMER_TEXT)}</p>"
        )

    status_color = "#2e7d32" if status.lower().startswith("solved") else "#b26a00"
    questions_html = "".join(f"<li>{html.escape(q)}</li>" for q in questions) or "<li>None captured</li>"

    ist_now = datetime.now(timezone.utc) + IST_OFFSET

    # Deliberately compact: owner on its own line so it is never ambiguous
    # who the next shift follows up with, one line handover, tight question
    # list, one line status, IST timestamp footer.
    body = (
        f"<p><b>Owner:</b> {html.escape(owner) if owner else 'Unassigned'}</p>"
        f"<p><b>Handover:</b> {html.escape(handover)}</p>"
        f"<ul style=\"margin:2px 0;padding-left:18px\">{questions_html}</ul>"
        f"<p><b>Status:</b> <span style=\"color:{status_color}\"><b>{html.escape(status)}</b></span>"
        f"{' - ' + html.escape(reason) if reason else ''}</p>"
        f"<p style=\"color:#999;font-size:10px;font-style:italic;margin:4px 0 0 0\">"
        f"{html.escape(DISCLAIMER_TEXT)}</p>"
        f"<p style=\"color:#aaa;font-size:10px;margin-top:2px\">"
        f"{ist_now.strftime('%b %d %H:%M IST')}</p>"
    )
    return body


# ---------------------------------------------------------------------
# State tracking (avoid duplicate notes on unchanged conversations)
# ---------------------------------------------------------------------

def load_state(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_state(path, state):
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def log_result(path, row):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp_utc", "conversation_id", "state", "status", "action", "detail"])
        writer.writerow(row)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Loop Support Shift Handover Bot")
    parser.add_argument("--dry-run", action="store_true", help="Do not actually post notes")
    parser.add_argument("--limit-ids", type=str, default="", help="Comma separated conversation IDs to restrict to")
    parser.add_argument("--state", type=str, default="both", choices=["open", "snoozed", "both"])
    parser.add_argument("--force", action="store_true", help="Ignore state file, post even if unchanged")
    args = parser.parse_args()

    intercom_token = os.environ.get("INTERCOM_ACCESS_TOKEN")
    admin_id = os.environ.get("INTERCOM_ADMIN_ID")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    state_file = os.environ.get("STATE_FILE", STATE_FILE_DEFAULT)
    log_file = os.environ.get("LOG_FILE", LOG_FILE_DEFAULT)

    missing = [name for name, val in [
        ("INTERCOM_ACCESS_TOKEN", intercom_token),
        ("INTERCOM_ADMIN_ID", admin_id),
        ("ANTHROPIC_API_KEY", anthropic_key),
    ] if not val]
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)

    intercom = IntercomClient(intercom_token)
    claude = AnthropicClient(anthropic_key, model)
    print(f"Using Anthropic model: {model}")

    state = load_state(state_file)
    admin_email_cache = {}

    limit_ids = set(x.strip() for x in args.limit_ids.split(",") if x.strip())
    states_to_process = ["open", "snoozed"] if args.state == "both" else [args.state]

    total_seen = 0
    total_posted = 0
    total_skipped_unchanged = 0
    total_skipped_not_human = 0
    total_errors = 0

    for state_name in states_to_process:
        print(f"\n=== Pulling {state_name} conversations ===")
        for conv_summary in intercom.search_conversations(state_name):
            conv_id = str(conv_summary.get("id"))
            if limit_ids and conv_id not in limit_ids:
                continue

            total_seen += 1
            try:
                conversation = intercom.get_conversation(conv_id)
            except Exception as e:
                print(f"  [{conv_id}] ERROR fetching conversation: {e}")
                log_result(log_file, [datetime.now(timezone.utc).isoformat(), conv_id, state_name, "error", "fetch_failed", str(e)])
                total_errors += 1
                continue

            # Skip conversations never escalated to a human (still purely
            # with the bot, no admin assigned), and skip conversations
            # currently owned by a non-human/system inbox account.
            current_admin_id = conversation.get("admin_assignee_id")
            if current_admin_id is None:
                total_skipped_not_human += 1
                continue
            owner_email = intercom.get_admin_email(current_admin_id, admin_email_cache)
            if owner_email in EXCLUDED_ADMIN_EMAILS:
                total_skipped_not_human += 1
                continue

            updated_at = conversation.get("updated_at")
            transcript_text, meta = build_conversation_view(conversation, bot_admin_id=admin_id)
            customer_message_count = meta.get("customer_message_count", 0)

            # A conversation can be assigned to a human admin while that
            # admin has not actually said anything yet, everything so far
            # is still just Loop AI. Skip until a real human message exists.
            if not meta.get("admin_names"):
                total_skipped_not_human += 1
                continue

            existing_state = state.get(conv_id)
            if not args.force and existing_state is not None:
                # A note already exists for this conversation. Only post a
                # fresh one if at least MIN_NEW_CUSTOMER_MESSAGES new
                # customer messages have arrived since that note, so a
                # conversation stuck in an automated bounce/auto-reply loop
                # (or just quiet with only agent activity) does not get a
                # near-duplicate note every run.
                previous_count = existing_state.get("customer_message_count", 0)
                new_messages = customer_message_count - previous_count
                if new_messages < MIN_NEW_CUSTOMER_MESSAGES_FOR_UPDATE:
                    total_skipped_unchanged += 1
                    continue

            try:
                raw_summary = claude.summarize(transcript_text, meta)
            except Exception as e:
                print(f"  [{conv_id}] ERROR calling Claude: {e}")
                log_result(log_file, [datetime.now(timezone.utc).isoformat(), conv_id, state_name, "error", "claude_failed", str(e)])
                total_errors += 1
                continue

            note_html = parse_claude_summary(raw_summary)

            if args.dry_run:
                # Dry runs must never mark a conversation as handled. Only an
                # actual successful post advances the state file, otherwise a
                # dry run would cause the next live run to skip conversations
                # it never actually posted to.
                print(f"\n--- [{conv_id}] DRY RUN, would post: ---")
                print(note_html)
                log_result(log_file, [datetime.now(timezone.utc).isoformat(), conv_id, state_name, "dry_run", "would_post", raw_summary[:300]])
                time.sleep(SLEEP_BETWEEN_CONVERSATIONS)
                continue

            try:
                intercom.post_note(conv_id, admin_id, note_html)
                print(f"  [{conv_id}] Note posted.")
                log_result(log_file, [datetime.now(timezone.utc).isoformat(), conv_id, state_name, "posted", "note_posted", raw_summary[:300]])
                total_posted += 1
            except Exception as e:
                print(f"  [{conv_id}] ERROR posting note: {e}")
                log_result(log_file, [datetime.now(timezone.utc).isoformat(), conv_id, state_name, "error", "post_failed", str(e)])
                total_errors += 1
                continue

            state[conv_id] = {
                "last_updated_at": updated_at,
                "customer_message_count": customer_message_count,
            }
            time.sleep(SLEEP_BETWEEN_CONVERSATIONS)

    if not args.dry_run:
        save_state(state_file, state)

    print("\n=== Summary ===")
    print(f"Conversations seen:            {total_seen}")
    print(f"Notes posted:                  {total_posted}")
    print(f"Skipped (fewer than {MIN_NEW_CUSTOMER_MESSAGES_FOR_UPDATE} new customer messages): {total_skipped_unchanged}")
    print(f"Skipped (not assigned to a human): {total_skipped_not_human}")
    print(f"Errors:                        {total_errors}")
    if args.dry_run:
        print("This was a DRY RUN. No notes were actually posted.")


if __name__ == "__main__":
    main()
