#!/usr/bin/env python3
"""
Loop Support Shift Handover Bot
=================================

What this does, every time it runs:
1. Pulls every Intercom conversation currently in "open" or "snoozed" state.
2. Skips any conversation that has not changed since the last time this
   script posted a note on it (so idle tickets do not get spammed with
   repeat notes on every run).
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
from datetime import datetime, timezone

INTERCOM_API_BASE = "https://api.intercom.io"
ANTHROPIC_API_BASE = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
STATE_FILE_DEFAULT = "shift_handover_state.json"
LOG_FILE_DEFAULT = "shift_handover_log.csv"

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
            "HANDOVER: <ONE short line, under 12 words if possible. Just "
            "who owns it now and the current state in the fewest words "
            "possible. If the conversation moved between two or more human "
            "agents, show that as 'Name1 to Name2' or similar, only if it "
            "actually happened. Do NOT narrate how it got there (no 'Loop "
            "AI could not resolve X and handed off to Y', no explaining "
            "what the bot tried, no quoting what the agent said). State "
            "the current fact only, not the backstory.>\n\n"
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


def build_conversation_view(conversation):
    """Returns (transcript_text, meta_dict)."""
    parts = conversation.get("conversation_parts", {}).get("conversation_parts", [])
    source = conversation.get("source", {}) or {}

    lines = []
    admin_names = []
    current_admin_name = None
    seen_authors = set()

    initial_body = strip_html(source.get("body", ""))
    initial_body = trim_quoted_reply(initial_body)
    author = source.get("author", {}) or {}
    if initial_body:
        lines.append(f"[Customer - {author.get('name', 'unknown')}]: {initial_body}")

    for part in parts:
        part_type = part.get("part_type")
        part_author = part.get("author", {}) or {}
        author_type = part_author.get("type")
        author_name = part_author.get("name", "")

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
    }
    return transcript_text, meta


# ---------------------------------------------------------------------
# Note formatting
# ---------------------------------------------------------------------

def parse_claude_summary(raw_text):
    """Parses the structured text Claude returns into an HTML note body.
    Falls back to dumping the raw text if parsing fails, rather than
    silently posting nothing."""
    handover = ""
    questions = []
    status = ""
    reason = ""

    handover_match = re.search(r"HANDOVER:\s*(.+?)(?=\n\s*QUESTIONS:|\Z)", raw_text, re.S)
    questions_match = re.search(r"QUESTIONS:\s*(.+?)(?=\n\s*STATUS:|\Z)", raw_text, re.S)
    status_match = re.search(r"STATUS:\s*(.+?)(?=\n\s*REASON:|\Z)", raw_text, re.S)
    reason_match = re.search(r"REASON:\s*(.+)", raw_text, re.S)

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

    if not handover and not questions and not status:
        # Parsing failed, fall back to raw text so nothing is silently lost
        safe = html.escape(raw_text)
        return f"<p><b>Shift handover summary</b></p><p>{safe}</p>"

    status_color = "#2e7d32" if status.lower().startswith("solved") else "#b26a00"
    questions_html = "".join(f"<li>{html.escape(q)}</li>" for q in questions) or "<li>None captured</li>"

    # Deliberately compact: one line handover, tight question list, one line
    # status. No section banner, no restated customer name/company (already
    # visible in the conversation header), no timestamp clutter beyond a
    # small footer.
    body = (
        f"<p><b>Handover:</b> {html.escape(handover)}</p>"
        f"<ul style=\"margin:2px 0;padding-left:18px\">{questions_html}</ul>"
        f"<p><b>Status:</b> <span style=\"color:{status_color}\"><b>{html.escape(status)}</b></span>"
        f"{' - ' + html.escape(reason) if reason else ''}</p>"
        f"<p style=\"color:#aaa;font-size:10px;margin-top:2px\">"
        f"{datetime.now(timezone.utc).strftime('%b %d %H:%M UTC')}</p>"
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

    state = load_state(state_file)

    limit_ids = set(x.strip() for x in args.limit_ids.split(",") if x.strip())
    states_to_process = ["open", "snoozed"] if args.state == "both" else [args.state]

    total_seen = 0
    total_posted = 0
    total_skipped_unchanged = 0
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

            updated_at = conversation.get("updated_at")
            if not args.force and str(state.get(conv_id, {}).get("last_updated_at")) == str(updated_at):
                total_skipped_unchanged += 1
                continue

            transcript_text, meta = build_conversation_view(conversation)

            try:
                raw_summary = claude.summarize(transcript_text, meta)
            except Exception as e:
                print(f"  [{conv_id}] ERROR calling Claude: {e}")
                log_result(log_file, [datetime.now(timezone.utc).isoformat(), conv_id, state_name, "error", "claude_failed", str(e)])
                total_errors += 1
                continue

            note_html = parse_claude_summary(raw_summary)

            if args.dry_run:
                print(f"\n--- [{conv_id}] DRY RUN, would post: ---")
                print(note_html)
                log_result(log_file, [datetime.now(timezone.utc).isoformat(), conv_id, state_name, "dry_run", "would_post", raw_summary[:300]])
            else:
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

            state[conv_id] = {"last_updated_at": updated_at}
            time.sleep(SLEEP_BETWEEN_CONVERSATIONS)

    save_state(state_file, state)

    print("\n=== Summary ===")
    print(f"Conversations seen:            {total_seen}")
    print(f"Notes posted:                  {total_posted}")
    print(f"Skipped (unchanged since last run): {total_skipped_unchanged}")
    print(f"Errors:                        {total_errors}")
    if args.dry_run:
        print("This was a DRY RUN. No notes were actually posted.")


if __name__ == "__main__":
    main()
