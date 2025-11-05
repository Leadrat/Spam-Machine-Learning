import os
import pickle
import base64
from pathlib import Path
from typing import List, Dict, Optional

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from bs4 import BeautifulSoup

# Read-only scope
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

BASE_DIR = Path(__file__).resolve().parents[1]
TOKEN_PATH = BASE_DIR / "token.pickle"
CREDS_PATH = BASE_DIR / "credentials.json"


def _get_credentials() -> Credentials:
    creds: Optional[Credentials] = None
    if TOKEN_PATH.exists():
        with open(TOKEN_PATH, "rb") as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDS_PATH.exists():
                raise FileNotFoundError(
                    f"credentials.json not found at {CREDS_PATH}. Download it from Google Cloud Console and place it there."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(TOKEN_PATH, "wb") as token:
            pickle.dump(creds, token)
    return creds


def get_gmail_service():
    creds = _get_credentials()
    service = build("gmail", "v1", credentials=creds)
    return service


def _decode_payload(data: str) -> str:
    try:
        return base64.urlsafe_b64decode(data.encode("UTF-8")).decode("UTF-8", errors="ignore")
    except Exception:
        return ""


def _extract_body_from_parts(parts) -> str:
    text_chunks: List[str] = []
    for part in parts or []:
        mime_type = part.get("mimeType", "")
        body = part.get("body", {})
        data = body.get("data")
        if data:
            content = _decode_payload(data)
            if mime_type == "text/plain":
                text_chunks.append(content)
            elif mime_type == "text/html":
                text_chunks.append(BeautifulSoup(content, "html.parser").get_text(" "))
        # Recurse into subparts
        if part.get("parts"):
            text_chunks.append(_extract_body_from_parts(part["parts"]))
    return "\n".join([c for c in text_chunks if c]).strip()


def fetch_latest_emails(max_results: int = 10) -> List[Dict[str, str]]:
    """Fetch the latest emails' subject and body from the user's inbox."""
    service = get_gmail_service()
    results = service.users().messages().list(userId="me", labelIds=["INBOX"], maxResults=max_results).execute()
    messages = results.get("messages", [])

    emails: List[Dict[str, str]] = []
    for msg in messages:
        m = service.users().messages().get(userId="me", id=msg["id"], format="full").execute()
        payload = m.get("payload", {})
        headers = payload.get("headers", [])
        subject = next((h["value"] for h in headers if h.get("name") == "Subject"), "(No subject)")

        body_text = ""
        if payload.get("parts"):
            body_text = _extract_body_from_parts(payload["parts"]) or ""
        else:
            data = payload.get("body", {}).get("data")
            if data:
                raw = _decode_payload(data)
                if payload.get("mimeType") == "text/html":
                    body_text = BeautifulSoup(raw, "html.parser").get_text(" ")
                else:
                    body_text = raw

        emails.append({"subject": subject or "(No subject)", "body": body_text or ""})
    return emails


if __name__ == "__main__":
    for e in fetch_latest_emails(5):
        print("Subject:", e["subject"])
        print("Body preview:", (e["body"][:200] + "...") if len(e["body"]) > 200 else e["body"])
        print("-" * 80)
