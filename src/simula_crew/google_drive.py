from __future__ import annotations

import csv
from io import StringIO
import re
from typing import Any
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


GOOGLE_SHEETS_SCOPE = "https://www.googleapis.com/auth/drive.readonly"


def google_sheet_export_url(sheet_url: str) -> str:
    sheet_id = _extract_sheet_id(sheet_url)
    gid = _extract_gid(sheet_url)
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


def fetch_google_sheet_csv(
    sheet_url: str,
    *,
    credentials_file: str | None = None,
) -> str:
    token = _google_access_token(credentials_file=credentials_file)
    request = Request(
        google_sheet_export_url(sheet_url),
        headers={"Authorization": f"Bearer {token}"},
    )
    with urlopen(request) as response:
        return response.read().decode("utf-8-sig")


def load_google_sheet_rows(
    sheet_url: str,
    *,
    credentials_file: str | None = None,
) -> list[dict[str, str]]:
    csv_text = fetch_google_sheet_csv(
        sheet_url,
        credentials_file=credentials_file,
    )
    reader = csv.DictReader(StringIO(csv_text))
    return [
        {str(key).strip(): str(value or "").strip() for key, value in row.items() if key is not None}
        for row in reader
        if any(str(value or "").strip() for value in row.values())
    ]


def _extract_sheet_id(sheet_url: str) -> str:
    match = re.search(r"/spreadsheets/d/([^/]+)", sheet_url)
    if match:
        return match.group(1)
    raise ValueError("Google Sheets URL must contain /spreadsheets/d/<sheet_id>.")


def _extract_gid(sheet_url: str) -> str:
    parsed = urlparse(sheet_url)
    query_gid = parse_qs(parsed.query).get("gid", [""])[0]
    if query_gid:
        return query_gid
    fragment_gid = parse_qs(parsed.fragment).get("gid", [""])[0]
    return fragment_gid or "0"


def _google_access_token(*, credentials_file: str | None) -> str:
    try:
        from google.auth import default
        from google.auth.transport.requests import Request as AuthRequest
        from google.oauth2 import service_account
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive ingestion requires: pip install -e '.[google]'"
        ) from exc

    if credentials_file:
        credentials: Any = service_account.Credentials.from_service_account_file(
            credentials_file,
            scopes=[GOOGLE_SHEETS_SCOPE],
        )
    else:
        credentials, _ = default(scopes=[GOOGLE_SHEETS_SCOPE])
    credentials.refresh(AuthRequest())
    token = getattr(credentials, "token", None)
    if not token:
        raise RuntimeError("Google credentials did not produce an access token.")
    return str(token)
