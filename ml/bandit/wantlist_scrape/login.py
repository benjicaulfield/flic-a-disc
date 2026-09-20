"""
One-time interactive login. Opens a real Chromium window, you log into
Discogs by hand (solve any captcha/2FA yourself), then press Enter here
and the session cookies are saved to auth_state.json in this folder.

Nothing is sent anywhere -- auth_state.json stays local on disk and is
gitignored.

Run:  /Users/benjamincaulfield/.pyenv/versions/3.11.5/bin/python3 login.py
"""
import pathlib
from playwright.sync_api import sync_playwright

STATE_PATH = pathlib.Path(__file__).parent / "auth_state.json"

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://www.discogs.com/login")
        input("Log in in the opened browser window, then press Enter here to save the session... ")
        context.storage_state(path=str(STATE_PATH))
        browser.close()
    print(f"Saved session to {STATE_PATH}")

if __name__ == "__main__":
    main()
