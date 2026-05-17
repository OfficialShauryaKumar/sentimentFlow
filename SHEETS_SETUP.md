# Google Sheets sync — setup

One-time setup so the daily archive pushes to a Google Sheet you can open from any browser.

## What you'll do

1. Create a Google Cloud project (free).
2. Enable the Google Sheets API.
3. Create a service account (a robot Google identity your script uses).
4. Download its credentials JSON.
5. Create a blank Google Sheet.
6. Share the sheet with the service account's email.
7. Paste the Sheet ID into your `.env`.
8. Install the two Python packages.

Takes ~15 minutes. After this, you do nothing — every cron run pushes new data automatically.

## Step-by-step

### 1. Create a Google Cloud project

Go to https://console.cloud.google.com/. Sign in with your Google account if asked.

At the top-left of the page, next to "Google Cloud", click the project dropdown → **"New Project"**. Name it `sentimentflow` (or anything). Click **Create**. Wait ~30 seconds for it to provision, then make sure that project is selected in the top dropdown.

### 2. Enable the Google Sheets API

In the left sidebar, click **APIs & Services** → **Library**. In the search box, type `Google Sheets API`. Click the result, then click the blue **Enable** button. Wait a few seconds.

### 3. Create a service account

In the left sidebar: **APIs & Services** → **Credentials**.

At the top, click **+ Create Credentials** → **Service account**.

- **Service account name:** `sentimentflow-writer`
- **Service account ID:** auto-fills
- Click **Create and Continue**
- **Role:** skip (click Continue)
- **Grant users access:** skip (click Done)

You're back on the Credentials page. You'll see your new service account listed under "Service Accounts" with an email like `sentimentflow-writer@sentimentflow-XXXXXX.iam.gserviceaccount.com`. **Copy that email** — you'll need it in step 6.

### 4. Download the credentials JSON

Click on the service account row to open it. Click the **Keys** tab. Click **Add Key** → **Create new key** → select **JSON** → **Create**.

A JSON file downloads to your computer (Downloads folder usually). It's named something like `sentimentflow-XXXXXX-XXXXX.json`.

**Move and rename it** to your project at exactly this path:

```
~/Downloads/sf-project/credentials/google_service_account.json
```

You'll need to create the `credentials/` folder first. From terminal:

```bash
mkdir -p ~/Downloads/sf-project/credentials
mv ~/Downloads/sentimentflow-*.json ~/Downloads/sf-project/credentials/google_service_account.json
```

This file is your script's password — never commit it. `.gitignore` already excludes the `credentials/` folder.

### 5. Create the Google Sheet

Go to https://sheets.google.com — create a new blank sheet. Name it `SentimentFlow Archive` (or whatever).

Look at the URL. It'll look like:

```
https://docs.google.com/spreadsheets/d/1AbCdEfGhIjKlMnOpQrStUvWxYz1234567890/edit#gid=0
```

The long string between `/d/` and `/edit` is the **Sheet ID**. Copy it.

### 6. Share the sheet with your service account

Click the green **Share** button (top right of the sheet). Paste the service account email you copied in step 3. Set permission to **Editor**. **Uncheck "Notify people"** (the email is a robot, it won't read the notification). Click **Share**.

### 7. Paste the Sheet ID into `.env`

Open `~/Downloads/sf-project/.env` in VS Code. Add or update this line:

```
GOOGLE_SHEET_ID=1AbCdEfGhIjKlMnOpQrStUvWxYz1234567890
```

Use the actual Sheet ID you copied. Save the file.

### 8. Install the two Python packages

```bash
cd ~/Downloads/sf-project
source venv/bin/activate
pip install gspread google-auth
```

## Test it

Run the daily archive script manually:

```bash
python daily_archive.py
```

You should see a log line: `INFO sheets_sync: Pushed N rows to Google Sheets`. Open your Google Sheet in a browser — you should see today's ~14 rows appear with a header row.

If the log says `Could not connect to Google Sheets: ...`, the most likely causes:

- Sheet ID is wrong in `.env` (check no trailing space, no quotes around it).
- You didn't share the sheet with the service account email.
- `credentials/google_service_account.json` doesn't exist or is at the wrong path.

## After setup

The dashboard's `_archive_snapshot()` automatically calls the Sheets push. So both your daily cron and any manual dashboard load now push to Sheets. Dedup is per (date, ticker) — running the script 10 times on the same day produces one row per ticker, not 10.

To see a live view from any device, just bookmark the Sheet URL and open it whenever.
