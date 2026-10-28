# Gunicorn reads this file automatically from the working directory, so these
# settings apply even when the host's start command is just `gunicorn app:app`
# (as on Render, which ignores the Procfile).
#
# A full scrape (Reddit + news + yfinance for every ticker) runs inside the
# request and can take well over gunicorn's default 30 s on a small instance,
# which gets the worker killed mid-scrape. Allow it time to finish.
timeout = 180
graceful_timeout = 30

# One worker keeps memory inside free-tier limits. The bind address comes
# from $PORT, which gunicorn uses automatically when it is set.
workers = 1
