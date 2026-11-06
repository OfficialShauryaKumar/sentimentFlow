# Gunicorn reads this file automatically from the working directory, so these
# settings apply even when the host's start command is just `gunicorn app:app`
# (as on Render, which ignores the Procfile).
#
# A full scrape (Reddit + news + yfinance for every ticker) runs inside the
# request and can take well over gunicorn's default 30 s on a small instance,
# which gets the worker killed mid-scrape. Allow it time to finish.
timeout = 180
graceful_timeout = 30

# One worker keeps memory inside free-tier limits; its threads let quick requests
# (paper results, holdings) answer while a long scrape runs, instead of queueing
# behind it. The bind address comes from $PORT, which gunicorn uses when set.
workers = 1
worker_class = "gthread"
threads = 4
