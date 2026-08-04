"""
Keeps the database current: syncs the current Premier League season on a
schedule, so finished matches get their real scores and new fixtures get
added automatically — no manual re-running of the backfill script.

Usage (runs forever, in the foreground):
    python -m data_collection.scheduler

For a quick one-off sync without starting the scheduler loop:
    python -m data_collection.scheduler --once
"""

import argparse
import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from data_collection.api_clients.football_data_org import FootballDataClient
from data_collection.sync import current_season_start_year, get_or_create_league, sync_season
from database.models import Team
from database.session import SessionLocal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# football-data.org free tier delays scores by roughly 10 minutes to an
# hour depending on competition. Running every 3 hours keeps us far under
# the 10 req/min rate limit (this job makes exactly 1 request per run)
# while staying reasonably current for a non-live use case.
SYNC_INTERVAL_HOURS = 3


def run_sync() -> None:
    client = FootballDataClient()
    session = SessionLocal()
    try:
        league = get_or_create_league(session, client)

        # Rebuild the teams cache from the DB each run rather than keeping
        # it in memory across runs — cheap, and avoids the cache going
        # stale if teams are ever added/edited some other way.
        teams_by_external_id = {
            team.external_id: team
            for team in session.query(Team).filter_by(league_id=league.id).all()
        }

        season = current_season_start_year()
        created, updated = sync_season(
            session, client, league, season, teams_by_external_id
        )
        logger.info(
            "Sync complete for season %s: %d created, %d updated.",
            season,
            created,
            updated,
        )
    except Exception:
        logger.exception("Sync job failed")
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single sync immediately and exit, instead of starting the scheduler loop.",
    )
    args = parser.parse_args()

    if args.once:
        run_sync()
        return

    scheduler = BlockingScheduler()
    scheduler.add_job(
        run_sync,
        trigger=IntervalTrigger(hours=SYNC_INTERVAL_HOURS),
        next_run_time=None,  # don't fire immediately; see run_sync() call below
    )
    logger.info(
        "Scheduler started. Syncing every %d hours. Press Ctrl+C to stop.",
        SYNC_INTERVAL_HOURS,
    )
    run_sync()  # sync once immediately on startup, then let the interval take over
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()