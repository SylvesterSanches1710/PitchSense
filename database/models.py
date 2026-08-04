"""
Core database models.

Design note: `external_id` fields store the ID from whichever data source
we're using (set in Phase 1's API client). Our own `id` (primary key) is
never exposed to or dependent on any external API, so swapping data
providers later never requires touching foreign keys or migrations.

MatchFeature holds one row per match with every engineered feature as a
column. We start with just Elo; later Phase 2 steps (form, home/away
splits, head-to-head, rest days, etc.) add columns here rather than
creating a new table per feature — one row per match, all features
together, is what makes assembling the final training dataset a single
clean query instead of a chain of joins across a dozen tables.
 
All feature columns are nullable: a match without enough prior history
(e.g. a team's very first game in our dataset) legitimately has no valid
feature value yet, and the model-training step needs to be able to tell
"missing" apart from "zero".
"""

from __future__ import annotations

import datetime
import enum

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class League(Base):
    __tablename__ = "leagues"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(50), unique=True)
    name: Mapped[str] = mapped_column(String(100))
    country: Mapped[str] = mapped_column(String(100))

    teams: Mapped[list["Team"]] = relationship(back_populates="league")
    matches: Mapped[list["Match"]] = relationship(back_populates="league")


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(50), unique=True)
    name: Mapped[str] = mapped_column(String(100))
    short_name: Mapped[str | None] = mapped_column(String(20), nullable=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"))

    league: Mapped["League"] = relationship(back_populates="teams")

    home_matches: Mapped[list["Match"]] = relationship(
        back_populates="home_team", foreign_keys="Match.home_team_id"
    )
    away_matches: Mapped[list["Match"]] = relationship(
        back_populates="away_team", foreign_keys="Match.away_team_id"
    )
    injuries: Mapped[list["Injury"]] = relationship(back_populates="team")


class MatchStatus(str, enum.Enum):
    SCHEDULED = "scheduled"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint("external_id", name="uq_match_external_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(50))
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"))
    season: Mapped[str] = mapped_column(String(20))  # e.g. "2024-2025"

    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))

    kickoff_utc: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[MatchStatus] = mapped_column(
        Enum(MatchStatus), default=MatchStatus.SCHEDULED
    )

    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Nullable: not every data source provides these, and Phase 2 features
    # need to handle their absence gracefully rather than assuming presence.
    home_xg: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_xg: Mapped[float | None] = mapped_column(Float, nullable=True)

    venue: Mapped[str | None] = mapped_column(String(150), nullable=True)

    league: Mapped["League"] = relationship(back_populates="matches")
    home_team: Mapped["Team"] = relationship(
        back_populates="home_matches", foreign_keys=[home_team_id]
    )
    away_team: Mapped["Team"] = relationship(
        back_populates="away_matches", foreign_keys=[away_team_id]
    )
    odds: Mapped[list["Odds"]] = relationship(back_populates="match")


class Odds(Base):
    """
    One row per bookmaker, per market, per snapshot in time. We deliberately
    do NOT overwrite odds on update — inserting a new row with a fresh
    `fetched_at` lets us later analyze odds movement, not just the closing
    line. Storage is cheap; lost history isn't recoverable.
    """

    __tablename__ = "odds"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    bookmaker: Mapped[str] = mapped_column(String(50))
    market: Mapped[str] = mapped_column(String(30))  # "h2h", "totals", "btts"

    home_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    draw_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_odds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # For totals markets (e.g. Over/Under 2.5)
    line: Mapped[float | None] = mapped_column(Float, nullable=True)
    over_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    under_odds: Mapped[float | None] = mapped_column(Float, nullable=True)

    fetched_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.datetime.utcnow
    )

    match: Mapped["Match"] = relationship(back_populates="odds")


class Injury(Base):
    __tablename__ = "injuries"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    player_name: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(50))  # "injured", "suspended", "doubtful"
    reported_date: Mapped[datetime.date] = mapped_column(DateTime, nullable=True)
    expected_return: Mapped[datetime.date | None] = mapped_column(
        DateTime, nullable=True
    )

    team: Mapped["Team"] = relationship(back_populates="injuries")

class MatchFeature(Base):
    __tablename__ = "match_features"
 
    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), unique=True)
 
    # Elo ratings going INTO the match (pre-match) — this is the actual
    # predictive feature. Post-match ratings are used only to seed the
    # next match's pre-match value, not stored here.
    elo_home_pre: Mapped[float | None] = mapped_column(Float, nullable=True)
    elo_away_pre: Mapped[float | None] = mapped_column(Float, nullable=True)
 
    match: Mapped["Match"] = relationship()