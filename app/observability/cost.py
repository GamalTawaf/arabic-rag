"""Cost accounting and the daily spend cap.

Two things live here:

- :func:`estimate_cost_usd` — token counts to dollars, from one price table.
- :class:`SpendTracker`     — today's spend, and the kill switch that stops the
  service quietly burning budget.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

from app.config import settings
from app.constants import PRICES_USD_PER_MTOK
from app.data import Spend

_PER_MTOK = 1_000_000


class SpendCapExceeded(Exception):
    """Raised when a call would push the day past the configured cap."""

    def __init__(self, cap_usd: float, spend: Spend) -> None:
        self.cap_usd = cap_usd
        self.spend = spend
        self.remaining = max(0.0, cap_usd - spend.usd)
        super().__init__(
            f"daily spend cap reached: ${spend.usd:.4f} of ${cap_usd:.2f} "
            f"spent on {spend.date} across {spend.calls} calls"
        )


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD for one call, from :data:`PRICES_USD_PER_MTOK`.

    Raises :class:`ValueError` for an unpriced model — a silent 0.0 would make
    the spend cap a no-op for exactly the model nobody checked.
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError(f"token counts must be >= 0, got {input_tokens}/{output_tokens}")
    prices = _lookup_price(model)
    if prices is None:
        raise ValueError(
            f"no price for model {model!r}; add it to PRICES_USD_PER_MTOK "
            f"(known prefixes: {', '.join(sorted(PRICES_USD_PER_MTOK))})"
        )
    price_in, price_out = prices
    return (input_tokens * price_in + output_tokens * price_out) / _PER_MTOK


def _lookup_price(model: str) -> tuple[float, float] | None:
    """Longest matching prefix wins, so 'claude-opus-5' can't shadow a longer key."""
    matches = [key for key in PRICES_USD_PER_MTOK if model.startswith(key)]
    if not matches:
        return None
    return PRICES_USD_PER_MTOK[max(matches, key=len)]


def _utc_today() -> str:
    return datetime.now(UTC).date().isoformat()


class SpendTracker:
    """Per-process daily spend counter and cap.

    Keyed by UTC date: the first call after midnight UTC resets the counter, so
    "today" always means the current UTC day, not a rolling 24 hours.

    trade-off: single-process, in-memory only. Two Cloud Run instances each get
    their own counter, so the effective cap is ``cap x instances``, and a
    restart forgets the day's spend. That is a deliberate simplification, not an
    oversight — with scale-to-zero and a demo-sized workload the failure mode is
    "cap is loose", never "cap is wrong in the dangerous direction" within one
    process. Upgrade path: a Redis INCR keyed ``spend:{date}`` (or a Postgres
    row with an atomic UPDATE ... RETURNING), same public interface.

    Not thread-safe by design: the service is single-threaded asyncio, and no
    method here awaits, so no two coroutines interleave inside one. That is why
    the cap is enforced through :meth:`reserve`/:meth:`settle` rather than
    check-then-record — the *caller's* await is what breaks the invariant, not
    anything in this class.
    """

    def __init__(self, cap_usd: float, *, clock: Callable[[], str] = _utc_today) -> None:
        if cap_usd < 0:
            raise ValueError(f"cap_usd must be >= 0, got {cap_usd}")
        self._cap_usd = cap_usd
        self._clock = clock
        self._spend = Spend(date=clock(), usd=0.0, calls=0)

    @property
    def cap_usd(self) -> float:
        return self._cap_usd

    def record(self, usd: float) -> None:
        """Add one call's cost to today's total."""
        if usd < 0:
            raise ValueError(f"usd must be >= 0, got {usd}")
        current = self.today()
        self._spend = replace(current, usd=current.usd + usd, calls=current.calls + 1)

    def reserve(self, estimated_usd: float) -> float:
        """Check the cap and immediately hold ``estimated_usd`` against it.

        The pair :meth:`reserve`/:meth:`settle` exists because check-then-record
        straddles the ``await`` on the provider: every request in flight reads the
        same pre-call total, so N concurrent requests all pass a cap with room
        for one. Measured before this existed: 10 concurrent calls against a cap
        with room for 3 spent 2.56x the cap. Holding the estimate at check time
        makes the decision and the debit one uninterrupted step.

        Returns the amount held, to be handed back to :meth:`settle`. Raises
        :class:`SpendCapExceeded` without holding anything when the cap is spent.
        """
        if estimated_usd < 0:
            raise ValueError(f"estimated_usd must be >= 0, got {estimated_usd}")
        self.check(estimated_usd)
        current = self.today()
        self._spend = replace(current, usd=current.usd + estimated_usd)
        return estimated_usd

    def settle(self, reserved_usd: float, actual_usd: float) -> None:
        """Swap a reservation for the real cost, counting one call.

        ``actual_usd`` of 0.0 releases the hold — the right answer when the
        provider failed, when the client vanished mid-stream, or when no usage
        was reported. Clamped at zero because the UTC-day rollover can reset the
        counter between the reserve and the settle.
        """
        if actual_usd < 0:
            raise ValueError(f"actual_usd must be >= 0, got {actual_usd}")
        current = self.today()
        self._spend = replace(
            current,
            usd=max(0.0, current.usd - reserved_usd + actual_usd),
            calls=current.calls + 1,
        )

    def would_exceed(self, estimated_usd: float = 0.0) -> bool:
        """True when spending ``estimated_usd`` would reach or pass the cap.

        Reaching the cap exactly counts as exceeding it: the cap is a ceiling to
        stop at, not one to sit on.
        """
        return self.today().usd + estimated_usd >= self._cap_usd

    def check(self, estimated_usd: float = 0.0) -> None:
        """Raise :class:`SpendCapExceeded` when the cap is spent. Call before an LLM call."""
        if self.would_exceed(estimated_usd):
            raise SpendCapExceeded(self._cap_usd, self.today())

    def today(self) -> Spend:
        """Today's spend, rolling over first if the UTC date changed."""
        now = self._clock()
        if self._spend.date != now:
            self._spend = Spend(date=now, usd=0.0, calls=0)
        return self._spend

    def remaining(self) -> float:
        """Budget left today. Never negative — an overshoot reads as 0.0."""
        return max(0.0, self._cap_usd - self.today().usd)


_tracker: SpendTracker | None = None


def get_spend_tracker() -> SpendTracker:
    """The process-wide tracker, built from ``settings.daily_spend_cap_usd``."""
    global _tracker
    if _tracker is None:
        _tracker = SpendTracker(settings.daily_spend_cap_usd)
    return _tracker


def reset_spend_tracker() -> None:
    """Drop the process-wide tracker. For tests."""
    global _tracker
    _tracker = None
