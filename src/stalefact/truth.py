"""Ground truth for generated conversations: every fact event and the value in force after it."""

from __future__ import annotations

from pydantic import BaseModel, Field

STATING_KINDS = ("assert", "restate", "correct")


class FactEvent(BaseModel):
    """A fact-bearing turn. `value_after` is the value in force once the turn has been said."""

    call_no: int
    turn_no: int
    slot: str
    subject: str
    kind: str  # assert | restate | correct | retract
    value: str
    value_after: str


class ConversationTruth(BaseModel):
    """The facts of one claim over time, plus the patterns planted in it."""

    conversation_id: str
    claim_type: str
    n_calls: int
    patterns: list[str] = Field(default_factory=list)
    once_slots: list[str] = Field(default_factory=list)
    events: list[FactEvent] = Field(default_factory=list)

    def slot_events(self, slot: str, up_to_call: int | None = None) -> list[FactEvent]:
        """Events of a slot in time order, optionally only those in calls <= up_to_call."""
        selected = [
            e
            for e in self.events
            if e.slot == slot and (up_to_call is None or e.call_no <= up_to_call)
        ]
        return sorted(selected, key=lambda e: (e.call_no, e.turn_no))

    def gold_at(self, slot: str, call_no: int) -> str | None:
        """The value in force for a slot at the end of a call, or None if never stated."""
        events = self.slot_events(slot, call_no)
        return events[-1].value_after if events else None

    def earlier_values(self, slot: str, call_no: int) -> set[str]:
        """Values of the slot stated up to the end of a call that differ from the gold value."""
        gold = self.gold_at(slot, call_no)
        return {
            e.value
            for e in self.slot_events(slot, call_no)
            if e.kind in STATING_KINDS and e.value != gold
        }

    def changed_by(self, slot: str, call_no: int) -> bool:
        """True if the slot had a correction or a retraction by the end of the call."""
        return any(e.kind in ("correct", "retract") for e in self.slot_events(slot, call_no))

    def age_at(self, slot: str, call_no: int) -> int:
        """Calls since the value in force was last stated (0 = stated in this very call)."""
        gold = self.gold_at(slot, call_no)
        stated = [
            e.call_no
            for e in self.slot_events(slot, call_no)
            if e.kind in STATING_KINDS and e.value == gold
        ]
        return call_no - max(stated)

    def slots_stated_by(self, call_no: int) -> list[str]:
        """Slots with a value in force at the end of the call, in first-mention order."""
        seen: list[str] = []
        for event in sorted(self.events, key=lambda e: (e.call_no, e.turn_no)):
            if event.call_no <= call_no and event.slot not in seen:
                seen.append(event.slot)
        return [s for s in seen if self.gold_at(s, call_no)]


class Truth(BaseModel):
    """Ground truth for a whole dataset."""

    seed: int
    conversations: list[ConversationTruth]

    def by_id(self) -> dict[str, ConversationTruth]:
        return {c.conversation_id: c for c in self.conversations}
