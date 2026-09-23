"""只追加事件账本与班级实况还原。

连续发生“场地冲突 → 临时占课 → 离线补签”后，仍能按时间顺序重放事件，
还原每个班真正完成的内容、缺口与补课安排。账本只追加、不修改不删除；
所有判定（确认、异常、覆盖）都是重放的派生结果，可随时重新计算。

事件类型：

- teacher_report / venue_observation / sample_attendance
- schedule_change（合规调课）
- takeover（临时占课：其他学科占用，记录占用科目与依据单号，可为空表示突发）
- weather_trigger（降雨/高温触发，关联预案）
- makeup_plan（补课安排，挂接缺课场次）
- review_resolved（教研结论）

重放顺序无关：账本先把事件整理成以 Occasion 为键的索引（dict/set），
确认与状态推导只依赖“最终事件集合”，补传乱序到达不改变结论。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .events import (
    ConfirmationResult,
    Occasion,
    SampleAttendance,
    SessionMode,
    TeacherReport,
    VenueObservation,
    assess_session,
)
from .models import ActivityKind


@dataclass(frozen=True)
class LedgerEntry:
    seq: int
    event_type: str
    payload: Any  # 领域对象或 dict（均为不可变）


@dataclass(frozen=True)
class ReconstructedSession:
    occasion: Occasion
    class_id: str
    kind: ActivityKind
    taught_skill: str
    mode: SessionMode
    minutes: int
    confirmed: bool
    reasons: tuple[str, ...]
    flags: tuple[str, ...]
    notes: tuple[str, ...]
    makeup_for: Optional[Occasion]
    per_student: dict[str, int]


class EventLedger:
    def __init__(self):
        self._entries: list[LedgerEntry] = []

    def append(self, event_type: str, payload) -> LedgerEntry:
        entry = LedgerEntry(len(self._entries) + 1, event_type, payload)
        self._entries.append(entry)
        return entry

    def entries(self, event_type: str | None = None) -> tuple[LedgerEntry, ...]:
        if event_type is None:
            return tuple(self._entries)
        return tuple(e for e in self._entries if e.event_type == event_type)

    # ------------------------------------------------------------------
    def _index_events(self, slot_ids: set[str]) -> dict:
        """把账本整理为以 Occasion 为键的索引。

        同场次重复教师记录以最后追加的一条为准（实课一场只有一条教师记录）；
        其余结构均为 dict/set，结论只取决于最终事件集合而非到达顺序。
        """
        reports: dict[Occasion, TeacherReport] = {}
        observations: dict[Occasion, VenueObservation] = {}
        attendances: dict[Occasion, list[SampleAttendance]] = defaultdict(list)
        rescheduled: dict[Occasion, str] = {}      # -> 依据
        takeovers: list[dict] = []
        makeup_links: dict[Occasion, Occasion] = {}  # 原场次 -> 补课场次
        weather: dict[Occasion, str] = {}
        conflicts: set[Occasion] = set()

        for entry in self._entries:
            p, et = entry.payload, entry.event_type
            if et == "teacher_report" and p.occasion.slot_id in slot_ids:
                reports[p.occasion] = p
            elif et == "venue_observation" and p.occasion.slot_id in slot_ids:
                observations[p.occasion] = p
            elif et == "sample_attendance" and p.occasion.slot_id in slot_ids:
                attendances[p.occasion].append(p)
            elif et == "schedule_change" and p.occasion.slot_id in slot_ids:
                rescheduled[p.occasion] = p.basis_ref
            elif et == "takeover" and p["slot_id"] in slot_ids:
                takeovers.append(p)
            elif et == "weather_trigger" and p["occasion"].slot_id in slot_ids:
                weather[p["occasion"]] = p["policy_ref"]
            elif et == "venue_conflict_reported" and p["occasion"].slot_id in slot_ids:
                conflicts.add(p["occasion"])
            elif et == "makeup_plan" and p["makeup_for"].slot_id in slot_ids:
                makeup_links[p["makeup_for"]] = p["occasion"]

        return {
            "reports": reports,
            "observations": observations,
            "attendances": dict(attendances),
            "rescheduled": rescheduled,
            "takeovers": takeovers,
            "makeup_links": makeup_links,
            "weather": weather,
            "conflicts": conflicts,
        }

    # ------------------------------------------------------------------
    def rebuild_class(
        self,
        class_id: str,
        class_slots,
        class_headcount: int,
        adaptations: dict,
        token_to_student: dict[str, str],
        venues: dict,
        current_week: int,
        *,
        assessor: Callable[..., ConfirmationResult] = assess_session,
        cached_results: dict[Occasion, ConfirmationResult] | None = None,
    ) -> dict:
        """重放账本，还原单班实况。

        class_slots: 该班的 PlanSlot 集合（生效方案版本）。
        cached_results: 场次 -> 已算好的 ConfirmationResult（增量重算时复用
        未受影响场次的结论）；缺省/缺失的场次照常重新确认。
        """
        slot_ids = {s.slot_id for s in class_slots}
        index = self._index_events(slot_ids)

        results: dict[Occasion, ConfirmationResult] = {}
        for occ in index["reports"]:
            if cached_results is not None and occ in cached_results:
                results[occ] = cached_results[occ]
                continue
            results[occ] = assessor(
                index["reports"][occ],
                index["observations"].get(occ),
                index["attendances"].get(occ, []),
                class_headcount, adaptations, token_to_student,
            )

        return self._assemble(
            class_id=class_id, class_slots=class_slots, venues=venues,
            plan_week=current_week, index=index, results=results,
        )

    # ------------------------------------------------------------------
    def reconfirm_occasion(
        self,
        class_id: str,
        class_slots,
        class_headcount: int,
        adaptations: dict,
        token_to_student: dict[str, str],
        venues: dict,
        current_week: int,
        previous: dict,
        occasion: Occasion,
        *,
        assessor: Callable[..., ConfirmationResult] = assess_session,
    ) -> dict:
        """补传到达后只重算对应场次（含补课挂接两端），其余场次沿用旧结论。

        previous 为该班最近一次 rebuild_class()/reconfirm_occasion() 的结果。
        本方法的输出必须与对当前账本全量 rebuild_class() 完全一致——
        增量只省确认计算，不改变任何派生状态。
        """
        slot_ids = {s.slot_id for s in class_slots}
        index = self._index_events(slot_ids)

        # 受影响场次：补传目标本身 + 与它有补课挂接的场次（补课确认会回填原场次）
        affected: set[Occasion] = {occasion}
        for original, makeup in index["makeup_links"].items():
            if original == occasion:
                affected.add(makeup)
            if makeup == occasion:
                affected.add(original)

        cached: dict[Occasion, ConfirmationResult] = {}
        for s in previous["sessions"]:
            if s.occasion in affected:
                continue
            cached[s.occasion] = ConfirmationResult(
                occasion=s.occasion, confirmed=s.confirmed, mode=s.mode,
                effective_minutes=s.minutes, per_student_minutes=s.per_student,
                reasons=s.reasons, flags=s.flags,
            )

        return self.rebuild_class(
            class_id, class_slots, class_headcount, adaptations,
            token_to_student, venues, current_week,
            assessor=assessor, cached_results=cached,
        )

    # ------------------------------------------------------------------
    def _assemble(
        self,
        *,
        class_id: str,
        class_slots,
        venues: dict,
        plan_week: int,
        index: dict,
        results: dict[Occasion, ConfirmationResult],
    ) -> dict:
        """由事件索引 + 每场确认结果推导场次状态。纯函数，可安全增量复用。"""
        reports = index["reports"]
        observations = index["observations"]
        rescheduled = index["rescheduled"]
        takeovers = index["takeovers"]
        makeup_links = index["makeup_links"]
        weather = index["weather"]
        conflicts = index["conflicts"]

        sessions: list[ReconstructedSession] = []
        occasion_states: dict[str, str] = {}
        made_up: set[str] = set()
        takeover_keys = {Occasion(t["slot_id"], t["week"]).key() for t in takeovers}
        flags_index: dict[str, tuple[str, ...]] = {}

        # 按场次做三方确认（场次顺序固定：周次 -> slot_id）
        for occ, report in sorted(reports.items(), key=lambda kv: (kv[0].week, kv[0].slot_id)):
            result = results[occ]
            notes: list[str] = []
            obs = observations.get(occ)
            if obs is not None:
                venue = venues.get(obs.venue_id)
                if venue is not None and obs.observed_headcount > venue.safe_capacity:
                    notes.append("capacity_breach")
            if occ in conflicts:
                notes.append("venue_conflict_actual")

            key = occ.key()
            flags_index[key] = result.flags
            sessions.append(ReconstructedSession(
                occasion=occ, class_id=class_id, kind=report.kind,
                taught_skill=report.taught_skill, mode=report.mode,
                minutes=result.effective_minutes, confirmed=result.confirmed,
                reasons=result.reasons, flags=result.flags,
                notes=tuple(notes), makeup_for=report.makeup_for,
                per_student=result.per_student_minutes,
            ))
            if result.confirmed:
                if report.mode == SessionMode.MAKEUP and report.makeup_for is not None:
                    occasion_states[report.makeup_for.key()] = "completed"
                    made_up.add(report.makeup_for.key())
                    occasion_states[key] = "completed_makeup"
                else:
                    occasion_states[key] = "completed"
            else:
                occasion_states[key] = "in_review"

        # 展开计划场次状态（计划有但无报告）
        missing: list[str] = []
        for slot in class_slots:
            for week in range(1, plan_week + 1):
                if not slot.active_in_week(week):
                    continue
                occ = Occasion(slot.slot_id, week)
                key = occ.key()
                if key in occasion_states:
                    continue
                if occ in rescheduled:
                    occasion_states[key] = "rescheduled"
                    continue
                if key in takeover_keys:
                    occasion_states[key] = "taken_over"
                    continue
                if occ in weather and occ not in reports:
                    occasion_states[key] = "weather_pending"  # 触发但未执行替代
                    continue
                occasion_states[key] = "missing"
                missing.append(key)

        # 占课 / 已排补课的缺课 -> 待补课
        pending_makeup = [
            k for k, st in occasion_states.items()
            if st in ("taken_over", "missing", "weather_pending") or
            (st == "in_review")
        ]
        makeup_schedule = {
            original.key(): makeup.key()
            for original, makeup in makeup_links.items()
        }

        return {
            "class_id": class_id,
            "states": occasion_states,
            "sessions": sessions,
            "missing_occasions": tuple(sorted(missing)),
            "pending_makeup": tuple(sorted(pending_makeup)),
            "makeup_schedule": makeup_schedule,
            "made_up": made_up,
            "rescheduled": {k.key(): ref for k, ref in rescheduled.items()},
            "takeovers": tuple(sorted(takeovers, key=lambda t: (t["week"], t["slot_id"]))),
            "flags": flags_index,
        }
