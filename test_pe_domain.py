"""pe_domain 领域规则测试：方案版本、三方确认、防虚增、还原、复核与可见性。"""

import unittest

from pe_domain.coverage import compute_class_coverage
from pe_domain.events import (
    MIN_SAMPLE_RATIO,
    Occasion,
    SampleAttendance,
    SessionMode,
    TeacherReport,
    VenueObservation,
    assess_session,
    parse_aware_iso,
    pseudonym,
)
from pe_domain.ledger import EventLedger
from pe_domain.models import (
    ActivityKind,
    InjuryAdaptation,
    PlanSlot,
    SemesterPlan,
    SkillGoal,
    Teacher,
    Venue,
    VenueType,
    WeatherAlternative,
)
from pe_domain.plans import PlanRegistry, validate_plan
from pe_domain.review import AnomalyKind, ReviewBoard, ReviewStatus, detect_anomalies
from pe_domain.visibility import (
    AccessDenied,
    IdentityVault,
    build_parent_view,
    build_public_summary,
)

# ---------------------------------------------------------------- 夹具

FIELD = Venue("V-FIELD", "室外操场", VenueType.OUTDOOR, 50)
FIELD2 = Venue("V-FIELD-2", "第二操场", VenueType.OUTDOOR, 50)
FIELD3 = Venue("V-FIELD-3", "西操场", VenueType.OUTDOOR, 50)
GYM = Venue("V-GYM", "体育馆", VenueType.INDOOR, 45)
ROOM = Venue("V-ROOM", "形体房", VenueType.INDOOR, 20)
VENUES = {v.venue_id: v for v in (FIELD, FIELD2, FIELD3, GYM, ROOM)}
CLASS_VENUE = {"C1": "V-FIELD", "C2": "V-FIELD-2", "C3": "V-FIELD-3"}

T_WANG = Teacher("T-WANG", "王老师", frozenset({"BB", "急救"}))
T_CHEN = Teacher("T-CHEN", "陈老师", frozenset({"BB", "急救"}))
T_ZHAO = Teacher("T-ZHAO", "赵老师", frozenset({"BB", "急救"}))
T_LI = Teacher("T-LI", "李老师", frozenset({"田径"}))
TEACHERS = {t.teacher_id: t for t in (T_WANG, T_CHEN, T_ZHAO, T_LI)}
CLASS_TEACHER = {"C1": "T-WANG", "C2": "T-CHEN", "C3": "T-ZHAO"}

RAIN_GYM = WeatherAlternative("rain", "V-GYM", "室内球性练习", "POL-RAIN-01")
HEAT_GYM = WeatherAlternative("heat", "V-GYM", "室内低强度活动", "POL-HEAT-01")

GOAL_BB = SkillGoal("BB", "篮球", teach_weeks=(1, 2), practice_weeks=(1, 2, 3, 4), match_weeks=(3, 4))
HEADCOUNT = 40


def pe_slot(slot_id, class_id, weekday, teacher=T_WANG, skill="BB", venue="V-FIELD", alts=(RAIN_GYM, HEAT_GYM)):
    return PlanSlot(
        slot_id=slot_id, class_id=class_id, weekday=weekday,
        kind=ActivityKind.PE_CLASS, week_parity="all",
        venue_id=venue, teacher_id=teacher.teacher_id, skill_code=skill,
        headcount=HEADCOUNT, weather_alternatives=tuple(alts),
    )


def five_pe_slots(class_id):
    return [pe_slot(f"{class_id}-PE-{d}", class_id, d,
                    teacher=TEACHERS[CLASS_TEACHER[class_id]],
                    venue=CLASS_VENUE[class_id])
            for d in range(1, 6)]


def make_plan(class_ids=("C1",), version=0, status="draft"):
    slots = [s for cid in class_ids for s in five_pe_slots(cid)]
    return SemesterPlan(
        school_id="S", semester="2026-1", version=version,
        slots=tuple(slots), skill_goals=(GOAL_BB,), status=status,
    )


def _with_zone(value: str) -> str:
    """正常用法：给不带时区的钟面时间补 UTC（Z）；已带偏移的原样保留。"""
    from datetime import datetime
    text = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    if datetime.fromisoformat(text).tzinfo is None:
        return value + "Z"
    return value


def att(token, occ, when, source="online", received=None):
    when = _with_zone(when)
    received = _with_zone(received or when)
    return SampleAttendance(occ, token, when, received, source)


class FixtureTest(unittest.TestCase):
    def test_fixtures_satisfy_validation(self):
        issues = validate_plan(make_plan(["C1", "C2", "C3"]), VENUES, TEACHERS)
        self.assertEqual(issues, [])


# ---------------------------------------------------------------- 方案校验与版本

class PlanValidationTest(unittest.TestCase):
    def test_daily_pe_shortfall(self):
        plan = make_plan()
        # 删掉周五的课 -> 每周仅 4 天
        slots = tuple(s for s in plan.slots if s.slot_id != "C1-PE-5")
        issues = validate_plan(SemesterPlan("S", "2026-1", 0, slots, (GOAL_BB,)), VENUES, TEACHERS)
        self.assertTrue(any(i.code == "daily_pe_shortfall" for i in issues))

    def test_capacity_and_qualification(self):
        bad_slot = pe_slot("X", "C1", 1, teacher=T_WANG, venue="V-ROOM")
        other_slots = [s for s in five_pe_slots("C1") if s.weekday != 1]
        plan = SemesterPlan("S", "2026-1", 0, tuple(other_slots + [bad_slot]), (GOAL_BB,))
        issues = validate_plan(plan, VENUES, TEACHERS)
        codes = {i.code for i in issues}
        self.assertIn("capacity_exceeded", codes)  # 40 > 形体房 20

        unqual = pe_slot("Y", "C1", 1, teacher=T_LI)
        plan2 = SemesterPlan("S", "2026-1", 0,
                             tuple(s for s in five_pe_slots("C1") if s.weekday != 1) + (unqual,),
                             (GOAL_BB,))
        codes2 = {i.code for i in validate_plan(plan2, VENUES, TEACHERS)}
        self.assertIn("qualification_mismatch", codes2)

    def test_venue_conflict_detected(self):
        # 两班同天都用操场
        s1 = pe_slot("C1-PE-1", "C1", 1)
        s2 = pe_slot("C2-PE-1", "C2", 1)
        plan = SemesterPlan("S", "2026-1", 0, (s1, s2), (GOAL_BB,))
        issues = validate_plan(plan, VENUES, TEACHERS)
        self.assertTrue(any(i.code == "venue_conflict" for i in issues))

    def test_outdoor_slot_requires_weather_alternatives(self):
        no_alt = pe_slot("C1-PE-1", "C1", 1, alts=())
        rest = [s for s in five_pe_slots("C1") if s.weekday != 1]
        plan = SemesterPlan("S", "2026-1", 0, tuple(rest + [no_alt]), (GOAL_BB,))
        codes = {i.code for i in validate_plan(plan, VENUES, TEACHERS)}
        self.assertIn("weather_alt_missing", codes)

    def test_invalid_plan_cannot_submit(self):
        registry = PlanRegistry()
        slots = tuple(s for s in make_plan().slots if s.weekday != 5)
        bad = SemesterPlan("S", "2026-1", 0, slots, (GOAL_BB,))
        with self.assertRaises(ValueError):
            registry.submit(bad, VENUES, TEACHERS, submitted_by="admin")

    def test_versions_are_immutable_and_superseded(self):
        registry = PlanRegistry()
        v1 = registry.submit(make_plan(status="draft"), VENUES, TEACHERS, submitted_by="admin")
        self.assertEqual(v1.version, 1)
        self.assertEqual(v1.status, "submitted")
        approved1 = registry.approve("S", "2026-1", 1)
        self.assertEqual(approved1.status, "approved")

        v2 = registry.submit(make_plan(status="draft"), VENUES, TEACHERS, submitted_by="admin")
        registry.approve("S", "2026-1", v2.version)
        self.assertEqual(registry.get("S", "2026-1", 1).status, "superseded")
               # 历史完整保留，可对照阴阳课表
        self.assertEqual(len(registry.history("S", "2026-1")), 2)
        # 冻结方案不可直接改字段
        with self.assertRaises(Exception):
            approved1.status = "draft"


# ---------------------------------------------------------------- 三方确认与时长

class ConfirmationTest(unittest.TestCase):
    def setUp(self):
        self.occ = Occasion("C1-PE-1", 1)
        self.when = "2026-09-07T10:00:00"

    def _report(self, mode=SessionMode.NORMAL, minutes=40, skill="BB",
                venue_id="V-FIELD", **kw):
        return TeacherReport(
            self.occ, "T-WANG", ActivityKind.PE_CLASS, skill, minutes, mode,
            venue_id, **kw,
        )

    def test_three_party_confirmation(self):
        tokens = [f"tok{i}" for i in range(4)]
        result = assess_session(
            self._report(),
            VenueObservation(self.occ, "V-FIELD", 40),
            [att(t, self.occ, self.when) for t in tokens],
            HEADCOUNT, {}, {},
        )
        self.assertTrue(result.confirmed)
        self.assertEqual(result.effective_minutes, 40)
        self.assertEqual(set(result.per_student_minutes), set(tokens))

    def test_missing_venue_observation_is_not_confirmed(self):
        result = assess_session(
            self._report(), None,
            [att(f"t{i}", self.occ, self.when) for i in range(4)],
            HEADCOUNT, {}, {},
        )
        self.assertFalse(result.confirmed)
        self.assertEqual(result.effective_minutes, 0)
        self.assertTrue(any("场地" in r for r in result.reasons))

    def test_sample_below_minimum_not_confirmed(self):
        required = max(3, int(HEADCOUNT * MIN_SAMPLE_RATIO + 0.999))
        result = assess_session(
            self._report(),
            VenueObservation(self.occ, "V-FIELD", 40),
            [att(f"t{i}", self.occ, self.when) for i in range(required - 1)],
            HEADCOUNT, {}, {},
        )
        self.assertFalse(result.confirmed)
        self.assertTrue(any("抽样" in r for r in result.reasons))

    def test_duplicate_signin_does_not_inflate_minutes_or_count(self):
        t = "tokA"
        result = assess_session(
            self._report(),
            VenueObservation(self.occ, "V-FIELD", 40),
            [
                att(t, self.occ, self.when),
                att(t, self.occ, self.when),  # 同场次重复签到
                att("tokB", self.occ, self.when),
                att("tokC", self.occ, self.when),
                att("tokD", self.occ, self.when),
            ],
            HEADCOUNT, {}, {},
        )
        self.assertTrue(result.confirmed)
        self.assertEqual(len(result.per_student_minutes), 4)  # 去重后 4 人
        self.assertTrue(any(f.startswith(f"duplicate_signin:{t}") for f in result.flags))

    def test_offline_late_signin_is_logged_but_not_counted(self):
        result = assess_session(
            self._report(),
            VenueObservation(self.occ, "V-FIELD", 40),
            [
                att("late", self.occ, self.when, source="offline",
                    received="2026-09-09T11:00:00"),  # 49 小时后
                att("ok1", self.occ, self.when),
                att("ok2", self.occ, self.when),
                att("ok3", self.occ, self.when),
            ],
            HEADCOUNT, {}, {},
        )
        self.assertFalse(result.confirmed)  # 有效样本只剩 3
        self.assertIn("offline_late:late", result.flags)

    def test_offline_signin_within_window_counts(self):
        result = assess_session(
            self._report(),
            VenueObservation(self.occ, "V-FIELD", 40),
            [
                att("off", self.occ, self.when, source="offline",
                    received="2026-09-08T15:00:00"),  # 29 小时后
                att("ok1", self.occ, self.when),
                att("ok2", self.occ, self.when),
                att("ok3", self.occ, self.when),
            ],
            HEADCOUNT, {}, {},
        )
        self.assertTrue(result.confirmed)

    def test_reported_minutes_capped_to_standard(self):
        result = assess_session(
            self._report(minutes=120),
            VenueObservation(self.occ, "V-FIELD", 40),
            [att(f"t{i}", self.occ, self.when) for i in range(4)],
            HEADCOUNT, {}, {},
        )
        self.assertEqual(result.effective_minutes, 40)
        self.assertTrue(any("minutes_capped" in f for f in result.flags))

    def test_rain_alternative_requires_basis_and_counts(self):
        no_basis = assess_session(
            self._report(mode=SessionMode.RAIN_ALT, venue_id="V-GYM"),
            VenueObservation(self.occ, "V-GYM", 40),
            [att(f"t{i}", self.occ, self.when) for i in range(4)],
            HEADCOUNT, {}, {},
        )
        self.assertFalse(no_basis.confirmed)

        ok = assess_session(
            self._report(mode=SessionMode.RAIN_ALT, venue_id="V-GYM",
                         basis_ref="POL-RAIN-01"),
            VenueObservation(self.occ, "V-GYM", 40),
            [att(f"t{i}", self.occ, self.when) for i in range(4)],
            HEADCOUNT, {}, {},
        )
        self.assertTrue(ok.confirmed)
        self.assertEqual(ok.mode, SessionMode.RAIN_ALT)

    def test_injury_adaptation_reduces_only_that_student(self):
        adaptation = InjuryAdaptation("student-7", "BB", adjusted_minutes=10,
                                      basis_ref="MED-2026-07", note="踝伤")
        vault_map = {"tok7": "student-7"}
        result = assess_session(
            self._report(),
            VenueObservation(self.occ, "V-FIELD", 40),
            [att("tok7", self.occ, self.when)] +
            [att(f"t{i}", self.occ, self.when) for i in range(3)],
            HEADCOUNT, {"student-7": adaptation}, vault_map,
        )
        self.assertTrue(result.confirmed)
        self.assertEqual(result.per_student_minutes["tok7"], 10)
        self.assertEqual(result.per_student_minutes["t0"], 40)

    def test_free_play_and_exam_drill_flagged(self):
        for mode, flag in ((SessionMode.FREE, "whole_free_play"),
                           (SessionMode.EXAM_DRILL, "exam_drill_only")):
            result = assess_session(
                self._report(mode=mode),
                VenueObservation(self.occ, "V-FIELD", 40),
                [att(f"t{i}", self.occ, self.when) for i in range(4)],
                HEADCOUNT, {}, {},
            )
            self.assertTrue(result.confirmed)
            self.assertIn(flag, result.flags)

    def test_makeup_must_link_original_occasion(self):
        result = assess_session(
            self._report(mode=SessionMode.MAKEUP),
            VenueObservation(self.occ, "V-FIELD", 40),
            [att(f"t{i}", self.occ, self.when) for i in range(4)],
            HEADCOUNT, {}, {},
        )
        self.assertFalse(result.confirmed)
        self.assertTrue(any("补课" in r for r in result.reasons))


# ---------------------------------------------------------------- 时区时间线与去重顺序

class TimezoneSigninTest(unittest.TestCase):
    """跨时区交流周：带偏移 ISO 时间统一到 UTC 时间线，先各自判定再稳定去重。"""

    def setUp(self):
        self.occ = Occasion("C1-PE-1", 1)
        self.ven = VenueObservation(self.occ, "V-FIELD", 40)

    def _report(self):
        return TeacherReport(
            self.occ, "T-WANG", ActivityKind.PE_CLASS, "BB", 40,
            SessionMode.NORMAL, "V-FIELD",
        )

    def _four(self, extra):
        base = [att("ok1", self.occ, "2026-09-07T10:00:00Z"),
                att("ok2", self.occ, "2026-09-07T10:00:00Z"),
                att("ok3", self.occ, "2026-09-07T10:00:00Z")]
        return base + extra

    def test_parse_aware_iso_unifies_to_utc_timeline(self):
        from datetime import timezone
        self.assertEqual(
            parse_aware_iso("2026-09-07T10:00:00+08:00"),
            parse_aware_iso("2026-09-07T02:00:00Z"),
        )
        self.assertEqual(
            parse_aware_iso("2026-09-07T10:00:00Z").utcoffset(),
            timezone.utc.utcoffset(None),
        )
        with self.assertRaises(ValueError):
            parse_aware_iso("2026-09-07T10:00:00")        # 无时区
        with self.assertRaises(ValueError):
            parse_aware_iso("2026-13-07T10:00:00+08:00")  # 无法解析（13 月）
        with self.assertRaises(ValueError):
            parse_aware_iso("not-a-time")

    def test_cross_midnight_offline_within_window_counts(self):
        # 23:30 发生、次日 01:00 才收到：跨午夜但只有 1.5 小时
        late = att("off", self.occ, "2026-09-07T23:30:00Z", source="offline",
                   received="2026-09-08T01:00:00Z")
        result = assess_session(self._report(), self.ven, self._four([late]),
                                HEADCOUNT, {}, {})
        self.assertTrue(result.confirmed)
        self.assertNotIn("offline_late:off", result.flags)

    def test_offline_offset_that_naive_clock_face_would_accept_is_rejected(self):
        # 发生在夏令时 +02，补传收到于冬令时 +01：钟面差正好 48 小时，
        # 真实 UTC 时间线是 49 小时——按截断钟面会误计入，统一时间线后必须拒绝。
        late = att("abroad", self.occ, "2026-10-24T10:00:00+02:00",
                   source="offline", received="2026-10-26T10:00:00+01:00")
        result = assess_session(self._report(), self.ven, self._four([late]),
                                HEADCOUNT, {}, {})
        self.assertFalse(result.confirmed)  # 有效只剩 3 人
        self.assertIn("offline_late:abroad", result.flags)

    def test_naive_unparseable_and_future_receive_are_rejected(self):
        bad = [
            SampleAttendance(self.occ, "naive", "2026-09-07T10:00:00",
                             "2026-09-07T10:00:00", "online"),
            SampleAttendance(self.occ, "garbage", "星期一早上",
                             "2026-09-07T10:00:00Z", "online"),
            att("future", self.occ, "2026-09-07T10:00:00Z",
                received="2026-09-07T09:00:00Z"),  # 接收早于发生
        ]
        result = assess_session(self._report(), self.ven, self._four(bad),
                                HEADCOUNT, {}, {})
        # 三条全部无效：样本仍是 3，达不到 4
        self.assertFalse(result.confirmed)
        self.assertIn("clock_anomaly:unparseable:naive", result.flags)
        self.assertIn("clock_anomaly:unparseable:garbage", result.flags)
        self.assertIn("clock_anomaly:received_before_occurred:future", result.flags)

    def test_invalid_early_duplicate_never_replaces_valid_record(self):
        # 合法签到 10:00 已在样本；随后补传一条“发生更早(08:00)但 49 小时后
        # 才收到”的离线重复——旧逻辑会先用它替换再跳过时限校验，把无效记录
        # 带回样本；新逻辑先各自判定，无效记录不得参与选取。
        poisoned = att("dup", self.occ, "2026-09-07T08:00:00Z",
                        source="offline", received="2026-09-09T09:00:00Z")
        valid = att("dup", self.occ, "2026-09-07T10:00:00Z")
        result = assess_session(
            self._report(), self.ven,
            self._four([valid, poisoned]), HEADCOUNT, {}, {},
        )
        self.assertTrue(result.confirmed)
        self.assertEqual(result.per_student_minutes.get("dup"), 40)
        self.assertIn("duplicate_signin:dup", result.flags)
        self.assertIn("offline_late:dup", result.flags)

    def test_mixed_valid_duplicates_select_earliest_valid_stably(self):
        # 三条同 token：晚到但有效的在线(11:00)、更早且窗口内的离线(09:00)、
        # 更早却超窗的离线(07:00)。应选“最早的有效候选”09:00 那条。
        late_early = att("mix", self.occ, "2026-09-07T07:00:00Z",
                         source="offline", received="2026-09-09T08:00:00Z")
        valid_early = att("mix", self.occ, "2026-09-07T09:00:00Z",
                          source="offline", received="2026-09-07T15:00:00Z")
        valid_late = att("mix", self.occ, "2026-09-07T11:00:00Z")
        forward = assess_session(
            self._report(), self.ven, self._four([valid_late, valid_early, late_early]),
            HEADCOUNT, {}, {},
        )
        reverse = assess_session(
            self._report(), self.ven, self._four([late_early, valid_early, valid_late]),
            HEADCOUNT, {}, {},
        )
        self.assertTrue(forward.confirmed)
        self.assertEqual(forward.per_student_minutes["mix"], 40)
        # 乱序到达：标记与结论完全一致
        self.assertEqual(forward.flags, reverse.flags)
        self.assertEqual(forward.confirmed, reverse.confirmed)
        self.assertIn("duplicate_signin:mix", forward.flags)
        self.assertIn("offline_late:mix", forward.flags)

    def test_flags_never_leak_real_student_id(self):
        adaptation = InjuryAdaptation("student-secret-7", "BB", 10, "MED-7")
        poisoned = SampleAttendance(
            self.occ, "tok7", "2026-09-07T08:00:00",  # naive，时钟异常
            "2026-09-07T10:00:00Z", "online",
        )
        result = assess_session(
            self._report(), self.ven, self._four([
                att("tok7", self.occ, "2026-09-07T10:00:00Z"), poisoned,
            ]),
            HEADCOUNT, {"student-secret-7": adaptation}, {"tok7": "student-secret-7"},
        )
        joined = " ".join(result.flags) + " " + " ".join(result.reasons)
        self.assertNotIn("student-secret-7", joined)
        self.assertIn("clock_anomaly:unparseable:tok7", result.flags)
        self.assertEqual(result.per_student_minutes["tok7"], 10)  # 适配折减照旧


# ---------------------------------------------------------------- 账本还原

class LedgerRebuildTest(unittest.TestCase):
    def setUp(self):
        self.ledger = EventLedger()
        self.slots = five_pe_slots("C1")
        self.when = lambda week, day=1: f"2026-09-{day + (week - 1) * 7:02d}T10:00:00"

    def _record_normal(self, slot_id, week, tokens=4, skill="BB"):
        occ = Occasion(slot_id, week)
        self.ledger.append("teacher_report", TeacherReport(
            occ, "T-WANG", ActivityKind.PE_CLASS, skill, 40,
            SessionMode.NORMAL, "V-FIELD"))
        self.ledger.append("venue_observation", VenueObservation(occ, "V-FIELD", 40))
        for i in range(tokens):
            self.ledger.append("sample_attendance", att(f"tok{i}", occ, self.when(week)))
        return occ

    def _rebuild(self, current_week=4):
        return self.ledger.rebuild_class(
            "C1", self.slots, HEADCOUNT, {}, {}, VENUES, current_week=current_week,
        )

    def test_normal_weeks_complete(self):
        for w in range(1, 4):
            for slot in self.slots:
                self._record_normal(slot.slot_id, w)
        rebuilt = self._rebuild(current_week=3)
        self.assertEqual(
            sum(1 for st in rebuilt["states"].values() if st == "completed"),
            15,
        )
        self.assertEqual(rebuilt["missing_occasions"], ())

    def test_conflict_takeover_offline_chain_is_reconstructable(self):
        # 第 1 周正常
        self._record_normal("C1-PE-1", 1)
        # 第 1 周周二：场地被他班占用 -> 数学课临时占课 -> 事后离线补签也救不回这节
        occ_tue = Occasion("C1-PE-2", 1)
        self.ledger.append("venue_conflict_reported",
                           {"occasion": occ_tue, "with_class": "C3"})
        self.ledger.append("takeover",
                           {"slot_id": "C1-PE-2", "week": 1, "subject": "数学",
                            "ref": ""})
        self.ledger.append("sample_attendance",
                           att("tok0", occ_tue, self.when(1, 2),
                               source="offline", received="2026-09-05T10:00:00"))
        rebuilt = self._rebuild(current_week=1)
        self.assertEqual(rebuilt["states"]["C1-PE-2#w1"], "taken_over")
        self.assertIn("C1-PE-2#w1", rebuilt["pending_makeup"])
        self.assertEqual(len(rebuilt["takeovers"]), 1)

        # 第 2 周安排补课并挂接原场次 -> 缺口被回填
        makeup_occ = Occasion("C1-PE-1", 2)
        self.ledger.append("makeup_plan", {"occasion": makeup_occ, "makeup_for": occ_tue})
        self.ledger.append("teacher_report", TeacherReport(
            makeup_occ, "T-WANG", ActivityKind.PE_CLASS, "BB", 40,
            SessionMode.MAKEUP, "V-GYM", basis_ref="MK-001", makeup_for=occ_tue))
        self.ledger.append("venue_observation", VenueObservation(makeup_occ, "V-GYM", 40))
        for i in range(4):
            self.ledger.append("sample_attendance",
                               att(f"mk{i}", makeup_occ, "2026-09-14T10:00:00"))
        rebuilt2 = self._rebuild(current_week=2)
        self.assertEqual(rebuilt2["states"]["C1-PE-2#w1"], "completed")
        self.assertEqual(rebuilt2["makeup_schedule"]["C1-PE-2#w1"], "C1-PE-1#w2")
        self.assertIn("C1-PE-2#w1", rebuilt2["made_up"])

    def test_weather_trigger_without_alternative_is_pending(self):
        occ = Occasion("C1-PE-1", 1)
        self.ledger.append("weather_trigger",
                           {"occasion": occ, "condition": "rain", "policy_ref": "POL-RAIN-01"})
        rebuilt = self._rebuild(current_week=1)
        self.assertEqual(rebuilt["states"]["C1-PE-1#w1"], "weather_pending")
        self.assertIn("C1-PE-1#w1", rebuilt["pending_makeup"])

    def test_compliant_reschedule_is_not_missing(self):
        occ = Occasion("C1-PE-1", 1)
        from pe_domain.events import ScheduleChange
        self.ledger.append("schedule_change", ScheduleChange(
            occ, new_weekday=3, new_venue_id="V-GYM",
            new_teacher_id="T-WANG", approver="principal", basis_ref="ADJ-09"))
        rebuilt = self._rebuild(current_week=1)
        self.assertEqual(rebuilt["states"]["C1-PE-1#w1"], "rescheduled")
        self.assertIn("C1-PE-1#w1", rebuilt["rescheduled"])

    def _record_normal_raw(self, ledger, slot_id, week, tokens=4, when=None):
        """与 _record_normal 相同，但可写入任意账本（乱序重放用）。"""
        occ = Occasion(slot_id, week)
        ledger.append("teacher_report", TeacherReport(
            occ, "T-WANG", ActivityKind.PE_CLASS, "BB", 40,
            SessionMode.NORMAL, "V-FIELD"))
        ledger.append("venue_observation", VenueObservation(occ, "V-FIELD", 40))
        for i in range(tokens):
            ledger.append("sample_attendance",
                          att(f"tok{i}", occ, when or self.when(week)))
        return occ

    def test_replay_order_does_not_change_confirmation(self):
        # 同批事件以不同追加顺序构建两个账本，结论必须一致
        import random
        payloads: list[tuple[str, object]] = []
        for w in range(1, 4):
            for slot in self.slots:
                occ = Occasion(slot.slot_id, w)
                payloads.append(("teacher_report", TeacherReport(
                    occ, "T-WANG", ActivityKind.PE_CLASS, "BB", 40,
                    SessionMode.NORMAL, "V-FIELD")))
                payloads.append(("venue_observation",
                                 VenueObservation(occ, "V-FIELD", 40)))
                for i in range(4):
                    payloads.append(("sample_attendance",
                                     att(f"tok{i}", occ, self.when(w))))
        # 同一学生在一周内混入更早有效/更晚重复/超窗离线，顺序敏感
        dup_occ = Occasion("C1-PE-2", 2)
        payloads += [
            ("sample_attendance", att("dup", dup_occ, "2026-09-14T11:00:00Z")),
            ("sample_attendance", att("dup", dup_occ, "2026-09-14T09:00:00Z",
                                      source="offline",
                                      received="2026-09-14T15:00:00Z")),
            ("sample_attendance", att("dup", dup_occ, "2026-09-14T07:00:00Z",
                                      source="offline",
                                      received="2026-09-16T08:00:00Z")),
        ]
        summaries = []
        for seed in range(5):
            shuffled = payloads[:]
            random.Random(seed).shuffle(shuffled)
            ledger = EventLedger()
            for et, p in shuffled:
                ledger.append(et, p)
            rebuilt = ledger.rebuild_class(
                "C1", self.slots, HEADCOUNT, {}, {}, VENUES, current_week=3)
            summaries.append((
                rebuilt["states"],
                tuple((s.occasion.key(), s.confirmed, s.flags) for s in rebuilt["sessions"]),
            ))
        first = summaries[0]
        for other in summaries[1:]:
            self.assertEqual(other[0], first[0])  # 状态一致
            self.assertEqual(other[1], first[1])  # 会话/标记一致

    def test_late_offline_arrival_reconfirms_only_that_occasion(self):
        # 第 1 周先只有 3 个有效样本 -> 周一场次不成立
        occ = Occasion("C1-PE-1", 1)
        self.ledger.append("teacher_report", TeacherReport(
            occ, "T-WANG", ActivityKind.PE_CLASS, "BB", 40,
            SessionMode.NORMAL, "V-FIELD"))
        self.ledger.append("venue_observation", VenueObservation(occ, "V-FIELD", 40))
        for i in range(3):
            self.ledger.append("sample_attendance",
                               att(f"tok{i}", occ, self.when(1)))
        # 同周其他场次正常，作为“未受影响场次”
        self._record_normal("C1-PE-2", 1)
        before = self._rebuild(current_week=1)
        self.assertEqual(before["states"]["C1-PE-1#w1"], "in_review")
        self.assertEqual(before["states"]["C1-PE-2#w1"], "completed")

        # 第 4 名学生的离线补传在窗口内到达 -> 只重算 C1-PE-1
        self.ledger.append("sample_attendance",
                           att("tok3", occ, self.when(1), source="offline",
                               received="2026-09-02T10:00:00Z"))
        incremental = self.ledger.reconfirm_occasion(
            "C1", self.slots, HEADCOUNT, {}, {}, VENUES, 1, before, occ)
        full = self._rebuild(current_week=1)
        self.assertEqual(incremental["states"], full["states"])
        self.assertEqual(incremental["states"]["C1-PE-1#w1"], "completed")
        # 其他场次结论原样保留
        self.assertEqual(incremental["states"]["C1-PE-2#w1"], "completed")
        self.assertEqual(
            [(s.occasion.key(), s.confirmed, s.per_student) for s in incremental["sessions"]],
            [(s.occasion.key(), s.confirmed, s.per_student) for s in full["sessions"]],
        )

    def test_makeup_attendance_backfills_only_original_occasion(self):
        # 原场次被占课待补；补课场次已安排但缺学生确认
        original = Occasion("C1-PE-2", 1)
        makeup = Occasion("C1-PE-1", 2)
        self._record_normal("C1-PE-1", 1)
        self.ledger.append("takeover",
                           {"slot_id": "C1-PE-2", "week": 1, "subject": "数学", "ref": ""})
        self.ledger.append("makeup_plan", {"occasion": makeup, "makeup_for": original})
        self.ledger.append("teacher_report", TeacherReport(
            makeup, "T-WANG", ActivityKind.PE_CLASS, "BB", 40,
            SessionMode.MAKEUP, "V-GYM", basis_ref="MK-002", makeup_for=original))
        self.ledger.append("venue_observation", VenueObservation(makeup, "V-GYM", 40))
        for i in range(3):
            self.ledger.append("sample_attendance",
                               att(f"mk{i}", makeup, "2026-09-14T10:00:00Z"))
        before = self._rebuild(current_week=2)
        self.assertEqual(before["states"]["C1-PE-2#w1"], "taken_over")
        self.assertEqual(before["states"]["C1-PE-1#w2"], "in_review")

        # 第 4 名学生补传到补课场次 -> 补课成立，应回填原场次为 completed
        self.ledger.append("sample_attendance",
                           att("mk3", makeup, "2026-09-14T10:00:00Z",
                               source="offline", received="2026-09-15T10:00:00Z"))
        incremental = self.ledger.reconfirm_occasion(
            "C1", self.slots, HEADCOUNT, {}, {}, VENUES, 2, before, makeup)
        full = self._rebuild(current_week=2)
        self.assertEqual(incremental["states"], full["states"])
        self.assertEqual(incremental["states"]["C1-PE-1#w2"], "completed_makeup")
        self.assertEqual(incremental["states"]["C1-PE-2#w1"], "completed")
        self.assertIn("C1-PE-2#w1", incremental["made_up"])


# ---------------------------------------------------------------- 覆盖规则

class CoverageTest(unittest.TestCase):
    def _sessions(self, rebuilt):
        return rebuilt["sessions"]

    def test_full_delivery_has_no_gaps(self):
        ledger = EventLedger()
        slots = five_pe_slots("C1")
        goal = GOAL_BB
        # teach: 周1-2 体育课；practice: 周1-4 每天都有；match: 周3-4 加赛事
        for w in range(1, 5):
            for slot in slots:
                occ = Occasion(slot.slot_id, w)
                ledger.append("teacher_report", TeacherReport(
                    occ, "T-WANG", ActivityKind.PE_CLASS, "BB", 40,
                    SessionMode.NORMAL, "V-FIELD"))
                ledger.append("venue_observation", VenueObservation(occ, "V-FIELD", 40))
                for i in range(4):
                    ledger.append("sample_attendance", att(f"t{i}", occ, f"2026-09-0{w}T10:00:00"))
        rebuilt = ledger.rebuild_class("C1", slots, HEADCOUNT, {}, {}, VENUES, current_week=4)
        coverage = compute_class_coverage("C1", rebuilt, (goal,))
        bb = coverage.skill_coverage[0]
        self.assertTrue(bb.taught)
        self.assertEqual(bb.practiced_ratio, 1.0)
        # 没有 CLASS_MATCH 事件，常赛为缺口
        self.assertEqual(bb.matched_ratio, 0.0)
        self.assertTrue(any("常赛缺口" in g for g in bb.gaps))

    def test_exam_drill_and_free_play_grant_no_skill_coverage(self):
        ledger = EventLedger()
        slots = five_pe_slots("C1")
        occ = Occasion("C1-PE-1", 1)
        ledger.append("teacher_report", TeacherReport(
            occ, "T-WANG", ActivityKind.PE_CLASS, "BB", 40,
            SessionMode.EXAM_DRILL, "V-FIELD"))
        ledger.append("venue_observation", VenueObservation(occ, "V-FIELD", 40))
        for i in range(4):
            ledger.append("sample_attendance", att(f"t{i}", occ, "2026-09-07T10:00:00"))
        rebuilt = ledger.rebuild_class("C1", slots, HEADCOUNT, {}, {}, VENUES, current_week=1)
        coverage = compute_class_coverage("C1", rebuilt, (GOAL_BB,))
        bb = coverage.skill_coverage[0]
        self.assertFalse(bb.taught)

    def test_makeup_backfills_teach_week(self):
        ledger = EventLedger()
        slots = five_pe_slots("C1")
        missing = Occasion("C1-PE-1", 1)
        makeup = Occasion("C1-PE-1", 3)
        ledger.append("makeup_plan", {"occasion": makeup, "makeup_for": missing})
        ledger.append("teacher_report", TeacherReport(
            makeup, "T-WANG", ActivityKind.PE_CLASS, "BB", 40,
            SessionMode.MAKEUP, "V-GYM", basis_ref="MK-01", makeup_for=missing))
        ledger.append("venue_observation", VenueObservation(makeup, "V-GYM", 40))
        for i in range(4):
            ledger.append("sample_attendance", att(f"t{i}", makeup, "2026-09-21T10:00:00"))
        rebuilt = ledger.rebuild_class("C1", slots, HEADCOUNT, {}, {}, VENUES, current_week=3)
        coverage = compute_class_coverage("C1", rebuilt, (GOAL_BB,))
        bb = coverage.skill_coverage[0]
        self.assertTrue(bb.taught)  # 补课按第 1 周归类，教会目标兑现


# ---------------------------------------------------------------- 异常与复核

class ReviewTest(unittest.TestCase):
    def test_yin_yang_and_patterns_open_review_not_penalty(self):
        ledger = EventLedger()
        slots = five_pe_slots("C1")
        # 周一二：只练考试；周三四：整节自由活动；周五：完全无事件（阴阳课表）
        for day, mode in ((1, SessionMode.EXAM_DRILL), (2, SessionMode.EXAM_DRILL),
                          (3, SessionMode.FREE), (4, SessionMode.FREE)):
            occ = Occasion(f"C1-PE-{day}", 1)
            ledger.append("teacher_report", TeacherReport(
                occ, "T-WANG", ActivityKind.PE_CLASS, "BB", 40, mode, "V-FIELD"))
            ledger.append("venue_observation", VenueObservation(occ, "V-FIELD", 40))
            for i in range(4):
                ledger.append("sample_attendance", att(f"t{i}", occ, "2026-09-07T10:00:00"))
        rebuilt = ledger.rebuild_class("C1", slots, HEADCOUNT, {}, {}, VENUES, current_week=1)
        anomalies = detect_anomalies("C1", rebuilt)
        kinds = {a.kind for a in anomalies}
        self.assertIn(AnomalyKind.YIN_YANG, kinds)      # 周四、五无事件
        self.assertIn(AnomalyKind.EXAM_DRILL, kinds)   # 连续应考
        self.assertIn(AnomalyKind.FREE_PLAY, kinds)

        board = ReviewBoard()
        case = board.open_case("C1", anomalies)
        self.assertEqual(board.open_classes(), ("C1",))
        self.assertEqual(case.status, ReviewStatus.OPEN)
        # 教研复核：确认占课有据 -> 只安排补课，不处罚
        board.resolve(case.case_id, ReviewStatus.MAKEUP_ORDERED, "教研员刘",
                      "两日考试项目训练属实，安排第 3 周补技能课")
        self.assertEqual(board.get(case.case_id).status, ReviewStatus.MAKEUP_ORDERED)
        with self.assertRaises(ValueError):
            board.resolve(case.case_id, ReviewStatus.CLEARED, "教研员刘", "不能二次裁定")

    def test_takeover_chain_detected(self):
        ledger = EventLedger()
        slots = five_pe_slots("C1")
        for day in (1, 2):
            ledger.append("takeover",
                          {"slot_id": f"C1-PE-{day}", "week": 1,
                           "subject": "数学", "ref": ""})
        rebuilt = ledger.rebuild_class("C1", slots, HEADCOUNT, {}, {}, VENUES, current_week=1)
        kinds = {a.kind for a in detect_anomalies("C1", rebuilt)}
        self.assertIn(AnomalyKind.TAKEOVER, kinds)

    def test_clock_anomaly_and_late_signin_flagged_for_review(self):
        ledger = EventLedger()
        slots = five_pe_slots("C1")
        occ = Occasion("C1-PE-1", 1)
        ledger.append("teacher_report", TeacherReport(
            occ, "T-WANG", ActivityKind.PE_CLASS, "BB", 40,
            SessionMode.NORMAL, "V-FIELD"))
        ledger.append("venue_observation", VenueObservation(occ, "V-FIELD", 40))
        for i in range(4):
            ledger.append("sample_attendance",
                          att(f"t{i}", occ, "2026-09-01T10:00:00Z"))
        # 时钟异常（接收早于发生）与超窗离线各一条，均只留痕
        ledger.append("sample_attendance",
                      att("bad-clock", occ, "2026-09-01T10:00:00Z",
                          received="2026-09-01T09:00:00Z"))
        ledger.append("sample_attendance",
                      att("late", occ, "2026-09-01T10:00:00Z", source="offline",
                          received="2026-09-04T10:00:00Z"))
        rebuilt = ledger.rebuild_class("C1", slots, HEADCOUNT, {}, {}, VENUES, current_week=1)
        flags = rebuilt["flags"]["C1-PE-1#w1"]
        self.assertIn("clock_anomaly:received_before_occurred:bad-clock", flags)
        self.assertIn("offline_late:late", flags)
        kinds = {a.kind for a in detect_anomalies("C1", rebuilt)}
        self.assertIn(AnomalyKind.SIGNIN, kinds)


# ---------------------------------------------------------------- 可见性

class VisibilityTest(unittest.TestCase):
    def test_pseudonym_salt_breaks_cross_semester_link(self):
        a = pseudonym("stu-1", "salt-2026-1")
        b = pseudonym("stu-1", "salt-2026-1")
        c = pseudonym("stu-1", "salt-2026-2")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_parent_can_only_read_own_child(self):
        vault = IdentityVault("salt")
        vault.enroll("stu-1")
        vault.enroll("stu-2")
        vault.link_parent("parent-1-cred", "stu-1")
        with self.assertRaises(AccessDenied):
            build_parent_view("forged-cred", vault, {}, [])
        view = build_parent_view("parent-1-cred", vault,
                                 {"stu-1": (InjuryAdaptation("stu-1", "BB", 10, "MED-1"),)}, [])
        self.assertEqual(view.student_label, "本人子女")
        self.assertEqual(len(view.adaptations), 1)

    def _class_coverage(self, cid, total, completed, pending):
        from pe_domain.coverage import ClassCoverage, SkillCoverage
        return ClassCoverage(
            class_id=cid, total_occasions=total, completed=completed,
            pending_makeup=pending, in_review=0,
            skill_coverage=(SkillCoverage("BB", True, 1.0, 1.0, (), ()),),
        )

    def test_public_summary_suppressed_below_three_classes(self):
        coverages = {"C1": self._class_coverage("C1", 20, 18, 2),
                     "C2": self._class_coverage("C2", 20, 20, 0)}
        summary = build_public_summary("S", coverages, {})
        self.assertFalse(summary.published)

    def test_public_summary_aggregates_and_excludes_open_review(self):
        coverages = {f"C{i}": self._class_coverage(f"C{i}", 20, 18, 2) for i in range(1, 4)}
        board = ReviewBoard()
        case = board.open_case("C3", [])
        # C3 在复核中 -> 整班排除，只剩 2 个班 -> 抑制
        summary = build_public_summary("S", coverages, {"C3": board.get(case.case_id)})
        self.assertFalse(summary.published)

        board.resolve(case.case_id, ReviewStatus.CLEARED, "教研员", "天气替代有据")
        coverages2 = dict(coverages)
        coverages2["C4"] = self._class_coverage("C4", 20, 18, 2)
        cases = {"C3": board.get(case.case_id)}
        summary = build_public_summary("S", coverages2, cases)
        self.assertTrue(summary.published)
        self.assertEqual(summary.classes_counted, 4)
        self.assertAlmostEqual(summary.occasions_completed_ratio, 0.9)

    def test_parent_view_and_coverage_sync_after_late_arrival(self):
        # 补传修正确认结果后，家长视图与班级覆盖率同步看到修正后的有效时长
        vault = IdentityVault("salt-2026-1")
        token = vault.enroll("stu-1")
        vault.link_parent("parent-cred", "stu-1")
        ledger = EventLedger()
        slots = five_pe_slots("C1")
        occ = Occasion("C1-PE-1", 1)
        ledger.append("teacher_report", TeacherReport(
            occ, "T-WANG", ActivityKind.PE_CLASS, "BB", 40,
            SessionMode.NORMAL, "V-FIELD"))
        ledger.append("venue_observation", VenueObservation(occ, "V-FIELD", 40))
        # 该生在线签到 + 2 名同学 -> 3 个有效样本，未达 4 人下限
        ledger.append("sample_attendance", att(token, occ, "2026-09-01T10:00:00Z"))
        for i in range(2):
            ledger.append("sample_attendance",
                          att(f"peer{i}", occ, "2026-09-01T10:00:00Z"))
        # 一条超窗离线补传（50 小时）只留痕不计入
        ledger.append("sample_attendance",
                      att("peer2", occ, "2026-09-01T10:00:00Z", source="offline",
                          received="2026-09-03T12:00:00Z"))
        before = ledger.rebuild_class("C1", slots, HEADCOUNT, {}, vault.token_to_student,
                                      VENUES, current_week=1)
        self.assertEqual(before["states"]["C1-PE-1#w1"], "in_review")
        self.assertIn("offline_late:peer2", before["flags"]["C1-PE-1#w1"])
        view_before = build_parent_view("parent-cred", vault, {}, before["sessions"])
        self.assertEqual(view_before.recent_minutes, ())  # 未确认场次不展示
        cov_before = compute_class_coverage("C1", before, (GOAL_BB,))
        self.assertEqual(cov_before.in_review, 1)

        # 第 4 名同学的有效离线补传到达 -> 场次成立
        ledger.append("sample_attendance",
                      att("peer3", occ, "2026-09-01T10:00:00Z", source="offline",
                          received="2026-09-02T10:00:00Z"))
        after = ledger.reconfirm_occasion(
            "C1", slots, HEADCOUNT, {}, vault.token_to_student, VENUES, 1, before, occ)
        self.assertEqual(after["states"]["C1-PE-1#w1"], "completed")
        view_after = build_parent_view("parent-cred", vault, {}, after["sessions"])
        self.assertEqual(view_after.recent_minutes, (("C1-PE-1#w1", 40, "normal"),))
        cov_after = compute_class_coverage("C1", after, (GOAL_BB,))
        self.assertEqual(cov_after.in_review, 0)
        self.assertEqual(cov_after.completed, 1)


if __name__ == "__main__":
    unittest.main()
