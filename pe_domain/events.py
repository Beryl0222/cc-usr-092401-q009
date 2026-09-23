"""实课事件：最小化采集、三方交叉确认、有效运动时长核算。

隐私最小化（对应“不能把未成年人的体测和课堂影像集中暴露”）：

- 不采集任何影像、不采集体测分数；
- 学生侧只抽取“最小样本”，且以学期假名（token）出现，
  真实身份映射保存在独立的身份保管处（visibility.IdentityVault）；
- 未被抽中的学生不产生任何学生侧事件。

三方交叉确认：一节实课须同时具备
  1. 教师授课记录（实际内容、时长、授课形态）；
  2. 场地观测（实际场地、在场人数、容量）；
  3. 抽样学生到场确认（达到最小样本量）。
任何一方缺失都不成立“已完成”，转待确认/教研复核，而不是直接判违规。

防虚增：
  - 有效运动时长以“教师申报 × 标准时长上限 × 学生适配”三者取小；
  - 所有签到时间必须是带时区偏移的 ISO 8601 时间，解析后统一到 UTC
    同一时间线再比较——跨时区记录绝不能按截断后的本地钟面计算；
  - 同一 (学生, 场次) 的候选先各自判定有效性（无时区/无法解析/接收早于
    发生/离线超窗均判无效并分类留痕），再按“最早发生、稳定兜底”规则只取
    一条；重复事件保留并标记，无效记录不会因替换去重被重新带回样本；
  - 离线补签有受理时限，且同样遵循去重，不能把一节课签成多节课。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import blake2b

from .models import STANDARD_MINUTES, ActivityKind, InjuryAdaptation


class SessionMode(str, Enum):
    NORMAL = "normal"          # 按方案授课
    RAIN_ALT = "rain_alt"      # 降雨替代（须引用预案）
    HEAT_ALT = "heat_alt"      # 高温替代（须引用预案）
    RESCHEDULED = "rescheduled"  # 调课（须引用审批依据）
    FREE = "free"              # 整节自由活动（不给技能覆盖，触发关注）
    EXAM_DRILL = "exam_drill"   # 只练考试项目（计入运动，不给计划技能覆盖）
    MAKEUP = "makeup"          # 补课（须挂接被缺场次）


# 离线补签受理时限（小时）：超过则仅留痕，不计时长
OFFLINE_ACCEPT_HOURS = 48

# 抽样确认的最小样本数与最低比例（取两者计算后的较大值，且至少 3 人）
MIN_SAMPLE_COUNT = 3
MIN_SAMPLE_RATIO = 0.1


def pseudonym(student_id: str, semester_salt: str) -> str:
    """生成学期假名：同学期稳定、跨学期不可链接。"""
    digest = blake2b(
        f"{semester_salt}|{student_id}".encode("utf-8"), digest_size=8
    ).hexdigest()
    return f"S{digest}"


@dataclass(frozen=True)
class Occasion:
    """一个排定时段在某一周的具体场次。"""

    slot_id: str
    week: int

    def key(self) -> str:
        return f"{self.slot_id}#w{self.week}"


@dataclass(frozen=True)
class TeacherReport:
    occasion: Occasion
    teacher_id: str
    kind: ActivityKind
    taught_skill: str          # 实际教授的技能（可能与计划不同）
    minutes: int               # 教师申报的实际时长
    mode: SessionMode
    venue_id: str
    basis_ref: str = ""        # 替代/调课依据（预案条款或审批单号）
    makeup_for: Occasion | None = None  # 补课挂接的缺课场次
    note: str = ""


@dataclass(frozen=True)
class VenueObservation:
    occasion: Occasion
    venue_id: str
    observed_headcount: int


@dataclass(frozen=True)
class SampleAttendance:
    occasion: Occasion
    token: str
    occurred_at: str   # ISO 时间，实际到场时刻
    received_at: str   # ISO 时间，服务器接收时刻（在线时二者相同）
    source: str        # "online" / "offline"
    device_id: str = ""


@dataclass(frozen=True)
class ScheduleChange:
    """调课记录：把原场次改到新时间/新场地/新教师，须有审批依据。"""

    occasion: Occasion
    new_weekday: int
    new_venue_id: str
    new_teacher_id: str
    approver: str
    basis_ref: str


@dataclass(frozen=True)
class ConfirmationResult:
    occasion: Occasion
    confirmed: bool
    mode: SessionMode
    effective_minutes: int               # 本班该场核定时长
    per_student_minutes: dict[str, int]  # token -> 有效时长（含伤病适配）
    reasons: tuple[str, ...]             # 未确认/折减的可解释原因
    flags: tuple[str, ...]               # 重复签到、自由活动等关注标记


def parse_aware_iso(value: str) -> datetime:
    """解析带时区偏移的 ISO 8601 时间并统一到 UTC 时间线。

    无时区信息（naive）或无法解析的时间一律拒绝，由调用方留痕。
    """
    if not isinstance(value, str):
        raise ValueError("时间必须是 ISO 字符串")
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"无法解析的时间：{value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"时间缺少时区偏移：{value!r}")
    return parsed.astimezone(timezone.utc)


def _hours_between(later: datetime, earlier: datetime) -> float:
    return (later - earlier).total_seconds() / 3600


@dataclass(frozen=True)
class _Candidate:
    """一条签到候选的有效性判定结果（同一生同一场次可有多条）。"""

    att: SampleAttendance
    occurred: datetime
    received: datetime
    valid: bool
    issue: str = ""  # clock_bad / received_before_occurred / offline_late


def _evaluate_candidate(att: SampleAttendance) -> _Candidate:
    """先独立判定：时间可解析、带时区、接收不早于发生、离线补签在窗口内。

    判定只依赖候选自身，与候选到达顺序无关，保证重放顺序变化不改变结果。
    """
    try:
        occurred = parse_aware_iso(att.occurred_at)
    except ValueError as exc:
        return _Candidate(att, datetime.min.replace(tzinfo=timezone.utc),
                          datetime.min.replace(tzinfo=timezone.utc), False,
                          f"clock_bad:{exc}")
    try:
        received = parse_aware_iso(att.received_at)
    except ValueError as exc:
        return _Candidate(att, occurred, datetime.min.replace(tzinfo=timezone.utc),
                          False, f"clock_bad:{exc}")
    if received < occurred:
        return _Candidate(att, occurred, received, False,
                          "received_before_occurred")
    if att.source == "offline":
        delay = _hours_between(received, occurred)
        if delay > OFFLINE_ACCEPT_HOURS:
            return _Candidate(att, occurred, received, False,
                              f"offline_late:{delay:.1f}h")
    return _Candidate(att, occurred, received, True)


def assess_session(
    report: TeacherReport,
    venue_obs: VenueObservation | None,
    attendances: list[SampleAttendance],
    class_headcount: int,
    adaptations: dict[str, InjuryAdaptation],
    token_to_student: dict[str, str],
    *,
    now_iso: str | None = None,
) -> ConfirmationResult:
    """对一场课做三方交叉确认并核算有效运动时长。

    adaptations / token_to_student 只在核算内部使用，结果中只回传 token。
    """
    reasons: list[str] = []
    flags: list[str] = []
    occ = report.occasion

    # ---- 学生侧：先各自判定有效，再按稳定规则去重 ----
    # 每 (token) 一组候选：组内先独立判有效（时钟/时限各自留痕），
    # 有效候选中按 (发生时刻, 接收时刻, 设备号) 稳定取最早一条；
    # 无效候选绝不因替换去重被带回样本。判定与遍历顺序无关，重放可重入。
    candidates: dict[str, list[_Candidate]] = {}
    for att in attendances:
        if att.occasion != occ:
            continue
        candidates.setdefault(att.token, []).append(_evaluate_candidate(att))

    unique: dict[str, SampleAttendance] = {}
    for token in sorted(candidates):
        group = candidates[token]
        valid = [c for c in group if c.valid]
        # 重复、迟到、时钟异常分别留痕（账本里只有 token，不泄露真实学号）；
        # 无效标记按类别排序后输出，与候选到达顺序无关
        if len(group) > 1:
            flags.append(f"duplicate_signin:{token}")
        invalid_flags: list[str] = []
        for c in group:
            if c.valid:
                continue
            if c.issue.startswith("offline_late:"):
                invalid_flags.append(f"offline_late:{token}")
            elif c.issue == "received_before_occurred":
                invalid_flags.append(f"clock_anomaly:received_before_occurred:{token}")
            else:
                invalid_flags.append(f"clock_anomaly:unparseable:{token}")
        flags.extend(sorted(invalid_flags))
        if not valid:
            continue
        chosen = min(
            valid,
            key=lambda c: (c.occurred, c.received, c.att.device_id, c.att.source),
        ).att
        unique[token] = chosen

    # ---- 三方：学生样本量 ----
    required = max(MIN_SAMPLE_COUNT, int(class_headcount * MIN_SAMPLE_RATIO + 0.999))
    if len(unique) < required:
        reasons.append(f"抽样确认不足：{len(unique)} < 要求 {required}")

    # ---- 三方：场地 ----
    if venue_obs is None:
        reasons.append("缺少场地观测")
    else:
        if venue_obs.venue_id != report.venue_id:
            reasons.append(
                f"场地不一致：教师申报 {report.venue_id}，场地观测 {venue_obs.venue_id}"
            )
        if venue_obs.observed_headcount > class_headcount:
            reasons.append(
                f"场地观测人数 {venue_obs.observed_headcount} 超出班级人数 {class_headcount}"
            )
        if venue_obs.observed_headcount <= 0:
            reasons.append("场地观测到场人数为 0")

    # ---- 依据校验 ----
    if report.mode in (SessionMode.RAIN_ALT, SessionMode.HEAT_ALT, SessionMode.RESCHEDULED):
        if not report.basis_ref:
            reasons.append(f"{report.mode.value} 缺少依据编号")
    if report.mode == SessionMode.MAKEUP and report.makeup_for is None:
        reasons.append("补课未挂接被缺场次")

    # ---- 时长：教师申报受标准上限约束 ----
    cap = STANDARD_MINUTES[report.kind]
    if report.minutes <= 0:
        reasons.append("教师申报时长非正")
        minutes = 0
    elif report.minutes > cap:
        flags.append(f"minutes_capped:{report.minutes}>{cap}")
        minutes = cap
    else:
        minutes = report.minutes

    # 自由活动 / 单纯应考训练：给身体活动时长，但不产生技能覆盖（在 coverage 体现）
    if report.mode == SessionMode.FREE:
        flags.append("whole_free_play")
    if report.mode == SessionMode.EXAM_DRILL:
        flags.append("exam_drill_only")

    # ---- 每生时长：伤病适配折减（只对抽样确认到的学生出结果）----
    per_student: dict[str, int] = {}
    for token in sorted(unique):
        student_id = token_to_student.get(token)
        adapted = adaptations.get(student_id) if student_id else None
        if adapted is not None and adapted.skill_code == report.taught_skill:
            per_student[token] = min(minutes, adapted.adjusted_minutes)
            flags.append(f"adapted:{token}")
        else:
            per_student[token] = minutes

    confirmed = not reasons
    return ConfirmationResult(
        occasion=occ,
        confirmed=confirmed,
        mode=report.mode,
        effective_minutes=minutes if confirmed else 0,
        per_student_minutes=per_student if confirmed else {},
        reasons=tuple(reasons),
        flags=tuple(flags),
    )
