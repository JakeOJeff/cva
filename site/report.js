/*
 * The report renderer, shared by two callers that must never disagree:
 *
 *   site/index.html        the static demo, reading committed JSON
 *   web/static/job.html    the live server, reading /api/jobs/{id}/result
 *
 * Both are handed the same `result` document, so the rendering lives here
 * once. The only difference between them is whether the teacher can be
 * corrected in place, which needs a server - pass `onTeacher` to enable it.
 *
 * Plain global script on purpose: no modules, no build step, so the static
 * page keeps working when it is copied somewhere and opened.
 */
(function (global) {
"use strict";

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const mmss = (t) => `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
const pct = (x) => `${Math.round((x || 0) * 100)}%`;

/* ---------------------------------------------------------------- colour */

// Colour follows the speaker, not their rank in whatever is on screen now.
// Fixed order, never cycled: past five speakers everyone shares the muted
// slot. Slot 1 is always the teacher, so a speaker keeps its colour when the
// transcript is re-filtered or the teacher is corrected by hand.
function assignColors(res) {
  const teacher = res.teacher.teacher;
  const others = Object.keys(res.speakers)
    .filter(s => s !== teacher)
    .sort((a, b) => res.speakers[b].talk_time - res.speakers[a].talk_time);
  const colorOf = { [teacher]: "var(--sp1)" };
  others.forEach((s, i) => {
    colorOf[s] = i < 4 ? `var(--sp${i + 2})` : "var(--muted)";
  });
  return colorOf;
}

// Colours for provisional, chunk-local speakers during streaming. They are
// deliberately NOT the final palette: until reconcile runs, "c3_SPEAKER_00"
// and "c7_SPEAKER_00" are unrelated, and showing them in the same colour
// would assert something we do not yet know.
function provisionalColor(speaker) {
  let h = 0;
  for (const ch of speaker) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return `var(--sp${(h % 5) + 1})`;
}

/* --------------------------------------------------------------- tooltip */

let tip = null;
function ensureTip() {
  if (tip) return tip;
  tip = document.getElementById("tip");
  if (!tip) {
    tip = document.createElement("div");
    tip.id = "tip";
    tip.hidden = true;
    document.body.appendChild(tip);
  }
  return tip;
}

// One delegated hover handler covers every mark on every page.
function installTips() {
  document.addEventListener("mousemove", (e) => {
    const el = e.target.closest("[data-tip]");
    if (!el) { ensureTip().hidden = true; return; }
    const t = ensureTip();
    t.innerHTML = el.dataset.tip;
    t.hidden = false;
    const pad = 14;
    const r = t.getBoundingClientRect();
    t.style.left = Math.min(e.clientX + pad, innerWidth - r.width - 8) + "px";
    t.style.top = Math.min(e.clientY + pad, innerHeight - r.height - 8) + "px";
  });
  document.addEventListener("mouseleave", () => { ensureTip().hidden = true; });
}

/* -------------------------------------------------------------- warnings */

// Neither of these failures is loud on its own. A wrong --lang makes Whisper
// invent fluent words rather than error, and a model too small for the
// language writes confident English-looking text. Both produce a transcript
// that reads fine until you read it, so they lead the page.
function warnings(res) {
  let html = "";

  if (res.meta.language_mismatch) {
    html += `<div class="panel"><div class="warnbox"><b>Language mismatch.</b>
      This was transcribed as <b>${esc(res.meta.requested_language)}</b>, but the
      audio sounds like <b>${esc(res.meta.detected_language)}</b> (confidence
      ${esc(res.meta.detected_confidence)}). Whisper does not fail on the wrong
      language, it invents plausible words &mdash; so treat the transcript and
      everything derived from it as unreliable, and re-run in
      ${esc(res.meta.detected_language)}.</div></div>`;
  }

  if (res.meta.wrong_script) {
    html += `<div class="panel"><div class="warnbox">
      <b>This transcript is not in the right script.</b> Only
      ${Math.round((res.meta.script_ratio || 0) * 100)}% of it is in the script
      ${esc(res.meta.language)} is written in, which means the
      <b>${esc(res.meta.model)}</b> model is too small for this language and has
      written plausible-looking nonsense instead. The speaker timeline and talk
      ratios are still meaningful &mdash; they come from the audio, not the
      words &mdash; but the text, question counts and any AI review are not.
      Re-run with <b>small</b> or larger.</div></div>`;
  }

  return html;
}

/* -------------------------------------------------------------- sections */

function tiles(m) {
  const wait = m.median_wait_time == null ? "—" : `${m.median_wait_time.toFixed(1)}s`;
  const waitNote = m.median_wait_time == null ? "no questions detected"
    : m.median_wait_time < 3 ? "under the 3s threshold"
    : "3s or more — good";
  const items = [
    ["Teacher talk", pct(m.teacher_talk_ratio), "share of all speech"],
    ["Students", pct(m.student_talk_ratio), `${m.n_students_heard} heard`],
    ["Questions", m.teacher_questions, `${m.questions_per_10min}/10 min`],
    ["Median wait", wait, waitNote],
    ["Answered", pct(m.question_response_rate), `${m.questions_answered} of ${m.teacher_questions}`],
    ["Lesson", mmss(m.duration), `${pct(m.silence_ratio)} silence`],
  ];
  return `<div class="tiles">${items.map(([k, v, n]) =>
    `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div>
     <div class="n">${esc(n)}</div></div>`).join("")}</div>`;
}

function proportion(m) {
  const parts = [
    ["Teacher", m.teacher_talk_time, "var(--sp1)"],
    ["Students", m.student_talk_time, "var(--sp2)"],
    ["Silence", m.silence_time, "var(--line)"],
  ].filter(p => p[1] > 0);
  const total = parts.reduce((s, p) => s + p[1], 0) || 1;

  return `<div class="panel"><h2>Where the lesson went</h2>
    <div class="prop">${parts.map(([label, v, c]) =>
      `<div data-tip="<b>${label}</b><br>${mmss(v)} · ${pct(v / total)} of the lesson"
            style="flex:${v};background:${c}"></div>`).join("")}</div>
    <div class="legend">${parts.map(([label, v, c]) =>
      `<span><i style="background:${c}"></i>${label} ${pct(v / total)}</span>`).join("")}</div>
  </div>`;
}

function timeline(res, colorOf) {
  const dur = res.meta.duration || 1;
  const bands = res.utterances.map(u =>
    `<i style="left:${(u.start / dur) * 100}%;width:${Math.max(((u.end - u.start) / dur) * 100, 0.12)}%;
       background:${colorOf[u.speaker] || "var(--muted)"}"
       data-tip="<b>${esc(u.label)}</b> · ${mmss(u.start)}–${mmss(u.end)}<br>${esc(u.text.slice(0, 110))}${u.text.length > 110 ? "…" : ""}"></i>`
  ).join("");

  const legend = Object.entries(res.speakers)
    .map(([sp, s]) => `<span><i style="background:${colorOf[sp]}"></i>${esc(s.label)} · ${pct(s.talk_share)}</span>`)
    .join("");

  return `<div class="panel"><h2>Who spoke when</h2>
    <div class="timeline">${bands}</div>
    <div class="axis"><span>00:00</span><span>${mmss(dur / 2)}</span><span>${mmss(dur)}</span></div>
    <div class="legend">${legend}</div>
  </div>`;
}

function verdict(res, colorOf, allowOverride) {
  const v = res.teacher;

  const picker = allowOverride ? `<div style="min-width:230px">
      <label class="field"><span>Wrong? Set it by hand</span>
        <select id="teacherPick">${Object.entries(res.speakers).map(([sp, s]) =>
          `<option value="${sp}" ${sp === v.teacher ? "selected" : ""}>${esc(s.label)} — ${sp} (${pct(s.talk_share)})</option>`
        ).join("")}</select></label>
      <div style="margin-top:9px;display:flex;gap:8px;align-items:center">
        <button id="applyTeacher">Re-analyse</button>
        <span class="sub" id="teacherMsg"></span>
      </div>
    </div>` : "";

  const low = v.confidence === "low" ? `<div class="warnbox">
    <b>Low confidence.</b> The top two speakers scored within ${(v.margin ?? 0).toFixed(2)} of
    each other. That usually means co-teaching, one very talkative student, or a diarization
    that split one person across two labels. Check the timeline${
      allowOverride ? " and correct it below if needed" : ""}.
  </div>` : "";

  return `<div class="panel"><h2>Teacher</h2>
    <div class="verdict">
      <div style="flex:1;min-width:240px">
        <div class="who" style="color:${colorOf[v.teacher]}">${esc(res.speakers[v.teacher]?.label || v.teacher)}
          <span class="sub mono" style="color:var(--muted);font-size:12px">${esc(v.teacher)}</span></div>
        <div class="sub">${esc(v.confidence)} confidence · ${esc(v.method)}</div>
        <ul>${(v.reasons || []).map(r => `<li>${esc(r)}</li>`).join("")}</ul>
      </div>
      ${picker}
    </div>
    ${low}
  </div>`;
}

function review(res) {
  const r = res.review;
  if (!r) {
    return `<div class="panel"><h2>Classroom summary</h2>
      <p class="empty">This session was analysed without the AI review, so there is
      no written summary &mdash; every metric above is still computed. Re-run with
      <span class="mono">--llm</span> to add one.</p></div>`;
  }
  if (r.error) {
    return `<div class="panel"><h2>Classroom summary</h2>
      <p class="empty">${esc(r.error)}</p></div>`;
  }
  const sect = (title, html) => `<h3>${title}</h3>${html}`;
  const list = (arr) => `<ul>${(arr || []).map(x => `<li>${esc(x)}</li>`).join("")}</ul>`;

  return `<div class="panel review"><h2>Classroom summary</h2>
    ${sect("Summary", `<p>${esc(r.summary)}</p>`)}
    ${sect("Topics covered", `<div class="tags">${(r.topics_covered || [])
      .map(t => `<span class="tag">${esc(t)}</span>`).join("")}</div>`)}

    ${sect(`Questioning <span class="tag">${esc(r.questioning.dominant_question_type.replace(/_/g, " "))}</span>`,
      `<p>${esc(r.questioning.assessment)}</p>${list(r.questioning.examples)}`)}

    ${sect("Explanation", `<p>${esc(r.explanation_quality.assessment)}</p>
      <div class="tags">
        <span class="tag">${r.explanation_quality.uses_examples ? "uses examples" : "few examples"}</span>
        <span class="tag">${r.explanation_quality.checks_understanding ? "checks understanding" : "rarely checks understanding"}</span>
      </div>`)}

    ${sect(`Feedback to students <span class="tag">${esc(r.feedback_to_students.specificity)}</span>`,
      `<p>${esc(r.feedback_to_students.assessment)}</p>`)}

    ${sect("Classroom management", `<p>${esc(r.classroom_management)}</p>`)}
    ${sect("Language use", `<p>${esc(r.language_use)}</p>`)}
    ${sect("What worked", list(r.strengths))}

    ${sect("What to try next", `<ul>${(r.areas_for_improvement || []).map(a =>
      `<li><b>${esc(a.issue)}</b><br>${esc(a.suggestion)}</li>`).join("")}</ul>`)}

    ${sect("Notable moments", (r.notable_moments || []).map(m =>
      `<div class="moment"><div class="t">${esc(m.timestamp)}</div>
       <div>${esc(m.what_happened)}</div>
       <div class="sub" style="color:var(--muted)">${esc(m.why_it_matters)}</div></div>`).join(""))}

    ${r.transcript_quality_caveat
      ? `<div class="warnbox"><b>On the transcript itself.</b> ${esc(r.transcript_quality_caveat)}</div>` : ""}

    <div class="sub" style="margin-top:16px;color:var(--muted);font-size:12px">
      ${esc(r._model || "")} · ${r._usage ? `${r._usage.input_tokens} in / ${r._usage.output_tokens} out tokens` : ""}
    </div>
  </div>`;
}

/* ------------------------------------------------------- metric glossary */

// The numbers above are useless to a teacher who cannot see how they were
// derived, and a reviewer should not have to read analyze.py to check one.
// Every row here is the actual formula in pipeline/analyze.py.
const GLOSSARY = [
  {
    name: "Teacher Dominance Ratio",
    key: "teacher_talk_ratio",
    formula: "teacher_talk_time / (teacher + student + unattributed talk time)",
    logic: `Sum the duration of every utterance attributed to the teacher, and
            divide by all speech in the lesson. Silence is excluded from the
            denominator, so this is a share of talk, not of wall clock.
            Speech the diarizer could not attribute to anybody stays in the
            denominator but counts for nobody &mdash; calling it student talk
            would flatter the number.`,
    reading: `Classroom research puts the typical figure around 0.70. Sustained
              above ~0.80 is lecture rather than discussion; below ~0.50 usually
              means group work, or that the teacher was mislabelled.`,
  },
  {
    name: "Student Participation Indicator",
    key: "student_participation",
    formula: "per student: that student's talk_time / all student talk_time",
    logic: `Student utterances are grouped by diarization cluster and each
            cluster's share of total student speech is reported alongside
            <span class="mono">n_students_heard</span>, the number of distinct
            clusters.`,
    reading: `One student holding 80% of the student share is not participation,
              it is one child answering everything &mdash; a headline
              "students talked 25%" hides that completely. Read the shares, not
              just the total. Note the floor: clusters are voices, not children,
              so two quiet students at the back are often counted as one.`,
  },
  {
    name: "Interaction Count (IRF triads)",
    key: "irf_triads",
    formula: "count of consecutive utterance triples where role = teacher → student → teacher",
    logic: `Initiation&ndash;Response&ndash;Feedback is the signature move of
            recitation teaching: the teacher asks, a student answers, the
            teacher evaluates. Counting the triples measures how often the
            lesson actually closed that loop.`,
    reading: `A high count with a high dominance ratio is a briskly-run
              recitation lesson &mdash; lots of exchanges, all of them
              teacher-controlled. A high count with a low dominance ratio is
              genuine dialogue. Near zero means students were spoken at.`,
  },
  {
    name: "Question Response Rate",
    key: "question_response_rate",
    formula: "questions followed by a student within 10s / teacher questions",
    logic: `A question is any teacher utterance containing "?" or a question
            word from the lexicon for that language in
            <span class="mono">pipeline/lang.py</span>. It counts as answered
            when the very next utterance is a student's and begins within the
            10-second response window.`,
    reading: `A low rate against a high question count means questions were
              rhetorical, or that they were pitched past the class. Because the
              detector is lexical, this number degrades exactly as far as the
              transcript does &mdash; check the script warning first.`,
  },
  {
    name: "Median Wait Time",
    key: "median_wait_time",
    formula: "median gap, in seconds, between a teacher question ending and the next utterance starting",
    logic: `Every teacher question contributes one gap, whoever speaks next.
            The median rather than the mean, because one long pause while a
            class writes would drag an average past anything meaningful.`,
    reading: `Rowe's classic finding: under about 3 seconds, students cannot
              formulate anything beyond recall. Pushing this above 3s is one of
              the few teaching changes with a large, well-replicated effect.`,
  },
  {
    name: "Silence Ratio",
    key: "silence_ratio",
    formula: "(lesson duration − total speech time) / lesson duration",
    logic: `Whatever is left of the recording once every attributed utterance is
            removed.`,
    reading: `Not automatically bad &mdash; deskwork, reading and thinking are
              all silence. It is a context number for the ratios above rather
              than a score, and on a noisy classroom recording it also absorbs
              whatever the diarizer declined to call speech.`,
  },
];

function glossary() {
  return `<div class="panel"><h2>How each metric is computed</h2>
    <p class="sub" style="margin:0 0 4px">Every formula below is the one in
      <span class="mono">pipeline/analyze.py</span>. Nothing here is learned or
      weighted &mdash; the metrics are deterministic so they can be compared
      across sessions and weeks.</p>
    ${GLOSSARY.map(g => `
      <div class="metricdef">
        <h3>${g.name}</h3>
        <div class="formula mono">${g.formula}</div>
        <p>${g.logic}</p>
        <p class="reading"><b>Reading it.</b> ${g.reading}</p>
      </div>`).join("")}
  </div>`;
}

/* ------------------------------------------------------------ transcript */

function transcript() {
  return `<div class="panel"><h2>Transcript</h2>
    <div class="filters">
      <button data-role="" class="on">Everyone</button>
      <button data-role="teacher">Teacher only</button>
      <button data-role="student">Students only</button>
      <span class="sub" id="uttCount"></span>
    </div>
    <div id="utts"></div>
  </div>`;
}

function renderUtts(res, colorOf, role) {
  const utts = role ? res.utterances.filter(u => u.role === role) : res.utterances;
  document.getElementById("uttCount").textContent = `${utts.length} utterances`;
  document.getElementById("utts").innerHTML = utts.map(u => `
    <div class="utt ${u.speaker_conf < 0.75 ? "low" : ""}">
      <div class="t">${mmss(u.start)}</div>
      <div class="who" style="color:${colorOf[u.speaker] || "var(--muted)"}">${esc(u.label)}</div>
      <div class="tx">${esc(u.text)}</div>
    </div>`).join("") || '<p class="empty">Nothing here.</p>';
}

/* ------------------------------------------------------------------ main */

/**
 * Render a finished result into `el`.
 *
 * opts.onTeacher(speaker) - if given, the teacher can be corrected in place
 *   and this is called with the chosen speaker id. Omit on the static site,
 *   where re-deriving would need a server.
 */
function render(el, res, opts) {
  opts = opts || {};
  const colorOf = assignColors(res);

  el.innerHTML =
    warnings(res) +
    `<div class="panel"><h2>At a glance</h2>${tiles(res.metrics)}</div>` +
    verdict(res, colorOf, !!opts.onTeacher) +
    proportion(res.metrics) +
    timeline(res, colorOf) +
    review(res) +
    glossary() +
    transcript();

  renderUtts(res, colorOf, "");

  el.querySelectorAll(".filters button").forEach(b => {
    b.onclick = () => {
      el.querySelectorAll(".filters button").forEach(x => x.classList.remove("on"));
      b.classList.add("on");
      renderUtts(res, colorOf, b.dataset.role);
    };
  });

  if (opts.onTeacher) {
    document.getElementById("applyTeacher").onclick = () =>
      opts.onTeacher(document.getElementById("teacherPick").value);
  }
}

/** One-line description of a result, for a page subtitle. */
function subtitle(res) {
  const m = res.meta;
  return `${mmss(m.duration)} · ${m.n_speakers} speakers · ${
    m.backend === "scribe" ? "scribe" : "whisper " + m.model} · ${m.language}` +
    (m.elapsed ? ` · analysed in ${Math.round(m.elapsed)}s` : "");
}

global.Report = {
  render, subtitle, renderUtts, assignColors, provisionalColor,
  installTips, esc, mmss, pct,
};

})(window);
