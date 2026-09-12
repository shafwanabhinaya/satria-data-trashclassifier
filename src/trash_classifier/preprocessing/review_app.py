"""Flask app factory buat tinjau manual multi-reviewer -- dipakai untuk lebih dari
satu putaran review (audit awal A/B/C/D, lalu penentuan label tujuan kategori A)
dengan rubrik & sumber kandidat yang beda-beda, tanpa duplikasi kode Flask/HTML/JS.

Sengaja generik: `decision_options` (kode -> deskripsi) menentukan tombol yang
tampil DAN validasi keputusan di `ReviewQueue`. Keyboard shortcut dipetakan
berdasar urutan (tombol ke-1 = tekan '1', dst), bukan huruf tetap, supaya
cocok untuk rubrik apapun (huruf A/B/C/D atau nama kelas panjang).
"""
from __future__ import annotations

import secrets
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template_string, request, send_file, session, url_for

from .review import ReviewQueue


def build_review_app(
    *,
    candidates_df,
    review_csv_path: Path,
    train_dir: Path,
    decision_options: dict[str, str],
    passcode: str,
    page_title: str,
    given_label_caption: str,
) -> Flask:
    queue = ReviewQueue(candidates_df, review_csv_path, decision_options=decision_options)
    train_dir = train_dir.resolve()
    decisions = list(decision_options.items())

    app = Flask(__name__)
    app.secret_key = secrets.token_hex(16)

    def _authed() -> bool:
        return session.get("ok") is True

    @app.before_request
    def require_login():
        if request.endpoint in ("login", "static"):
            return None
        if not _authed():
            return redirect(url_for("login"))
        return None

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        if request.method == "POST":
            if request.form.get("passcode") == passcode:
                session["ok"] = True
                return redirect(url_for("index"))
            error = "Passcode salah."
        return render_template_string(LOGIN_HTML, error=error, page_title=page_title)

    @app.route("/")
    def index():
        return render_template_string(
            INDEX_HTML, decisions=decisions, page_title=page_title, given_label_caption=given_label_caption)

    @app.route("/image")
    def image():
        raw_path = request.args.get("path", "")
        candidate = Path(raw_path).resolve()
        if not candidate.is_relative_to(train_dir):
            return "Path tidak diizinkan", 403
        if not candidate.exists():
            return "File tidak ditemukan", 404
        return send_file(candidate)

    @app.route("/api/next")
    def api_next():
        reviewer = (request.args.get("name") or "anonim").strip() or "anonim"
        item = queue.claim_next(reviewer)
        if item is None:
            return jsonify({"done": True, "progress": queue.progress()})
        return jsonify({
            "done": False,
            "filepath": item["filepath"],
            "given_label": item["given_label"],
            "progress": queue.progress(),
        })

    @app.route("/api/skip", methods=["POST"])
    def api_skip():
        data = request.get_json(force=True)
        queue.release(data["filepath"])
        return jsonify({"ok": True})

    @app.route("/api/decision", methods=["POST"])
    def api_decision():
        data = request.get_json(force=True)
        reviewer = (data.get("name") or "anonim").strip() or "anonim"
        record = queue.submit_decision(
            filepath=data["filepath"], reviewer=reviewer,
            decision=data["decision"], note=data.get("note", ""),
        )
        return jsonify({
            "ok": True,
            "predicted_label": record["predicted_label"],
            "label_quality_score": record["label_quality_score"],
            "progress": queue.progress(),
        })

    @app.route("/api/progress")
    def api_progress():
        return jsonify(queue.progress())

    return app


LOGIN_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Masuk - {{ page_title }}</title>
<style>
body{font-family:system-ui,sans-serif;max-width:360px;margin:96px auto;padding:0 16px;color:#222}
input{width:100%;box-sizing:border-box;padding:10px;font-size:16px;margin:10px 0;border:1px solid #ccc;border-radius:6px}
button{width:100%;padding:10px;font-size:16px;border:0;border-radius:6px;background:#2563eb;color:#fff;cursor:pointer}
.error{color:#c0392b}
</style></head>
<body>
<h2>{{ page_title }}</h2>
{% if error %}<p class="error">{{ error }}</p>{% endif %}
<form method="post">
<input type="password" name="passcode" placeholder="Passcode" autofocus>
<button type="submit">Masuk</button>
</form>
</body></html>
"""

INDEX_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>{{ page_title }}</title>
<style>
  body{font-family:system-ui,sans-serif;margin:0;padding:0;color:#1a1a1a;background:#f7f7f8}
  .wrap{max-width:920px;margin:0 auto;padding:16px}
  header{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
  #progress{font-variant-numeric:tabular-nums;color:#444}
  .card{background:#fff;border-radius:10px;padding:16px;margin-top:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
  .rubric{display:grid;grid-template-columns:1fr;gap:6px;font-size:14px;color:#333}
  .rubric b{color:#2563eb}
  #imgbox{text-align:center}
  #imgbox img{max-width:100%;max-height:60vh;border-radius:8px;border:1px solid #ddd}
  .given{font-size:18px;margin:10px 0}
  .given b{color:#1a1a1a}
  .btns{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
  .btns button{flex:1;min-width:120px;padding:14px;font-size:16px;border:2px solid #2563eb;border-radius:8px;background:#fff;color:#2563eb;cursor:pointer;font-weight:600}
  .btns button:hover{background:#2563eb;color:#fff}
  .btns button:disabled{opacity:.35;cursor:not-allowed;background:#fff;color:#999}
  .btns button:disabled:hover{background:#fff;color:#999}
  textarea{width:100%;box-sizing:border-box;margin-top:10px;padding:8px;font-size:14px;border-radius:6px;border:1px solid #ccc;min-height:50px}
  #skipbtn{margin-top:10px;background:none;border:1px solid #999;color:#555;padding:8px 14px;border-radius:6px;cursor:pointer}
  #reveal{display:none;background:#eef6ff;border:1px solid #b6d6ff;border-radius:8px;padding:12px;margin-top:14px}
  #nextbtn{margin-top:10px;padding:10px 18px;background:#16a34a;color:#fff;border:0;border-radius:6px;cursor:pointer;font-size:15px}
  #nameRow{margin-bottom:8px}
  #nameRow input{padding:6px 8px;border-radius:6px;border:1px solid #ccc}
  #done{display:none;text-align:center;padding:40px 0}
  kbd{background:#eee;border-radius:4px;padding:1px 6px;font-family:monospace}
</style></head>
<body>
<div class="wrap">
  <header>
    <h2>{{ page_title }}</h2>
    <div id="progress">memuat...</div>
  </header>

  <div id="nameRow">Nama reviewer: <input id="nameInput" placeholder="mis. Andi"></div>

  <div class="card rubric">
    {% for code, desc in decisions %}
    <div><b>{{ loop.index }}</b> ({{ code }}) -- {{ desc }}</div>
    {% endfor %}
  </div>

  <div class="card" id="reviewCard">
    <div id="imgbox"><img id="img" src="" alt="gambar"></div>
    <div class="given">{{ given_label_caption }} <b id="givenLabel">-</b></div>
    <div class="btns">
      {% for code, desc in decisions %}
      <button data-d="{{ code }}" title="tekan {{ loop.index }}">{{ loop.index }} - {{ desc }}</button>
      {% endfor %}
    </div>
    <textarea id="note" placeholder="Catatan singkat (opsional)..."></textarea>
    <div><button id="skipbtn">Lewati (serahkan ke reviewer lain)</button></div>

    <div id="reveal">
      <div>Prediksi model: <b id="predLabel"></b> (skor: <span id="predScore"></span>)</div>
      <button id="nextbtn">Lanjut ke gambar berikutnya &rarr;</button>
    </div>
  </div>

  <div id="done" class="card">
    <h3>Semua kandidat sudah ditinjau. Terima kasih!</h3>
    <div id="byReviewer"></div>
  </div>

  <p style="color:#888;font-size:13px">Shortcut keyboard: angka sesuai nomor tombol di atas, <kbd>Enter</kbd> untuk lanjut setelah reveal.</p>
</div>

<script>
let current = null; // {filepath, given_label}
let awaitingNext = false;
const KEY_TO_DECISION = {};
document.querySelectorAll('.btns button').forEach((btn, i) => { KEY_TO_DECISION[String(i + 1)] = btn.dataset.d; });

const nameInput = document.getElementById('nameInput');
nameInput.value = localStorage.getItem('reviewerName') || '';
nameInput.addEventListener('change', () => localStorage.setItem('reviewerName', nameInput.value));

function reviewerName() {
  return nameInput.value.trim() || 'anonim';
}

function renderProgress(p) {
  document.getElementById('progress').textContent =
    p.reviewed + ' / ' + p.total + ' ditinjau (' + p.in_progress + ' sedang dipegang reviewer lain)';
}

async function fetchNext() {
  document.getElementById('reveal').style.display = 'none';
  awaitingNext = false;
  const res = await fetch('/api/next?name=' + encodeURIComponent(reviewerName()));
  const data = await res.json();
  renderProgress(data.progress);
  if (data.done) {
    document.getElementById('reviewCard').style.display = 'none';
    document.getElementById('done').style.display = 'block';
    const byRev = data.progress.by_reviewer || {};
    document.getElementById('byReviewer').textContent =
      Object.entries(byRev).map(([k, v]) => k + ': ' + v).join(', ');
    current = null;
    return;
  }
  current = data;
  document.getElementById('img').src = '/image?path=' + encodeURIComponent(data.filepath);
  document.getElementById('givenLabel').textContent = data.given_label;
  document.getElementById('note').value = '';
  // Nonaktifkan tombol yang kodenya persis sama dengan given_label -- tidak masuk akal
  // pilih "benar" balik ke label yang justru sedang ditinjau/dikonfirmasi salah.
  document.querySelectorAll('.btns button').forEach(btn => {
    btn.disabled = btn.dataset.d === data.given_label;
  });
}

async function submitDecision(decision) {
  if (!current || awaitingNext) return;
  awaitingNext = true; // kunci SEBELUM request dikirim -- cegah double-klik/keypress ganda mengirim 2x
  const note = document.getElementById('note').value;
  const res = await fetch('/api/decision', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({filepath: current.filepath, name: reviewerName(), decision, note}),
  });
  const data = await res.json();
  renderProgress(data.progress);
  document.getElementById('predLabel').textContent = data.predicted_label;
  document.getElementById('predScore').textContent = Number(data.label_quality_score).toFixed(4);
  document.getElementById('reveal').style.display = 'block';
}

document.querySelectorAll('.btns button').forEach(btn => {
  btn.addEventListener('click', () => submitDecision(btn.dataset.d));
});

document.getElementById('skipbtn').addEventListener('click', async () => {
  if (!current) return;
  await fetch('/api/skip', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({filepath: current.filepath}),
  });
  fetchNext();
});

document.getElementById('nextbtn').addEventListener('click', fetchNext);

document.addEventListener('keydown', (e) => {
  if (document.activeElement === document.getElementById('note')) return;
  if (e.key in KEY_TO_DECISION && !awaitingNext) {
    submitDecision(KEY_TO_DECISION[e.key]);
  } else if (e.key === 'Enter' && awaitingNext) {
    fetchNext();
  }
});

fetchNext();
</script>
</body></html>
"""
