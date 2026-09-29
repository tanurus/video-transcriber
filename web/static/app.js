/* Transcriber web UI — one small script, no build step. */
(function () {
  "use strict";
  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };
  var sleep = function (ms) { return new Promise(function (r) { setTimeout(r, ms); }); };
  var page = document.body.getAttribute("data-page");

  // ---- helpers ------------------------------------------------------------------------
  function toast(message, bad) {
    var box = $("#toasts");
    var el = document.createElement("div");
    el.className = "toast" + (bad ? " bad" : "");
    el.textContent = message;
    box.appendChild(el);
    setTimeout(function () { el.remove(); }, bad ? 8000 : 4500);
  }

  async function api(url, opts) {
    opts = opts || {};
    var init = { method: opts.method || "GET", headers: {} };
    if (opts.json !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(opts.json);
    } else if (opts.body !== undefined) {
      init.body = opts.body;
    }
    var resp = await fetch(url, init);
    var data = null;
    try { data = await resp.json(); } catch (_) { /* not JSON */ }
    if (!resp.ok) {
      var err = new Error((data && data.error) || ("HTTP " + resp.status));
      err.status = resp.status;
      err.data = data;
      throw err;
    }
    return data;
  }

  function fmtBytes(n) {
    if (n < 1048576) return (n / 1024).toFixed(0) + " KB";
    if (n < 1073741824) return (n / 1048576).toFixed(1) + " MB";
    return (n / 1073741824).toFixed(2) + " GB";
  }

  function pillsFor(job) {
    var out = [];
    var st = {
      queued: ["Queued", "run"], running: ["Transcribing…", "run"], done: ["Transcribed", "ok"],
      error: ["Failed", "bad"], interrupted: ["Interrupted", "warn"]
    }[job.status] || [job.status, ""];
    out.push(st);
    var ai = job.ai_status;
    if (ai === "queued" || ai === "running") out.push(["AI…", "run"]);
    else if (ai === "done") out.push(["AI ✓", "ok"]);
    else if (ai === "error") out.push(["AI failed", "bad"]);
    var sy = job.sync_status;
    if (sy === "pending") out.push(job.sync_error ? ["Supabase: retrying", "warn"] : ["Sending…", "run"]);
    else if (sy === "synced") out.push(["In Supabase", "ok"]);
    else if (sy === "error") out.push(["Supabase failed", "bad"]);
    return out;
  }

  function renderPills(el, job) {
    if (!el) return;
    el.innerHTML = "";
    pillsFor(job).forEach(function (p) {
      var s = document.createElement("span");
      s.className = "pill " + p[1];
      s.textContent = p[0];
      var tip = (p[0].indexOf("AI") === 0 ? job.ai_error : p[0].indexOf("Supabase") === 0 ? job.sync_error : job.error);
      if (tip) s.title = tip;
      el.appendChild(s);
    });
  }

  function isActive(job) {
    return job.status === "queued" || job.status === "running" || job.ai_status === "queued" ||
      job.ai_status === "running" || (job.sync_status === "pending" && !job.sync_error);
  }

  // Live status pills for any list of [data-id] rows.
  function watchRows(rows) {
    var ids = rows.map(function (r) { return r.getAttribute("data-id"); }).filter(Boolean);
    if (!ids.length) return;
    var byId = {};
    rows.forEach(function (r) { byId[r.getAttribute("data-id")] = r; });
    async function tick() {
      var data;
      try { data = await api("/api/jobs?ids=" + ids.join(",")); } catch (_) { setTimeout(tick, 10000); return; }
      var active = false;
      data.jobs.forEach(function (j) {
        var row = byId[j.id];
        if (!row) return;
        renderPills($("[data-pills]", row), j);
        row.setAttribute("data-status", j.status);
        var t = $(".title", row);
        if (t && j.title && t.textContent !== j.title) t.textContent = j.title;
        if (isActive(j)) active = true;
      });
      setTimeout(tick, active ? 3000 : 20000);
    }
    tick();
  }

  // Clipboard API needs HTTPS; the tailnet URL is plain HTTP, so fall back.
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    var ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.focus(); ta.select();
    var ok = false;
    try { ok = document.execCommand("copy"); } catch (_) {}
    ta.remove();
    return ok ? Promise.resolve() : Promise.reject(new Error("copy blocked"));
  }

  // ---- health dot -----------------------------------------------------------------------
  async function health() {
    var el = $("#health");
    try {
      var h = await api("/api/health");
      if (h.gpu) {
        el.className = "health " + (h.gpu.ok ? "ok" : "down");
        el.title = h.gpu.ok ? "GPU server ready" : "GPU server: " + h.gpu.message;
      } else { el.className = "health"; el.title = "Cloud transcription"; }
      var sys = $("#sys-gpu");
      if (sys) sys.textContent = h.gpu ? (h.gpu.ok ? "ready" : "not responding — " + h.gpu.message) : "not configured";
    } catch (_) { el.className = "health down"; el.title = "Server unreachable"; }
    setTimeout(health, 30000);
  }
  health();

  // ---- settings form (options) ----------------------------------------------------------------
  function initOptions(root) {
    if (!root) return;
    var meta = {};
    try { meta = JSON.parse(root.getAttribute("data-meta") || "{}"); } catch (_) {}
    var presets = meta.presets || {}, defaults = meta.defaults || {};
    var touched = {};
    Object.keys(presets).forEach(function (k) { Object.keys(presets[k]).forEach(function (f) { touched[f] = 1; }); });

    function input(key) { return root.querySelector('[name="opt_' + key + '"]:not([type=hidden])'); }
    function setValue(key, value) {
      var el = input(key);
      if (!el) return;
      if (el.type === "checkbox") el.checked = !!value; else el.value = value;
    }
    function applyVisibility() {
      $$("[data-show-if]", root).forEach(function (f) {
        var src = input(f.getAttribute("data-show-if"));
        var allowed = f.getAttribute("data-show-values").split(",");
        f.hidden = !(src && allowed.indexOf(src.value) >= 0);
      });
    }
    $$('input[name="preset"]', root).forEach(function (r) {
      r.addEventListener("change", function () {
        if (!r.value || !presets.hasOwnProperty(r.value)) return;
        Object.keys(touched).forEach(function (k) {
          setValue(k, presets[r.value].hasOwnProperty(k) ? presets[r.value][k] : defaults[k]);
        });
        applyVisibility();
      });
    });
    root.addEventListener("input", function (e) {
      var name = e.target.getAttribute("name") || "";
      if (name.indexOf("opt_") === 0 && touched[name.slice(4)]) {
        var custom = root.querySelector('input[name="preset"][value=""]');
        if (custom) custom.checked = true;
      }
      applyVisibility();
    });
    applyVisibility();
  }
  $$("[data-options]").forEach(initOptions);

  function collectOptions(scope) {
    var out = {}, preset = null;
    $$("input, select, textarea", scope).forEach(function (el) {
      var name = el.getAttribute("name") || "";
      if (name === "preset" && el.checked) preset = el.value || null;
      if (name.indexOf("opt_") !== 0 || name.slice(-9) === "__present") return;
      var key = name.slice(4);
      out[key] = el.type === "checkbox" ? el.checked : el.value;
    });
    return { options: out, preset: preset };
  }

  // ---- menus & confirmations ------------------------------------------------------------------
  document.addEventListener("click", function (e) {
    var btn = e.target.closest("[data-menu]");
    $$(".menu-items").forEach(function (m) { if (!btn || m !== btn.nextElementSibling) m.hidden = true; });
    if (btn) { var items = btn.nextElementSibling; items.hidden = !items.hidden; e.preventDefault(); }
    var c = e.target.closest("[data-confirm]");
    if (c && !confirm(c.getAttribute("data-confirm"))) e.preventDefault();
    var cp = e.target.closest("[data-copy]");
    if (cp) {
      var src = $(cp.getAttribute("data-copy"));
      copyText(src.textContent).then(function () { toast("Copied."); }, function () { toast("Copy blocked by the browser — select and copy manually.", true); });
    }
  });

  // ---- upload page ----------------------------------------------------------------------------
  function initUpload() {
    var form = $("#upload-form"), input = $("#file-input"), zone = $("#dropzone");
    var list = $("#file-list"), submit = $("#submit-btn");
    var queue = [];
    var busy = false;

    function addFiles(files) {
      Array.prototype.forEach.call(files, function (f) {
        if (queue.some(function (q) { return q.file.name === f.name && q.file.size === f.size && q.state !== "done"; })) return;
        var li = document.createElement("li");
        li.innerHTML = '<span class="name"></span><button type="button" class="remove ghost">Remove</button>' +
          '<span class="sub"></span><span class="bar"><i></i></span>';
        $(".name", li).textContent = f.name;
        $(".sub", li).textContent = fmtBytes(f.size) + " · ready";
        var item = { file: f, li: li, state: "ready" };
        $(".remove", li).addEventListener("click", function () {
          if (item.state === "uploading") return;
          queue.splice(queue.indexOf(item), 1); li.remove(); refresh();
        });
        queue.push(item);
        list.appendChild(li);
      });
      refresh();
    }
    function refresh() {
      var ready = queue.filter(function (q) { return q.state === "ready" || q.state === "failed"; }).length;
      list.hidden = queue.length === 0;
      submit.disabled = busy || ready === 0;
      submit.textContent = busy ? "Uploading…" : ready > 1 ? "Transcribe " + ready + " files" : "Start transcription";
    }
    input.addEventListener("change", function () { addFiles(input.files); input.value = ""; });
    ["dragenter", "dragover"].forEach(function (ev) {
      zone.addEventListener(ev, function (e) { e.preventDefault(); zone.classList.add("over"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      zone.addEventListener(ev, function (e) { e.preventDefault(); zone.classList.remove("over"); });
    });
    zone.addEventListener("drop", function (e) { if (e.dataTransfer && e.dataTransfer.files) addFiles(e.dataTransfer.files); });
    window.addEventListener("beforeunload", function (e) { if (busy) { e.preventDefault(); e.returnValue = ""; } });

    async function sendChunk(id, offset, blob) {
      var resp = await fetch("/api/uploads/" + id + "?offset=" + offset, { method: "PUT", body: blob });
      var data = {};
      try { data = await resp.json(); } catch (_) {}
      if (resp.status === 409) return { resync: true };
      if (!resp.ok) { var e = new Error(data.error || "HTTP " + resp.status); e.fatal = resp.status < 500 && resp.status !== 408 && resp.status !== 429; throw e; }
      return data;
    }

    // Resumable upload: 8 MB pieces, each retried with backoff; after a dropped
    // connection the server says how much it has and we continue from there.
    async function uploadOne(item, settings) {
      var f = item.file, sub = $(".sub", item.li), bar = $(".bar i", item.li);
      item.state = "uploading"; item.li.className = "";
      var init = await api("/api/uploads", { method: "POST", json: { filename: f.name, size: f.size, last_modified: f.lastModified } });
      var id = init.upload_id, size = init.chunk_size, offset = 0, t0 = Date.now();
      while (offset < f.size) {
        var attempt = 0;
        for (;;) {
          try {
            var r = await sendChunk(id, offset, f.slice(offset, Math.min(offset + size, f.size)));
            if (r.resync) { offset = (await api("/api/uploads/" + id)).received; } else { offset = r.received; }
            break;
          } catch (e) {
            if (e.fatal) throw e;
            attempt += 1;
            if (attempt > 12) throw new Error("network kept failing: " + e.message);
            sub.textContent = "Connection problem — retrying (" + attempt + ")…";
            await sleep(Math.min(30000, 1000 * Math.pow(2, attempt)));
            try { offset = (await api("/api/uploads/" + id)).received; } catch (_) {}
          }
        }
        var pct = Math.round(offset / f.size * 100);
        bar.style.width = pct + "%";
        var secs = (Date.now() - t0) / 1000;
        var rate = offset / Math.max(1, secs);
        sub.textContent = fmtBytes(offset) + " of " + fmtBytes(f.size) + " · " + pct + "% · " + fmtBytes(rate) + "/s";
      }
      sub.textContent = "Uploaded — queuing…";
      var done = await api("/api/uploads/" + id + "/complete", { method: "POST", json: settings });
      item.state = "done"; item.li.className = "done"; bar.style.width = "100%";
      var job = done.job;
      sub.innerHTML = "";
      var a = document.createElement("a");
      a.href = job.url; a.textContent = done.duplicate ? "Already in your library — open it" : "Open transcript";
      var pills = document.createElement("span"); pills.className = "pills"; pills.setAttribute("data-pills", "");
      sub.appendChild(a); sub.appendChild(document.createTextNode(" ")); sub.appendChild(pills);
      item.li.setAttribute("data-id", job.id);
      $(".remove", item.li).hidden = true;
      return job;
    }

    form.addEventListener("submit", async function (e) {
      if (!window.fetch || !window.Blob) return; // plain multipart fallback
      e.preventDefault();
      if (busy) return;
      var settings = collectOptions(form);
      busy = true; refresh();
      var ok = 0, failed = 0, rows = [];
      for (var i = 0; i < queue.length; i++) {
        var item = queue[i];
        if (item.state !== "ready" && item.state !== "failed") continue;
        try { await uploadOne(item, settings); ok += 1; rows.push(item.li); }
        catch (err) {
          failed += 1; item.state = "failed"; item.li.className = "failed";
          $(".sub", item.li).textContent = "Failed: " + err.message + " — press the button to retry.";
        }
      }
      busy = false; refresh();
      if (ok) toast(ok + " file(s) queued for transcription.");
      if (failed) toast(failed + " upload(s) failed — they stay in the list to retry.", true);
      watchRows(rows);
    });

    $("#save-defaults").addEventListener("click", async function () {
      try { var r = await api("/api/settings/defaults", { method: "POST", json: collectOptions(form) }); toast(r.message); }
      catch (e) { toast(e.message, true); }
    });
    watchRows($$("#recent li[data-id]"));
  }

  // ---- library ----------------------------------------------------------------------------------
  function initLibrary() {
    var boxes = $$(".pick-box"), all = $("#select-all"), bar = $("#bulkbar"), count = $("#sel-count");
    function selected() { return boxes.filter(function (b) { return b.checked; }).map(function (b) { return b.value; }); }
    function sync() {
      var n = selected().length;
      count.textContent = n; bar.hidden = n === 0;
      all.checked = n > 0 && n === boxes.length; all.indeterminate = n > 0 && n < boxes.length;
      boxes.forEach(function (b) { b.closest(".item").classList.toggle("selected", b.checked); });
    }
    boxes.forEach(function (b) { b.addEventListener("change", sync); });
    all.addEventListener("change", function () { boxes.forEach(function (b) { b.checked = all.checked; }); sync(); });

    async function bulk(action, extra) {
      var ids = selected();
      try {
        var r = await api("/api/jobs/bulk", { method: "POST", json: Object.assign({ action: action, ids: ids }, extra || {}) });
        toast(r.message);
        if (action === "delete" || action === "regenerate") setTimeout(function () { location.reload(); }, 700);
      } catch (e) { toast(e.message, true); }
    }
    $$("[data-bulk]", bar).forEach(function (btn) {
      btn.addEventListener("click", function () {
        var action = btn.getAttribute("data-bulk");
        if (action === "delete") {
          if (confirm("Delete " + selected().length + " transcript(s) from this server? Rows already in Supabase stay there.")) bulk("delete");
        } else if (action === "regenerate") {
          $("#regen-count").textContent = selected().length;
          $("#regen-dialog").showModal();
        } else { bulk(action); }
      });
    });
    $$("[data-zip]", bar).forEach(function (a) {
      a.addEventListener("click", function () {
        location.href = "/download.zip?kind=" + a.getAttribute("data-zip") + "&ids=" + selected().join(",");
      });
    });
    var dlg = $("#regen-dialog");
    $("#regen-go").addEventListener("click", function (e) {
      e.preventDefault();
      var s = collectOptions(dlg);
      s.keep_base = !!$('[name="keep_base"]', dlg).checked;
      dlg.close();
      bulk("regenerate", s);
    });
    watchRows($$("#library .item"));
    sync();
  }

  // ---- job page -----------------------------------------------------------------------------------
  function initJob() {
    var art = $(".job"), id = art.getAttribute("data-id"), apiUrl = art.getAttribute("data-api");
    var log = $("#log"), since = 0, lastStatus = art.getAttribute("data-status"), lastAi = null;

    async function poll() {
      var j;
      try { j = await api(apiUrl + "?since=" + since); } catch (_) { setTimeout(poll, 8000); return; }
      if (j.lines.length) {
        log.textContent += j.lines.join("\n") + "\n";
        log.scrollTop = log.scrollHeight;
      }
      since = j.next_index;
      renderPills($("[data-pills]", art), j);
      $("#spinner").hidden = !(j.status === "queued" || j.status === "running");
      if (j.error) { $("#error").textContent = j.error; $("#error").hidden = false; }
      // Re-render once the transcript or the AI result appears.
      if ((lastStatus !== "done" && j.status === "done") || (lastAi && lastAi !== "done" && j.ai_status === "done")) {
        location.reload(); return;
      }
      lastStatus = j.status; lastAi = j.ai_status;
      setTimeout(poll, isActive(j) ? 2500 : 30000);
    }
    poll();

    $$(".tabs button").forEach(function (t) {
      t.addEventListener("click", function () {
        $$(".tabs button").forEach(function (x) { x.classList.toggle("active", x === t); });
        $$("[data-pane]").forEach(function (p) { p.hidden = p.getAttribute("data-pane") !== t.getAttribute("data-tab"); });
      });
    });
    var copy = $("#btn-copy");
    if (copy) copy.addEventListener("click", function () {
      var pane = $$("[data-pane]").filter(function (p) { return !p.hidden; })[0];
      copyText(pane ? pane.textContent : "").then(function () { toast("Transcript copied."); },
        function () { toast("Copy blocked by the browser — select the text and copy.", true); });
    });

    $$(".editable").forEach(function (el) {
      var before = el.textContent;
      el.addEventListener("keydown", function (e) { if (e.key === "Enter" && el.tagName === "H1") { e.preventDefault(); el.blur(); } });
      el.addEventListener("blur", async function () {
        var value = el.textContent.trim();
        if (value === before.trim()) return;
        var body = {}; body[el.getAttribute("data-field")] = value;
        try { await api("/api/job/" + id + "/meta", { method: "POST", json: body }); before = value; toast("Saved."); }
        catch (e) { toast(e.message, true); el.textContent = before; }
      });
    });

    async function act(action, extra) {
      try {
        var r = await api("/api/jobs/bulk", { method: "POST", json: Object.assign({ action: action, ids: [id] }, extra || {}) });
        toast(r.message);
        if (action === "delete") { location.href = "/library"; return; }
        if (action === "regenerate" && r.created && r.created[0]) { location.href = "/job/" + r.created[0]; return; }
        setTimeout(poll, 800);
      } catch (e) { toast(e.message, true); }
    }
    $$("[data-action]").forEach(function (b) {
      b.addEventListener("click", function () {
        var a = b.getAttribute("data-action");
        if (a === "delete") { if (confirm("Delete this transcript version from this server?")) act("delete"); }
        else if (a === "regenerate") $("#regen-dialog").showModal();
        else act(a);
      });
    });
    var dlg = $("#regen-dialog");
    if (dlg) $("#regen-go").addEventListener("click", function (e) {
      e.preventDefault();
      var s = collectOptions(dlg);
      s.keep_base = true;
      dlg.close();
      act("regenerate", s);
    });
  }

  // ---- settings -------------------------------------------------------------------------------------
  function initSettings() {
    $$("[data-test]").forEach(function (b) {
      b.addEventListener("click", async function () {
        var kind = b.getAttribute("data-test"), out = $('[data-result="' + kind + '"]');
        out.className = "test-result"; out.textContent = "Testing… (save first if you changed anything)";
        try {
          var r = await api("/api/settings/test/" + kind, { method: "POST" });
          out.className = "test-result " + (r.ok ? "ok" : "bad"); out.textContent = r.message;
          if (r.models) {
            var dl = $("#model-list"); dl.innerHTML = "";
            r.models.forEach(function (m) { var o = document.createElement("option"); o.value = m; dl.appendChild(o); });
          }
        } catch (e) { out.className = "test-result bad"; out.textContent = e.message; }
      });
    });
  }

  if (page === "upload") initUpload();
  if (page === "library") initLibrary();
  if (page === "job") initJob();
  if (page === "settings") initSettings();
})();
