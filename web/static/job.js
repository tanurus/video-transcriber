(function () {
  var script = document.currentScript;
  var api = script.getAttribute("data-api");
  var logEl = document.getElementById("log");
  var statusEl = document.getElementById("status");
  var downloadEl = document.getElementById("download");
  var errorEl = document.getElementById("error");
  var spinnerEl = document.getElementById("spinner");
  var since = 0;
  var terminal = { done: 1, error: 1, interrupted: 1 };

  function poll() {
    fetch(api + "?since=" + since)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.lines && data.lines.length) {
          data.lines.forEach(function (line) { logEl.textContent += line + "\n"; });
          since = data.next_index;
        }
        statusEl.textContent = data.status;
        if (data.error && errorEl) {
          errorEl.textContent = data.error;
          errorEl.removeAttribute("hidden");
        }
        if (data.download_ready) { downloadEl.removeAttribute("hidden"); }
        if (terminal[data.status]) {
          if (spinnerEl) { spinnerEl.setAttribute("hidden", ""); }
          return;
        }
        setTimeout(poll, 1000);
      })
      .catch(function () { setTimeout(poll, 2000); });
  }
  poll();
})();
