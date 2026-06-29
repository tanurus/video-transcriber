// Progressive enhancement: upload the file via XHR so we can show real upload
// progress, then hand off to the job page (which streams transcription logs).
// With JS disabled the plain form POST still works and redirects server-side.
(function () {
  var form = document.getElementById("upload-form");
  if (!form || !window.XMLHttpRequest || !("upload" in new XMLHttpRequest())) {
    return; // no XHR upload support -> fall back to native form submit
  }

  var input = document.getElementById("file-input");
  var btn = document.getElementById("submit-btn");
  var wrap = document.getElementById("upload-progress");
  var bar = document.getElementById("progress-bar");
  var pct = document.getElementById("progress-pct");
  var text = document.getElementById("progress-text");

  function mb(bytes) { return (bytes / (1024 * 1024)).toFixed(1) + " MB"; }

  function fail(message) {
    text.textContent = message;
    pct.textContent = "";
    bar.className = "progress-fill failed";
    bar.style.width = "100%";
    btn.disabled = false;
    btn.textContent = "Start transcription";
  }

  form.addEventListener("submit", function (e) {
    if (!input.files || !input.files.length) { return; } // let native "required" handle it
    e.preventDefault();

    var xhr = new XMLHttpRequest();
    xhr.open("POST", form.getAttribute("action"));
    xhr.setRequestHeader("X-Requested-With", "XMLHttpRequest");

    btn.disabled = true;
    btn.textContent = "Uploading…";
    bar.className = "progress-fill";
    bar.style.width = "0%";
    pct.textContent = "0%";
    text.textContent = "Uploading…";
    wrap.removeAttribute("hidden");

    xhr.upload.addEventListener("progress", function (ev) {
      if (ev.lengthComputable) {
        var p = Math.round((ev.loaded / ev.total) * 100);
        bar.style.width = p + "%";
        pct.textContent = p + "%";
        text.textContent = "Uploading " + mb(ev.loaded) + " of " + mb(ev.total);
      }
    });

    // Bytes finished sending; server is now saving the file + creating the job.
    xhr.upload.addEventListener("load", function () {
      bar.className = "progress-fill indeterminate";
      bar.style.width = "100%";
      pct.textContent = "";
      text.textContent = "Upload complete — starting transcription…";
    });

    xhr.addEventListener("load", function () {
      if (xhr.status >= 200 && xhr.status < 300) {
        var url = null;
        try { url = JSON.parse(xhr.responseText).job_url; } catch (_) {}
        window.location = url || xhr.responseURL || "/";
      } else {
        var msg = "Upload failed (HTTP " + xhr.status + ").";
        try { msg = JSON.parse(xhr.responseText).error || msg; } catch (_) {}
        fail(msg);
      }
    });
    xhr.addEventListener("error", function () { fail("Network error during upload."); });
    xhr.addEventListener("abort", function () { fail("Upload cancelled."); });

    xhr.send(new FormData(form));
  });
})();
