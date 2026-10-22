/* ==========================================================================
   speech-input.js — thin wrapper around the Web Speech API
   (SpeechRecognition / webkitSpeechRecognition).

   Exposes a small state machine so the UI never touches the raw API:
     createSpeechInput({ lang, autoSubmit, onInterim, onFinal,
                         onStateChange, onError })
       .supported  — feature detection (Firefox → false)
       .start() / .stop() / .toggle() / .abort()
       .listening  — boolean

   States: "idle" | "listening" | "error"
   Error codes from the API are mapped to short, user-friendly messages.

   Honesty note (mirrored in the UI): in Chrome/Edge this API streams audio
   to the browser vendor's speech service — it is not an offline feature.
   ========================================================================== */

(function (global) {
  "use strict";

  var LANG = "en-US";

  var ERROR_MESSAGES = {
    "not-allowed": "Microphone access was denied — allow it in your browser settings and try again.",
    "service-not-allowed": "Microphone access was denied — allow it in your browser settings and try again.",
    "no-speech": "Didn't catch that — no speech was detected.",
    "network": "Speech service network error — check your connection.",
    "audio-capture": "No microphone was found.",
    "aborted": null, // expected after stop(); never surfaced
  };

  function getCtor(globalObj) {
    return globalObj.SpeechRecognition || globalObj.webkitSpeechRecognition || null;
  }

  function createSpeechInput(options) {
    var config = Object.assign(
      {
        lang: LANG,
        interimResults: true,
        continuous: false,
        autoSubmit: false, // constant-style switch; off by default
        onInterim: null,   // (text) => void — streamed while speaking
        onFinal: null,     // (text) => void — final transcript
        onStateChange: null, // ("idle" | "listening" | "error") => void
        onError: null,     // ({ code, message }) => void
      },
      options || {}
    );

    var globalObj = typeof window !== "undefined" ? window : global;
    var Ctor = getCtor(globalObj);
    var recognition = null;
    var listening = false;
    var stoppedByUser = false;

    function setState(state) {
      if (typeof config.onStateChange === "function") config.onStateChange(state);
    }

    function emitError(code) {
      var message = ERROR_MESSAGES[code] || "Voice input error (" + code + ").";
      setState("error");
      listening = false;
      if (typeof config.onError === "function") config.onError({ code: code, message: message });
    }

    function build() {
      var rec = new Ctor();
      rec.lang = config.lang;
      rec.interimResults = config.interimResults;
      rec.continuous = config.continuous;
      rec.maxAlternatives = 1;

      rec.onresult = function (event) {
        var interim = "";
        var finalText = "";
        for (var i = event.resultIndex; i < event.results.length; i++) {
          var result = event.results[i];
          if (result.isFinal) finalText += result[0].transcript;
          else interim += result[0].transcript;
        }
        if (interim && typeof config.onInterim === "function") {
          config.onInterim(interim.trim());
        }
        if (finalText && typeof config.onFinal === "function") {
          config.onFinal(finalText.trim());
        }
      };

      rec.onerror = function (event) {
        emitError(event && event.error ? event.error : "unknown");
      };

      rec.onend = function () {
        // Natural end (silence/timeout) or our stop(). Only the user's stop
        // suppresses the state change; errors already emitted their own.
        listening = false;
        if (!stoppedByUser) setState("idle");
      };
      return rec;
    }

    function start() {
      if (!Ctor) {
        emitError("unsupported");
        return false;
      }
      if (listening) return true; // double-start is a no-op
      stoppedByUser = false;
      recognition = build();
      try {
        recognition.start();
        listening = true;
        setState("listening");
        return true;
      } catch (err) {
        // InvalidStateError etc. — surface as a generic voice error
        emitError("start-failed");
        return false;
      }
    }

    function stop() {
      stoppedByUser = true;
      if (recognition && listening) {
        try {
          recognition.stop();
        } catch (err) {
          /* already ended */
        }
      }
      listening = false;
      setState("idle");
    }

    function abort() {
      // Hard cancel (page unload): discard any pending transcript silently.
      stoppedByUser = true;
      listening = false;
      if (recognition) {
        try {
          recognition.abort();
        } catch (err) {
          /* already ended */
        }
      }
    }

    function toggle() {
      if (listening) {
        stop();
        return false;
      }
      start();
      return listening;
    }

    return {
      supported: !!Ctor,
      get listening() {
        return listening;
      },
      start: start,
      stop: stop,
      abort: abort,
      toggle: toggle,
      /* exposed for tests */
      _config: config,
    };
  }

  var api = { createSpeechInput: createSpeechInput, LANG: LANG, ERROR_MESSAGES: ERROR_MESSAGES };

  /* browser global */
  global.QuillSpeech = api;
  /* CommonJS export so the module is unit-testable in Node */
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
