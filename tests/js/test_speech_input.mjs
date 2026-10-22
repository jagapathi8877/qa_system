/*
 * Unit tests for static/speech-input.js — run with:  node --test tests/js/
 *
 * The Web Speech API is mocked with a FakeRecognition class that records
 * lifecycle calls and lets tests fire the onresult / onerror / onend
 * handlers directly, covering: start, interim results, final results,
 * errors (denied / no-speech), stop, and unsupported browsers.
 */

import test from "node:test";
import assert from "node:assert/strict";
import QuillSpeech from "../../static/speech-input.js";

const { createSpeechInput, LANG } = QuillSpeech;

/* ------------------------------------------------------------------ mock */

class FakeRecognition {
  constructor() {
    this.started = false;
    this.stopped = false;
    this.aborted = false;
    FakeRecognition.instances.push(this);
  }
  start() {
    if (this.started) throw new Error("InvalidStateError");
    this.started = true;
  }
  stop() {
    this.stopped = true;
    this.started = false;
  }
  abort() {
    this.aborted = true;
    this.started = false;
  }
  /* test helpers: simulate the API's event handlers */
  emitResult(results, resultIndex = 0) {
    this.onresult({ resultIndex, results });
  }
  emitError(code) {
    this.onerror({ error: code });
  }
  emitEnd() {
    this.onend();
  }
}
FakeRecognition.instances = [];

function withFakeCtor() {
  globalThis.webkitSpeechRecognition = FakeRecognition;
  return () => {
    delete globalThis.webkitSpeechRecognition;
    FakeRecognition.instances = [];
  };
}

function results(interimText, finalText) {
  // minimal EventResults-like array
  const items = [];
  if (interimText) {
    items.push({ isFinal: false, 0: { transcript: interimText } });
  }
  if (finalText) {
    items.push({ isFinal: true, 0: { transcript: finalText } });
  }
  return items;
}

/* ---------------------------------------------------------------- tests */

test("reports unsupported when no SpeechRecognition exists", () => {
  const ctl = createSpeechInput({});
  assert.equal(ctl.supported, false);
  const errors = [];
  const ctl2 = createSpeechInput({ onError: (e) => errors.push(e) });
  assert.equal(ctl2.start(), false);
  assert.equal(errors[0].code, "unsupported");
});

test("start() enters listening state and configures recognition", () => {
  const cleanup = withFakeCtor();
  const states = [];
  const ctl = createSpeechInput({ onStateChange: (s) => states.push(s) });
  assert.equal(ctl.supported, true);
  ctl.start();
  assert.equal(ctl.listening, true);
  assert.equal(states.at(-1), "listening");
  const rec = FakeRecognition.instances.at(-1);
  assert.equal(rec.started, true);
  assert.equal(rec.lang, LANG); // en-US via constant
  assert.equal(rec.interimResults, true);
  cleanup();
});

test("double start() does not restart recognition", () => {
  const cleanup = withFakeCtor();
  const ctl = createSpeechInput({});
  ctl.start();
  ctl.start();
  assert.equal(FakeRecognition.instances.length, 1);
  cleanup();
});

test("interim results stream via onInterim", () => {
  const cleanup = withFakeCtor();
  const interim = [];
  const ctl = createSpeechInput({ onInterim: (t) => interim.push(t) });
  ctl.start();
  FakeRecognition.instances.at(-1).emitResult(results("who wrote"));
  assert.deepEqual(interim, ["who wrote"]);
  cleanup();
});

test("final results arrive via onFinal", () => {
  const cleanup = withFakeCtor();
  const finals = [];
  const ctl = createSpeechInput({ onFinal: (t) => finals.push(t) });
  ctl.start();
  FakeRecognition.instances.at(-1).emitResult(
    results("who wrote", "Who wrote Hamlet")
  );
  assert.deepEqual(finals, ["Who wrote Hamlet"]);
  cleanup();
});

test("denied permission maps to a friendly message and error state", () => {
  const cleanup = withFakeCtor();
  const events = { errors: [], states: [] };
  const ctl = createSpeechInput({
    onError: (e) => events.errors.push(e),
    onStateChange: (s) => events.states.push(s),
  });
  ctl.start();
  FakeRecognition.instances.at(-1).emitError("not-allowed");
  assert.match(events.errors[0].message, /denied/i);
  assert.equal(events.states.at(-1), "error");
  assert.equal(ctl.listening, false);
  cleanup();
});

test("no-speech error maps to its own message", () => {
  const cleanup = withFakeCtor();
  const errors = [];
  const ctl = createSpeechInput({ onError: (e) => errors.push(e) });
  ctl.start();
  FakeRecognition.instances.at(-1).emitError("no-speech");
  assert.match(errors[0].message, /no speech was detected/i);
  cleanup();
});

test("stop() calls recognition.stop and returns to idle", () => {
  const cleanup = withFakeCtor();
  const states = [];
  const ctl = createSpeechInput({ onStateChange: (s) => states.push(s) });
  ctl.start();
  ctl.stop();
  assert.equal(ctl.listening, false);
  assert.equal(FakeRecognition.instances.at(-1).stopped, true);
  assert.equal(states.at(-1), "idle");
  // the natural onend after a user stop must not flip state again
  FakeRecognition.instances.at(-1).emitEnd();
  assert.equal(states.at(-1), "idle");
  cleanup();
});

test("toggle alternates listening / stopped", () => {
  const cleanup = withFakeCtor();
  const ctl = createSpeechInput({});
  assert.equal(ctl.toggle(), true);
  assert.equal(ctl.listening, true);
  assert.equal(ctl.toggle(), false);
  assert.equal(ctl.listening, false);
  cleanup();
});

test("abort() hard-cancels without a state ping-pong", () => {
  const cleanup = withFakeCtor();
  const ctl = createSpeechInput({});
  ctl.start();
  ctl.abort();
  assert.equal(FakeRecognition.instances.at(-1).aborted, true);
  assert.equal(ctl.listening, false);
  cleanup();
});

test("autoSubmit defaults to false and is configurable", () => {
  assert.equal(createSpeechInput({})._config.autoSubmit, false);
  assert.equal(createSpeechInput({ autoSubmit: true })._config.autoSubmit, true);
});
