// vexp-hint: per-prompt orientation + idle verification (fail-open). Managed by vexp.
const VEXP_BIN = "c:/Users/hanaz/.vscode/extensions/vexp.vexp-vscode-2.5.3-win32-x64/binaries/vexp-core-win32-x64/vexp-core.exe";
export const VexpHint = async ({ directory, client }) => {
  const fs = await import("node:fs");
  const path = await import("node:path");
  const taskFileFor = (sid) =>
    path.join(directory, ".vexp", "task-" + String(sid || "unknown") + ".txt");
  const gateMarker = (sid) =>
    path.join(directory, ".vexp", "idle-gate-" + String(sid || "unknown") + ".done");
  return {
    "chat.message": async (input, output) => {
      try {
        const { execFileSync } = await import("node:child_process");
        const text = (output.parts || [])
          .filter((p) => p && p.type === "text" && typeof p.text === "string")
          .map((p) => p.text)
          .join("\n");
        if (!text || text.length < 40) return;
        // First prompt of the session = the task spec for the idle gate.
        const sid = (input && input.sessionID) || null;
        try {
          const tf = taskFileFor(sid);
          if (!fs.existsSync(tf)) {
            fs.mkdirSync(path.dirname(tf), { recursive: true });
            fs.writeFileSync(tf, text);
          }
        } catch (e) { /* fail open */ }
        const out = execFileSync(VEXP_BIN, ["prompt-hint"], {
          input: JSON.stringify({ prompt: text, session_id: sid }),
          timeout: 4000,
          env: { ...process.env, CLAUDE_PROJECT_DIR: directory },
          encoding: "utf8",
        });
        if (!out || !out.trim()) return;
        const hint = JSON.parse(out).hookSpecificOutput?.additionalContext;
        if (hint) output.parts.push({ type: "text", text: hint });
      } catch (e) { /* fail open */ }
    },
    event: async ({ event }) => {
      // Idle gate: the opencode twin of the Claude Stop hook. Runs the
      // mechanical completion check once per session; on gaps, sends ONE
      // follow-up prompt with the exact list (best effort - any failure
      // is silent and the session simply stays stopped).
      try {
        if (!event || event.type !== "session.idle") return;
        const sid = event.properties && event.properties.sessionID;
        if (!sid) return;
        const marker = gateMarker(sid);
        if (fs.existsSync(marker)) return;
        const tf = taskFileFor(sid);
        if (!fs.existsSync(tf)) return;
        const { execFileSync } = await import("node:child_process");
        const out = execFileSync(
          VEXP_BIN,
          ["verify", "--json", "--task-file", tf],
          { timeout: 15000, cwd: directory, encoding: "utf8" }
        );
        const rep = JSON.parse(out);
        const items = [];
        for (const f of (rep.spec && rep.spec.forbidden_touched) || [])
          items.push("- the task says NOT to modify `" + f + "` but it was changed - revert or justify");
        for (const a of (rep.spec && rep.spec.artifacts_missing) || [])
          items.push("- the task asks for `" + a + "` and it does not exist yet");
        for (const b of (rep.broken_imports || []).slice(0, 8))
          items.push("- " + b.file + ":" + b.line + " imports `" + b.imports + "` which no longer exists in " + b.from_changed_file);
        if (!items.length) return;
        fs.writeFileSync(marker, "1");
        await client.session.prompt({
          path: { id: sid },
          body: {
            parts: [{ type: "text", text:
              "vexp verify found mechanically checkable gaps between the session's work and the task:\n" +
              items.join("\n") + "\nFix each item above. This check will not repeat." }],
          },
        });
      } catch (e) { /* fail open */ }
    },
  };
};
