import os, glob, json, sys, subprocess, pathlib, yaml
from openai import OpenAI

ROOT = pathlib.Path(".")
PROMPT = os.environ["INPUT_PROMPT"].strip()
PATHS = [p.strip() for p in os.environ["INPUT_PATHS"].split(",") if p.strip()]
ALLOW_MAX_BYTES = 400_000  # guardrail: total file bytes
ALLOW_FILE_GLOBS = PATHS
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")  # override via repo secret/var if desired

# Optional repo policy file
policy_path = ROOT / ".ai-policy.yml"
policy = {}
if policy_path.exists():
    policy = yaml.safe_load(policy_path.read_text()) or {}
    ALLOW_MAX_BYTES = int(policy.get("max_bytes", ALLOW_MAX_BYTES))
    if "allow_globs" in policy:
        ALLOW_FILE_GLOBS = policy["allow_globs"]

def read_files():
    files = []
    seen = set()
    for g in ALLOW_FILE_GLOBS:
        for path in glob.glob(g, recursive=True):
            p = pathlib.Path(path)
            if not p.is_file():
                continue
            if p.suffix.lower() in {".png",".jpg",".jpeg",".gif",".pdf",".zip",".tar",".gz"}:
                continue
            if p.name.startswith(".ai-"):  # never edit config by accident
                continue
            if p in seen:
                continue
            seen.add(p)
            files.append(p)
    return files

files = read_files()
if not files:
    print("No files matched the allowed globs; nothing to edit.")
    sys.exit(0)

total_bytes = sum(p.stat().st_size for p in files)
if total_bytes > ALLOW_MAX_BYTES:
    print(f"Refusing: {total_bytes} bytes exceeds limit {ALLOW_MAX_BYTES}", file=sys.stderr)
    sys.exit(0)

def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()

# Capture original contents
original = {str(p): p.read_text(encoding="utf-8", errors="ignore") for p in files}

# Instruction for structured edits
system_msg = """You are a precise repo editor. Apply the user's request to the provided files.
- Make minimal, high-quality edits.
- Preserve formatting and front-matter.
- Do not invent facts or break links.
- Return ONLY a JSON object with an array 'changes' of {path, content} for files that should be replaced entirely.
- If no changes needed, return {"changes": []}.
"""

# Compact file bundle
file_bundle = "\n\n".join(
    f"=== FILE: {path} ===\n{content}"
    for path, content in original.items()
)

client = OpenAI()

response = client.responses.create(
    model=MODEL,
    reasoning={"effort": "medium"},
    input=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"User request:\n{PROMPT}\n\nProject files:\n{file_bundle}"},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "changes_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "changes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"}
                            },
                            "required": ["path", "content"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["changes"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
)

# Parse model output (new SDK first, then fallback)
data = None
try:
    data = response.output[0].content[0].parsed  # parsed JSON (newer SDKs)
except Exception:
    try:
        out_text = getattr(response, "output_text", None) or str(response)
        data = json.loads(out_text)
    except Exception:
        print("Failed to parse model output as JSON; aborting.", file=sys.stderr)
        sys.exit(0)

changes = data.get("changes", [])
if not isinstance(changes, list):
    print("Model returned invalid 'changes' (not a list); aborting.", file=sys.stderr)
    sys.exit(0)

# ---- APPLY CHANGES ----
def allowed_by_globs(p: pathlib.Path) -> bool:
    # allow if path matches any allowlist glob
    for g in ALLOW_FILE_GLOBS:
        if p.match(g) or p.as_posix().startswith(g.rstrip("*")):
            return True
    return False

applied = 0
for ch in changes:
    path = ch.get("path")
    content = ch.get("content")
    if not isinstance(path, str) or not isinstance(content, str):
        continue
    p = pathlib.Path(path)

    # enforce allowlist
    if not allowed_by_globs(p):
        continue

    # create dirs if needed (honor policy allow_creates)
    if not p.exists():
        if not policy.get("allow_creates", True):
            continue
        p.parent.mkdir(parents=True, exist_ok=True)

    # normalize line endings
    content = content.replace("\r\n", "\n")
    p.write_text(content, encoding="utf-8")
    applied += 1

print(f"Applied changes to {applied} file(s).")
# Exit 0 even if no changes; later steps will detect no diff.
